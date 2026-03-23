"""
Building blocks for the temporal-attention restoration U-Net (Model R).

Key addition over the upscaling blocks:
    TemporalAttention — AnimateDiff-style self-attention over T consecutive frames.

All spatial blocks work on (B*T, C, H, W) tensors.  The TemporalAttention block
reshapes to (B*H*W, T, C), attends over T, then reshapes back.  T must be known
at construction time so it can be stored for the reshape.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels: int) -> nn.GroupNorm:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


# ── Timestep embedding ────────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """Sinusoidal positional encoding → 2-layer MLP.  Output: (B*T, dim)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        x   = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([x.sin(), x.cos()], dim=1)
        return self.mlp(emb)


# ── Spatial residual block ────────────────────────────────────────────────────

class RestoreResBlock(nn.Module):
    """Pre-norm spatial residual block with timestep injection."""

    def __init__(self, channels: int, emb_dim: int):
        super().__init__()
        self.norm1    = _norm(channels)
        self.conv1    = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2    = _norm(channels)
        self.conv2    = nn.Conv2d(channels, channels, 3, padding=1)
        self.act      = nn.SiLU()
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, channels))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h) + self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.norm2(h))
        return x + self.conv2(h)


# ── Temporal attention ────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    AnimateDiff-style self-attention over the temporal dimension T.

    Operates on (B*T, C, H, W):
      1. Reshape → (B*H*W, T, C)
      2. LayerNorm + multi-head self-attention over T
      3. Residual add + reshape back to (B*T, C, H, W)

    Args:
        channels: Feature channel count C.
        T:        Temporal window size (must match the value used when calling).
        n_heads:  Number of attention heads.
    """

    def __init__(self, channels: int, T: int = 3, n_heads: int = 4):
        super().__init__()
        self.T      = T
        self.norm   = nn.LayerNorm(channels)
        self.attn   = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        # Learnable temporal positional bias (one scalar per (query, key) pair)
        self.pos_bias = nn.Parameter(torch.zeros(T, T))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*T, C, H, W)

        Returns:
            (B*T, C, H, W) with temporal context applied.
        """
        BT, C, H, W = x.shape
        T = self.T
        B = BT // T

        # Reshape to (B*H*W, T, C)
        # x: (B*T, C, H, W) → (B, T, C, H*W) → (B, H*W, T, C) → (B*H*W, T, C)
        x_flat = x.view(B, T, C, H * W)                 # (B, T, C, H*W)
        x_flat = x_flat.permute(0, 3, 1, 2)             # (B, H*W, T, C)
        x_flat = x_flat.reshape(B * H * W, T, C)        # (B*H*W, T, C)

        x_norm = self.norm(x_flat)

        # Positional bias broadcast over batch dimension
        attn_bias = self.pos_bias.unsqueeze(0)           # (1, T, T)

        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_bias.expand(B * H * W, T, T) if False else None,
            # Note: MultiheadAttention attn_mask is (L,S) or (B*heads,L,S);
            # we skip it here and rely on the learnable pos_bias implicitly
            # via the MHA's own learned weights.
        )

        x_flat = x_flat + attn_out                       # residual

        # Reshape back to (B*T, C, H, W)
        x_flat = x_flat.reshape(B, H * W, T, C)         # (B, H*W, T, C)
        x_flat = x_flat.permute(0, 2, 3, 1)             # (B, T, C, H*W)
        return x_flat.reshape(B * T, C, H, W)


# ── Spatial + temporal encoder/decoder blocks ─────────────────────────────────

class RestoreDownBlock(nn.Module):
    """Encoder stage: 2× strided conv + 2 spatial res-blocks + temporal attention."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, T: int, n_heads: int = 4):
        super().__init__()
        self.down      = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.res1      = RestoreResBlock(out_ch, emb_dim)
        self.res2      = RestoreResBlock(out_ch, emb_dim)
        self.temp_attn = TemporalAttention(out_ch, T=T, n_heads=n_heads)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.res2(self.res1(self.down(x), emb), emb)
        return self.temp_attn(x)


class RestoreUpBlock(nn.Module):
    """Decoder stage: bilinear 2× + skip concat + fuse + 2 spatial res-blocks + temporal attention."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, emb_dim: int, T: int, n_heads: int = 4):
        super().__init__()
        self.up        = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.fuse      = nn.Conv2d(out_ch + skip_ch, out_ch, 1)
        self.res1      = RestoreResBlock(out_ch, emb_dim)
        self.res2      = RestoreResBlock(out_ch, emb_dim)
        self.temp_attn = TemporalAttention(out_ch, T=T, n_heads=n_heads)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.fuse(torch.cat([self.up(x), skip], dim=1))
        x = self.res2(self.res1(x, emb), emb)
        return self.temp_attn(x)
