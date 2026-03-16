"""
Timestep-conditioned building blocks for the diffusion U-Net.

These are separate from blocks.py so the existing deterministic model
is not disturbed.
"""

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
    """
    Sinusoidal positional encoding over diffusion timestep t,
    followed by a 2-layer MLP.

    Output shape: (B, dim)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) long integer timesteps → (B, dim) float embedding."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )  # (half,)
        x = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        emb = torch.cat([x.sin(), x.cos()], dim=1)        # (B, dim)
        return self.mlp(emb)


# ── Residual block ────────────────────────────────────────────────────────────

class DiffResBlock(nn.Module):
    """
    Pre-norm residual block with timestep embedding injection.

    The embedding is projected to `channels` and added after the first
    conv (standard DDPM-style AdaGN-free conditioning).
    """

    def __init__(self, channels: int, emb_dim: int):
        super().__init__()
        self.norm1 = _norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = _norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU()
        # Project timestep embedding → channels (applied after first conv)
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, channels))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


# ── Encoder / Decoder blocks ──────────────────────────────────────────────────

class DiffDownBlock(nn.Module):
    """Encoder stage: 2× strided conv + 2 DiffResBlocks."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.res1 = DiffResBlock(out_ch, emb_dim)
        self.res2 = DiffResBlock(out_ch, emb_dim)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.res1(x, emb)
        x = self.res2(x, emb)
        return x


class DiffUpBlock(nn.Module):
    """Decoder stage: bilinear 2× upsample → concat skip → fuse conv → 2 DiffResBlocks."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        # 1×1 conv to fuse the concatenated (upsampled + skip) features
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, 1)
        self.res1 = DiffResBlock(out_ch, emb_dim)
        self.res2 = DiffResBlock(out_ch, emb_dim)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        x = self.up(x)
        x = self.fuse(torch.cat([x, skip], dim=1))
        x = self.res1(x, emb)
        x = self.res2(x, emb)
        return x
