"""
Spatial building blocks for the upscaling diffusion U-Net (Model S).
No temporal attention — frame-by-frame processing.
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
    """Sinusoidal encoding + 2-layer MLP. Output: (B, dim)."""

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


# ── Residual block ────────────────────────────────────────────────────────────

class DiffResBlock(nn.Module):
    """Pre-norm residual conv block with timestep injection."""

    def __init__(self, channels: int, emb_dim: int):
        super().__init__()
        self.norm1    = _norm(channels)
        self.conv1    = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2    = _norm(channels)
        self.conv2    = nn.Conv2d(channels, channels, 3, padding=1)
        self.act      = nn.SiLU()
        self.emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, channels))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h  = self.act(self.norm1(x))
        h  = self.conv1(h) + self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        h  = self.act(self.norm2(h))
        h  = self.conv2(h)
        return x + h


# ── Encoder / Decoder blocks ──────────────────────────────────────────────────

class DiffDownBlock(nn.Module):
    """2× strided conv downsample + 2 residual blocks."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.res1 = DiffResBlock(out_ch, emb_dim)
        self.res2 = DiffResBlock(out_ch, emb_dim)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        return self.res2(self.res1(self.down(x), emb), emb)


class DiffUpBlock(nn.Module):
    """Bilinear 2× upsample + skip concat + fuse + 2 residual blocks."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.up   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, 1)
        self.res1 = DiffResBlock(out_ch, emb_dim)
        self.res2 = DiffResBlock(out_ch, emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.fuse(torch.cat([self.up(x), skip], dim=1))
        return self.res2(self.res1(x, emb), emb)
