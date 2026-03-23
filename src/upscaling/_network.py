"""
SRUNet — 2× super-resolution diffusion U-Net (Model S).

Input : (B, 6, H_hr, W_hr)  =  [x_t (3), lr_bicubic_up (3)]
         The LR frame is bicubic-upsampled to HR size before concatenation.
         No ROI mask channel (full-frame SR).
Output: (B, 3, H_hr, W_hr)  =  predicted noise ε̂

Architecture is identical to the reference DiffusionUNet but:
- IN_CHANNELS = 6  (no ROI mask)
- Upsampling is handled externally (the caller bicubic-upsamples LR → HR size).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ._blocks import DiffDownBlock, DiffResBlock, DiffUpBlock, TimestepEmbedding, _norm


class SRUNet(nn.Module):
    """
    Timestep-conditioned U-Net for 2× super-resolution.

    Args:
        base_channels:    Feature width after stem conv.
        encoder_channels: Channel widths for the 3 encoder stages.
        num_res_blocks:   Number of DiffResBlocks in the bottleneck.
    """

    IN_CHANNELS = 6  # x_t(3) + lr_bicubic_up(3)

    def __init__(
        self,
        base_channels:    int            = 64,
        encoder_channels: tuple[int, ...] = (128, 256, 512),
        num_res_blocks:   int            = 4,
    ):
        super().__init__()
        ec      = list(encoder_channels)
        emb_dim = base_channels * 4

        self.time_emb = TimestepEmbedding(emb_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(self.IN_CHANNELS, base_channels, 3, padding=1),
            nn.SiLU(),
        )

        self.enc0 = DiffDownBlock(base_channels, ec[0], emb_dim)
        self.enc1 = DiffDownBlock(ec[0],         ec[1], emb_dim)
        self.enc2 = DiffDownBlock(ec[1],         ec[2], emb_dim)

        self.bottleneck = nn.ModuleList(
            [DiffResBlock(ec[2], emb_dim) for _ in range(num_res_blocks)]
        )

        self.dec0 = DiffUpBlock(ec[2], ec[1], ec[1], emb_dim)
        self.dec1 = DiffUpBlock(ec[1], ec[0], ec[0], emb_dim)
        self.dec2 = DiffUpBlock(ec[0], base_channels, base_channels, emb_dim)

        self.out_proj = nn.Sequential(
            _norm(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, H, W)  — [x_t, lr_bicubic_up] at HR resolution
            t: (B,) long      — diffusion timestep indices

        Returns:
            (B, 3, H, W) predicted noise ε̂
        """
        emb = self.time_emb(t)

        s  = self.stem(x)
        e0 = self.enc0(s,  emb)
        e1 = self.enc1(e0, emb)
        e2 = self.enc2(e1, emb)

        b = e2
        for blk in self.bottleneck:
            b = blk(b, emb)

        d0 = self.dec0(b,  e1, emb)
        d1 = self.dec1(d0, e0, emb)
        d2 = self.dec2(d1, s,  emb)

        return self.out_proj(d2)


def build_sr_model(cfg) -> SRUNet:
    """Construct SRUNet from an OmegaConf/dict config node."""
    m = cfg.model
    return SRUNet(
        base_channels=m.base_channels,
        encoder_channels=tuple(m.encoder_channels),
        num_res_blocks=m.num_res_blocks,
    )
