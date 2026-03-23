"""
RestoreUNet — temporal-attention diffusion U-Net for DCVC artifact removal (Model R).

Architecture overview
─────────────────────
Processes a sliding window of T consecutive frames simultaneously.
Temporal self-attention (AnimateDiff-style) is added after each encoder/decoder
stage so each frame's features are contextualised by its neighbours.

Tensor convention
─────────────────
• Spatial processing runs on (B*T, C, H, W).
• TemporalAttention reshapes to (B*H*W, T, C), attends, then reshapes back.
• The forward pass receives the full window; only the centre frame's noise
  prediction is used at inference (index T//2).

Input channels per frame:  x_t(3) + degraded(3) = 6
  x_t       — noisy version of the clean HR frame at diffusion step t
  degraded  — DCVC-decompressed (artifact-laden) frame (conditioning signal)

Output: (B*T, 3, H, W) predicted noise ε̂  (caller slices [centre] for inference)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ._blocks import (
    RestoreDownBlock,
    RestoreResBlock,
    RestoreUpBlock,
    TemporalAttention,
    TimestepEmbedding,
    _norm,
)


class RestoreUNet(nn.Module):
    """
    Timestep-conditioned, temporal-attention U-Net for artifact removal.

    Args:
        base_channels:    Feature width after the stem conv.
        encoder_channels: Channel widths for the 3 encoder stages.
        num_res_blocks:   Number of spatial res-blocks in the bottleneck.
        T:                Temporal window size (must be odd; centre frame = T//2).
        n_heads:          Number of temporal attention heads.
    """

    IN_CHANNELS = 6   # per frame: x_t(3) + degraded(3)

    def __init__(
        self,
        base_channels:    int             = 64,
        encoder_channels: tuple[int, ...] = (128, 256, 512),
        num_res_blocks:   int             = 4,
        T:                int             = 3,
        n_heads:          int             = 4,
    ):
        super().__init__()
        assert T % 2 == 1, "T must be odd so the centre frame is well-defined"
        self.T   = T
        ec       = list(encoder_channels)
        emb_dim  = base_channels * 4

        # ── Timestep embedding ────────────────────────────────────────────────
        # t is broadcast over all T frames in the window.
        self.time_emb = TimestepEmbedding(emb_dim)

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(self.IN_CHANNELS, base_channels, 3, padding=1),
            nn.SiLU(),
        )
        # Temporal attention on stem features
        self.stem_temp = TemporalAttention(base_channels, T=T, n_heads=n_heads)

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc0 = RestoreDownBlock(base_channels, ec[0], emb_dim, T=T, n_heads=n_heads)
        self.enc1 = RestoreDownBlock(ec[0],         ec[1], emb_dim, T=T, n_heads=n_heads)
        self.enc2 = RestoreDownBlock(ec[1],         ec[2], emb_dim, T=T, n_heads=n_heads)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.ModuleList(
            [RestoreResBlock(ec[2], emb_dim) for _ in range(num_res_blocks)]
        )
        self.bottle_temp = TemporalAttention(ec[2], T=T, n_heads=n_heads)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec0 = RestoreUpBlock(ec[2], ec[1], ec[1], emb_dim, T=T, n_heads=n_heads)
        self.dec1 = RestoreUpBlock(ec[1], ec[0], ec[0], emb_dim, T=T, n_heads=n_heads)
        self.dec2 = RestoreUpBlock(ec[0], base_channels, base_channels, emb_dim, T=T, n_heads=n_heads)

        # ── Output head ───────────────────────────────────────────────────────
        self.out_proj = nn.Sequential(
            _norm(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*T, 6, H, W)  — [x_t, degraded] for each frame in the window
            t: (B*T,) long      — diffusion timestep (same value broadcast over T)

        Returns:
            (B*T, 3, H, W)  predicted noise ε̂ for every frame in the window
        """
        emb = self.time_emb(t)       # (B*T, emb_dim)

        s  = self.stem_temp(self.stem(x))   # (B*T, base, H,   W)
        e0 = self.enc0(s,  emb)             # (B*T, ec0,  H/2, W/2)
        e1 = self.enc1(e0, emb)             # (B*T, ec1,  H/4, W/4)
        e2 = self.enc2(e1, emb)             # (B*T, ec2,  H/8, W/8)

        b = e2
        for blk in self.bottleneck:
            b = blk(b, emb)
        b = self.bottle_temp(b)             # (B*T, ec2,  H/8, W/8)

        d0 = self.dec0(b,  e1, emb)         # (B*T, ec1,  H/4, W/4)
        d1 = self.dec1(d0, e0, emb)         # (B*T, ec0,  H/2, W/2)
        d2 = self.dec2(d1, s,  emb)         # (B*T, base, H,   W)

        return self.out_proj(d2)             # (B*T, 3,    H,   W)


def build_restore_model(cfg) -> RestoreUNet:
    """Construct RestoreUNet from an OmegaConf/dict config node."""
    m = cfg.model
    return RestoreUNet(
        base_channels=m.base_channels,
        encoder_channels=tuple(m.encoder_channels),
        num_res_blocks=m.num_res_blocks,
        T=m.temporal_window,
        n_heads=m.n_heads,
    )
