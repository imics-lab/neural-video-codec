"""
DiffusionUNet — noise-predicting U-Net for conditional image SR via DDPM.

Architecture overview
─────────────────────
Input  : (B, 7, H, W)  =  [x_t (3), lr_upsampled (3), roi_mask (1)]
         All channels are at the *HR* resolution.  The network does NOT
         perform spatial upsampling — the LR frame is pre-upsampled with
         bicubic interpolation before being concatenated.
Timestep: (B,) long integer → sinusoidal embedding → MLP → emb injected
         into every DiffResBlock via a learned linear projection.
Output : (B, 3, H, W)  =  predicted noise ε̂ (same size as x_t / target)

During training the model minimises a ROI-weighted MSE between ε̂ and the
actual noise ε added to the HR frame.  During inference a DDIM reverse
diffusion loop is run (see _diffusion.py).
"""

import torch
import torch.nn as nn

from ._blocks import (
    _norm,
    DiffDownBlock,
    DiffResBlock,
    DiffUpBlock,
    TimestepEmbedding,
)


class DiffusionUNet(nn.Module):
    """
    Timestep-conditioned U-Net noise predictor.

    Args:
        base_channels:    Feature width after the stem conv.
        encoder_channels: Channel widths for the 3 encoder stages.
        num_res_blocks:   Number of DiffResBlocks in the bottleneck.
    """

    # Input channels: x_t(3) + lr_up(3) + roi_mask(1)
    IN_CHANNELS = 7

    def __init__(
        self,
        base_channels: int = 64,
        encoder_channels: tuple[int, ...] = (128, 256, 512),
        num_res_blocks: int = 4,
    ):
        super().__init__()
        ec = list(encoder_channels)
        emb_dim = base_channels * 4   # e.g. 64*4 = 256

        # ── Timestep embedding ────────────────────────────────────────────────
        self.time_emb = TimestepEmbedding(emb_dim)

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(self.IN_CHANNELS, base_channels, 3, padding=1),
            nn.SiLU(),
        )

        # ── Encoder ───────────────────────────────────────────────────────────
        # Each stage halves spatial resolution and grows channel width.
        # Stage 0: base_channels → ec[0]   (skip s  at base_channels)
        # Stage 1: ec[0]         → ec[1]   (skip e0 at ec[0])
        # Stage 2: ec[1]         → ec[2]   (skip e1 at ec[1])
        self.enc0 = DiffDownBlock(base_channels, ec[0], emb_dim)
        self.enc1 = DiffDownBlock(ec[0],         ec[1], emb_dim)
        self.enc2 = DiffDownBlock(ec[1],         ec[2], emb_dim)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.ModuleList(
            [DiffResBlock(ec[2], emb_dim) for _ in range(num_res_blocks)]
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        # Each stage doubles spatial resolution and receives the matching skip.
        # dec0: ec[2] + skip(e1, ec[1])         → ec[1]
        # dec1: ec[1] + skip(e0, ec[0])         → ec[0]
        # dec2: ec[0] + skip(s,  base_channels) → base_channels
        self.dec0 = DiffUpBlock(ec[2], ec[1], ec[1], emb_dim)
        self.dec1 = DiffUpBlock(ec[1], ec[0], ec[0], emb_dim)
        self.dec2 = DiffUpBlock(ec[0], base_channels, base_channels, emb_dim)

        # ── Output ────────────────────────────────────────────────────────────
        # Predict 3-channel noise at full HR resolution.
        self.out_proj = nn.Sequential(
            _norm(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 7, H, W) — [x_t, lr_upsampled, roi_mask] at HR resolution
            t: (B,) long     — diffusion timestep indices

        Returns:
            (B, 3, H, W) — predicted noise ε̂
        """
        emb = self.time_emb(t)     # (B, emb_dim)

        s  = self.stem(x)          # (B, base, H,   W)
        e0 = self.enc0(s,  emb)    # (B, ec0,  H/2, W/2)
        e1 = self.enc1(e0, emb)    # (B, ec1,  H/4, W/4)
        e2 = self.enc2(e1, emb)    # (B, ec2,  H/8, W/8)

        b = e2
        for block in self.bottleneck:
            b = block(b, emb)       # (B, ec2,  H/8, W/8)

        d0 = self.dec0(b,  e1, emb)   # (B, ec1,  H/4, W/4)
        d1 = self.dec1(d0, e0, emb)   # (B, ec0,  H/2, W/2)
        d2 = self.dec2(d1, s,  emb)   # (B, base, H,   W)

        return self.out_proj(d2)       # (B, 3,    H,   W)


def build_diff_model(cfg) -> DiffusionUNet:
    """Instantiate DiffusionUNet from an UpscalingConfig or OmegaConf config node."""
    return DiffusionUNet(
        base_channels=cfg.model.base_channels,
        encoder_channels=tuple(cfg.model.encoder_channels),
        num_res_blocks=cfg.model.num_res_blocks,
    )
