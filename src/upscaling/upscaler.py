"""
Upscaler — core inference class wrapping DiffusionUNet + GaussianDiffusion.

Handles:
  - Checkpoint loading and model construction
  - Single-frame upscaling with tiled DDIM (img2img + color_fix)
  - Gaussian-blended tile reassembly
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from ._network import DiffusionUNet
from ._diffusion import GaussianDiffusion


# ---------------------------------------------------------------------------
# Gaussian blending helpers
# ---------------------------------------------------------------------------

def _gaussian_window(size: int, sigma: Optional[float] = None) -> torch.Tensor:
    if sigma is None:
        sigma = size / 6.0
    coords = torch.arange(size, dtype=torch.float32) - size / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.max()


def _blend_weight(h: int, w: int) -> torch.Tensor:
    """2-D Gaussian weight map (H, W)."""
    gh = _gaussian_window(h)
    gw = _gaussian_window(w)
    return (gh.unsqueeze(1) * gw.unsqueeze(0)).clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Frame tensor conversion helpers
# ---------------------------------------------------------------------------

def _frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 (H, W, 3) → RGB float32 (3, H, W) in [0, 1]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)


def _tensor_to_frame(t: torch.Tensor) -> np.ndarray:
    """Float32 tensor (3, H, W) in [0, 1] → BGR uint8 (H, W, 3)."""
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Upscaler
# ---------------------------------------------------------------------------

class Upscaler:
    """
    Wraps DiffusionUNet + GaussianDiffusion for single-frame HR inference.

    Args:
        config: UpscalingConfig dataclass (from src.pipeline.config_schema).
    """

    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device(config.device)

        ckpt_path = Path(config.model.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {ckpt_path}\n"
                "Download instructions: see models/README.md"
            )

        logger.info(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)

        # Support both bare state-dict and wrapped {"model": ..., "model_cfg": ...}
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
            mcfg = ckpt.get("model_cfg", {})
            T = ckpt.get("diffusion_cfg", {}).get("timesteps", config.model.timesteps)
        else:
            state_dict = ckpt
            mcfg = {}
            T = config.model.timesteps

        self.model = DiffusionUNet(
            base_channels=mcfg.get("base_channels", config.model.base_channels),
            encoder_channels=tuple(
                mcfg.get("encoder_channels", config.model.encoder_channels)
            ),
            num_res_blocks=mcfg.get("num_res_blocks", config.model.num_res_blocks),
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        self.diffusion = GaussianDiffusion(T=T)
        logger.info(
            f"Upscaler ready | scale={config.scale}x | T={T} | "
            f"ddim_steps={config.inference.ddim_steps} | "
            f"t_start={config.inference.t_start} | device={self.device}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def upscale_frame(
        self,
        frame_bgr: np.ndarray,
        weight_mask: np.ndarray,   # (H_lr, W_lr) float32
    ) -> np.ndarray:
        """
        Upscale one LR BGR frame to HR.

        Steps:
          1. Bicubic upsample LR frame by config.scale.
          2. Run tiled DDIM with img2img (t_start) and optional color_fix.
          3. Return HR BGR uint8.

        Args:
            frame_bgr:   LR input frame (H, W, 3) BGR uint8.
            weight_mask: ROI weight mask at LR resolution (H, W) float32.

        Returns:
            HR frame (H*scale, W*scale, 3) BGR uint8.
        """
        cfg = self.config
        frame_t = _frame_to_tensor(frame_bgr).unsqueeze(0).to(self.device)
        mask_t = (
            torch.from_numpy(weight_mask)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        lr_up = F.interpolate(
            frame_t, scale_factor=cfg.scale, mode="bilinear", align_corners=False
        )
        wm_hr = F.interpolate(mask_t, scale_factor=cfg.scale, mode="nearest")

        H, W = lr_up.shape[2], lr_up.shape[3]
        tile_size = cfg.inference.tile_size
        overlap = cfg.inference.tile_overlap

        if tile_size > 0 and (H > tile_size or W > tile_size):
            out = self._tiled_ddim(lr_up, wm_hr)
        else:
            out = self.diffusion.ddim_sample(
                self.model,
                lr_up,
                wm_hr,
                steps=cfg.inference.ddim_steps,
                device=self.device,
                t_start=cfg.inference.t_start,
            )
            if cfg.inference.color_fix:
                lr_mean = lr_up.mean(dim=[2, 3], keepdim=True)
                out_mean = out.mean(dim=[2, 3], keepdim=True)
                out = (out - out_mean + lr_mean).clamp(0.0, 1.0)

        return _tensor_to_frame(out)

    # ------------------------------------------------------------------
    # Internal tiled DDIM
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _tiled_ddim(
        self,
        lr_up: torch.Tensor,    # (1, 3, H, W) at HR resolution
        wm_hr: torch.Tensor,    # (1, 1, H, W) at HR resolution
    ) -> torch.Tensor:
        """
        Process a large HR frame in overlapping tiles with Gaussian blending.

        Two mechanisms prevent the tiled-colour-shift artifact:
        1. Img2img mode (t_start): each tile begins from a noised version of
           the LR upsampled tile, anchoring colour structure.
        2. Per-tile colour correction (color_fix): tile channel means shifted
           to match LR upsampled tile means.

        Returns:
            (1, 3, H, W) in [0, 1].
        """
        cfg = self.config
        device = self.device
        tile_size = cfg.inference.tile_size
        overlap = cfg.inference.tile_overlap

        _, _, H, W = lr_up.shape

        # Global noise map for spatial consistency across tile boundaries
        global_noise = torch.randn(1, 3, H, W, device=device)

        output = torch.zeros(1, 3, H, W, device=device)
        weight = torch.zeros(1, 1, H, W, device=device)

        step = max(tile_size - overlap, 1)

        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                y1 = min(y0 + tile_size, H)
                x1 = min(x0 + tile_size, W)
                ya = max(0, y1 - tile_size)
                xa = max(0, x1 - tile_size)

                lr_tile   = lr_up[:, :, ya:y1, xa:x1]
                wm_tile   = wm_hr[:, :, ya:y1, xa:x1]
                init_tile = global_noise[:, :, ya:y1, xa:x1]

                tile_out = self.diffusion.ddim_sample(
                    self.model,
                    lr_tile,
                    wm_tile,
                    steps=cfg.inference.ddim_steps,
                    device=device,
                    init_noise=init_tile,
                    t_start=cfg.inference.t_start,
                )  # (1, 3, th, tw)

                # Per-tile colour correction: shift output means to match LR means.
                if cfg.inference.color_fix:
                    lr_mean  = lr_tile.mean(dim=[2, 3], keepdim=True)
                    out_mean = tile_out.mean(dim=[2, 3], keepdim=True)
                    tile_out = (tile_out - out_mean + lr_mean).clamp(0.0, 1.0)

                th, tw = tile_out.shape[2], tile_out.shape[3]
                blend = _blend_weight(th, tw).to(device).unsqueeze(0).unsqueeze(0)

                output[:, :, ya:y1, xa:x1] += tile_out * blend
                weight[:, :, ya:y1, xa:x1] += blend

        return (output / weight.clamp(min=1e-6)).clamp(0.0, 1.0)
