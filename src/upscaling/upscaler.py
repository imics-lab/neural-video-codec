"""
Upscaler — inference wrapper around SRUNet + GaussianDiffusion (Model S).

Handles:
  - Checkpoint loading and model construction
  - Per-frame 2× DDIM upscaling with tiled inference
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

from ._network import SRUNet
from ._diffusion import GaussianDiffusion


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)


def _to_frame(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _gaussian_weight(h: int, w: int) -> torch.Tensor:
    def _g(n: int) -> torch.Tensor:
        c = torch.arange(n, dtype=torch.float32) - n / 2.0
        return (torch.exp(-(c ** 2) / (2 * (n / 6.0) ** 2)) / 1.0).clamp(min=1e-6)
    return _g(h).unsqueeze(1) * _g(w).unsqueeze(0)


# ── Upscaler ──────────────────────────────────────────────────────────────────

class Upscaler:
    """
    Wraps SRUNet for single-frame 2× super-resolution inference.

    Args:
        config: Upscaling config (OmegaConf node or dict).
    """

    def __init__(self, config) -> None:
        self.cfg    = config
        self.device = torch.device(config.device)
        self.scale  = int(config.scale)

        ckpt_path = Path(config.model.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SR checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
            mcfg  = ckpt.get("model_cfg", {})
            T_ckpt = ckpt.get("diffusion_cfg", {}).get("timesteps", config.model.timesteps)
        else:
            state = ckpt
            mcfg  = {}
            T_ckpt = config.model.timesteps

        self.model = SRUNet(
            base_channels=mcfg.get("base_channels",    config.model.base_channels),
            encoder_channels=tuple(mcfg.get("encoder_channels", config.model.encoder_channels)),
            num_res_blocks=mcfg.get("num_res_blocks",  config.model.num_res_blocks),
        )
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.diffusion = GaussianDiffusion(T=T_ckpt)

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def upscale_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        2× upscale a single BGR frame.

        Args:
            frame_bgr: LR BGR uint8 (H, W, 3).

        Returns:
            HR BGR uint8 (H*scale, W*scale, 3).
        """
        cfg = self.cfg
        lr  = _to_tensor(frame_bgr).unsqueeze(0).to(self.device)   # (1,3,H,W)

        # Bicubic upsample LR → HR size (conditioning signal for the network)
        lr_up = F.interpolate(lr, scale_factor=self.scale,
                              mode="bilinear", align_corners=False)  # (1,3,H*s,W*s)
        _, _, H, W = lr_up.shape

        tile_sz = cfg.inference.tile_size
        overlap = cfg.inference.tile_overlap

        if tile_sz > 0 and (H > tile_sz or W > tile_sz):
            out = self._tiled_ddim(lr_up)
        else:
            out = self._ddim_full(lr_up)

        if cfg.inference.color_fix:
            lr_mean  = lr_up.mean(dim=[2, 3], keepdim=True)
            out_mean = out.mean(dim=[2, 3], keepdim=True)
            out      = (out - out_mean + lr_mean).clamp(0.0, 1.0)

        return _to_frame(out)

    # ── Internal ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _ddim_full(self, lr_up: torch.Tensor) -> torch.Tensor:
        """Run DDIM on the full HR frame. Returns (1, 3, H, W)."""
        B, _, H, W = lr_up.shape
        cfg = self.cfg
        ab  = self.diffusion.alpha_bar.to(self.device)

        t_start  = cfg.inference.t_start
        steps    = cfg.inference.ddim_steps
        step_idx = torch.linspace(t_start, 0, steps + 1).long()

        ab_s  = ab[t_start]
        noise = torch.randn(B, 3, H, W, device=self.device)
        x     = ab_s.sqrt() * lr_up + (1.0 - ab_s).sqrt() * noise

        for i in range(steps):
            t_cur  = int(step_idx[i].item())
            t_next = int(step_idx[i + 1].item())
            t_b    = torch.full((B,), t_cur, device=self.device, dtype=torch.long)

            inp     = torch.cat([x, lr_up], dim=1)   # (B, 6, H, W)
            eps_hat = self.model(inp, t_b)

            ab_t    = ab[t_cur]
            ab_prev = ab[t_next]
            x0_pred = (x - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 2.0)
            x       = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps_hat

        return x.clamp(0.0, 1.0)

    @torch.no_grad()
    def _tiled_ddim(self, lr_up: torch.Tensor) -> torch.Tensor:
        """Process large HR frame in Gaussian-blended tiles. Returns (1, 3, H, W)."""
        cfg     = self.cfg
        tile_sz = cfg.inference.tile_size
        overlap = cfg.inference.tile_overlap
        _, _, H, W = lr_up.shape

        global_noise = torch.randn(1, 3, H, W, device=self.device)
        output = torch.zeros(1, 3, H, W, device=self.device)
        weight = torch.zeros(1, 1, H, W, device=self.device)
        step   = max(tile_sz - overlap, 1)

        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                y1 = min(y0 + tile_sz, H)
                x1 = min(x0 + tile_sz, W)
                ya = max(0, y1 - tile_sz)
                xa = max(0, x1 - tile_sz)

                lr_tile   = lr_up[:, :, ya:y1, xa:x1]
                init_tile = global_noise[:, :, ya:y1, xa:x1]

                # Initialise tile from global noise for colour consistency
                ab_s  = self.diffusion.alpha_bar[cfg.inference.t_start].to(self.device)
                x_tile = ab_s.sqrt() * lr_tile + (1 - ab_s).sqrt() * init_tile
                tile_out = self._ddim_tile(x_tile, lr_tile)

                if cfg.inference.color_fix:
                    lm = lr_tile.mean(dim=[2, 3], keepdim=True)
                    om = tile_out.mean(dim=[2, 3], keepdim=True)
                    tile_out = (tile_out - om + lm).clamp(0.0, 1.0)

                th, tw = tile_out.shape[2], tile_out.shape[3]
                blend  = _gaussian_weight(th, tw).to(self.device).unsqueeze(0).unsqueeze(0)

                output[:, :, ya:y1, xa:x1] += tile_out * blend
                weight[:, :, ya:y1, xa:x1] += blend

        return (output / weight.clamp(min=1e-6)).clamp(0.0, 1.0)

    @torch.no_grad()
    def _ddim_tile(self, x_init: torch.Tensor, lr_tile: torch.Tensor) -> torch.Tensor:
        """Run DDIM steps on a single pre-initialised tile."""
        cfg = self.cfg
        ab  = self.diffusion.alpha_bar.to(self.device)
        B   = x_init.shape[0]
        t_start  = cfg.inference.t_start
        steps    = cfg.inference.ddim_steps
        step_idx = torch.linspace(t_start, 0, steps + 1).long()
        x = x_init
        for i in range(steps):
            t_cur  = int(step_idx[i].item())
            t_next = int(step_idx[i + 1].item())
            t_b    = torch.full((B,), t_cur, device=self.device, dtype=torch.long)
            inp     = torch.cat([x, lr_tile], dim=1)
            eps_hat = self.model(inp, t_b)
            ab_t    = ab[t_cur]
            ab_prev = ab[t_next]
            x0_pred = (x - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 2.0)
            x       = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps_hat
        return x.clamp(0.0, 1.0)
