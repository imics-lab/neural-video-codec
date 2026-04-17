"""
Restorer — inference wrapper around RestoreUNet + GaussianDiffusion (Model R).

Handles:
  - Checkpoint loading
  - Sliding-window temporal batching (window size T, step 1)
  - Tiled DDIM for large frames
  - Gaussian-blended tile reassembly
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ._network import RestoreUNet
from ._diffusion import GaussianDiffusion


# ── Frame ↔ tensor helpers ────────────────────────────────────────────────────

def _to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 (H,W,3) → RGB float32 (3,H,W) in [0,1]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)


def _to_frame(t: torch.Tensor) -> np.ndarray:
    """Float32 (3,H,W) in [0,1] → BGR uint8 (H,W,3)."""
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ── Gaussian tile blending ────────────────────────────────────────────────────

def _gaussian_weight(h: int, w: int) -> torch.Tensor:
    def _g(n: int) -> torch.Tensor:
        c = torch.arange(n, dtype=torch.float32) - n / 2.0
        g = torch.exp(-(c ** 2) / (2 * (n / 6.0) ** 2))
        return (g / g.max()).clamp(min=1e-6)
    return (_g(h).unsqueeze(1) * _g(w).unsqueeze(0))


# ── Restorer ──────────────────────────────────────────────────────────────────

class Restorer:
    """
    Wraps RestoreUNet for inference on a sequence of degraded frames.

    Args:
        config: Restoration config (OmegaConf node or dict).
    """

    def __init__(self, config) -> None:
        self.cfg    = config
        self.device = torch.device(config.device)
        self.T      = int(config.model.temporal_window)

        ckpt_path = Path(config.model.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Restoration checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
            mcfg  = ckpt.get("model_cfg", {})
            T_ckpt = ckpt.get("model_cfg", {}).get("temporal_window", self.T)
        else:
            state = ckpt
            mcfg  = {}
            T_ckpt = self.T

        # Strip DataParallel 'module.' prefix if present
        if any(k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}

        self.model = RestoreUNet(
            base_channels=mcfg.get("base_channels",    config.model.base_channels),
            encoder_channels=tuple(mcfg.get("encoder_channels", config.model.encoder_channels)),
            num_res_blocks=mcfg.get("num_res_blocks",  config.model.num_res_blocks),
            T=T_ckpt,
            n_heads=mcfg.get("n_heads", config.model.n_heads),
        )
        self.model.load_state_dict(state)
        self.model.to(self.device).half().eval()

        T_diff = getattr(config.model, "timesteps", 1000)
        self.diffusion = GaussianDiffusion(T=T_diff)

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def restore_sequence(self, degraded_frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Restore a list of degraded BGR frames.

        Uses a sliding window of size T (stride 1).  Boundary frames are padded
        by replicating the first/last frame.

        Args:
            degraded_frames: List of BGR uint8 arrays all the same spatial size.

        Returns:
            List of restored BGR uint8 arrays (same length and size).
        """
        N = len(degraded_frames)
        if N == 0:
            return []

        # Pre-convert all frames to tensors (fp16 to match model)
        deg_tensors = [_to_tensor(f).to(self.device, dtype=torch.float16) for f in degraded_frames]

        T          = self.T
        half       = T // 2
        _, H, W    = deg_tensors[0].shape
        batch_size = int(getattr(self.cfg.inference, 'batch_size', 1))
        cfg        = self.cfg
        tile_sz    = cfg.inference.tile_size
        restored: List[Optional[torch.Tensor]] = [None] * N

        import time as _time
        t0 = _time.perf_counter()
        processed = 0

        for batch_start in range(0, N, batch_size):
            centres = list(range(batch_start, min(batch_start + batch_size, N)))
            B = len(centres)

            # Build windows and stack: (B*T, 3, H, W)
            windows = []
            for centre in centres:
                idxs = [max(0, min(N - 1, centre + k - half)) for k in range(T)]
                windows.append(torch.stack([deg_tensors[i] for i in idxs], dim=0))
            cond = torch.stack(windows, dim=0).view(B * T, 3, H, W)

            if tile_sz > 0 and (H > tile_sz or W > tile_sz):
                # Tiled path: fall back to per-frame to keep memory bounded
                for i, centre in enumerate(centres):
                    restored[centre] = self._restore_window(windows[i], half)
            else:
                t_s  = cfg.inference.t_start
                ab_s = self.diffusion.alpha_bar[t_s].to(self.device)
                noise = torch.randn_like(cond)
                x = ab_s.sqrt() * cond + (1.0 - ab_s).sqrt() * noise

                x_out = self._denoise_full(x, cond)          # (B*T, 3, H, W)
                x_out = x_out.view(B, T, 3, H, W)
                cond_b = cond.view(B, T, 3, H, W)

                for i, centre in enumerate(centres):
                    out = x_out[i, half]
                    if cfg.inference.color_fix:
                        dm  = cond_b[i, half].mean(dim=[1, 2], keepdim=True)
                        om  = out.mean(dim=[1, 2], keepdim=True)
                        out = (out - om + dm).clamp(0.0, 1.0)
                    restored[centre] = out

            processed += B
            elapsed = _time.perf_counter() - t0
            fps = processed / elapsed
            eta = (N - processed) / fps if fps > 0 else 0
            print(f"[restore] {processed}/{N} frames  {fps:.2f} fps  ETA {eta:.0f}s",
                  flush=True)

        return [_to_frame(t) for t in restored if t is not None]

    # ── Internal ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _restore_window(self, window: torch.Tensor, centre_idx: int) -> torch.Tensor:
        """
        Denoise a T-frame window and return the centre frame.

        Args:
            window:      (T, 3, H, W) degraded frames in [0, 1], on device.
            centre_idx:  Index of the centre frame within the window.

        Returns:
            (3, H, W) restored centre frame in [0, 1].
        """
        T, _, H, W = window.shape
        cfg = self.cfg

        # Build input: for each frame in window, x_t starts at t_start noised
        # version of the degraded frame; degraded is the conditioning signal.
        # We add noise to ALL frames at t_start to initialise the reverse pass.
        t_start  = cfg.inference.t_start
        ab_s     = self.diffusion.alpha_bar[t_start].to(self.device)
        noise    = torch.randn(T, 3, H, W, device=self.device)
        x        = ab_s.sqrt() * window + (1.0 - ab_s).sqrt() * noise  # (T, 3, H, W)

        tile_sz  = cfg.inference.tile_size
        overlap  = cfg.inference.tile_overlap

        if tile_sz > 0 and (H > tile_sz or W > tile_sz):
            x_restored = self._tiled_denoise(x, window, tile_sz, overlap)
        else:
            x_restored = self._denoise_full(x, window)  # (T, 3, H, W)

        centre = x_restored[centre_idx]  # (3, H, W)

        if cfg.inference.color_fix:
            deg_mean  = window[centre_idx].mean(dim=[1, 2], keepdim=True)
            out_mean  = centre.mean(dim=[1, 2], keepdim=True)
            centre    = (centre - out_mean + deg_mean).clamp(0.0, 1.0)

        return centre

    @torch.no_grad()
    def _denoise_full(self, x_init: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Run DDIM on the full frame (no tiling). Returns (T, 3, H, W)."""
        T, _, H, W = x_init.shape
        cfg = self.cfg
        ab  = self.diffusion.alpha_bar.to(self.device)
        t_start = cfg.inference.t_start
        steps   = cfg.inference.ddim_steps
        step_idx = torch.linspace(t_start, 0, steps + 1).long()

        x = x_init  # (T, 3, H, W)
        for i in range(steps):
            t_cur  = int(step_idx[i].item())
            t_next = int(step_idx[i + 1].item())
            t_batch = torch.full((T,), t_cur, device=self.device, dtype=torch.long)

            inp     = torch.cat([x, cond], dim=1)       # (T, 6, H, W)
            with torch.amp.autocast("cuda"):
                eps_hat = self.model(inp, t_batch)      # (T, 3, H, W)
            eps_hat = eps_hat.to(x.dtype)

            ab_t    = ab[t_cur]
            ab_prev = ab[t_next]
            x0_pred = (x - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 2.0)
            x       = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps_hat

        return x.clamp(0.0, 1.0)

    @torch.no_grad()
    def _tiled_denoise(
        self,
        x_init: torch.Tensor,
        cond: torch.Tensor,
        tile_sz: int,
        overlap: int,
    ) -> torch.Tensor:
        """Run DDIM with tiled processing and Gaussian blending. Returns (T, 3, H, W)."""
        T, _, H, W = x_init.shape
        cfg    = self.cfg
        ab     = self.diffusion.alpha_bar.to(self.device)
        t_start  = cfg.inference.t_start
        steps    = cfg.inference.ddim_steps
        step_idx = torch.linspace(t_start, 0, steps + 1).long()

        global_noise = x_init.clone()  # same init per tile for colour consistency

        output = torch.zeros(T, 3, H, W, device=self.device)
        weight = torch.zeros(T, 1, H, W, device=self.device)
        step   = max(tile_sz - overlap, 1)

        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                y1 = min(y0 + tile_sz, H)
                x1 = min(x0 + tile_sz, W)
                ya = max(0, y1 - tile_sz)
                xa = max(0, x1 - tile_sz)

                x_tile    = global_noise[:, :, ya:y1, xa:x1]
                cond_tile = cond[:, :, ya:y1, xa:x1]

                tile_out = self._denoise_full(x_tile, cond_tile)  # (T, 3, th, tw)

                if cfg.inference.color_fix:
                    for ti in range(T):
                        cm = cond_tile[ti].mean(dim=[1, 2], keepdim=True)
                        om = tile_out[ti].mean(dim=[1, 2], keepdim=True)
                        tile_out[ti] = (tile_out[ti] - om + cm).clamp(0.0, 1.0)

                th, tw = tile_out.shape[2], tile_out.shape[3]
                blend  = _gaussian_weight(th, tw).to(self.device)  # (th, tw)
                blend  = blend.unsqueeze(0).unsqueeze(0)            # (1, 1, th, tw)

                output[:, :, ya:y1, xa:x1] += tile_out * blend
                weight[:, :, ya:y1, xa:x1] += blend

        return (output / weight.clamp(min=1e-6)).clamp(0.0, 1.0)
