"""
Restorer — inference wrapper around RestoreUNet + GaussianDiffusion (Model R).

Inference protocol mirrors training exactly:
  - Only the CENTRE frame of each T-window is noised (x_t).
  - Model receives T copies of [x_t, cond_frame_i] for i in 0..T-1.
  - Only the centre frame noise prediction (eps[T//2]) is used to update x_t.
  - Inputs/outputs in [-1,1] space (training: x0 = orig * 2 - 1).
  - Optional tiling at training patch size (256×256) via Gaussian blending.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from ._network import RestoreUNet
from ._diffusion import GaussianDiffusion


def _to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 (H,W,3) → RGB float32 (3,H,W) in [0,1]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)


def _to_frame(t: torch.Tensor) -> np.ndarray:
    """Float32 (3,H,W) in [0,1] → BGR uint8 (H,W,3)."""
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _gaussian_weight(h: int, w: int) -> torch.Tensor:
    def _g(n):
        c = torch.arange(n, dtype=torch.float32) - n / 2.0
        g = torch.exp(-(c ** 2) / (2 * (n / 6.0) ** 2))
        return (g / g.max()).clamp(min=1e-6)
    return _g(h).unsqueeze(1) * _g(w).unsqueeze(0)


class Restorer:

    def __init__(self, config) -> None:
        self.cfg    = config
        self.device = torch.device(config.device)
        self.T      = int(config.model.temporal_window)

        ckpt_path = Path(config.model.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Restoration checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state  = ckpt["model"]
            mcfg   = ckpt.get("model_cfg", {})
            T_ckpt = mcfg.get("temporal_window", self.T)
        else:
            state  = ckpt
            mcfg   = {}
            T_ckpt = self.T

        if any(k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}

        self.model = RestoreUNet(
            base_channels    = mcfg.get("base_channels",    config.model.base_channels),
            encoder_channels = tuple(mcfg.get("encoder_channels", config.model.encoder_channels)),
            num_res_blocks   = mcfg.get("num_res_blocks",   config.model.num_res_blocks),
            T                = T_ckpt,
            n_heads          = mcfg.get("n_heads", config.model.n_heads),
        )
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()  # fp32 weights; autocast handles fp16 compute

        T_diff = getattr(config.model, "timesteps", 1000)
        self.diffusion = GaussianDiffusion(T=T_diff)

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def restore_sequence(self, degraded_frames: List[np.ndarray]) -> List[np.ndarray]:
        N = len(degraded_frames)
        if N == 0:
            return []

        T    = self.T
        half = T // 2
        cfg  = self.cfg

        deg_01 = [_to_tensor(f).to(self.device) for f in degraded_frames]

        _, H, W    = deg_01[0].shape
        batch_size = int(getattr(cfg.inference, 'batch_size', 1))
        t_start    = cfg.inference.t_start
        ddim_steps = cfg.inference.ddim_steps
        tile_sz    = int(cfg.inference.tile_size)
        tile_ov    = int(cfg.inference.tile_overlap)

        ab       = self.diffusion.alpha_bar.to(self.device)
        ab_s     = ab[t_start]
        step_idx = torch.linspace(t_start, 0, ddim_steps + 1).long()

        restored: List[Optional[torch.Tensor]] = [None] * N

        import time as _time
        t0 = _time.perf_counter()
        processed = 0

        for batch_start in range(0, N, batch_size):
            centres = list(range(batch_start, min(batch_start + batch_size, N)))
            B = len(centres)

            # Build T-frame windows in [-1,1]
            windows_norm = []   # (B, T, 3, H, W) in [-1,1]
            for c in centres:
                idxs = [max(0, min(N - 1, c + k - half)) for k in range(T)]
                w    = torch.stack([deg_01[i] for i in idxs], dim=0)  # (T,3,H,W)
                windows_norm.append(w * 2.0 - 1.0)

            if tile_sz > 0 and (H > tile_sz or W > tile_sz):
                # ── Tiled path: process 256×256 patches matching training ────
                # Run each window independently (B=1 per tile for simplicity)
                for i, c in enumerate(centres):
                    restored[c] = self._restore_tiled(
                        windows_norm[i],   # (T, 3, H, W)
                        c, ab_s, ab, step_idx, ddim_steps, tile_sz, tile_ov,
                    )
            else:
                # ── Full-frame path ───────────────────────────────────────────
                # Noise only the centre frame of each window
                x_parts = []
                for i, c in enumerate(centres):
                    gen = torch.Generator(device=self.device).manual_seed(c)
                    noise = torch.randn(3, H, W, device=self.device, generator=gen)
                    x_c   = windows_norm[i][half]
                    x_parts.append(ab_s.sqrt() * x_c + (1.0 - ab_s).sqrt() * noise)
                x = torch.stack(x_parts, dim=0)   # (B, 3, H, W)

                x = self._ddim_loop(x, windows_norm, B, T, half, ab, step_idx, ddim_steps)

                x_out = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
                for i, c in enumerate(centres):
                    restored[c] = x_out[i]

            processed += B
            elapsed = _time.perf_counter() - t0
            fps = processed / elapsed
            eta = (N - processed) / fps if fps > 0 else 0
            print(f"[restore] {processed}/{N} frames  {fps:.2f} fps  ETA {eta:.0f}s",
                  flush=True)

        return [_to_frame(t) for t in restored if t is not None]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ddim_loop(
        self,
        x:            torch.Tensor,   # (B, 3, H, W) noised centre frame in [-1,1]
        windows_norm: list,           # list of B tensors (T, 3, h, w) in [-1,1]
        B: int, T: int, half: int,
        ab:       torch.Tensor,
        step_idx: torch.Tensor,
        ddim_steps: int,
    ) -> torch.Tensor:
        """Run DDIM; return denoised x in [-1,1]."""
        _, _, H, W = x.shape
        for si in range(ddim_steps):
            t_cur  = int(step_idx[si].item())
            t_next = int(step_idx[si + 1].item())

            # Build (B*T, 6, H, W): T copies of x[i] with different cond frames
            rows = []
            for i in range(B):
                for ti in range(T):
                    rows.append(torch.cat(
                        [x[i:i+1], windows_norm[i][ti:ti+1]], dim=1))
            model_in = torch.cat(rows, dim=0)
            t_b = torch.full((B * T,), t_cur, device=self.device, dtype=torch.long)

            with torch.amp.autocast("cuda"):
                eps_all = self.model(model_in, t_b)
            eps_all = eps_all.float().view(B, T, 3, H, W)
            eps_c   = eps_all[:, half]              # (B, 3, H, W)

            ab_t    = ab[t_cur]
            ab_prev = ab[t_next]
            x0_pred = (x - (1 - ab_t).sqrt() * eps_c) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 2.0)
            x       = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps_c
        return x

    @torch.no_grad()
    def _restore_tiled(
        self,
        window_norm: torch.Tensor,   # (T, 3, H, W) in [-1,1]
        frame_seed:  int,
        ab_s:        torch.Tensor,
        ab:          torch.Tensor,
        step_idx:    torch.Tensor,
        ddim_steps:  int,
        tile_sz:     int,
        overlap:     int,
    ) -> torch.Tensor:
        """
        Restore a single window by splitting into tile_sz×tile_sz tiles,
        running DDIM on each, and blending with Gaussian weights.
        Returns (3, H, W) in [0,1].
        """
        T, _, H, W = window_norm.shape
        half  = T // 2
        step  = max(tile_sz - overlap, 1)

        output = torch.zeros(3, H, W, device=self.device)
        weight = torch.zeros(1, H, W, device=self.device)

        tile_idx = 0
        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                y1 = min(y0 + tile_sz, H); ya = max(0, y1 - tile_sz)
                x1 = min(x0 + tile_sz, W); xa = max(0, x1 - tile_sz)

                cond_tile  = window_norm[:, :, ya:y1, xa:x1]  # (T,3,th,tw)
                _, _, th, tw = cond_tile.shape
                centre_tile = cond_tile[half]                  # (3,th,tw) in [-1,1]

                # Seed noise from frame + tile index for temporal consistency
                gen = torch.Generator(device=self.device).manual_seed(
                    frame_seed * 10000 + tile_idx)
                noise = torch.randn(3, th, tw, device=self.device, generator=gen)
                x_tile = (ab_s.sqrt() * centre_tile
                          + (1.0 - ab_s).sqrt() * noise).unsqueeze(0)  # (1,3,th,tw)

                # DDIM: B=1 window
                x_tile = self._ddim_loop(
                    x_tile,
                    [cond_tile],          # list of 1 tensor (T,3,th,tw)
                    B=1, T=T, half=half,
                    ab=ab, step_idx=step_idx, ddim_steps=ddim_steps,
                )
                tile_out = ((x_tile[0] + 1.0) * 0.5).clamp(0.0, 1.0)  # (3,th,tw)

                blend = _gaussian_weight(th, tw).to(self.device)        # (th,tw)
                output[:, ya:y1, xa:x1] += tile_out * blend
                weight[:, ya:y1, xa:x1] += blend
                tile_idx += 1

        return (output / weight.clamp(min=1e-6)).clamp(0.0, 1.0)
