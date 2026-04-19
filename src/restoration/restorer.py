"""
Restorer — inference wrapper around RestoreUNet + GaussianDiffusion (Model R).

Inference protocol (must mirror training exactly):
  - Only the CENTRE frame of each window is noised (x_t).
  - The model receives T copies of [x_t, cond_frame_i] for i in 0..T-1
    (same noised x, different conditioning per temporal position).
  - Only the centre frame's noise prediction (eps[T//2]) updates x_t.
  - Inputs/outputs are in [-1,1] space (matching training normalisation
    x0 = orig * 2 - 1).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

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


# ── Restorer ──────────────────────────────────────────────────────────────────

class Restorer:
    """
    Wraps RestoreUNet for inference on a sequence of degraded frames.

    Sliding window of size T (stride 1). Boundary frames padded by replication.
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
            T_ckpt = mcfg.get("temporal_window", self.T)
        else:
            state = ckpt
            mcfg  = {}
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
        self.model.to(self.device).half().eval()

        T_diff = getattr(config.model, "timesteps", 1000)
        self.diffusion = GaussianDiffusion(T=T_diff)

    # ── Public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def restore_sequence(self, degraded_frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Restore a list of degraded BGR frames.

        Returns a list of restored BGR uint8 arrays (same length and size).
        """
        N = len(degraded_frames)
        if N == 0:
            return []

        T    = self.T
        half = T // 2
        cfg  = self.cfg

        # Convert all frames to [0,1] tensors (fp16)
        deg_01 = [_to_tensor(f).to(self.device, dtype=torch.float16)
                  for f in degraded_frames]

        _, H, W = deg_01[0].shape
        batch_size = int(getattr(cfg.inference, 'batch_size', 1))
        t_start    = cfg.inference.t_start
        ddim_steps = cfg.inference.ddim_steps
        color_fix  = cfg.inference.color_fix

        ab        = self.diffusion.alpha_bar.to(self.device)
        ab_s      = ab[t_start]
        step_idx  = torch.linspace(t_start, 0, ddim_steps + 1).long()

        restored: List[Optional[torch.Tensor]] = [None] * N

        import time as _time
        t0 = _time.perf_counter()
        processed = 0

        for batch_start in range(0, N, batch_size):
            centres = list(range(batch_start, min(batch_start + batch_size, N)))
            B = len(centres)

            # ── Build T-frame windows in [-1,1] and [0,1] ─────────────────────
            # windows_norm: (B, T, 3, H, W) in [-1,1] — conditioning for model
            # centre_01:    (B, 3, H, W) in [0,1]     — for color-fix
            windows_norm = []
            centre_01    = []
            for c in centres:
                idxs = [max(0, min(N - 1, c + k - half)) for k in range(T)]
                w    = torch.stack([deg_01[i] for i in idxs], dim=0)  # (T,3,H,W)
                windows_norm.append(w * 2.0 - 1.0)
                centre_01.append(deg_01[c])

            # ── Noise ONLY the centre frame (matching training) ────────────────
            # x shape: (B, 3, H, W) — one noised frame per window
            x_parts = []
            for i, c in enumerate(centres):
                gen = torch.Generator(device=self.device).manual_seed(c)
                noise = torch.randn(3, H, W, device=self.device,
                                    dtype=torch.float16, generator=gen)
                x_c = windows_norm[i][half]   # (3,H,W) centre frame in [-1,1]
                x_parts.append(ab_s.sqrt() * x_c + (1.0 - ab_s).sqrt() * noise)
            x = torch.stack(x_parts, dim=0)   # (B, 3, H, W)

            # ── DDIM loop ──────────────────────────────────────────────────────
            # model input: (B*T, 6, H, W) ordered [w0t0, w0t1, w0t2, w1t0, ...]
            # Each row: [x_centre_i replicated, cond_frame_ti]  ← same as training
            for si in range(ddim_steps):
                t_cur  = int(step_idx[si].item())
                t_next = int(step_idx[si + 1].item())

                # Build (B*T, 6, H, W)
                rows = []
                for i in range(B):
                    for ti in range(T):
                        rows.append(torch.cat([x[i].unsqueeze(0),
                                               windows_norm[i][ti].unsqueeze(0)], dim=1))
                model_in = torch.cat(rows, dim=0)              # (B*T, 6, H, W)
                t_b = torch.full((B * T,), t_cur,
                                 device=self.device, dtype=torch.long)

                with torch.amp.autocast("cuda"):
                    eps_all = self.model(model_in, t_b)        # (B*T, 3, H, W)
                eps_all = eps_all.to(x.dtype)

                # Take only centre-frame prediction from each window
                eps_all = eps_all.view(B, T, 3, H, W)
                eps_c   = eps_all[:, half]                     # (B, 3, H, W)

                ab_t    = ab[t_cur]
                ab_prev = ab[t_next]
                x0_pred = (x - (1 - ab_t).sqrt() * eps_c) / ab_t.sqrt()
                x0_pred = x0_pred.clamp(-1.0, 2.0)
                x       = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps_c

            # ── Convert [-1,1] → [0,1] and optional color-fix ─────────────────
            x_out = ((x + 1.0) * 0.5).clamp(0.0, 1.0)        # (B, 3, H, W)

            for i, c in enumerate(centres):
                out = x_out[i]
                if color_fix:
                    dm  = centre_01[i].mean(dim=[1, 2], keepdim=True)
                    om  = out.mean(dim=[1, 2], keepdim=True)
                    out = (out - om + dm).clamp(0.0, 1.0)
                restored[c] = out

            processed += B
            elapsed = _time.perf_counter() - t0
            fps = processed / elapsed
            eta = (N - processed) / fps if fps > 0 else 0
            print(f"[restore] {processed}/{N} frames  {fps:.2f} fps  ETA {eta:.0f}s",
                  flush=True)

        return [_to_frame(t) for t in restored if t is not None]
