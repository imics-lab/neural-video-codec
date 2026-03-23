"""
GaussianDiffusion — cosine schedule, DDPM forward, DDIM reverse.
Shared by both the restoration and upscaling models.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _cosine_alpha_bar(t: torch.Tensor, T: int, s: float = 0.008) -> torch.Tensor:
    f  = torch.cos((t / T + s) / (1.0 + s) * math.pi / 2.0) ** 2
    f0 = math.cos(s / (1.0 + s) * math.pi / 2.0) ** 2
    return (f / f0).clamp(0.0, 1.0)


class GaussianDiffusion:
    """
    DDPM forward process + deterministic DDIM reverse.

    Args:
        T: Total diffusion timesteps (default 1000).
    """

    def __init__(self, T: int = 1000):
        self.T = T
        ts = torch.arange(T + 1, dtype=torch.float32)
        self.alpha_bar = _cosine_alpha_bar(ts, T)   # (T+1,) on CPU

    # ── Forward (noising) ─────────────────────────────────────────────────────

    def q_sample(
        self,
        x0:    torch.Tensor,             # (B, C, H, W) clean image in [0, 1]
        t:     torch.Tensor,             # (B,) long timestep indices
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε.

        Returns:
            x_t:   Noisy image.
            noise: Noise added (training target).
        """
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t.cpu()].to(x0.device).view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise, noise

    # ── Reverse (DDIM, η=0) ───────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        model,                              # noise-predicting nn.Module
        cond:       torch.Tensor,           # conditioning input (passed as extra channels)
        steps:      int = 20,
        device:     torch.device = torch.device("cuda"),
        init_noise: torch.Tensor | None = None,
        t_start:    int | None = None,      # img2img: start from noised cond at t_start
    ) -> torch.Tensor:
        """
        Run DDIM to produce a clean frame.

        The model receives  torch.cat([x_t, cond], dim=1)  as its spatial input.
        The caller is responsible for constructing `cond` appropriately.

        Args:
            model:      Trained noise predictor in eval mode.
            cond:       (B, C_cond, H, W) conditioning channels at target resolution.
            steps:      Number of DDIM denoising steps.
            device:     Computation device.
            init_noise: Override initial noise (useful for tiled consistency).
            t_start:    Img2img strength. None = full denoising from Gaussian noise.

        Returns:
            (B, 3, H, W) denoised output in [0, 1].
        """
        B, _, H, W = cond.shape
        ab = self.alpha_bar.to(device)

        if t_start is not None:
            noise = init_noise if init_noise is not None else torch.randn(B, 3, H, W, device=device)
            ab_s  = ab[t_start]
            x     = ab_s.sqrt() * cond[:, :3] + (1.0 - ab_s).sqrt() * noise
            steps_idx = torch.linspace(t_start, 0, steps + 1).long()
        else:
            x = init_noise if init_noise is not None else torch.randn(B, 3, H, W, device=device)
            steps_idx = torch.linspace(self.T - 1, 0, steps + 1).long()

        for i in range(steps):
            t_cur  = int(steps_idx[i].item())
            t_next = int(steps_idx[i + 1].item())
            t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)

            inp     = torch.cat([x, cond], dim=1)
            eps_hat = model(inp, t_batch)

            ab_t    = ab[t_cur]
            ab_prev = ab[t_next]
            x0_pred = (x - (1.0 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 2.0)
            x       = ab_prev.sqrt() * x0_pred + (1.0 - ab_prev).sqrt() * eps_hat

        return x.clamp(0.0, 1.0)
