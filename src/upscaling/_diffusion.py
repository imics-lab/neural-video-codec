"""
GaussianDiffusion — cosine noise schedule, DDPM forward process, DDIM reverse.

This is a standalone helper class (not nn.Module).  The U-Net model is
passed in as an argument to ddim_sample so that the caller controls
which device and dtype the model lives on.

Noise schedule
──────────────
We use the cosine schedule from Nichol & Dhariwal (2021):

    ᾱ_t = cos((t/T + s) / (1+s) · π/2)²  /  cos((s/(1+s)) · π/2)²

which yields a smoother degradation curve than the linear schedule and
keeps more signal at small t (important for high-frequency detail).

DDIM sampling (η = 0, deterministic)
─────────────────────────────────────
Given x_t and the predicted noise ε̂_θ:

    x̂_0  = (x_t - √(1−ᾱ_t) · ε̂_θ) / √ᾱ_t
    x_prev = √ᾱ_{t-1} · x̂_0  +  √(1−ᾱ_{t-1}) · ε̂_θ

Repeat for `steps` evenly spaced timesteps from T-1 → 0.
"""

import math

import torch
import torch.nn.functional as F


def _cosine_alpha_bar(t: torch.Tensor, T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule: ᾱ_t ∈ [0, 1]."""
    f = torch.cos((t / T + s) / (1.0 + s) * math.pi / 2.0) ** 2
    f0 = math.cos(s / (1.0 + s) * math.pi / 2.0) ** 2
    return (f / f0).clamp(0.0, 1.0)


class GaussianDiffusion:
    """
    Implements the DDPM forward (noising) process and DDIM reverse (denoising).

    Args:
        T: Total number of diffusion timesteps (default 1000).
    """

    def __init__(self, T: int = 1000):
        self.T = T
        ts = torch.arange(T + 1, dtype=torch.float32)
        # ᾱ_t for t = 0 … T  (T+1 values so we can index t and t-1)
        self.alpha_bar = _cosine_alpha_bar(ts, T)   # (T+1,)  on CPU

    # ── Forward process ───────────────────────────────────────────────────────

    def q_sample(
        self,
        x0: torch.Tensor,           # (B, 3, H, W) clean HR in [0, 1]
        t:  torch.Tensor,           # (B,) long timestep indices
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add Gaussian noise to x0 at diffusion step t.

        Returns:
            x_t:   Noisy version of x0.
            noise: The actual noise added (used as training target).
        """
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t.cpu()].to(x0.device).view(-1, 1, 1, 1)   # (B, 1, 1, 1)
        x_t = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
        return x_t, noise

    # ── Reverse process (DDIM) ────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        model,                              # DiffusionUNet
        lr_up:    torch.Tensor,             # (B, 3, H, W)  bicubic-upsampled LR
        roi_mask: torch.Tensor,             # (B, 1, H, W)  ROI weight map
        steps:    int = 20,
        device:   torch.device = torch.device("cuda"),
        init_noise: torch.Tensor | None = None,
        t_start:  int | None = None,        # img2img: begin from this timestep
    ) -> torch.Tensor:
        """
        Run DDIM reverse diffusion to produce a clean HR image.

        When t_start is given (img2img mode), the reverse process begins from
        a noised version of lr_up at timestep t_start rather than from pure
        Gaussian noise at T-1.  This anchors each tile to the LR color structure
        and eliminates the tiled-colour-shift artifact.

        Args:
            model:      Trained DiffusionUNet in eval mode.
            lr_up:      Bicubic-upsampled LR conditioning frame (at HR size).
            roi_mask:   ROI weight mask (at HR size).
            steps:      Number of DDIM denoising steps.
            device:     Torch device.
            init_noise: Noise tensor to use as (or mix into) the starting x.
                        Pass a crop from a global noise map for tile consistency.
            t_start:    Starting timestep for img2img.  Lower = more faithful
                        to LR colour/structure; higher = more freedom.
                        Recommended: 500–700.  None = start from T-1 (full noise).

        Returns:
            (B, 3, H, W) restored HR frame in [0, 1].
        """
        B, _, H, W = lr_up.shape
        ab = self.alpha_bar.to(device)   # (T+1,)

        if t_start is not None:
            # Img2img: mix lr_up with noise at t_start, then denoise from there.
            # x_{t_start} = sqrt(ab_t) * lr_up + sqrt(1-ab_t) * noise
            noise = init_noise if init_noise is not None else torch.randn(B, 3, H, W, device=device)
            ab_s  = ab[t_start]
            x     = ab_s.sqrt() * lr_up + (1.0 - ab_s).sqrt() * noise
            step_indices = torch.linspace(t_start, 0, steps + 1).long()
        else:
            x = init_noise if init_noise is not None else torch.randn(B, 3, H, W, device=device)
            step_indices = torch.linspace(self.T - 1, 0, steps + 1).long()

        for i in range(steps):
            t_cur  = int(step_indices[i].item())
            t_next = int(step_indices[i + 1].item())

            t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)

            inp     = torch.cat([x, lr_up, roi_mask], dim=1)   # (B, 7, H, W)
            eps_hat = model(inp, t_batch)                       # (B, 3, H, W)

            ab_t    = ab[t_cur]
            ab_prev = ab[t_next]

            x0_pred = (x - (1.0 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 2.0)

            x = ab_prev.sqrt() * x0_pred + (1.0 - ab_prev).sqrt() * eps_hat

        return x.clamp(0.0, 1.0)
