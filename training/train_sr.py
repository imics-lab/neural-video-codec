#!/usr/bin/env python3
"""
Train Model S — enhanced 2x diffusion SR: upsample + denoise + sharpen.

Dataset layout (produced by data_prep/prepare_sr_enhanced.py):
    <data_dir>/hr/<video>/<frame>.png   <- clean HR ground truth
    <data_dir>/lr/<video>/<frame>.png   <- DCVC-degraded + downscaled LR input

The model learns in one pass to: upsample 2x, remove compression artifacts,
and restore sharp high-frequency detail.

Loss: L1 + lambda_vgg*VGG + lambda_lpips*LPIPS + lambda_grad*Gradient
      + lambda_freq*FrequencyDomain

Usage:
    python training/train_sr.py
        --data    /path/to/sr_pairs
        --config  configs/gpu/upscaling.yaml
        [--epochs 100]
        [--batch-size 4]
        [--lr 1e-4]
        [--ckpt-dir checkpoints/sr]
        [--resume checkpoints/sr/latest.pt]
        [--val-split 0.05]
        [--workers 4]
        [--log-every 50]
        [--save-every 5]
        [--verbose]
"""
from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOGGER = logging.getLogger("train.sr")


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    LOGGER.setLevel(level)


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        import json
        with open(path) as f:
            return json.load(f)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SRDataset(Dataset):
    """
    Returns (lr_up, hr) pairs where lr_up is bicubic-upsampled LR.

    Each item:
        lr_up : Tensor (3, H, W)  — LR bicubically upsampled to HR size, [0,1]
        hr    : Tensor (3, H, W)  — HR ground truth, [0,1]
    """

    def __init__(
        self,
        data_dir: Path,
        scale: int = 2,
        patch_size: int = 256,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.scale      = scale
        self.patch_size = patch_size
        self.augment    = augment

        hr_root = data_dir / "hr"
        lr_root = data_dir / "lr"

        self.pairs: List[Tuple[Path, Path]] = []
        for hr_dir in sorted(hr_root.iterdir()):
            if not hr_dir.is_dir():
                continue
            lr_dir = lr_root / hr_dir.name
            if not lr_dir.exists():
                continue
            for hr_img in sorted(hr_dir.glob("*.png")):
                lr_img = lr_dir / hr_img.name
                if lr_img.exists():
                    self.pairs.append((hr_img, lr_img))

        LOGGER.info(f"SRDataset: {len(self.pairs)} pairs from {data_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr_path, lr_path = self.pairs[idx]
        hr = cv2.imread(str(hr_path))
        lr = cv2.imread(str(lr_path))

        ps = self.patch_size
        H, W = hr.shape[:2]

        # Random crop on HR; corresponding crop on LR
        if H >= ps and W >= ps:
            y0_hr = random.randint(0, H - ps)
            x0_hr = random.randint(0, W - ps)
            # Align crop to scale grid
            y0_hr = (y0_hr // self.scale) * self.scale
            x0_hr = (x0_hr // self.scale) * self.scale
            hr = hr[y0_hr:y0_hr + ps, x0_hr:x0_hr + ps]

            lr_ps  = ps // self.scale
            y0_lr  = y0_hr // self.scale
            x0_lr  = x0_hr // self.scale
            lr = lr[y0_lr:y0_lr + lr_ps, x0_lr:x0_lr + lr_ps]
        else:
            hr = cv2.resize(hr, (ps, ps))
            lr = cv2.resize(lr, (ps // self.scale, ps // self.scale))

        # Augmentation: horizontal/vertical flip + 90° rotation
        if self.augment:
            if random.random() < 0.5:
                hr = np.fliplr(hr).copy()
                lr = np.fliplr(lr).copy()
            if random.random() < 0.5:
                hr = np.flipud(hr).copy()
                lr = np.flipud(lr).copy()
            k = random.randint(0, 3)
            if k:
                hr = np.rot90(hr, k).copy()
                lr = np.rot90(lr, k).copy()

        # Bicubic upsample LR → HR size
        lr_up = cv2.resize(
            lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC
        )

        def to_tensor(img: np.ndarray) -> torch.Tensor:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return torch.from_numpy(img).permute(2, 0, 1)

        return to_tensor(lr_up), to_tensor(hr)


# ── Perceptual losses ─────────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.slice = nn.Sequential(*list(vgg.features)[:16]).eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = (pred   - self.mean) / self.std
        target = (target - self.mean) / self.std
        return F.l1_loss(self.slice(pred), self.slice(target))


class LPIPSLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            import lpips
            self.fn = lpips.LPIPS(net="vgg").eval()
            for p in self.fn.parameters():
                p.requires_grad_(False)
            self._available = True
        except ImportError:
            LOGGER.warning("lpips package not found — LPIPS loss disabled.")
            self._available = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self._available:
            return torch.tensor(0.0, device=pred.device)
        # lpips expects [-1, 1]
        pred   = pred   * 2.0 - 1.0
        target = target * 2.0 - 1.0
        return self.fn(pred, target).mean()


# ── Sharpness & frequency losses ─────────────────────────────────────────────

class GradientLoss(nn.Module):
    """Sobel gradient loss — penalises blurry edges."""

    def __init__(self) -> None:
        super().__init__()
        # Sobel kernels (same for all channels)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        # shape (1,1,3,3) — applied depthwise via groups=C
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _grad(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B * C, 1, H, W)
        gx = F.conv2d(x_flat, self.kx, padding=1)
        gy = F.conv2d(x_flat, self.ky, padding=1)
        return (gx ** 2 + gy ** 2 + 1e-8).sqrt().view(B, C, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._grad(pred), self._grad(target))


class LaplacianLoss(nn.Module):
    """Laplacian sharpness loss — emphasises fine high-frequency detail."""

    def __init__(self) -> None:
        super().__init__()
        k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("kernel", k.view(1, 1, 3, 3))

    def _lap(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B * C, 1, H, W)
        return F.conv2d(x_flat, self.kernel, padding=1).view(B, C, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._lap(pred), self._lap(target))


class FrequencyLoss(nn.Module):
    """FFT-domain loss weighted toward high frequencies.

    Computes the 2D FFT of pred and target, applies a radial high-frequency
    weight mask, and penalises amplitude differences.  Encourages the model
    to recover texture and fine detail that L1 in pixel-space tends to ignore.
    """

    def __init__(self, high_freq_weight: float = 4.0) -> None:
        super().__init__()
        self.hfw = high_freq_weight  # multiplier on frequencies above Nyquist/2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        # Work in float32 regardless of input dtype
        p = pred.float()
        t = target.float()

        fp = torch.fft.rfft2(p, norm="ortho")
        ft = torch.fft.rfft2(t, norm="ortho")

        # Build radial frequency weight mask (same device as input)
        fy = torch.fft.fftfreq(H, device=pred.device).abs()
        fx = torch.fft.rfftfreq(W, device=pred.device).abs()
        freq_r = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()  # (H, W//2+1)
        weight = 1.0 + (self.hfw - 1.0) * (freq_r > 0.25).float()
        weight = weight.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W//2+1)

        diff = (fp - ft).abs()
        return (diff * weight).mean()


# ── Diffusion schedule ────────────────────────────────────────────────────────

def _cosine_betas(T: int = 1000) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    acp   = torch.cos((steps / T + 0.008) / 1.008 * math.pi / 2) ** 2
    acp   = acp / acp[0]
    betas = 1.0 - acp[1:] / acp[:-1]
    return betas.clamp(0.0, 0.999).float()


class DiffusionSchedule(nn.Module):
    def __init__(self, T: int = 1000) -> None:
        super().__init__()
        betas  = _cosine_betas(T)
        alphas = 1.0 - betas
        acp    = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_acp",           acp.sqrt())
        self.register_buffer("sqrt_one_minus_acp", (1.0 - acp).sqrt())
        self.T = T

    def q_sample(
        self,
        x0:    torch.Tensor,
        t:     torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        s  = self.sqrt_acp[t].view(-1, 1, 1, 1)
        so = self.sqrt_one_minus_acp[t].view(-1, 1, 1, 1)
        return s * x0 + so * noise, noise


# ── Training loop ─────────────────────────────────────────────────────────────

def _build_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    from src.upscaling._network import SRUNet
    model_cfg = cfg.get("model", {})
    model = SRUNet(
        base_channels    = model_cfg.get("base_channels",    64),
        encoder_channels = tuple(model_cfg.get("encoder_channels", [128, 256, 512])),
        num_res_blocks   = model_cfg.get("num_res_blocks",  4),
    )
    return model.to(device)


def train(args: argparse.Namespace) -> None:
    cfg    = _load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Device: {device}")

    scale      = cfg.get("scale", 2)
    patch_size = args.patch_size or cfg.get("training", {}).get("patch_size", 256)
    full_ds    = SRDataset(
        Path(args.data), scale=scale, patch_size=patch_size, augment=True
    )
    val_len = max(1, int(len(full_ds) * args.val_split))
    trn_len = len(full_ds) - val_len
    trn_ds, val_ds = random_split(full_ds, [trn_len, val_len])
    LOGGER.info(f"Dataset: {trn_len} train / {val_len} val samples")

    trn_loader = DataLoader(
        trn_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model      = _build_model(cfg, device)
    schedule   = DiffusionSchedule(T=1000).to(device)
    vgg_loss   = VGGPerceptualLoss().to(device)
    lpips_loss = LPIPSLoss().to(device)
    grad_loss  = GradientLoss().to(device)
    lap_loss   = LaplacianLoss().to(device)
    freq_loss  = FrequencyLoss().to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"SRUNet params: {param_count/1e6:.1f}M")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val    = float("inf")

    if args.resume and Path(args.resume).exists():
        LOGGER.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        lr_sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))

    trn_cfg  = cfg.get("training", {})
    vgg_w    = float(trn_cfg.get("vgg_weight",   0.1))
    lpips_w  = float(trn_cfg.get("lpips_weight", 0.1))
    grad_w   = float(trn_cfg.get("grad_weight",  0.05))
    lap_w    = float(trn_cfg.get("lap_weight",   0.05))
    freq_w   = float(trn_cfg.get("freq_weight",  0.05))

    for epoch in range(start_epoch, args.epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        trn_loss = 0.0
        t_start  = time.perf_counter()

        for step, (lr_up, hr) in enumerate(trn_loader):
            lr_up = lr_up.to(device)  # (B,3,H,W) [0,1]
            hr    = hr.to(device)     # (B,3,H,W) [0,1]
            B     = hr.size(0)

            t_idx = torch.randint(0, schedule.T, (B,), device=device)

            x0 = hr * 2.0 - 1.0  # [-1,1]
            x_t, noise = schedule.q_sample(x0, t_idx)

            cond     = lr_up * 2.0 - 1.0  # [-1,1]
            model_in = torch.cat([x_t, cond], dim=1)  # (B, 6, H, W)

            pred_noise = model(model_in, t_idx)

            l1   = F.l1_loss(pred_noise, noise)

            # Reconstruct x0 estimate for perceptual losses
            s    = schedule.sqrt_acp[t_idx].view(B, 1, 1, 1)
            so   = schedule.sqrt_one_minus_acp[t_idx].view(B, 1, 1, 1)
            x0_hat = (x_t - so * pred_noise) / s.clamp(min=1e-8)
            x0_hat = x0_hat.clamp(-1.0, 1.0)
            hr01   = (x0_hat + 1) / 2
            gt01   = (x0    + 1) / 2

            perc  = vgg_loss(hr01, gt01)
            lp    = lpips_loss(hr01, gt01)
            grd   = grad_loss(hr01, gt01)
            lap   = lap_loss(hr01, gt01)
            frq   = freq_loss(hr01, gt01)

            loss  = (l1
                     + vgg_w  * perc
                     + lpips_w * lp
                     + grad_w  * grd
                     + lap_w   * lap
                     + freq_w  * frq)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            trn_loss += loss.item()

            if (step + 1) % args.log_every == 0:
                avg = trn_loss / (step + 1)
                LOGGER.info(
                    f"Epoch {epoch+1}/{args.epochs} step {step+1}/{len(trn_loader)} "
                    f"loss={avg:.4f} l1={l1.item():.4f} "
                    f"vgg={perc.item():.4f} lpips={lp.item():.4f} "
                    f"grad={grd.item():.4f} lap={lap.item():.4f} freq={frq.item():.4f}"
                )

        trn_loss /= max(len(trn_loader), 1)
        lr_sched.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_up, hr in val_loader:
                lr_up = lr_up.to(device)
                hr    = hr.to(device)
                B     = hr.size(0)

                t_idx = torch.randint(0, schedule.T, (B,), device=device)
                x0    = hr * 2.0 - 1.0
                x_t, noise = schedule.q_sample(x0, t_idx)
                cond = lr_up * 2.0 - 1.0
                pred = model(torch.cat([x_t, cond], dim=1), t_idx)
                val_loss += F.l1_loss(pred, noise).item()

        val_loss /= max(len(val_loader), 1)
        elapsed   = time.perf_counter() - t_start

        LOGGER.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"trn={trn_loss:.4f} val={val_loss:.4f} | "
            f"lr={optimiser.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.0f}s"
        )

        # ── Checkpoint ─────────────────────────────────────────────────────
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": lr_sched.state_dict(),
            "best_val":  best_val,
            "config":    cfg,
        }
        torch.save(state, ckpt_dir / "latest.pt")
        if is_best:
            torch.save(state, ckpt_dir / "best.pt")
            LOGGER.info(f"  *** New best val={best_val:.4f} → saved best.pt")

        if (epoch + 1) % args.save_every == 0:
            torch.save(state, ckpt_dir / f"epoch_{epoch+1:04d}.pt")

    LOGGER.info("Training complete.")
    LOGGER.info(f"Best val loss: {best_val:.4f}")
    LOGGER.info(f"Checkpoints in: {ckpt_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Model S (SR diffusion).")
    p.add_argument("--data",       required=True, help="Path to SR pairs dir")
    p.add_argument("--config",     default="configs/gpu/upscaling.yaml")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--patch-size", type=int,   default=None)
    p.add_argument("--ckpt-dir",   default="checkpoints/sr")
    p.add_argument("--resume",     default=None)
    p.add_argument("--val-split",  type=float, default=0.05)
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--log-every",  type=int,   default=50)
    p.add_argument("--save-every", type=int,   default=5)
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
