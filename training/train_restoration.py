#!/usr/bin/env python3
"""
Train Model R — temporal-attention restoration diffusion model.

Dataset layout (produced by data_prep/prepare_restoration.py):
    <data_dir>/original/<video>/<frame>.png   ← clean reference
    <data_dir>/degraded/<video>/<frame>.png   ← DCVC-degraded input

Training uses a sliding temporal window of size T.  For each sample the centre
frame is the target; all T frames supply the conditioning (concatenated channel-
wise with the noised target).

Usage:
    python training/train_restoration.py
        --data    /path/to/restoration_pairs
        --config  configs/gpu/restoration.yaml
        [--epochs 100]
        [--batch-size 4]
        [--lr 1e-4]
        [--ckpt-dir checkpoints/restoration]
        [--resume checkpoints/restoration/latest.pt]
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
sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("train.restoration")


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

class RestorationDataset(Dataset):
    """
    Returns a temporal window of (degraded, original) patch pairs.

    Each item: (deg_window, orig_centre)
        deg_window  : Tensor (T*3, H, W)  — T degraded frames stacked channel-wise
        orig_centre : Tensor (3, H, W)    — clean centre frame
    """

    def __init__(
        self,
        data_dir: Path,
        T: int = 3,
        patch_size: int = 256,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.T = T
        self.patch_size = patch_size
        self.augment = augment
        self.centre = T // 2

        orig_root = data_dir / "original"
        deg_root  = data_dir / "degraded"

        # Build list of (video_stem, sorted_frame_names)
        self.sequences: List[Tuple[Path, Path, List[str]]] = []
        for video_dir in sorted(orig_root.iterdir()):
            if not video_dir.is_dir():
                continue
            deg_dir = deg_root / video_dir.name
            if not deg_dir.exists():
                continue
            names = sorted(p.name for p in video_dir.glob("*.png"))
            if len(names) >= T:
                self.sequences.append((video_dir, deg_dir, names))

        # Flat index: (seq_idx, centre_frame_idx_in_seq)
        self.index: List[Tuple[int, int]] = []
        for si, (_, _, names) in enumerate(self.sequences):
            for fi in range(self.centre, len(names) - self.centre):
                self.index.append((si, fi))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        si, fi = self.index[idx]
        orig_dir, deg_dir, names = self.sequences[si]

        # Collect T frames around centre
        deg_frames: List[np.ndarray] = []
        for offset in range(-self.centre, self.centre + 1):
            frame_idx = min(max(fi + offset, 0), len(names) - 1)
            deg_img = cv2.imread(str(deg_dir / names[frame_idx]))
            deg_frames.append(deg_img)
        orig_img = cv2.imread(str(orig_dir / names[fi]))

        # Random crop (all frames share the same crop)
        H, W = orig_img.shape[:2]
        ps = self.patch_size
        if H > ps and W > ps:
            y0 = random.randint(0, H - ps)
            x0 = random.randint(0, W - ps)
            deg_frames = [f[y0:y0+ps, x0:x0+ps] for f in deg_frames]
            orig_img   = orig_img[y0:y0+ps, x0:x0+ps]
        else:
            deg_frames = [cv2.resize(f, (ps, ps)) for f in deg_frames]
            orig_img   = cv2.resize(orig_img, (ps, ps))

        # Augmentation: horizontal flip
        if self.augment and random.random() < 0.5:
            deg_frames = [np.fliplr(f).copy() for f in deg_frames]
            orig_img   = np.fliplr(orig_img).copy()

        # Convert BGR→RGB, HWC→CHW, /255, [0,1]
        def to_tensor(img: np.ndarray) -> torch.Tensor:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return torch.from_numpy(img).permute(2, 0, 1)

        deg_tensors = [to_tensor(f) for f in deg_frames]      # T × (3,H,W)
        deg_window  = torch.cat(deg_tensors, dim=0)            # (T*3, H, W)
        orig_tensor = to_tensor(orig_img)                      # (3, H, W)

        return deg_window, orig_tensor


# ── VGG perceptual loss ───────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        # Use features up to relu3_3 (index 16)
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


# ── Diffusion utilities ───────────────────────────────────────────────────────

def _cosine_betas(T: int = 1000) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos((steps / T + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0.0, 0.999).float()


class DiffusionSchedule(nn.Module):
    def __init__(self, T: int = 1000) -> None:
        super().__init__()
        betas = _cosine_betas(T)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("sqrt_acp", acp.sqrt())
        self.register_buffer("sqrt_one_minus_acp", (1.0 - acp).sqrt())
        self.T = T

    def q_sample(
        self,
        x0: torch.Tensor,
        t:  torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        s_acp = self.sqrt_acp[t].view(-1, 1, 1, 1)
        s_omacp = self.sqrt_one_minus_acp[t].view(-1, 1, 1, 1)
        return s_acp * x0 + s_omacp * noise, noise


# ── Training loop ─────────────────────────────────────────────────────────────

def _build_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    from src.restoration._network import RestoreUNet
    model_cfg = cfg.get("model", {})
    model = RestoreUNet(
        base_channels   = model_cfg.get("base_channels",    64),
        encoder_channels = tuple(model_cfg.get("encoder_channels", [128, 256, 512])),
        num_res_blocks   = model_cfg.get("num_res_blocks",  4),
        T                = cfg.get("temporal_window",       3),
        n_heads          = model_cfg.get("n_heads",         4),
    )
    return model.to(device)


def train(args: argparse.Namespace) -> None:
    cfg    = _load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Device: {device}")

    # Dataset
    T_win      = cfg.get("temporal_window", 3)
    patch_size = args.patch_size or cfg.get("training", {}).get("patch_size", 256)
    full_ds    = RestorationDataset(
        Path(args.data), T=T_win, patch_size=patch_size, augment=True
    )
    val_len  = max(1, int(len(full_ds) * args.val_split))
    trn_len  = len(full_ds) - val_len
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

    # Model + schedule
    model    = _build_model(cfg, device)
    schedule = DiffusionSchedule(T=1000).to(device)
    vgg_loss = VGGPerceptualLoss().to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"RestoreUNet params: {param_count/1e6:.1f}M")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val    = float("inf")

    # Resume
    if args.resume and Path(args.resume).exists():
        LOGGER.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))

    vgg_w = float(cfg.get("training", {}).get("vgg_weight", 0.1))

    for epoch in range(start_epoch, args.epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        trn_loss = 0.0
        t_start  = time.perf_counter()

        for step, (deg_window, orig_centre) in enumerate(trn_loader):
            # deg_window : (B, T*3, H, W)
            # orig_centre: (B, 3, H, W)
            B  = orig_centre.size(0)
            deg_window   = deg_window.to(device)
            orig_centre  = orig_centre.to(device)

            # Sample random diffusion timestep
            t_idx = torch.randint(0, schedule.T, (B,), device=device)

            # Forward diffusion on clean centre frame (scaled to [-1,1])
            x0    = orig_centre * 2.0 - 1.0
            x_t, noise = schedule.q_sample(x0, t_idx)

            # Degrade window to [-1,1] for conditioning
            # Pick centre-frame slice from deg_window: channel offset = centre*3
            c = T_win // 2
            deg_cond = deg_window[:, c*3:(c+1)*3] * 2.0 - 1.0  # (B,3,H,W)

            # Build model input: (B, T, 6, H, W) — each temporal slot gets
            # x_t + its degraded frame.
            # For non-centre slots the "target noise" is irrelevant but the
            # network needs consistent input shape.
            inputs = []
            for ti in range(T_win):
                deg_i = deg_window[:, ti*3:(ti+1)*3] * 2.0 - 1.0
                if ti == c:
                    noised_i = x_t
                else:
                    # Noise centre-frame but expose neighbouring degraded frames
                    noised_i = x_t  # same noise — simplified; network learns context
                inputs.append(torch.cat([noised_i, deg_i], dim=1))  # (B,6,H,W)

            # Stack along batch: (B*T, 6, H, W)
            model_in = torch.stack(inputs, dim=1).view(B * T_win, 6, *x_t.shape[-2:])
            t_rep    = t_idx.unsqueeze(1).expand(B, T_win).reshape(B * T_win)

            pred_noise = model(model_in, t_rep)  # (B*T, 3, H, W)

            # Only supervise on centre frame
            pred_centre = pred_noise.view(B, T_win, 3, *x_t.shape[-2:])[:, c]

            l1_loss  = F.l1_loss(pred_centre, noise)
            # VGG on reconstructed x0 estimate
            sqrt_acp = schedule.sqrt_acp[t_idx].view(B, 1, 1, 1)
            sqrt_omacp = schedule.sqrt_one_minus_acp[t_idx].view(B, 1, 1, 1)
            x0_hat = (x_t - sqrt_omacp * pred_centre) / sqrt_acp.clamp(min=1e-8)
            x0_hat = x0_hat.clamp(-1.0, 1.0)
            # Convert to [0,1] for VGG
            perc_loss = vgg_loss((x0_hat + 1) / 2, (x0 + 1) / 2)

            loss = l1_loss + vgg_w * perc_loss

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            trn_loss += loss.item()

            if (step + 1) % args.log_every == 0:
                avg = trn_loss / (step + 1)
                LOGGER.info(
                    f"Epoch {epoch+1}/{args.epochs} step {step+1}/{len(trn_loader)} "
                    f"loss={avg:.4f} l1={l1_loss.item():.4f} vgg={perc_loss.item():.4f}"
                )

        trn_loss /= max(len(trn_loader), 1)
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for deg_window, orig_centre in val_loader:
                B  = orig_centre.size(0)
                deg_window  = deg_window.to(device)
                orig_centre = orig_centre.to(device)

                t_idx = torch.randint(0, schedule.T, (B,), device=device)
                x0    = orig_centre * 2.0 - 1.0
                x_t, noise = schedule.q_sample(x0, t_idx)

                c = T_win // 2
                inputs = []
                for ti in range(T_win):
                    deg_i = deg_window[:, ti*3:(ti+1)*3] * 2.0 - 1.0
                    inputs.append(torch.cat([x_t, deg_i], dim=1))
                model_in = torch.stack(inputs, dim=1).view(B * T_win, 6, *x_t.shape[-2:])
                t_rep    = t_idx.unsqueeze(1).expand(B, T_win).reshape(B * T_win)

                pred_noise = model(model_in, t_rep)
                pred_centre = pred_noise.view(B, T_win, 3, *x_t.shape[-2:])[:, c]
                val_loss += F.l1_loss(pred_centre, noise).item()

        val_loss /= max(len(val_loader), 1)
        elapsed = time.perf_counter() - t_start

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
            "scheduler": scheduler.state_dict(),
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
    p = argparse.ArgumentParser(description="Train Model R (restoration diffusion).")
    p.add_argument("--data",          required=True, help="Path to restoration pairs dir")
    p.add_argument("--config",        default="configs/gpu/restoration.yaml")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch-size",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--patch-size",    type=int,   default=None,
                   help="Override patch size from config")
    p.add_argument("--ckpt-dir",      default="checkpoints/restoration")
    p.add_argument("--resume",        default=None, help="Path to checkpoint to resume")
    p.add_argument("--val-split",     type=float, default=0.05)
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--log-every",     type=int,   default=50)
    p.add_argument("--save-every",    type=int,   default=5)
    p.add_argument("--verbose",       action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
