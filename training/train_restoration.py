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


def _to_np_rgb(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) float [0,1] → HWC uint8 RGB."""
    return (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


@torch.no_grad()
def _save_samples(
    model: nn.Module,
    schedule: "DiffusionSchedule",
    val_ds: Dataset,
    device: torch.device,
    out_dir: Path,
    epoch: int,
    n_samples: int = 4,       # unused (kept for call-site compat); video length controls frames
    t_start: int = 200,
    ddim_steps: int = 20,
    fps: int = 30,
    video_seconds: float = 1.0,
) -> None:
    """
    Render a short side-by-side video (degraded | restored | original) for one
    validation sequence so training progress is easy to assess visually.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_model = model.module if isinstance(model, nn.DataParallel) else model
    eval_model.eval()

    ab       = schedule.alphas_cumprod.to(device)
    step_idx = torch.linspace(t_start, 0, ddim_steps + 1).long()

    # ── Pick a sequence directly from the underlying dataset ─────────────────
    # val_ds may be a Subset; unwrap to reach .sequences
    base_ds = val_ds.dataset if hasattr(val_ds, "dataset") else val_ds
    if not hasattr(base_ds, "sequences") or not base_ds.sequences:
        LOGGER.warning("  _save_samples: no sequences available, skipping")
        eval_model.train()
        return

    T_win = base_ds.T
    half  = T_win // 2
    ps    = base_ds.patch_size
    n_frames = max(T_win, int(round(fps * video_seconds)))

    # Pick the sequence with the most frames (most informative)
    orig_dir, deg_dir, names = max(base_ds.sequences, key=lambda s: len(s[2]))
    if len(names) < T_win:
        LOGGER.warning("  _save_samples: sequence too short, skipping")
        eval_model.train()
        return

    # Use a fixed centre crop for temporal consistency across all frames
    sample_bgr = cv2.imread(str(orig_dir / names[0]))
    H_full, W_full = sample_bgr.shape[:2]
    if H_full > ps and W_full > ps:
        y0 = (H_full - ps) // 2
        x0 = (W_full - ps) // 2
    else:
        y0, x0 = 0, 0

    def _crop(bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        if h > ps and w > ps:
            return bgr[y0:y0+ps, x0:x0+ps]
        return cv2.resize(bgr, (ps, ps))

    def _load_tensor(path: Path) -> torch.Tensor:
        bgr  = cv2.imread(str(path))
        crop = _crop(bgr)
        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1).to(device)

    # Frames to render: pick n_frames starting from centre-index half
    start_fi = half
    end_fi   = min(start_fi + n_frames, len(names) - half)
    frame_indices = list(range(start_fi, end_fi))

    # Set up video writer (3× width side-by-side: degraded | restored | original)
    vid_path = out_dir / f"epoch{epoch+1:04d}.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(vid_path), fourcc, fps, (ps * 3, ps))

    for fi in frame_indices:
        # Build T-frame window of degraded frames in [-1,1]
        cond_tensors = []
        for offset in range(-half, half + 1):
            clamped = min(max(fi + offset, 0), len(names) - 1)
            cond_tensors.append(_load_tensor(deg_dir / names[clamped]))
        orig_t   = _load_tensor(orig_dir / names[fi])
        centre_t = cond_tensors[half]    # (3, H, W) in [0,1]

        cond_norm = [(t * 2.0 - 1.0).unsqueeze(0) for t in cond_tensors]  # each (1,3,H,W)
        centre_norm = centre_t * 2.0 - 1.0                                 # (3,H,W)

        gen   = torch.Generator(device=device).manual_seed(fi)
        noise = torch.randn(1, 3, ps, ps, device=device, generator=gen)
        x     = ab[t_start].sqrt() * centre_norm.unsqueeze(0) + (1 - ab[t_start]).sqrt() * noise

        for i in range(ddim_steps):
            t_cur  = int(step_idx[i].item())
            t_next = int(step_idx[i + 1].item())
            t_b    = torch.full((T_win,), t_cur, device=device, dtype=torch.long)
            inp    = torch.cat([torch.cat([x, cond_norm[ti]], dim=1) for ti in range(T_win)], dim=0)
            with torch.amp.autocast("cuda"):
                eps = eval_model(inp, t_b)
            eps_c   = eps[half:half+1].float()
            ab_t    = ab[t_cur];  ab_p = ab[t_next]
            x0_pred = (x - (1 - ab_t).sqrt() * eps_c) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 2)
            x       = ab_p.sqrt() * x0_pred + (1 - ab_p).sqrt() * eps_c

        restored = ((x.squeeze(0) + 1.0) * 0.5).clamp(0, 1)

        deg_np  = _to_np_rgb(centre_t)
        rest_np = _to_np_rgb(restored)
        orig_np = _to_np_rgb(orig_t)

        row_rgb = np.concatenate([deg_np, rest_np, orig_np], axis=1)
        writer.write(cv2.cvtColor(row_rgb, cv2.COLOR_RGB2BGR))

    writer.release()
    eval_model.train()
    LOGGER.info(f"  Sample video saved → {vid_path}  ({len(frame_indices)} frames)")


def train(args: argparse.Namespace) -> None:
    cfg    = _load_config(args.config)
    gpu_ids = None
    if args.gpus is not None:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
        torch.cuda.set_device(gpu_ids[0])
    elif args.gpu is not None:
        gpu_ids = [args.gpu]
        torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info(f"Device: {device}")

    # Dataset
    T_win = args.temporal_window or cfg.get("temporal_window", 3)
    patch_size = args.patch_size or cfg.get("training", {}).get("patch_size", 256)
    full_ds    = RestorationDataset(
        Path(args.data), T=T_win, patch_size=patch_size, augment=True
    )
    val_len  = max(1, int(len(full_ds) * args.val_split))
    trn_len  = len(full_ds) - val_len
    trn_ds, val_ds = random_split(full_ds, [trn_len, val_len])
    LOGGER.info(f"Dataset: {trn_len} train / {val_len} val samples")

    # Optionally subsample training set for faster iteration
    if args.max_samples and args.max_samples < len(trn_ds):
        indices = torch.randperm(len(trn_ds))[:args.max_samples].tolist()
        trn_ds  = torch.utils.data.Subset(trn_ds, indices)
        LOGGER.info(f"Subsampled training set to {args.max_samples} samples")

    trn_loader = DataLoader(
        trn_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )

    # Model + schedule
    model    = _build_model(cfg, device)
    if gpu_ids and len(gpu_ids) > 1 and args.batch_size >= len(gpu_ids):
        LOGGER.info(f"Using GPUs {gpu_ids} via DataParallel")
        model = nn.DataParallel(model, device_ids=gpu_ids)
    else:
        LOGGER.info(f"Using single GPU {device}")
    schedule = DiffusionSchedule(T=1000).to(device)
    vgg_loss = VGGPerceptualLoss().to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"RestoreUNet params: {param_count/1e6:.1f}M")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

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
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))

    vgg_w = float(cfg.get("training", {}).get("vgg_weight", 0.1))

    steps_per_epoch = args.steps_per_epoch or len(trn_loader)

    for epoch in range(start_epoch, args.epochs):
        # ── Train ──────────────────────────────────────────────────────────
        LOGGER.info(f"Epoch {epoch+1}/{args.epochs} starting ({steps_per_epoch} steps) ...")
        model.train()
        trn_loss = 0.0
        t_start  = time.perf_counter()

        for step, (deg_window, orig_centre) in enumerate(trn_loader):
            if step >= steps_per_epoch:
                break

            B  = orig_centre.size(0)
            deg_window   = deg_window.to(device, non_blocking=True)
            orig_centre  = orig_centre.to(device, non_blocking=True)

            t_idx = torch.randint(0, schedule.T, (B,), device=device)

            x0    = orig_centre * 2.0 - 1.0
            x_t, noise = schedule.q_sample(x0, t_idx)

            c = T_win // 2
            inputs = []
            for ti in range(T_win):
                deg_i    = deg_window[:, ti*3:(ti+1)*3] * 2.0 - 1.0
                inputs.append(torch.cat([x_t, deg_i], dim=1))

            model_in = torch.stack(inputs, dim=1).view(B * T_win, 6, *x_t.shape[-2:])
            t_rep    = t_idx.unsqueeze(1).expand(B, T_win).reshape(B * T_win)

            with torch.amp.autocast("cuda", enabled=args.amp):
                pred_noise  = model(model_in, t_rep).contiguous()
                pred_centre = pred_noise.view(B, T_win, 3, *x_t.shape[-2:])[:, c].contiguous()
                l1_loss     = F.l1_loss(pred_centre, noise.contiguous())
                sqrt_acp    = schedule.sqrt_acp[t_idx].view(B, 1, 1, 1)
                sqrt_omacp  = schedule.sqrt_one_minus_acp[t_idx].view(B, 1, 1, 1)
                x0_hat      = (x_t - sqrt_omacp * pred_centre) / sqrt_acp.clamp(min=1e-8)
                x0_hat      = x0_hat.clamp(-1.0, 1.0)
                perc_loss   = vgg_loss((x0_hat + 1) / 2, (x0 + 1) / 2)
                loss        = l1_loss + vgg_w * perc_loss

            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()

            trn_loss += loss.item()

            if (step + 1) % args.log_every == 0:
                elapsed  = time.perf_counter() - t_start
                sps      = (step + 1) / elapsed
                eta      = (steps_per_epoch - step - 1) / sps if sps > 0 else 0
                avg      = trn_loss / (step + 1)
                LOGGER.info(
                    f"Epoch {epoch+1}/{args.epochs} step {step+1}/{steps_per_epoch} "
                    f"loss={avg:.4f} l1={l1_loss.item():.4f} vgg={perc_loss.item():.4f} "
                    f"ETA {eta/60:.1f}min"
                )

        trn_loss /= max(len(trn_loader), 1)
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        val_loss = 0.0
        if not args.no_val:
            # Unwrap DataParallel for validation to avoid Blackwell alignment issues
            eval_model = model.module if isinstance(model, nn.DataParallel) else model
            eval_model.eval()
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

                    pred_noise  = eval_model(model_in, t_rep)
                    pred_centre = pred_noise.view(B, T_win, 3, *x_t.shape[-2:])[:, c]
                    val_loss += F.l1_loss(pred_centre, noise).item()

            val_loss /= max(len(val_loader), 1)
            eval_model.train()

        elapsed = time.perf_counter() - t_start
        LOGGER.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"trn={trn_loss:.4f} val={val_loss:.4f} | "
            f"lr={optimiser.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.0f}s"
        )

        # ── Checkpoint ─────────────────────────────────────────────────────
        is_best = (val_loss < best_val) if not args.no_val else True
        if is_best:
            best_val = val_loss

        state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler":    scaler.state_dict(),
            "best_val":  best_val,
            "config":    cfg,
        }
        torch.save(state, ckpt_dir / "latest.pt")
        if is_best:
            torch.save(state, ckpt_dir / "best.pt")
            LOGGER.info(f"  *** New best val={best_val:.4f} → saved best.pt")

        # ── Visual samples ─────────────────────────────────────────────────
        if args.sample_every > 0 and (epoch + 1) % args.sample_every == 0:
            _save_samples(
                model, schedule, val_ds, device,
                out_dir=ckpt_dir / "samples",
                epoch=epoch,
                n_samples=args.num_samples,
            )

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
    p.add_argument("--gpu",           type=int,   default=None,
                   help="Single GPU index (e.g. --gpu 2)")
    p.add_argument("--gpus",          default=None,
                   help="Comma-separated GPU indices for DataParallel (e.g. --gpus 1,2,3,4)")
    p.add_argument("--no-val",         action="store_true",
                   help="Skip validation (avoids DataParallel issues on some GPUs)")
    p.add_argument("--amp",            action="store_true", default=True,
                   help="Enable AMP fp16 training (default: on)")
    p.add_argument("--no-amp",         dest="amp", action="store_false",
                   help="Disable AMP")
    p.add_argument("--max-samples",    type=int, default=None,
                   help="Subsample dataset to N samples (e.g. 50000)")
    p.add_argument("--sample-every",   type=int, default=10,
                   help="Save visual samples every N epochs (0 = disable)")
    p.add_argument("--num-samples",    type=int, default=4,
                   help="Number of validation samples to visualize")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="Cap steps per epoch regardless of dataset size")
    p.add_argument("--temporal-window", type=int, default=None,
                   help="Override temporal window size T (e.g. 1 3 5)")
    p.add_argument("--verbose",        action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
