#!/usr/bin/env python3
"""
run_upscale.py — Super-resolution using S3Diff (one-step diffusion SR).

S3Diff is a degradation-guided one-step diffusion SR model (SD-Turbo based).
Pretrained weights: https://huggingface.co/zhangap/S3Diff
Paper: https://arxiv.org/abs/2405.10044

Usage:
    python run_upscale.py --input FRAMES_DIR_OR_VIDEO
                          [--output OUTPUT.mp4]
                          [--scale 4]
                          [--tile 512]
                          [--seed 42]
                          [--fps 30]
                          [--half]
                          [--verbose]
"""
from __future__ import annotations

import argparse
import logging
import math
import random
import sys
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

LOGGER = logging.getLogger("codec.upscale")
ROOT = Path(__file__).resolve().parent
S3DIFF_DIR = ROOT / "S3Diff"

# Weights paths
DE_NET_PATH    = ROOT / "weights" / "de_net.pth"
S3DIFF_PATH    = ROOT / "weights" / "s3diff.pkl"

DE_NET_URL  = "https://huggingface.co/zhangap/S3Diff/resolve/main/de_net.pth"
S3DIFF_URL  = "https://huggingface.co/zhangap/S3Diff/resolve/main/s3diff.pkl"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", force=True)


def _status(msg: str) -> None:
    print(msg, flush=True)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _download(url: str, dest: Path) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    _status(f"Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    _status(f"  Saved to {dest}")


def _ensure_s3diff_repo() -> None:
    if S3DIFF_DIR.exists():
        return
    import subprocess
    _status("Cloning S3Diff repository ...")
    subprocess.run(
        ["git", "clone", "https://github.com/ArcticHare105/S3Diff.git", str(S3DIFF_DIR)],
        check=True,
    )
    _status("S3Diff cloned.")


def _ensure_weights() -> None:
    if not DE_NET_PATH.exists():
        _download(DE_NET_URL, DE_NET_PATH)
    if not S3DIFF_PATH.exists():
        _download(S3DIFF_URL, S3DIFF_PATH)


def _load_models(device: torch.device, half: bool):
    _ensure_s3diff_repo()
    _ensure_weights()

    # Add S3Diff src to path
    s3diff_src = S3DIFF_DIR / "src"
    if str(s3diff_src) not in sys.path:
        sys.path.insert(0, str(s3diff_src))
    if str(S3DIFF_DIR) not in sys.path:
        sys.path.insert(0, str(S3DIFF_DIR))

    from huggingface_hub import snapshot_download
    from s3diff import S3Diff
    from de_net import DEResNet

    _status("Downloading stabilityai/sd-turbo (cached after first run) ...")
    sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")

    _status("Loading S3Diff ...")
    net_sr = S3Diff(
        lora_rank_unet=32,
        lora_rank_vae=16,
        sd_path=sd_path,
        pretrained_path=str(S3DIFF_PATH),
        device=device,
    )
    net_sr.set_eval()
    if half:
        net_sr.half()

    _status("Loading degradation estimator ...")
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    ckpt = torch.load(str(DE_NET_PATH), map_location="cpu")
    net_de.load_state_dict(ckpt.get("state_dict", ckpt))
    net_de.to(device).eval()

    return net_sr, net_de


def _upscale_frame(
    frame_bgr: np.ndarray,
    net_sr,
    net_de,
    scale: int,
    device: torch.device,
    half: bool,
    align: str = "wavelet",
) -> np.ndarray:
    # BGR → RGB tensor [0,1]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    im_lr = transforms.ToTensor()(frame_rgb).unsqueeze(0).to(device)
    if half:
        im_lr = im_lr.half()

    ori_h, ori_w = im_lr.shape[2:]

    # Bilinear upsample to target resolution
    im_lr_up = F.interpolate(
        im_lr, size=(ori_h * scale, ori_w * scale),
        mode="bilinear", align_corners=False,
    ).contiguous()

    # Normalize to [-1, 1] and pad to multiples of 64
    im_norm = im_lr_up * 2.0 - 1.0
    im_norm = im_norm.clamp(-1.0, 1.0)
    res_h, res_w = im_norm.shape[2:]
    pad_h = math.ceil(res_h / 64) * 64 - res_h
    pad_w = math.ceil(res_w / 64) * 64 - res_w
    im_padded = F.pad(im_norm, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        deg_score = net_de(im_lr)
        output = net_sr(im_padded, deg_score, prompt="a clear and high quality image")

    # Remove padding, denormalize
    output = output[:, :, :res_h, :res_w]
    out_tensor = (output * 0.5 + 0.5).clamp(0, 1).cpu().float()

    # Optional wavelet color fix for consistency
    if align == "wavelet":
        try:
            from s3diff_utils import wavelet_color_fix
            out_pil = transforms.ToPILImage()(out_tensor[0])
            ref_pil = transforms.ToPILImage()(im_lr_up[0].cpu().float())
            out_pil = wavelet_color_fix(out_pil, ref_pil)
            out_tensor = transforms.ToTensor()(out_pil).unsqueeze(0)
        except Exception:
            pass  # skip color fix if utils not available

    # Back to BGR numpy
    out_np = (out_tensor[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)


def _load_frames(source: Path) -> tuple[list[np.ndarray], float]:
    if source.is_dir():
        paths = sorted(source.glob("*.png")) + sorted(source.glob("*.jpg"))
        if not paths:
            raise ValueError(f"No PNG/JPG files in {source}")
        return [cv2.imread(str(p)) for p in paths], 30.0
    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, frm = cap.read()
        if not ok:
            break
        frames.append(frm)
    cap.release()
    return frames, fps


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-step diffusion SR with S3Diff.")
    p.add_argument("--input",   required=True,  help="Input video or frames directory")
    p.add_argument("--output",  default=None,   help="Output video path")
    p.add_argument("--scale",   type=int, default=4, choices=[2, 4],
                   help="Upscale factor (default: 4)")
    p.add_argument("--tile",    type=int, default=0,
                   help="Tile size to limit VRAM (0=no tiling, default: 0)")
    p.add_argument("--seed",    type=int, default=42,
                   help="Fixed seed for temporal consistency (default: 42)")
    p.add_argument("--half",    action="store_true", help="FP16 inference")
    p.add_argument("--fps",     type=float, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    input_path = Path(args.input)
    if not input_path.exists():
        _status(f"ERROR: Input not found: {input_path}")
        return 1

    out_path = Path(args.output) if args.output else (
        Path("outputs/upscaling") / (input_path.stem + f"_s3diff_x{args.scale}.mp4")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _status(f"Device: {device}")

    _set_seed(args.seed)

    _status("Loading S3Diff model ...")
    net_sr, net_de = _load_models(device, half=args.half)

    _status(f"Loading frames from {input_path} ...")
    frames, src_fps = _load_frames(input_path)
    fps = args.fps or src_fps
    _status(f"Loaded {len(frames)} frames at {fps:.1f} fps")

    _status(f"Upscaling {len(frames)} frames with S3Diff x{args.scale} (seed={args.seed}) ...")
    t0 = time.perf_counter()
    upscaled = []
    for i, frame in enumerate(frames):
        _set_seed(args.seed)  # fixed seed per frame for temporal consistency
        out_frame = _upscale_frame(frame, net_sr, net_de, args.scale, device, args.half)
        upscaled.append(out_frame)
        if (i + 1) % 25 == 0 or (i + 1) == len(frames):
            elapsed = time.perf_counter() - t0
            _status(f"  {i+1}/{len(frames)} frames  ({(i+1)/elapsed:.1f} fps)")

    elapsed = time.perf_counter() - t0
    _status(f"Upscaling done in {elapsed:.1f}s ({len(frames)/elapsed:.1f} fps avg)")

    _status(f"Writing video -> {out_path} ...")
    h, w = upscaled[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in upscaled:
        writer.write(frame)
    writer.release()
    _status(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
