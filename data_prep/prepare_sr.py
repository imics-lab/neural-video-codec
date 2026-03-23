#!/usr/bin/env python3
"""
Prepare training pairs for Model S (super-resolution).

For each video in the input folder:
  1. Extract frames at original resolution (HR ground truth)
  2. Bicubic-downscale by `scale` factor to create LR input
  3. Save paired PNGs:
       <output_dir>/hr/<video_stem>/<frame_idx:06d>.png
       <output_dir>/lr/<video_stem>/<frame_idx:06d>.png

Usage:
    python data_prep/prepare_sr.py
        --videos /path/to/videos
        --output /path/to/sr_pairs
        --scale  2
        [--patch-size 256]
        [--patches-per-frame 4]
        [--max-videos N]
        [--verbose]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("data_prep.sr")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", force=True)
    LOGGER.setLevel(level)


def _find_videos(folder: Path) -> List[Path]:
    videos = []
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        videos.extend(folder.rglob(f"*{ext}"))
    return sorted(videos)


def _extract_frames(video_path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frm = cap.read()
        if not ok:
            break
        frames.append(frm)
    cap.release()
    return frames


def process_video(
    video_path: Path,
    output_dir: Path,
    scale: int = 2,
    patch_size: int = 256,
    patches_per_frame: int = 4,
) -> int:
    stem = video_path.stem
    hr_dir = output_dir / "hr" / stem
    lr_dir = output_dir / "lr" / stem
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Processing {stem} ...")
    t0 = time.perf_counter()

    hr_frames = _extract_frames(video_path)
    if not hr_frames:
        LOGGER.warning(f"  No frames in {stem}")
        return 0

    pairs_written = 0
    for fi, hr in enumerate(hr_frames):
        H, W = hr.shape[:2]
        # Ensure dimensions are divisible by scale
        H_crop = (H // scale) * scale
        W_crop = (W // scale) * scale
        hr = hr[:H_crop, :W_crop]

        lr_w, lr_h = W_crop // scale, H_crop // scale
        lr = cv2.resize(hr, (lr_w, lr_h), interpolation=cv2.INTER_AREA)

        # Extract patches at HR size, then save corresponding LR patches
        if H_crop >= patch_size and W_crop >= patch_size:
            for _ in range(patches_per_frame):
                y0_hr = np.random.randint(0, H_crop - patch_size + 1)
                x0_hr = np.random.randint(0, W_crop - patch_size + 1)
                # Align to scale grid
                y0_hr = (y0_hr // scale) * scale
                x0_hr = (x0_hr // scale) * scale

                hr_patch = hr[y0_hr:y0_hr + patch_size, x0_hr:x0_hr + patch_size]
                y0_lr = y0_hr // scale
                x0_lr = x0_hr // scale
                lr_patch = lr[y0_lr:y0_lr + patch_size // scale,
                               x0_lr:x0_lr + patch_size // scale]

                name = f"{fi:06d}_{y0_hr}_{x0_hr}.png"
                cv2.imwrite(str(hr_dir / name), hr_patch)
                cv2.imwrite(str(lr_dir / name), lr_patch)
                pairs_written += 1
        else:
            name = f"{fi:06d}.png"
            cv2.imwrite(str(hr_dir / name), hr)
            cv2.imwrite(str(lr_dir / name), lr)
            pairs_written += 1

    elapsed = time.perf_counter() - t0
    LOGGER.info(f"  {stem}: {pairs_written} pairs in {elapsed:.1f}s")
    return pairs_written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--videos",     required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--scale",      type=int, default=2)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--patches-per-frame", type=int, default=4)
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    videos_dir = Path(args.videos)
    output_dir = Path(args.output)

    if not videos_dir.exists():
        print(f"ERROR: Videos folder not found: {videos_dir}", file=sys.stderr)
        return 1

    videos = _find_videos(videos_dir)
    if args.max_videos:
        videos = videos[:args.max_videos]

    print(f"Found {len(videos)} videos | scale={args.scale}x | output -> {output_dir}")
    total = 0
    for i, vp in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {vp.name}")
        try:
            total += process_video(vp, output_dir, args.scale,
                                   args.patch_size, args.patches_per_frame)
        except Exception as e:
            LOGGER.error(f"Failed on {vp.name}: {e}")

    print(f"\nDone. Total pairs: {total}")
    print(f"  HR: {output_dir}/hr/")
    print(f"  LR: {output_dir}/lr/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
