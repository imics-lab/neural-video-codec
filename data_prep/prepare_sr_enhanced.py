#!/usr/bin/env python3
"""
Prepare training pairs for Model S (enhanced SR: upsample + denoise + sharpen).

Uses the restoration dataset (already compressed+decompressed) as the degraded
input, paired with the original HR frame as target. This trains Model S to
simultaneously:
  - 2x upsample
  - Remove DCVC compression artifacts
  - Sharpen / restore fine detail

Pair layout (reads from restoration pairs, writes to enhanced SR pairs):
    Input (LR degraded): restoration_pairs/degraded/<stem>/<name>.png  -- downscaled 2x
    Target (HR clean):   restoration_pairs/original/<stem>/<name>.png  -- full res

Output:
    <output_dir>/hr/<stem>/<name>.png   -- original clean HR
    <output_dir>/lr/<stem>/<name>.png   -- DCVC-degraded + bicubic downscaled

Usage:
    python data_prep/prepare_sr_enhanced.py
        --restoration-dir data/restoration_pairs
        --output          data/sr_enhanced_pairs
        [--scale 2]
        [--max-videos N]
        [--verbose]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("data_prep.sr_enhanced")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", force=True)
    LOGGER.setLevel(level)


def _find_stems(restoration_dir: Path) -> List[str]:
    orig_root = restoration_dir / "original"
    if not orig_root.exists():
        return []
    return sorted(d.name for d in orig_root.iterdir() if d.is_dir())


def process_stem(
    stem: str,
    restoration_dir: Path,
    output_dir: Path,
    scale: int,
) -> int:
    orig_dir = restoration_dir / "original" / stem
    deg_dir  = restoration_dir / "degraded"  / stem

    if not orig_dir.exists() or not deg_dir.exists():
        LOGGER.warning(f"  Missing dirs for {stem}, skipping.")
        return 0

    hr_out = output_dir / "hr" / stem
    lr_out = output_dir / "lr" / stem
    hr_out.mkdir(parents=True, exist_ok=True)
    lr_out.mkdir(parents=True, exist_ok=True)

    pairs_written = 0
    for orig_path in sorted(orig_dir.glob("*.png")):
        deg_path = deg_dir / orig_path.name
        if not deg_path.exists():
            continue

        hr = cv2.imread(str(orig_path))   # clean HR  (H, W, 3)
        deg = cv2.imread(str(deg_path))   # compressed degraded, same resolution

        if hr is None or deg is None:
            continue

        H, W = hr.shape[:2]
        # Ensure dimensions divisible by scale
        H_c = (H // scale) * scale
        W_c = (W // scale) * scale
        hr  = hr[:H_c, :W_c]
        deg = deg[:H_c, :W_c]

        # Downscale the degraded frame to create the LR input
        lr_w, lr_h = W_c // scale, H_c // scale
        lr_deg = cv2.resize(deg, (lr_w, lr_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(hr_out / orig_path.name), hr)
        cv2.imwrite(str(lr_out / orig_path.name), lr_deg)
        pairs_written += 1

    return pairs_written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--restoration-dir", required=True,
                   help="Path to restoration_pairs dir (has original/ and degraded/)")
    p.add_argument("--output",  required=True, help="Output sr_enhanced_pairs dir")
    p.add_argument("--scale",   type=int, default=2)
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    rest_dir   = Path(args.restoration_dir)
    output_dir = Path(args.output)

    if not rest_dir.exists():
        print(f"ERROR: restoration dir not found: {rest_dir}", file=sys.stderr)
        return 1

    stems = _find_stems(rest_dir)
    if not stems:
        print(f"No video stems found under {rest_dir}/original/", file=sys.stderr)
        return 1

    if args.max_videos:
        stems = stems[:args.max_videos]

    print(f"Found {len(stems)} video stems | scale={args.scale}x | output -> {output_dir}")

    total = 0
    t0 = time.perf_counter()
    for i, stem in enumerate(stems, 1):
        print(f"[{i}/{len(stems)}] {stem}")
        try:
            n = process_stem(stem, rest_dir, output_dir, args.scale)
            total += n
            LOGGER.info(f"  {stem}: {n} pairs")
        except Exception as e:
            LOGGER.error(f"Failed on {stem}: {e}")

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.0f}s. Total pairs: {total}")
    print(f"  HR (clean):      {output_dir}/hr/")
    print(f"  LR (degraded):   {output_dir}/lr/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
