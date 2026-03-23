#!/usr/bin/env python3
"""
benchmark_qp.py — Sweep DCVC QP levels and report quality/bitrate table.

For each QP combination (roi_qp, bg_qp) in the sweep grid:
  1. Run compress → decompress on a sample video
  2. Compute PSNR / SSIM / LPIPS vs original
  3. Record archive size (proxy for bitrate)
  4. Print a results table and optionally save CSV

Usage:
    python benchmark_qp.py
        --video   /path/to/sample.mp4
        --config  configs/gpu/compression.yaml
        [--roi-qps  63]
        [--bg-qps   10 20 30 40]
        [--max-frames 30]
        [--out-csv results/qp_sweep.csv]
        [--verbose]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from eval_metrics import evaluate as _eval_frames, _load_frames


def _load_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        import json
        with open(path) as f:
            return json.load(f)


def _deep_set(d: Dict, keys: List[str], value: Any) -> Dict:
    """Set nested dict key in-place and return d."""
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
    return d


def _run_single(
    video_path: str,
    cfg: Dict[str, Any],
    roi_qp: int,
    bg_qp: int,
    max_frames: Optional[int],
) -> Dict[str, Any]:
    import copy
    run_cfg = copy.deepcopy(cfg)

    # Override QPs
    _deep_set(run_cfg, ["compression", "quality", "roi_qp"], roi_qp)
    _deep_set(run_cfg, ["compression", "quality", "bg_qp"],  bg_qp)

    from src.compression.phase_compress import compress_video
    from src.decompression.phase_decompress import decompress_archive

    t0 = time.perf_counter()
    archive_bytes = compress_video(video_path, run_cfg)
    comp_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    decomp_cfg = {"decompression": run_cfg.get("compression", {}).copy()}
    result = decompress_archive(archive_bytes, decomp_cfg)
    decomp_time = time.perf_counter() - t1

    pred_frames = result["frames"]
    if max_frames:
        pred_frames = pred_frames[:max_frames]

    gt_frames = _load_frames(video_path, max_frames)
    n = min(len(pred_frames), len(gt_frames))
    pred_frames = pred_frames[:n]
    gt_frames   = gt_frames[:n]

    _, agg = _eval_frames(pred_frames, gt_frames, verbose=False)
    agg["roi_qp"]      = roi_qp
    agg["bg_qp"]       = bg_qp
    agg["archive_kb"]  = round(len(archive_bytes) / 1024, 1)
    agg["comp_s"]      = round(comp_time, 2)
    agg["decomp_s"]    = round(decomp_time, 2)
    return agg


def _print_table(rows: List[Dict[str, Any]]) -> None:
    header = (
        f"{'roi_qp':>7} {'bg_qp':>7} {'arch_KB':>9} "
        f"{'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} "
        f"{'comp_s':>7} {'decomp_s':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        lpips = f"{r['lpips_mean']:.4f}" if not np.isnan(r.get("lpips_mean", float("nan"))) else " N/A "
        print(
            f"{r['roi_qp']:>7} {r['bg_qp']:>7} {r['archive_kb']:>9.1f} "
            f"{r['psnr_mean']:>7.2f} {r['ssim_mean']:>7.4f} {lpips:>7} "
            f"{r['comp_s']:>7.2f} {r['decomp_s']:>8.2f}"
        )
    print()


def _save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "roi_qp", "bg_qp", "archive_kb",
        "psnr_mean", "psnr_std",
        "ssim_mean", "ssim_std",
        "lpips_mean", "lpips_std",
        "comp_s", "decomp_s", "n_frames",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"CSV saved → {out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep DCVC QP levels.")
    p.add_argument("--video",      required=True)
    p.add_argument("--config",     default="configs/gpu/compression.yaml")
    p.add_argument("--roi-qps",    type=int, nargs="+", default=[63],
                   help="ROI QP values to sweep (default: 63)")
    p.add_argument("--bg-qps",     type=int, nargs="+", default=[10, 20, 30, 40],
                   help="Background QP values to sweep")
    p.add_argument("--max-frames", type=int, default=30)
    p.add_argument("--out-csv",    default=None)
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

    cfg   = _load_config(args.config)
    pairs = list(product(args.roi_qps, args.bg_qps))
    print(f"Sweeping {len(pairs)} QP combination(s) on {args.video}")

    results = []
    for roi_qp, bg_qp in pairs:
        print(f"  roi_qp={roi_qp} bg_qp={bg_qp} ...", end=" ", flush=True)
        try:
            row = _run_single(args.video, cfg, roi_qp, bg_qp, args.max_frames)
            results.append(row)
            print(f"PSNR={row['psnr_mean']:.2f}dB  arch={row['archive_kb']:.1f}KB")
        except Exception as e:
            print(f"FAILED: {e}")

    if results:
        _print_table(results)
        if args.out_csv:
            _save_csv(results, args.out_csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
