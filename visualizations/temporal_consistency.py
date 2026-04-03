"""
temporal_consistency.py — Compute optical-flow warping error (temporal flickering).

For each consecutive frame pair (F_t, F_{t+1}):
    1. Compute optical flow w_{t→t+1} using Farneback or RAFT
    2. Warp F_{t+1} back to frame t using w
    3. Compute MAE between F_t and warped(F_{t+1}) in non-occluded regions

Lower score = more temporally consistent.

Usage:
    # Single method
    python visualizations/temporal_consistency.py \
        --frames  outputs/frames/ncc_sr \
        --name    "NCC Full"

    # Compare multiple methods
    python visualizations/temporal_consistency.py \
        --compare \
        dcvc_uniform=outputs/frames/dcvc_uniform \
        ncc_no_ta=outputs/frames/ncc_no_temporal_attn \
        ncc_full=outputs/frames/ncc_sr
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _load_frames(frame_dir: str) -> list[np.ndarray]:
    d = Path(frame_dir)
    files = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
    if not files:
        raise FileNotFoundError(f"No images in {d}")
    frames = []
    for f in files:
        bgr = cv2.imread(str(f))
        if bgr is not None:
            frames.append(bgr)
    return frames


def _warp(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp frame by flow using remap."""
    h, w = flow.shape[:2]
    map_x = (flow[:, :, 0] + np.arange(w)[None, :]).astype(np.float32)
    map_y = (flow[:, :, 1] + np.arange(h)[:, None]).astype(np.float32)
    return cv2.remap(frame, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _occlusion_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray,
                    threshold: float = 1.0) -> np.ndarray:
    """Simple occlusion mask: pixels where forward+backward flow don't agree."""
    warped_bwd = _warp(flow_bwd, flow_fwd)
    diff = np.linalg.norm(flow_fwd + warped_bwd, axis=2)
    return diff < threshold  # True = non-occluded


def compute_warp_error_farneback(frames: list[np.ndarray]) -> tuple[float, float]:
    """
    Compute mean warping error over consecutive frame pairs using
    Farneback dense optical flow.

    Returns (mean_mae, std_mae).
    """
    errors = []
    for i in range(len(frames) - 1):
        f0 = frames[i]
        f1 = frames[i + 1]

        gray0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

        # Forward flow: t → t+1
        flow_fwd = cv2.calcOpticalFlowFarneback(
            gray0, gray1, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        # Backward flow: t+1 → t (for occlusion mask)
        flow_bwd = cv2.calcOpticalFlowFarneback(
            gray1, gray0, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        occ_mask = _occlusion_mask(flow_fwd, flow_bwd)

        # Warp f1 back to f0's space using forward flow
        warped_f1 = _warp(f1, flow_fwd)

        # MAE in non-occluded region
        diff = np.abs(f0.astype(np.float32) - warped_f1.astype(np.float32))
        mae = diff[occ_mask].mean()
        errors.append(mae)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(frames)-1} pairs", flush=True)

    return float(np.mean(errors)), float(np.std(errors))


def evaluate_method(frame_dir: str, name: str) -> dict:
    print(f"Loading frames from {frame_dir} ...")
    frames = _load_frames(frame_dir)
    print(f"  {len(frames)} frames. Computing warping error ...")
    mean_err, std_err = compute_warp_error_farneback(frames)
    return {"name": name, "n_frames": len(frames),
            "warp_error_mean": mean_err, "warp_error_std": std_err}


def print_table(results: list[dict]) -> None:
    header = f"{'Method':<40} {'Frames':>7} {'Warp Error':>12} {'±':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['name']:<40} {r['n_frames']:>7} "
              f"{r['warp_error_mean']:>12.4f} {r['warp_error_std']:>6.4f}")


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--frames", default=None,
                   help="Frame directory for single-method evaluation")
    p.add_argument("--name",   default="")
    p.add_argument("--compare", nargs="*", metavar="NAME=DIR",
                   help="Multiple methods: name=path name=path ...")
    p.add_argument("--out",    default=None, help="Save CSV results")
    args = p.parse_args()

    results = []

    if args.compare:
        for item in args.compare:
            name, path = item.split("=", 1)
            results.append(evaluate_method(path, name))
    elif args.frames:
        results.append(evaluate_method(args.frames, args.name or args.frames))
    else:
        p.print_help()
        sys.exit(1)

    print()
    print_table(results)

    if args.out:
        import csv
        from pathlib import Path as P
        P(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV saved: {args.out}")


if __name__ == "__main__":
    main()
