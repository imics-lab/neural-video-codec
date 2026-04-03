"""
compute_metrics.py — Compute PSNR / SSIM / LPIPS for Tables 1 and 2.

Evaluates frame pairs (predicted vs. ground-truth) from a directory of images
and prints a summary table.  Optionally restricts evaluation to the ROI region
using per-frame detection JSON produced by the pipeline.

Usage:
    # Full-frame metrics
    python visualizations/compute_metrics.py \
        --pred  outputs/frames/ncc_sr \
        --gt    outputs/frames/original \
        --name  "NCC-full"

    # ROI-only metrics (requires detections.json from pipeline archive)
    python visualizations/compute_metrics.py \
        --pred  outputs/frames/ncc_sr \
        --gt    outputs/frames/original \
        --detections outputs/detections.json \
        --name  "NCC-full (ROI)"

    # Batch — compare multiple methods at once
    python visualizations/compute_metrics.py \
        --batch configs/eval_batch.yaml \
        --out   visualizations/images/metrics.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:
    ssim_fn = None

try:
    import torch
    import lpips as lpips_lib
    _lpips_net = None
    def _get_lpips():
        global _lpips_net
        if _lpips_net is None:
            _lpips_net = lpips_lib.LPIPS(net="alex")
            if torch.cuda.is_available():
                _lpips_net = _lpips_net.cuda()
            _lpips_net.eval()
        return _lpips_net
except ImportError:
    lpips_lib = None


# ── Metric functions ──────────────────────────────────────────────────────────

def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * math.log10(255.0 ** 2 / mse)


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    if ssim_fn is None:
        return float("nan")
    return float(ssim_fn(pred, gt, channel_axis=2, data_range=255))


def lpips_score(pred: np.ndarray, gt: np.ndarray) -> float:
    if lpips_lib is None:
        return float("nan")
    import torch
    net = _get_lpips()
    def _to_t(x):
        t = torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0)
        t = t.permute(2, 0, 1).unsqueeze(0)
        if torch.cuda.is_available():
            t = t.cuda()
        return t
    with torch.no_grad():
        score = net(_to_t(pred), _to_t(gt)).item()
    return score


# ── ROI masking ───────────────────────────────────────────────────────────────

def _roi_mask(h: int, w: int, detections: list[dict]) -> Optional[np.ndarray]:
    """Build a binary mask (H,W) from detection bboxes for a single frame."""
    mask = np.zeros((h, w), dtype=bool)
    for det in detections:
        x0 = max(0, int(det.get("x0", det.get("x1", 0))))
        y0 = max(0, int(det.get("y0", det.get("y1", 0))))
        x1 = min(w, int(det.get("x1", det.get("x2", w))))
        y1 = min(h, int(det.get("y1", det.get("y2", h))))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask if mask.any() else None


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    pred_dir: str,
    gt_dir: str,
    detections_path: Optional[str] = None,
    name: str = "",
) -> dict:
    pred_files = sorted(Path(pred_dir).glob("*.png")) + \
                 sorted(Path(pred_dir).glob("*.jpg"))
    gt_files   = sorted(Path(gt_dir).glob("*.png")) + \
                 sorted(Path(gt_dir).glob("*.jpg"))
    pred_files = sorted(pred_files)
    gt_files   = sorted(gt_files)

    n = min(len(pred_files), len(gt_files))
    if n == 0:
        raise FileNotFoundError(f"No matching frames in {pred_dir} / {gt_dir}")

    detections = {}
    if detections_path and Path(detections_path).exists():
        with open(detections_path) as f:
            data = json.load(f)
        # Support both list and dict-keyed formats
        if isinstance(data, list):
            detections = {i: data[i] for i in range(len(data))}
        else:
            detections = {int(k): v for k, v in data.items()}

    psnr_list, ssim_list, lpips_list = [], [], []

    for i, (pf, gf) in enumerate(zip(pred_files, gt_files)):
        pred_bgr = cv2.imread(str(pf))
        gt_bgr   = cv2.imread(str(gf))
        if pred_bgr is None or gt_bgr is None:
            continue

        # Resize pred to match gt if needed
        if pred_bgr.shape != gt_bgr.shape:
            pred_bgr = cv2.resize(pred_bgr, (gt_bgr.shape[1], gt_bgr.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)

        pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)
        gt_rgb   = cv2.cvtColor(gt_bgr,   cv2.COLOR_BGR2RGB)

        # Apply ROI mask if available
        if detections and i in detections:
            h, w = gt_rgb.shape[:2]
            mask = _roi_mask(h, w, detections.get(i, []))
            if mask is not None:
                ys, xs = np.where(mask)
                y0, y1 = ys.min(), ys.max() + 1
                x0, x1 = xs.min(), xs.max() + 1
                pred_rgb = pred_rgb[y0:y1, x0:x1]
                gt_rgb   = gt_rgb  [y0:y1, x0:x1]

        psnr_list.append(psnr(pred_rgb, gt_rgb))
        ssim_list.append(ssim(pred_rgb, gt_rgb))
        lpips_list.append(lpips_score(pred_rgb, gt_rgb))

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n} frames evaluated", flush=True)

    def _mean(lst):
        valid = [x for x in lst if not math.isnan(x) and not math.isinf(x)]
        return float(np.mean(valid)) if valid else float("nan")

    def _std(lst):
        valid = [x for x in lst if not math.isnan(x) and not math.isinf(x)]
        return float(np.std(valid)) if valid else float("nan")

    return {
        "name":       name or pred_dir,
        "n_frames":   n,
        "psnr_mean":  _mean(psnr_list),
        "psnr_std":   _std(psnr_list),
        "ssim_mean":  _mean(ssim_list),
        "ssim_std":   _std(ssim_list),
        "lpips_mean": _mean(lpips_list),
        "lpips_std":  _std(lpips_list),
    }


def print_table(results: list[dict]) -> None:
    header = f"{'Method':<35} {'PSNR':>8} {'±':>5} {'SSIM':>7} {'±':>5} {'LPIPS':>7} {'±':>5}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['name']:<35} "
              f"{r['psnr_mean']:>8.3f} {r['psnr_std']:>5.3f} "
              f"{r['ssim_mean']:>7.4f} {r['ssim_std']:>5.4f} "
              f"{r['lpips_mean']:>7.4f} {r['lpips_std']:>5.4f}")


def save_csv(results: list[dict], out_path: str) -> None:
    import csv
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved: {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred",        default=None)
    p.add_argument("--gt",          default=None)
    p.add_argument("--detections",  default=None,
                   help="Path to detections.json for ROI masking")
    p.add_argument("--name",        default="")
    p.add_argument("--batch",       default=None,
                   help="YAML file with list of {name, pred, gt, detections}")
    p.add_argument("--out",         default=None,
                   help="Save results to CSV")
    args = p.parse_args()

    results = []

    if args.batch:
        try:
            import yaml
            with open(args.batch) as f:
                entries = yaml.safe_load(f)
        except ImportError:
            print("ERROR: pyyaml required for --batch mode")
            sys.exit(1)
        for entry in entries:
            print(f"\nEvaluating: {entry['name']} ...")
            r = evaluate(entry["pred"], entry["gt"],
                         entry.get("detections"), entry["name"])
            results.append(r)
    elif args.pred and args.gt:
        print(f"Evaluating: {args.name or args.pred} ...")
        r = evaluate(args.pred, args.gt, args.detections, args.name)
        results.append(r)
    else:
        p.print_help()
        sys.exit(1)

    print()
    print_table(results)

    if args.out:
        save_csv(results, args.out)


if __name__ == "__main__":
    main()
