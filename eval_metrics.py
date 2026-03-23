#!/usr/bin/env python3
"""
eval_metrics.py — Evaluate pipeline output against ground-truth originals.

Computes per-frame and aggregate PSNR, SSIM, LPIPS between:
  - pipeline output video / frame directory
  - ground-truth original video / frame directory

Usage:
    python eval_metrics.py
        --pred   outputs/pipeline/sample_pipeline.mp4
        --gt     /path/to/original_sample.mp4
        [--out-csv results/metrics.csv]
        [--max-frames N]
        [--verbose]

Outputs a table to stdout and optionally a CSV file.
"""
from __future__ import annotations

import argparse
import csv
import glob
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

LOGGER = logging.getLogger("eval.metrics")


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    LOGGER.setLevel(level)


# ── Frame loading ─────────────────────────────────────────────────────────────

def _load_frames(source: str, max_frames: Optional[int]) -> List[np.ndarray]:
    p = Path(source)
    if p.is_dir():
        paths = sorted(
            glob.glob(str(p / "*.png")) + glob.glob(str(p / "*.jpg"))
        )
        frames = [cv2.imread(fp) for fp in paths]
    elif p.is_file():
        cap = cv2.VideoCapture(str(p))
        frames = []
        while True:
            ok, frm = cap.read()
            if not ok:
                break
            frames.append(frm)
        cap.release()
    else:
        raise FileNotFoundError(f"Not found: {p}")

    if max_frames:
        frames = frames[:max_frames]
    return frames


# ── Metric functions ──────────────────────────────────────────────────────────

def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """PSNR in dB (uint8 images, BGR)."""
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def _ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """SSIM (per-channel, then averaged)."""
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        # Convert BGR→RGB and compute SSIM
        pred_rgb = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        gt_rgb   = cv2.cvtColor(gt,   cv2.COLOR_BGR2RGB)
        score, _ = sk_ssim(
            pred_rgb, gt_rgb,
            multichannel=True,
            data_range=255,
            full=True,
            channel_axis=2,
        )
        return float(score)
    except ImportError:
        # Fallback: manual SSIM on grayscale
        pred_g = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gt_g   = cv2.cvtColor(gt,   cv2.COLOR_BGR2GRAY).astype(np.float64)
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        mu1, mu2 = pred_g.mean(), gt_g.mean()
        s1  = pred_g.std()  ** 2
        s2  = gt_g.std()    ** 2
        s12 = np.mean((pred_g - mu1) * (gt_g - mu2))
        return float(
            (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
            / ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2))
        )


class _LPIPSCalc:
    """Lazy-loaded LPIPS calculator (avoids import error if not installed)."""

    def __init__(self) -> None:
        self._fn = None
        self._device = None

    def _init(self) -> bool:
        if self._fn is not None:
            return True
        try:
            import torch
            import lpips
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._fn = lpips.LPIPS(net="vgg").to(self._device).eval()
            return True
        except ImportError:
            return False

    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
        if not self._init():
            return None
        import torch
        def to_tensor(img: np.ndarray) -> "torch.Tensor":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)
        with torch.no_grad():
            val = self._fn(to_tensor(pred), to_tensor(gt))
        return float(val.item())


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    pred_frames: List[np.ndarray],
    gt_frames:   List[np.ndarray],
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    n = min(len(pred_frames), len(gt_frames))
    if n == 0:
        raise ValueError("No frames to evaluate.")

    lpips_calc = _LPIPSCalc()
    rows: List[Dict[str, Any]] = []

    for i in range(n):
        pred = pred_frames[i]
        gt   = gt_frames[i]

        # Resize pred to match gt if needed
        if pred.shape[:2] != gt.shape[:2]:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)

        row: Dict[str, Any] = {
            "frame": i,
            "psnr":  _psnr(pred, gt),
            "ssim":  _ssim(pred, gt),
        }
        lp = lpips_calc(pred, gt)
        row["lpips"] = lp if lp is not None else float("nan")

        rows.append(row)
        if verbose:
            LOGGER.debug(
                f"Frame {i:5d}  PSNR={row['psnr']:.2f}  "
                f"SSIM={row['ssim']:.4f}  LPIPS={row['lpips']:.4f}"
            )

    # Aggregate
    agg: Dict[str, Any] = {
        "n_frames":   n,
        "psnr_mean":  float(np.mean([r["psnr"]  for r in rows])),
        "psnr_std":   float(np.std( [r["psnr"]  for r in rows])),
        "ssim_mean":  float(np.mean([r["ssim"]  for r in rows])),
        "ssim_std":   float(np.std( [r["ssim"]  for r in rows])),
    }
    lpips_vals = [r["lpips"] for r in rows if not np.isnan(r["lpips"])]
    if lpips_vals:
        agg["lpips_mean"] = float(np.mean(lpips_vals))
        agg["lpips_std"]  = float(np.std(lpips_vals))
    else:
        agg["lpips_mean"] = float("nan")
        agg["lpips_std"]  = float("nan")

    return rows, agg


def _print_table(agg: Dict[str, Any]) -> None:
    print("\n" + "=" * 50)
    print(f"  Frames evaluated : {agg['n_frames']}")
    print(f"  PSNR  (dB)       : {agg['psnr_mean']:.2f} ± {agg['psnr_std']:.2f}")
    print(f"  SSIM             : {agg['ssim_mean']:.4f} ± {agg['ssim_std']:.4f}")
    if not np.isnan(agg["lpips_mean"]):
        print(f"  LPIPS ↓          : {agg['lpips_mean']:.4f} ± {agg['lpips_std']:.4f}")
    else:
        print("  LPIPS            : N/A (lpips package not installed)")
    print("=" * 50 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pipeline quality metrics.")
    p.add_argument("--pred",       required=True, help="Predicted video or frame dir")
    p.add_argument("--gt",         required=True, help="Ground-truth video or frame dir")
    p.add_argument("--out-csv",    default=None,  help="Save per-frame CSV to this path")
    p.add_argument("--max-frames", type=int,      default=None)
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    print(f"Loading predicted frames from {args.pred} ...")
    pred_frames = _load_frames(args.pred, args.max_frames)
    print(f"Loading ground-truth frames from {args.gt} ...")
    gt_frames   = _load_frames(args.gt,   args.max_frames)

    print(f"Evaluating {min(len(pred_frames), len(gt_frames))} frames ...")
    rows, agg = evaluate(pred_frames, gt_frames, verbose=args.verbose)
    _print_table(agg)

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "psnr", "ssim", "lpips"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-frame CSV saved → {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
