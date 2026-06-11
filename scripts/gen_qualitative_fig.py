#!/usr/bin/env python3
"""
gen_qualitative_fig.py  --  Build the qualitative comparison figure for the paper.

Creates a side-by-side panel: original | decomp only | ours (full pipeline),
with zoomed insets showing the animal region (top) and background (bottom).

Usage:
    python scripts/gen_qualitative_fig.py \
        --original  outputs/pipeline/bird1_original.mp4 \
        --decomp    outputs/pipeline/bird1_decompressed.mp4 \
        --restored  outputs/pipeline/bird1_restored.mp4 \
        --frame     30 \
        --roi       0.45 0.25 0.30 0.40   # x y w h as fractions of frame
    python scripts/gen_qualitative_fig.py --help
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "NCC__Neural_Compression_Codec" / "graphics"


def _read_frame(video: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _crop(img: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    H, W = img.shape[:2]
    x0 = max(0, int(x * W))
    y0 = max(0, int(y * H))
    x1 = min(W, int((x + w) * W))
    y1 = min(H, int((y + h) * H))
    return img[y0:y1, x0:x1]


def _bg_crop(img: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    """A patch clearly outside the ROI, roughly opposite corner."""
    bx = (x + 0.5) % 0.5
    by = (y + 0.5) % 0.5
    return _crop(img, bx, by, w, h)


def build_figure(
    original: np.ndarray,
    decomp:   np.ndarray,
    restored: np.ndarray,
    roi_xywh: tuple[float, float, float, float],
    out_path: Path,
) -> None:
    x, y, w, h = roi_xywh
    cols = [
        ("Original", original),
        ("DCVC dual (no restore)", decomp),
        ("Ours (full pipeline)", restored),
    ]

    fig, axes = plt.subplots(
        3, 3,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.05, "wspace": 0.02},
    )

    for col_idx, (title, img) in enumerate(cols):
        # Full frame
        ax = axes[0, col_idx]
        ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=3)
        ax.axis("off")
        # ROI rectangle overlay
        H, W = img.shape[:2]
        rect = mpatches.Rectangle(
            (x * W, y * H), w * W, h * H,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)

        # ROI crop
        ax_roi = axes[1, col_idx]
        ax_roi.imshow(_crop(img, x, y, w, h))
        ax_roi.axis("off")
        if col_idx == 0:
            ax_roi.set_ylabel("ROI crop", fontsize=8, labelpad=2)

        # Background crop (top-left corner, away from ROI)
        ax_bg = axes[2, col_idx]
        ax_bg.imshow(_bg_crop(img, x, y, w, h))
        ax_bg.axis("off")
        if col_idx == 0:
            ax_bg.set_ylabel("BG crop", fontsize=8, labelpad=2)

    plt.suptitle(
        "Qualitative comparison: original vs.\ DCVC dual (no restore) vs.\ full pipeline",
        fontsize=11, y=1.01,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate qualitative comparison figure")
    ap.add_argument("--original",  required=True, help="Original video path")
    ap.add_argument("--decomp",    required=True, help="Decompressed-only video path")
    ap.add_argument("--restored",  required=True, help="Full pipeline video path")
    ap.add_argument("--frame",     type=int, default=30, help="Frame index to extract")
    ap.add_argument("--roi",       type=float, nargs=4, default=[0.35, 0.2, 0.3, 0.4],
                    metavar=("X", "Y", "W", "H"),
                    help="ROI region as fractions of frame (x y w h)")
    ap.add_argument("--out",       default=None, help="Output PDF path")
    args = ap.parse_args()

    orig = _read_frame(Path(args.original), args.frame)
    dcp  = _read_frame(Path(args.decomp),   args.frame)
    rest = _read_frame(Path(args.restored),  args.frame)

    out = Path(args.out) if args.out else OUT_DIR / "qualitative_comparison.pdf"
    build_figure(orig, dcp, rest, tuple(args.roi), out)
    # Also save PNG
    png = out.with_suffix(".png")
    build_figure(orig, dcp, rest, tuple(args.roi), png)


if __name__ == "__main__":
    main()
