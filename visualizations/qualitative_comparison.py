"""
qualitative_comparison.py — Generate graphics/qualitative.pdf (Figure 2).

Creates a side-by-side frame comparison showing one (or two) representative
frames through the pipeline with 4× magnified crop insets.

Columns (per row):
    1. Original (uncompressed)
    2. DCVC-Uniform at matched bitrate
    3. NCC dual-stream (post-blend, pre-restoration)
    4. NCC after RestoreUNet
    5. NCC after S3Diff

Usage:
    python visualizations/qualitative_comparison.py \
        --original   outputs/frames/original \
        --dcvc_uni   outputs/frames/dcvc_uniform \
        --ncc_blend  outputs/frames/ncc_blend \
        --ncc_rest   outputs/frames/ncc_restored \
        --ncc_sr     outputs/frames/ncc_sr \
        --frame_idx  42 \
        --crop       800,200,1200,600 \
        --out        visualizations/images/qualitative.pdf

    # Second row (optional):
        --original2  outputs/frames2/original \
        ... (same flags with "2" suffix)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


COLUMN_LABELS = [
    "Original",
    "DCVC-Uniform",
    "NCC (post-blend)",
    "NCC + RestoreUNet",
    "NCC + S3Diff",
]

CROP_COLOR  = "#FF4444"
INSET_SCALE = 4


def _load_frame(frame_dir: str, idx: int) -> np.ndarray:
    """Load frame `idx` from a directory of PNGs/JPGs (sorted)."""
    d = Path(frame_dir)
    files = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
    if not files:
        raise FileNotFoundError(f"No images in {d}")
    idx = min(idx, len(files) - 1)
    bgr = cv2.imread(str(files[idx]))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _crop_and_resize(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                     scale: int = 4) -> np.ndarray:
    crop = img[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    return cv2.resize(crop, (w * scale, h * scale),
                      interpolation=cv2.INTER_NEAREST)


def _draw_row(axes_row, frames, crop_box, row_label=None):
    x0, y0, x1, y1 = crop_box
    for col_idx, (ax, img) in enumerate(zip(axes_row, frames)):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])

        # Draw crop rectangle
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  linewidth=1.5, edgecolor=CROP_COLOR,
                                  facecolor="none")
        ax.add_patch(rect)

        # Inset crop in bottom-right corner
        inset = _crop_and_resize(img, x0, y0, x1, y1, INSET_SCALE)
        ih, iw = inset.shape[:2]
        fw, fh = img.shape[1], img.shape[0]
        inset_ax = ax.inset_axes(
            [(fw - iw - 4) / fw, 4 / fh, iw / fw, ih / fh],
            transform=ax.transData,
        )
        inset_ax.imshow(inset)
        inset_ax.set_xticks([]); inset_ax.set_yticks([])
        for spine in inset_ax.spines.values():
            spine.set_edgecolor(CROP_COLOR)
            spine.set_linewidth(1.5)

    if row_label:
        axes_row[0].set_ylabel(row_label, fontsize=8, fontweight="bold",
                                rotation=90, labelpad=4)


def make_figure(
    dirs: list[str],
    frame_idx: int,
    crop_box: tuple[int, int, int, int],
    dirs2: Optional[list[str]] = None,
    frame_idx2: int = 0,
    crop_box2: Optional[tuple[int, int, int, int]] = None,
    out_path: str = "visualizations/images/qualitative.pdf",
) -> None:
    n_rows = 1 if dirs2 is None else 2
    n_cols = 5

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.0, n_rows * 2.2),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.03})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Column headers
    for ci, label in enumerate(COLUMN_LABELS):
        axes[0, ci].set_title(label, fontsize=8, fontweight="bold", pad=3)

    # Row 1
    frames1 = [_load_frame(d, frame_idx) for d in dirs]
    _draw_row(axes[0], frames1, crop_box, row_label="Clip 1")

    # Row 2 (optional)
    if dirs2 is not None:
        frames2 = [_load_frame(d, frame_idx2) for d in dirs2]
        cb2 = crop_box2 if crop_box2 is not None else crop_box
        _draw_row(axes[1], frames2, cb2, row_label="Clip 2")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def _parse_crop(s: str) -> tuple[int, int, int, int]:
    parts = [int(x) for x in s.split(",")]
    assert len(parts) == 4, "crop must be x0,y0,x1,y1"
    return tuple(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--original",  required=True)
    p.add_argument("--dcvc_uni",  required=True)
    p.add_argument("--ncc_blend", required=True)
    p.add_argument("--ncc_rest",  required=True)
    p.add_argument("--ncc_sr",    required=True)
    p.add_argument("--frame_idx", type=int, default=0)
    p.add_argument("--crop",      default="400,100,800,500",
                   help="x0,y0,x1,y1 crop box for inset")
    # Second row (optional)
    p.add_argument("--original2",  default=None)
    p.add_argument("--dcvc_uni2",  default=None)
    p.add_argument("--ncc_blend2", default=None)
    p.add_argument("--ncc_rest2",  default=None)
    p.add_argument("--ncc_sr2",    default=None)
    p.add_argument("--frame_idx2", type=int, default=0)
    p.add_argument("--crop2",      default=None)
    p.add_argument("--out", default="visualizations/images/qualitative.pdf")
    args = p.parse_args()

    dirs  = [args.original, args.dcvc_uni, args.ncc_blend,
             args.ncc_rest, args.ncc_sr]
    dirs2 = None
    if args.original2:
        dirs2 = [args.original2, args.dcvc_uni2, args.ncc_blend2,
                 args.ncc_rest2, args.ncc_sr2]

    make_figure(
        dirs=dirs,
        frame_idx=args.frame_idx,
        crop_box=_parse_crop(args.crop),
        dirs2=dirs2,
        frame_idx2=args.frame_idx2,
        crop_box2=_parse_crop(args.crop2) if args.crop2 else None,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
