"""
temporal_grid.py — Generate graphics/temporal.pdf (Figure 3, optional).

Creates a 3×4 grid showing temporal coherence across methods:
    Rows:    DCVC-Uniform | NCC (no temporal attn) | NCC Full
    Columns: frame t | t+1 | t+2 | t+3

Usage:
    python visualizations/temporal_grid.py \
        --dcvc_uni   outputs/frames/dcvc_uniform \
        --ncc_no_ta  outputs/frames/ncc_no_temporal_attn \
        --ncc_full   outputs/frames/ncc_sr \
        --start_idx  40 \
        --crop       700,150,1050,500 \
        --out        visualizations/images/temporal.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROW_LABELS = [
    "DCVC-Uniform",
    "NCC (no temporal attn)",
    "NCC Full",
]
N_FRAMES = 4


def _load_frame(frame_dir: str, idx: int) -> np.ndarray:
    d = Path(frame_dir)
    files = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
    if not files:
        raise FileNotFoundError(f"No images in {d}")
    idx = min(idx, len(files) - 1)
    bgr = cv2.imread(str(files[idx]))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _crop(img: np.ndarray, box: tuple) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dcvc_uni",  required=True)
    p.add_argument("--ncc_no_ta", required=True,
                   help="Frames from NCC without temporal attention")
    p.add_argument("--ncc_full",  required=True)
    p.add_argument("--start_idx", type=int, default=40)
    p.add_argument("--crop",      default="400,100,900,550",
                   help="x0,y0,x1,y1 crop region to display")
    p.add_argument("--out", default="visualizations/images/temporal.pdf")
    args = p.parse_args()

    crop_box = tuple(int(x) for x in args.crop.split(","))
    dirs = [args.dcvc_uni, args.ncc_no_ta, args.ncc_full]

    n_rows, n_cols = 3, N_FRAMES
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.8, n_rows * 2.0),
                             gridspec_kw={"hspace": 0.06, "wspace": 0.03})

    # Column headers
    for ci in range(N_FRAMES):
        axes[0, ci].set_title(f"$t+{ci}$", fontsize=9, fontweight="bold", pad=3)

    for ri, (row_label, frame_dir) in enumerate(zip(ROW_LABELS, dirs)):
        for ci in range(N_FRAMES):
            img = _load_frame(frame_dir, args.start_idx + ci)
            cropped = _crop(img, crop_box)
            axes[ri, ci].imshow(cropped)
            axes[ri, ci].set_xticks([])
            axes[ri, ci].set_yticks([])
            if ci == 0:
                axes[ri, ci].set_ylabel(row_label, fontsize=8,
                                         fontweight="bold", rotation=90,
                                         labelpad=4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
