#!/usr/bin/env python3
"""
gen_pipeline_fig.py  --  Generate the pipeline overview figure for the paper.

Produces: NCC__Neural_Compression_Codec/graphics/pipeline_overview.pdf
          NCC__Neural_Compression_Codec/graphics/pipeline_overview.png

Usage:
    python scripts/gen_pipeline_fig.py
    python scripts/gen_pipeline_fig.py --out path/to/figure.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "NCC__Neural_Compression_Codec" / "graphics"


def _box(ax, x, y, w, h, text, color, fontsize=9, text_color="white"):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor="white",
        facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold", wrap=True,
            multialignment="center")


def _arrow(ax, x0, y0, x1, y1, color="#444444"):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.5,
        ),
    )


def draw_pipeline(ax) -> None:
    # Color palette
    edge_col   = "#2c7bb6"   # blue for edge components
    server_col = "#d7191c"   # red for server components
    data_col   = "#1a9641"   # green for data stores
    bg_col     = "#f0f0f0"

    # Background bands
    ax.axvspan(0.0, 4.55, ymin=0, ymax=1, color="#e8f4fc", zorder=0)
    ax.axvspan(4.55, 10.0, ymin=0, ymax=1, color="#fff0f0", zorder=0)

    # Band labels
    ax.text(2.25, 4.7, "Edge Node (Jetson Orin Nano)", ha="center", va="bottom",
            fontsize=10, color=edge_col, fontweight="bold")
    ax.text(7.25, 4.7, "Server (Cloud GPU)", ha="center", va="bottom",
            fontsize=10, color=server_col, fontweight="bold")

    bw, bh = 1.7, 0.65

    # Edge components (left side)
    _box(ax, 1.2, 3.8, bw, bh, "Raw Video\nCapture", edge_col)
    _box(ax, 1.2, 2.8, bw, bh, "MegaDetector\n+ KLT Tracking", edge_col)
    _box(ax, 1.2, 1.8, bw, bh, "ROI Mask\nGeneration", edge_col)
    _box(ax, 3.5, 2.8, bw, bh, "DCVC ROI\nStream (QP 63)", edge_col)
    _box(ax, 3.5, 1.8, bw, bh, "DCVC BG\nStream (QP 5)", edge_col)
    _box(ax, 3.5, 0.8, 1.7, 0.5, "ZIP Archive", data_col, fontsize=8)

    # Server components (right side)
    _box(ax, 6.0, 2.3, bw, bh, "DCVC Decode\n+ Composite", server_col)
    _box(ax, 8.0, 2.3, bw, bh, "RestoreUNet\n(L1 diffusion)", server_col)
    _box(ax, 9.5, 2.3, bw, bh, "S3Diff\nUpscale (2x)", server_col)

    # Arrows: edge side
    _arrow(ax, 1.2, 3.47, 1.2, 3.13, edge_col)         # capture -> detector
    _arrow(ax, 1.2, 2.47, 1.2, 2.13, edge_col)         # detector -> roi mask
    _arrow(ax, 2.05, 2.5,  2.6, 2.8,  edge_col)        # mask -> roi stream
    _arrow(ax, 2.05, 2.1,  2.6, 1.8,  edge_col)        # mask -> bg stream
    _arrow(ax, 3.5, 2.47, 3.5, 1.05, edge_col)         # roi stream -> zip
    _arrow(ax, 3.5, 1.47, 3.5, 1.05, edge_col)         # bg stream -> zip

    # Transmission arrow
    ax.annotate(
        "",
        xy=(5.05, 2.3),
        xytext=(4.35, 0.8),
        arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.5,
                        connectionstyle="arc3,rad=-0.3"),
    )
    ax.text(4.7, 1.5, "low-BW\nuplink", ha="center", va="center",
            fontsize=7.5, color="#444444", style="italic")

    # Arrows: server side
    _arrow(ax, 6.85, 2.3, 7.15, 2.3, server_col)       # decomp -> restore
    _arrow(ax, 8.85, 2.3, 9.15, 2.3, server_col)       # restore -> upscale

    # Side note: detections.json
    _box(ax, 1.2, 0.5, 1.5, 0.4, "detections.json", data_col, fontsize=7.5)
    ax.annotate("", xy=(1.2, 0.7), xytext=(1.2, 1.47),
                arrowprops=dict(arrowstyle="-|>", color=data_col, lw=1.0,
                                linestyle="dashed"))

    # Output label
    ax.text(9.5, 1.7, "Enhanced\nVideo", ha="center", va="top",
            fontsize=8, color=server_col)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="Output path (default: graphics/)")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")
    draw_pipeline(ax)
    plt.tight_layout(pad=0.2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stems = [args.out] if args.out else [
        str(OUT_DIR / "pipeline_overview.pdf"),
        str(OUT_DIR / "pipeline_overview.png"),
    ]
    for stem in stems:
        fig.savefig(stem, dpi=200, bbox_inches="tight")
        print(f"Saved: {stem}")
    plt.close(fig)


if __name__ == "__main__":
    main()
