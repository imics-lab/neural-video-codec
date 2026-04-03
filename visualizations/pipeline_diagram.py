"""
pipeline_diagram.py — Generate graphics/pipeline.pdf (Figure 1).

Draws the NCC system overview as a horizontal block diagram using matplotlib.
Outputs both PDF (for LaTeX) and PNG (for preview).

Usage:
    python visualizations/pipeline_diagram.py
    python visualizations/pipeline_diagram.py --out graphics/pipeline.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ── Colour palette ────────────────────────────────────────────────────────────
C_DETECT  = "#4C72B0"   # blue   — detection/tracking
C_ENCODE  = "#DD8452"   # orange — encoding
C_ARCHIVE = "#8C8C8C"   # grey   — bitstream archive
C_DECODE  = "#55A868"   # green  — decoding / blend
C_RESTORE = "#C44E52"   # red    — restoration
C_SR      = "#8172B2"   # purple — super-resolution
C_IO      = "#2D2D2D"   # dark   — video I/O
C_MASK    = "#937860"   # brown  — mask flow

TEXT_KW   = dict(ha="center", va="center", fontsize=7.5, color="white",
                 fontweight="bold", wrap=True)
ARROW_KW  = dict(arrowstyle="-|>", color="#333333", lw=1.2,
                 mutation_scale=12)


def _box(ax, cx, cy, w, h, label, sublabel="", color="#4C72B0", fontsize=7.5):
    rect = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle="round,pad=0.02",
                          facecolor=color, edgecolor="white", linewidth=1.2)
    ax.add_patch(rect)
    if sublabel:
        ax.text(cx, cy + h * 0.13, label,
                ha="center", va="center", fontsize=fontsize,
                color="white", fontweight="bold")
        ax.text(cx, cy - h * 0.18, sublabel,
                ha="center", va="center", fontsize=fontsize - 1.5,
                color="#EEEEEE", style="italic")
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold")


def _arrow(ax, x0, y0, x1, y1, label="", color="#333333", lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=11))
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.05, label, ha="center", va="bottom",
                fontsize=6.5, color=color, style="italic")


def main(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    BH = 0.70   # box height
    BW = 1.55   # box width (standard)
    CY = 2.1    # centre y for main flow

    # ── Video In ──────────────────────────────────────────────────────────────
    _box(ax, 0.75, CY, 1.0, BH, "Video In", color=C_IO, fontsize=7)
    _arrow(ax, 1.25, CY, 1.65, CY)

    # ── ROI Detection ─────────────────────────────────────────────────────────
    _box(ax, 2.35, CY, 1.35, BH, "ROI Detection",
         "MegaDetector + KLT", color=C_DETECT)
    _arrow(ax, 3.025, CY, 3.5, CY)

    # Mask label on arrow
    ax.text(3.26, CY + 0.22, r"$\mathbf{M}_t$", ha="center",
            fontsize=7.5, color=C_MASK, style="italic")

    # ── Dual-Stream Encoder ───────────────────────────────────────────────────
    # Draw two sub-stream boxes
    _box(ax, 4.35, CY + 0.42, 1.55, 0.52,
         "ROI Stream", "QP = 63", color=C_ENCODE, fontsize=7)
    _box(ax, 4.35, CY - 0.42, 1.55, 0.52,
         "BG Stream", "QP = 25", color="#B07040", fontsize=7)
    # Brace label
    ax.text(4.35, CY, "DCVC Encoder", ha="center", va="center",
            fontsize=6.5, color="#333333", style="italic")

    _arrow(ax, 3.5, CY + 0.20, 3.82, CY + 0.42)
    _arrow(ax, 3.5, CY - 0.20, 3.82, CY - 0.42)

    _arrow(ax, 5.125, CY + 0.42, 5.55, CY + 0.28)
    _arrow(ax, 5.125, CY - 0.42, 5.55, CY - 0.28)

    # ── Archive ───────────────────────────────────────────────────────────────
    _box(ax, 6.12, CY, 1.05, 1.0,
         "Archive", "roi.bin\nbg.bin\nmeta.json", color=C_ARCHIVE, fontsize=7)
    _arrow(ax, 6.65, CY, 7.1, CY)

    # ── Decoder + Blend ───────────────────────────────────────────────────────
    _box(ax, 7.85, CY, 1.40, BH,
         "DCVC Decode", "+ Feathered Blend", color=C_DECODE)
    _arrow(ax, 8.55, CY, 9.0, CY)

    # Mask re-enters blend
    ax.annotate("", xy=(7.85, CY - BH / 2),
                xytext=(7.85, CY - 1.0),
                arrowprops=dict(arrowstyle="-|>", color=C_MASK, lw=1.0,
                                mutation_scale=9))
    ax.plot([2.35, 2.35, 7.85], [CY - 1.0, CY - 1.0, CY - 1.0],
            color=C_MASK, lw=0.9, ls="--")
    ax.text(5.1, CY - 1.12, r"$\mathbf{M}_t$ (blend)", ha="center",
            fontsize=6.5, color=C_MASK, style="italic")

    # ── RestoreUNet ───────────────────────────────────────────────────────────
    _box(ax, 9.75, CY, 1.40, BH,
         "RestoreUNet", "DDIM  T=3", color=C_RESTORE)
    _arrow(ax, 10.45, CY, 10.9, CY)

    # ── S3Diff SR ─────────────────────────────────────────────────────────────
    _box(ax, 11.65, CY, 1.40, BH,
         "S3Diff SR", "1-step diffusion ×4", color=C_SR)
    _arrow(ax, 12.35, CY, 12.75, CY)

    # ── Video Out ─────────────────────────────────────────────────────────────
    _box(ax, 13.0 - 0.38, CY, 0.75, BH, "Video\nOut", color=C_IO, fontsize=7)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title("NCC Pipeline Overview", fontsize=10, fontweight="bold",
                 pad=4, color="#1A1A1A")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=C_DETECT,  label="Detection & Tracking"),
        mpatches.Patch(color=C_ENCODE,  label="DCVC Encoding"),
        mpatches.Patch(color=C_ARCHIVE, label="Bitstream Archive"),
        mpatches.Patch(color=C_DECODE,  label="Decoding & Blending"),
        mpatches.Patch(color=C_RESTORE, label="Restoration (Model R)"),
        mpatches.Patch(color=C_SR,      label="Super-Resolution (Model S)"),
    ]
    ax.legend(handles=legend_items, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=6,
              fontsize=6.5, framealpha=0.85, edgecolor="#CCCCCC")

    plt.tight_layout(pad=0.3)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    # Also save PNG preview
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"Saved: {png}")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="visualizations/images/pipeline.pdf")
    args = p.parse_args()
    main(args.out)
