#!/usr/bin/env python3
"""
gen_latex_tables.py  --  Read results/ CSVs and print ready-to-paste LaTeX tables.

Usage (run after run_experiments.py finishes):
    python scripts/gen_latex_tables.py
    python scripts/gen_latex_tables.py --results results/  # custom results dir
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "results"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(open(path, newline="")))


def _f(val: str | float, fmt: str = ".2f", bold: bool = False) -> str:
    try:
        v = float(val)
        if math.isnan(v):
            return "---"
        s = format(v, fmt)
    except (ValueError, TypeError):
        s = str(val)
    return rf"\textbf{{{s}}}" if bold else s


def _best(rows: list[dict], key: str, higher_better: bool = True) -> str:
    vals = []
    for r in rows:
        try:
            v = float(r.get(key, "nan"))
            if not math.isnan(v):
                vals.append(v)
        except (ValueError, TypeError):
            pass
    if not vals:
        return ""
    return str(max(vals) if higher_better else min(vals))


def table_stage_ablation(rows: list[dict]) -> str:
    if not rows:
        return "% stage_ablation.csv not found\n"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Stage ablation: metrics at each pipeline stage. "
        r"WE = warping error ($\times10^{-4}$), lower is better for LPIPS and WE.}",
        r"\label{tab:stage_ablation}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Stage & PSNR & SSIM & LPIPS & VMAF & WE ($\times10^{-4}$) \\",
        r"\midrule",
    ]
    best_psnr  = _best(rows, "psnr",  True)
    best_ssim  = _best(rows, "ssim",  True)
    best_lpips = _best(rows, "lpips", False)
    best_vmaf  = _best(rows, "vmaf",  True)
    best_we    = _best(rows, "warp_err_1e4", False)

    for r in rows:
        stage = r.get("stage", r.get("method", ""))
        psnr  = _f(r.get("psnr",  "nan"), ".2f", r.get("psnr")  == best_psnr)
        ssim  = _f(r.get("ssim",  "nan"), ".4f", r.get("ssim")  == best_ssim)
        lpips = _f(r.get("lpips", "nan"), ".4f", r.get("lpips") == best_lpips)
        vmaf  = _f(r.get("vmaf",  "nan"), ".2f", r.get("vmaf")  == best_vmaf)
        we    = _f(r.get("warp_err_1e4", "nan"), ".2f",
                   r.get("warp_err_1e4") == best_we)
        lines.append(f"{stage} & {psnr} & {ssim} & {lpips} & {vmaf} & {we} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_rd_baselines(rows: list[dict]) -> str:
    if not rows:
        return "% rd_baselines.csv not found\n"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Rate-distortion comparison. Archive size in KB for the test clip "
        r"(1280$\times$720). Higher PSNR/SSIM/VMAF and lower LPIPS are better.}",
        r"\label{tab:rd}",
        r"\begin{tabular}{lrcccc}",
        r"\toprule",
        r"Method & KB & PSNR & SSIM & LPIPS & VMAF \\",
        r"\midrule",
    ]
    best_psnr  = _best(rows, "psnr",  True)
    best_ssim  = _best(rows, "ssim",  True)
    best_lpips = _best(rows, "lpips", False)
    best_vmaf  = _best(rows, "vmaf",  True)

    for r in rows:
        method = r.get("method", "")
        kb     = _f(r.get("archive_kb", "nan"), ".0f")
        psnr   = _f(r.get("psnr",  "nan"), ".2f", r.get("psnr")  == best_psnr)
        ssim   = _f(r.get("ssim",  "nan"), ".4f", r.get("ssim")  == best_ssim)
        lpips  = _f(r.get("lpips", "nan"), ".4f", r.get("lpips") == best_lpips)
        vmaf   = _f(r.get("vmaf",  "nan"), ".2f", r.get("vmaf")  == best_vmaf)
        lines.append(f"{method} & {kb} & {psnr} & {ssim} & {lpips} & {vmaf} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_qp_sweep(rows: list[dict]) -> str:
    if not rows:
        return "% qp_sweep.csv not found\n"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Effect of background QP on overall quality "
        r"(QP$_{\text{ROI}}{=}63$ throughout). Archive size and metrics "
        r"measured on a representative 1280$\times$720 clip.}",
        r"\label{tab:roi_bg}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"QP$_{\text{BG}}$ & Archive (KB) & PSNR (dB) & SSIM & LPIPS \\",
        r"\midrule",
    ]
    best_psnr  = _best(rows, "psnr",  True)
    best_ssim  = _best(rows, "ssim",  True)
    best_lpips = _best(rows, "lpips", False)

    for r in rows:
        qp    = r.get("bg_qp", "")
        kb    = _f(r.get("archive_kb", "nan"), ".0f")
        psnr  = _f(r.get("psnr",  "nan"), ".2f", r.get("psnr")  == best_psnr)
        ssim  = _f(r.get("ssim",  "nan"), ".4f", r.get("ssim")  == best_ssim)
        lpips = _f(r.get("lpips", "nan"), ".4f", r.get("lpips") == best_lpips)
        lines.append(f"{qp} & {kb} & {psnr} & {ssim} & {lpips} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_temporal_ablation(rows: list[dict]) -> str:
    if not rows:
        return "% temporal_ablation.csv not found\n"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation: effect of temporal window size $T$. "
        r"Warping error measures temporal consistency ($\downarrow$ better).}",
        r"\label{tab:ablation_T}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"$T$ & PSNR (dB) & SSIM & LPIPS & WE ($\times10^{-4}$) \\",
        r"\midrule",
    ]
    best_psnr  = _best(rows, "psnr",  True)
    best_ssim  = _best(rows, "ssim",  True)
    best_lpips = _best(rows, "lpips", False)
    best_we    = _best(rows, "warp_err_1e4", False)

    for r in rows:
        T     = r.get("T", "")
        psnr  = _f(r.get("psnr",  "nan"), ".2f", r.get("psnr")  == best_psnr)
        ssim  = _f(r.get("ssim",  "nan"), ".4f", r.get("ssim")  == best_ssim)
        lpips = _f(r.get("lpips", "nan"), ".4f", r.get("lpips") == best_lpips)
        we    = _f(r.get("warp_err_1e4", "nan"), ".2f",
                   r.get("warp_err_1e4") == best_we)
        label = {
            "1": r"1 (no temporal)",
            "3": r"3 (proposed)",
            "5": "5",
        }.get(str(T), str(T))
        lines.append(f"{label} & {psnr} & {ssim} & {lpips} & {we} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_ddim_ablation(rows: list[dict]) -> str:
    if not rows:
        return "% ddim_ablation.csv not found\n"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation: DDIM steps $d$ vs.\ quality and per-frame inference time "
        r"(1280$\times$720, NVIDIA A5000 24\,GB).}",
        r"\label{tab:ablation_steps}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"$d$ & PSNR (dB) & SSIM & LPIPS & ms/frame \\",
        r"\midrule",
    ]
    best_psnr  = _best(rows, "psnr",  True)
    best_ssim  = _best(rows, "ssim",  True)
    best_lpips = _best(rows, "lpips", False)

    for r in rows:
        d     = r.get("ddim_steps", "")
        psnr  = _f(r.get("psnr",  "nan"), ".2f", r.get("psnr")  == best_psnr)
        ssim  = _f(r.get("ssim",  "nan"), ".4f", r.get("ssim")  == best_ssim)
        lpips = _f(r.get("lpips", "nan"), ".4f", r.get("lpips") == best_lpips)
        elapsed = r.get("elapsed_s", "nan")
        try:
            # elapsed_s is total; convert to ms/frame by reading the video frame count
            # If not available, just show total seconds
            ms = _f(elapsed, ".0f")
        except Exception:
            ms = "---"
        lines.append(f"{d} & {psnr} & {ssim} & {lpips} & {ms} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_codec_comparison(rows: list[dict]) -> str:
    if not rows:
        return "% codec_comparison.csv not found\n"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Codec backend comparison (compress + decompress + restore). "
        r"Archive size in KB; metrics on 1280$\times$720 test clip.}",
        r"\label{tab:codec_comparison}",
        r"\begin{tabular}{lrcccc}",
        r"\toprule",
        r"Codec & KB & PSNR & SSIM & LPIPS & VMAF \\",
        r"\midrule",
    ]
    best_psnr  = _best(rows, "psnr",  True)
    best_ssim  = _best(rows, "ssim",  True)
    best_lpips = _best(rows, "lpips", False)
    best_vmaf  = _best(rows, "vmaf",  True)

    for r in rows:
        codec = r.get("codec", "").upper()
        kb    = _f(r.get("archive_kb", "nan"), ".0f")
        psnr  = _f(r.get("psnr",  "nan"), ".2f", r.get("psnr")  == best_psnr)
        ssim  = _f(r.get("ssim",  "nan"), ".4f", r.get("ssim")  == best_ssim)
        lpips = _f(r.get("lpips", "nan"), ".4f", r.get("lpips") == best_lpips)
        vmaf  = _f(r.get("vmaf",  "nan"), ".2f", r.get("vmaf")  == best_vmaf)
        lines.append(f"{codec} & {kb} & {psnr} & {ssim} & {lpips} & {vmaf} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate LaTeX tables from results CSVs")
    ap.add_argument("--results", default=str(DEFAULT_RESULTS),
                    help="Path to results/ directory")
    args = ap.parse_args()
    R = Path(args.results)

    tables = [
        ("Stage ablation",    table_stage_ablation,   R / "stage_ablation.csv"),
        ("RD baselines",      table_rd_baselines,     R / "rd_baselines.csv"),
        ("QP sweep",          table_qp_sweep,         R / "qp_sweep.csv"),
        ("Temporal ablation", table_temporal_ablation, R / "temporal_ablation.csv"),
        ("DDIM ablation",     table_ddim_ablation,    R / "ddim_ablation.csv"),
        ("Codec comparison",  table_codec_comparison, R / "codec_comparison.csv"),
    ]

    for title, fn, csv_path in tables:
        rows = _load(csv_path)
        print(f"\n% {'='*60}")
        print(f"% {title}")
        print(f"% {'='*60}\n")
        print(fn(rows))

    missing = [csv_path.name for _, _, csv_path in tables if not csv_path.exists()]
    if missing:
        print(f"\n% Missing CSVs (run run_experiments.py first): {', '.join(missing)}")


if __name__ == "__main__":
    main()
