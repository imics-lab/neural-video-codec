#!/usr/bin/env python3
"""
run_experiments.py - Schedule and run all paper experiments.

Experiments
  --stage-ablation   PSNR/SSIM/LPIPS/WE at each pipeline stage
  --qp-sweep         Background QP sweep with ROI QP fixed at 63  (Table 2)
  --ddim-ablation    DDIM step count vs quality and timing          (Table 4)
  --temporal         Temporal window T vs quality and WE            (Table 3)
  --rd               Rate-distortion vs H.264 and uniform DCVC      (Table 1)
  --all              Run every experiment above

Each experiment writes a CSV to results/ and reads from there for --summary.
DDIM and temporal ablations reuse a single compress+decompress pass to avoid
running the encoder N times unnecessarily.

Usage:
  python run_experiments.py --all
  python run_experiments.py --stage-ablation --qp-sweep
  python run_experiments.py --summary
  python run_experiments.py --all --video dataset/deer01.mp4 --gpu 1
  python run_experiments.py --all --dry-run
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

import numpy as np

ROOT       = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"
OUT_DIR    = ROOT / "outputs" / "experiments"

GT_VIDEO       = ROOT / "dataset" / "cow15.mp4"
COMPRESS_CFG   = ROOT / "configs" / "gpu" / "compression.yaml"
RESTORE_CFG    = {"1": ROOT / "configs" / "gpu" / "restoration_T1.yaml",
                  "3": ROOT / "configs" / "gpu" / "restoration.yaml",
                  "5": ROOT / "configs" / "gpu" / "restoration_T5.yaml"}

DRY_RUN = False
GPU     = None          # set by --gpu; injected into env as CUDA_VISIBLE_DEVICES


# ── Subprocess helpers ────────────────────────────────────────────────────────

def _env() -> Dict[str, str]:
    e = os.environ.copy()
    if GPU is not None:
        e["CUDA_VISIBLE_DEVICES"] = str(GPU)
    return e


def _run(cmd: List[str], capture: bool = False) -> Optional[str]:
    printable = " ".join(str(c) for c in cmd)
    print(f"  $ {printable}", flush=True)
    if DRY_RUN:
        return None
    kw: Dict[str, Any] = {"check": True, "env": _env()}
    if capture:
        kw["capture_output"] = True
        kw["text"] = True
    result = subprocess.run(cmd, **kw)
    return result.stdout if capture else None


def _py(*args) -> List[str]:
    return [sys.executable, *[str(a) for a in args]]


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _write_tmp_cfg(cfg: Dict[str, Any]) -> Path:
    fd, p = tempfile.mkstemp(suffix=".yaml", prefix="exp_cfg_")
    with os.fdopen(fd, "w") as f:
        yaml.dump(cfg, f)
    return Path(p)


def _set_qp(cfg: Dict[str, Any], roi_qp: int, bg_qp: int) -> None:
    q = cfg.setdefault("compression", {}).setdefault("quality", {})
    q["roi_qp_i"] = roi_qp
    q["roi_qp_p"] = roi_qp
    q["bg_qp_i"]  = bg_qp
    q["bg_qp_p"]  = bg_qp


# ── Metric helpers ────────────────────────────────────────────────────────────

def _eval_metrics(pred: Path, csv_out: Path, *, vmaf: bool = False) -> Dict[str, float]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = _py("eval_metrics.py", "--pred", pred, "--gt", GT_VIDEO, "--out-csv", csv_out)
    if not vmaf:
        cmd += ["--no-vmaf"]
    _run(cmd)
    if DRY_RUN:
        return {"psnr": 0.0, "ssim": 0.0, "ms_ssim": 0.0, "lpips": 0.0}
    rows = list(csv.DictReader(open(csv_out)))
    if not rows:
        nan = float("nan")
        return {"psnr": nan, "ssim": nan, "ms_ssim": nan, "lpips": nan}

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in rows
                if r.get(key, "nan") not in ("nan", "", None)]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "psnr":    _mean("psnr"),
        "ssim":    _mean("ssim"),
        "ms_ssim": _mean("ms_ssim"),
        "lpips":   _mean("lpips"),
    }


def _eval_temporal(video: Path) -> float:
    out = _run(_py("eval_temporal.py", "--videos", video), capture=True)
    if DRY_RUN or out is None:
        return 0.0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            try:
                return float(parts[-3])
            except (ValueError, IndexError):
                pass
    return float("nan")


def _archive_kb(cfg_path: Path) -> float:
    """Compress GT_VIDEO with cfg_path and return the archive size in KB.

    Uses run_pipeline.py --stages compress --save-intermediate to avoid
    run_compression.py's strict ONNX-path validation. The intermediate archive
    is written by the pipeline to {out_dir}/{stem}.zip per its convention.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_data = _load_yaml(cfg_path)
    pipeline_out_dir = Path(
        (cfg_data.get("output") or {}).get("out_dir", "outputs/pipeline")
    )
    archive_path = pipeline_out_dir / f"{GT_VIDEO.stem}.zip"
    tmp_video = OUT_DIR / "_size_probe.mp4"
    _run(_py("run_pipeline.py",
             "--video", GT_VIDEO, "--config", cfg_path,
             "--stages", "compress",
             "--save-intermediate",
             "--output", tmp_video))
    if DRY_RUN:
        return 0.0
    if not archive_path.exists():
        return float("nan")
    size = archive_path.stat().st_size / 1024.0
    archive_path.unlink(missing_ok=True)
    tmp_video.unlink(missing_ok=True)
    return size


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}")


# ── Pipeline wrapper ──────────────────────────────────────────────────────────

def _pipeline(stages: List[str], output: Path, cfg: Path, *,
              input_video: Optional[Path] = None,
              restore_config: Optional[Path] = None,
              ddim_steps: Optional[int] = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = _py("run_pipeline.py",
              "--video" if input_video is None else "--input",
              input_video if input_video is not None else GT_VIDEO,
              "--config", cfg,
              "--stages", *stages,
              "--output", output)
    if restore_config:
        cmd += ["--restore-config", str(restore_config)]
    if ddim_steps is not None:
        cmd += ["--ddim-steps", str(ddim_steps)]
    _run(cmd)


# ── Experiments ───────────────────────────────────────────────────────────────

def exp_stage_ablation() -> None:
    """
    Show how quality and temporal consistency change at each pipeline stage.
    Runs compress+decompress once, then restore once; both measured against GT.
    """
    print("\n=== Stage ablation ===")

    decomp_video   = OUT_DIR / "stage_decomp.mp4"
    restored_video = OUT_DIR / "stage_restored.mp4"

    print("\n-- compress + decompress --")
    _pipeline(["compress", "decompress"], decomp_video, COMPRESS_CFG)

    print("\n-- compress + decompress + restore --")
    _pipeline(["compress", "decompress", "restore"], restored_video, COMPRESS_CFG)

    rows = []
    for label, video in [("decomp", decomp_video), ("restored", restored_video)]:
        m  = _eval_metrics(video, RESULT_DIR / f"stage_{label}.csv")
        we = _eval_temporal(video)
        row = {"stage": label, **m, "warp_err_1e4": round(we, 4)}
        rows.append(row)
        print(f"  {label:12s}  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  "
              f"MS-SSIM={m['ms_ssim']:.4f}  LPIPS={m['lpips']:.4f}  WE={we:.4f}")

    _write_csv(RESULT_DIR / "stage_ablation.csv",
               ["stage", "psnr", "ssim", "ms_ssim", "lpips", "warp_err_1e4"], rows)


def exp_qp_sweep() -> None:
    """Background QP sweep: ROI fixed at 63, BG varied. Measures Table 2."""
    print("\n=== QP sweep ===")

    rows = []
    for bg_qp in [5, 15, 25, 40]:
        print(f"\n-- bg_qp={bg_qp} --")
        cfg = copy.deepcopy(_load_yaml(COMPRESS_CFG))
        _set_qp(cfg, roi_qp=63, bg_qp=bg_qp)
        tmp = _write_tmp_cfg(cfg)
        out = OUT_DIR / f"qp_bg{bg_qp}.mp4"
        try:
            _pipeline(["compress", "decompress", "restore"], out, tmp)
            m   = _eval_metrics(out, RESULT_DIR / f"qp_bg{bg_qp}.csv")
            kb  = _archive_kb(tmp)
        finally:
            tmp.unlink(missing_ok=True)

        row = {"bg_qp": bg_qp, "archive_kb": round(kb, 1), **m}
        rows.append(row)
        print(f"  bg_qp={bg_qp}  {kb:.0f} KB  PSNR={m['psnr']:.2f}  "
              f"SSIM={m['ssim']:.4f}  LPIPS={m['lpips']:.4f}")

    _write_csv(RESULT_DIR / "qp_sweep.csv",
               ["bg_qp", "archive_kb", "psnr", "ssim", "ms_ssim", "lpips"], rows)


def exp_ddim_ablation() -> None:
    """
    DDIM step count vs quality and inference time. Table 4.
    Compresses once, then runs restore with d=1/3/6/10/20 from the saved
    decompressed video to avoid redundant encoding.
    """
    print("\n=== DDIM ablation ===")

    decomp_video = OUT_DIR / "ddim_decomp_base.mp4"
    if not decomp_video.exists() or DRY_RUN:
        print("\n-- compress + decompress (base) --")
        _pipeline(["compress", "decompress"], decomp_video, COMPRESS_CFG)

    rows = []
    for d in [1, 3, 6, 10, 20]:
        print(f"\n-- ddim_steps={d} --")
        out = OUT_DIR / f"ddim_d{d}.mp4"
        t0  = time.time()
        _pipeline(["restore"], out, COMPRESS_CFG,
                  input_video=decomp_video, ddim_steps=d)
        elapsed = round(time.time() - t0, 1)
        m = _eval_metrics(out, RESULT_DIR / f"ddim_d{d}.csv")
        row = {"ddim_steps": d, "elapsed_s": elapsed, **m}
        rows.append(row)
        print(f"  d={d:2d}  {elapsed:.1f}s  PSNR={m['psnr']:.2f}  "
              f"SSIM={m['ssim']:.4f}  LPIPS={m['lpips']:.4f}")

    _write_csv(RESULT_DIR / "ddim_ablation.csv",
               ["ddim_steps", "elapsed_s", "psnr", "ssim", "ms_ssim", "lpips"], rows)


def exp_temporal_ablation() -> None:
    """
    Temporal window T=1/3/5 vs per-frame quality and warping error. Table 3.
    Reuses the same decompressed base video as the DDIM ablation if present.
    """
    print("\n=== Temporal ablation ===")

    decomp_video = OUT_DIR / "ddim_decomp_base.mp4"
    if not decomp_video.exists() or DRY_RUN:
        print("\n-- compress + decompress (base) --")
        _pipeline(["compress", "decompress"], decomp_video, COMPRESS_CFG)

    rows = []
    for T in [1, 3, 5]:
        print(f"\n-- T={T} --")
        out = OUT_DIR / f"temporal_T{T}.mp4"
        _pipeline(["restore"], out, COMPRESS_CFG,
                  input_video=decomp_video,
                  restore_config=RESTORE_CFG[str(T)])
        m  = _eval_metrics(out, RESULT_DIR / f"temporal_T{T}.csv")
        we = _eval_temporal(out)
        row = {"T": T, **m, "warp_err_1e4": round(we, 4)}
        rows.append(row)
        print(f"  T={T}  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  "
              f"LPIPS={m['lpips']:.4f}  WE={we:.4f}")

    _write_csv(RESULT_DIR / "temporal_ablation.csv",
               ["T", "psnr", "ssim", "ms_ssim", "lpips", "warp_err_1e4"], rows)


def exp_rd_baselines() -> None:
    """
    Rate-distortion comparison. Table 1.
      H.264 CRF 18 / 28 / 40       (ffmpeg libx264)
      DCVC uniform QP 25 / 40 / 63  (no ROI awareness)
      Ours ROI-aware + restore
    """
    print("\n=== Rate-distortion baselines ===")

    rows: List[Dict[str, Any]] = []

    # H.264 baselines
    for crf in [18, 28, 40]:
        print(f"\n-- H.264 CRF={crf} --")
        enc = OUT_DIR / f"h264_crf{crf}.mp4"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        _run(["ffmpeg", "-y", "-i", str(GT_VIDEO),
              "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
              "-pix_fmt", "yuv420p", str(enc)])
        size_kb = (0.0 if DRY_RUN else enc.stat().st_size / 1024.0)
        m = _eval_metrics(enc, RESULT_DIR / f"h264_crf{crf}.csv")
        rows.append({"method": f"H.264 CRF {crf}",
                     "archive_kb": round(size_kb, 1), **m})
        print(f"  H.264 CRF={crf}  {size_kb:.0f} KB  "
              f"PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}")

    # DCVC uniform QP (no ROI awareness — equal QP on both streams)
    for uniform_qp in [25, 40, 63]:
        print(f"\n-- DCVC uniform QP={uniform_qp} --")
        cfg = copy.deepcopy(_load_yaml(COMPRESS_CFG))
        _set_qp(cfg, roi_qp=uniform_qp, bg_qp=uniform_qp)
        tmp = _write_tmp_cfg(cfg)
        out = OUT_DIR / f"dcvc_uniform_qp{uniform_qp}.mp4"
        try:
            _pipeline(["compress", "decompress"], out, tmp)
            m  = _eval_metrics(out, RESULT_DIR / f"dcvc_uniform_qp{uniform_qp}.csv")
            kb = _archive_kb(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        rows.append({"method": f"DCVC uniform QP {uniform_qp}",
                     "archive_kb": round(kb, 1), **m})
        print(f"  DCVC uniform QP={uniform_qp}  {kb:.0f} KB  "
              f"PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}")

    # Ours: ROI-aware + restore
    print("\n-- Ours (ROI-aware + restore) --")
    out = OUT_DIR / "ours_full.mp4"
    _pipeline(["compress", "decompress", "restore"], out, COMPRESS_CFG)
    m  = _eval_metrics(out, RESULT_DIR / "ours_full.csv")
    kb = _archive_kb(COMPRESS_CFG)
    rows.append({"method": "Ours (ROI-aware + restore)",
                 "archive_kb": round(kb, 1), **m})
    print(f"  Ours  {kb:.0f} KB  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}")

    _write_csv(RESULT_DIR / "rd_baselines.csv",
               ["method", "archive_kb", "psnr", "ssim", "ms_ssim", "lpips"], rows)


def exp_codec_comparison() -> None:
    """
    Compare ROI-aware pipeline across codec backends: DCVC, AV1, HEVC, H.264.
    Runs compress+decompress+restore for each codec at its default quality settings,
    measures PSNR/SSIM/LPIPS and archive size.
    Results go into the paper appendix; best codec is used for main results.
    """
    print("\n=== Codec comparison ===")

    codecs = [
        ("dcvc", {},),
        ("av1",  {"ffmpeg": {"av1_encoder": "libsvtav1"}}),
        ("hevc", {"ffmpeg": {"hevc_encoder": "libx265"}}),
        ("h264", {"ffmpeg": {"h264_encoder": "libx264"}}),
    ]

    rows = []
    for codec, extra_cfg in codecs:
        print(f"\n-- codec={codec} --")
        base_cfg = copy.deepcopy(_load_yaml(COMPRESS_CFG))
        base_cfg.setdefault("compression", {})["codec"] = codec
        for k, v in extra_cfg.items():
            base_cfg["compression"].setdefault(k, {}).update(v)
        tmp = _write_tmp_cfg(base_cfg)
        out = OUT_DIR / f"codec_{codec}.mp4"
        try:
            _pipeline(["compress", "decompress", "restore"], out, tmp)
            m  = _eval_metrics(out, RESULT_DIR / f"codec_{codec}.csv")
            kb = _archive_kb(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        row = {"codec": codec, "archive_kb": round(kb, 1), **m}
        rows.append(row)
        print(f"  {codec:6s}  {kb:.0f} KB  PSNR={m['psnr']:.2f}  "
              f"SSIM={m['ssim']:.4f}  LPIPS={m['lpips']:.4f}")

    _write_csv(RESULT_DIR / "codec_comparison.csv",
               ["codec", "archive_kb", "psnr", "ssim", "ms_ssim", "lpips"], rows)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary() -> None:
    sections = [
        ("Stage ablation",      RESULT_DIR / "stage_ablation.csv"),
        ("QP sweep",            RESULT_DIR / "qp_sweep.csv"),
        ("DDIM ablation",       RESULT_DIR / "ddim_ablation.csv"),
        ("Temporal ablation",   RESULT_DIR / "temporal_ablation.csv"),
        ("R-D baselines",       RESULT_DIR / "rd_baselines.csv"),
        ("Codec comparison",    RESULT_DIR / "codec_comparison.csv"),
    ]
    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    for title, path in sections:
        if not path.exists():
            continue
        rows = list(csv.DictReader(open(path)))
        if not rows:
            continue
        keys = list(rows[0].keys())
        print(f"\n{title}:")
        header = "  " + "  ".join(f"{k:>16}" for k in keys)
        print(header)
        print("  " + "-" * (18 * len(keys)))
        for row in rows:
            line = "  " + "  ".join(f"{str(row.get(k,''))[:16]:>16}" for k in keys)
            print(line)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global DRY_RUN, GPU, GT_VIDEO

    ap = argparse.ArgumentParser(
        description="Run paper experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--all",               action="store_true", help="Run all experiments")
    ap.add_argument("--stage-ablation",    action="store_true")
    ap.add_argument("--qp-sweep",          action="store_true")
    ap.add_argument("--ddim-ablation",     action="store_true")
    ap.add_argument("--temporal",          action="store_true")
    ap.add_argument("--rd",                action="store_true")
    ap.add_argument("--codec-comparison",  action="store_true")
    ap.add_argument("--summary",        action="store_true",
                    help="Print collected results without running anything")
    ap.add_argument("--video", default=str(GT_VIDEO),
                    metavar="PATH", help="Input video (default: dataset/cow15.mp4)")
    ap.add_argument("--gpu", default=None, type=int,
                    metavar="N", help="CUDA device index (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing")
    args = ap.parse_args()

    DRY_RUN = bool(args.dry_run)
    GPU     = args.gpu
    GT_VIDEO = Path(args.video)

    if args.summary:
        print_summary()
        return

    run_all = args.all or not any([
        args.stage_ablation, args.qp_sweep, args.ddim_ablation,
        args.temporal, args.rd, args.codec_comparison,
    ])

    if run_all or args.stage_ablation:
        exp_stage_ablation()
    if run_all or args.qp_sweep:
        exp_qp_sweep()
    if run_all or args.ddim_ablation:
        exp_ddim_ablation()
    if run_all or args.temporal:
        exp_temporal_ablation()
    if run_all or args.rd:
        exp_rd_baselines()
    if run_all or args.codec_comparison:
        exp_codec_comparison()

    print_summary()


if __name__ == "__main__":
    main()
