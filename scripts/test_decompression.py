from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _phase0_utils import (  # noqa: E402
    DEFAULT_DECOMPRESSION_CONFIG,
    human_bytes,
    load_yaml,
    read_archive_payloads,
    resolve_from_root,
    resolve_out_dir,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 sanity check: decompression")
    parser.add_argument("archive", type=str, help="Path to compressed archive (.zip)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_DECOMPRESSION_CONFIG),
        help="Decompression config YAML path",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/sanity_checks/decompression",
        help="Directory for outputs",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repeated runs with the same archive/config (recommended: 1-2)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If > 0, pass --max-frames N to run_decompression.py",
    )
    parser.add_argument(
        "--no-interpolate",
        action="store_true",
        help="Disable interpolation for faster smoke testing",
    )
    parser.add_argument(
        "--amt-batch-size",
        type=int,
        default=None,
        help="Optional AMT batch size override passed to run_decompression.py",
    )
    parser.add_argument(
        "--amt-crop-margin",
        type=int,
        default=None,
        help="Optional AMT crop margin override passed to run_decompression.py",
    )
    parser.add_argument(
        "--write-lossless-yuv420",
        action="store_true",
        help="Also write a lossless FFV1 Matroska file encoded as yuv420p.",
    )
    return parser.parse_args()


def _tail(text: str, lines: int = 30) -> str:
    parts = [x for x in text.splitlines() if x.strip()]
    if not parts:
        return "(no output)"
    return "\n".join(parts[-lines:])


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_archive_meta(archive_path: Path) -> Dict[str, Any]:
    meta = json.loads(read_archive_payloads(archive_path)["meta.json"].decode("utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid meta.json in archive: {archive_path}")
    return meta


def _probe_video(path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open reconstructed video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames = 0
    px_hash = hashlib.sha256()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            px_hash.update(frame.tobytes())
            frames += 1
    finally:
        cap.release()

    if frames <= 0:
        raise RuntimeError(f"Reconstructed video has zero frames: {path}")

    return {
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "frames": int(frames),
        "pixels_sha256": px_hash.hexdigest(),
        "file_sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _run_decompression_once(
    *,
    archive_path: Path,
    config_path: Path,
    out_path: Path,
    lossless_out_path: Path,
    lossless_yuv420_out_path: Path | None,
    max_frames: int,
    no_interpolate: bool,
    amt_batch_size: int | None,
    amt_crop_margin: int | None,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "run_decompression.py"),
        str(archive_path),
        "--config",
        str(config_path),
        "--output",
        str(out_path),
        "--lossless-output",
        str(lossless_out_path),
    ]
    if amt_batch_size is not None:
        cmd.extend(["--amt-batch-size", str(int(amt_batch_size))])
    if amt_crop_margin is not None:
        cmd.extend(["--amt-crop-margin", str(int(amt_crop_margin))])
    if lossless_yuv420_out_path is not None:
        cmd.extend(["--lossless-yuv420-output", str(lossless_yuv420_out_path)])
    if int(max_frames) > 0:
        cmd.extend(["--max-frames", str(int(max_frames))])
    if bool(no_interpolate):
        cmd.append("--no-interpolate")

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            "run_decompression.py failed.\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout_tail:\n{_tail(proc.stdout)}\n"
            f"stderr_tail:\n{_tail(proc.stderr)}"
        )
    if not out_path.exists():
        raise FileNotFoundError(f"Expected reconstructed output was not created: {out_path}")
    if not lossless_out_path.exists():
        raise FileNotFoundError(f"Expected lossless reconstructed output was not created: {lossless_out_path}")
    if lossless_yuv420_out_path is not None and not lossless_yuv420_out_path.exists():
        raise FileNotFoundError(
            f"Expected lossless yuv420 reconstructed output was not created: {lossless_yuv420_out_path}"
        )

    video = _probe_video(out_path)
    lossless_file_sha256 = _sha256_file(lossless_out_path)
    result = {
        "output_path": str(out_path),
        "lossless_output_path": str(lossless_out_path),
        "lossless_size_bytes": int(lossless_out_path.stat().st_size),
        "lossless_file_sha256": str(lossless_file_sha256),
        "runtime_sec": round(elapsed, 3),
        "stdout_tail": _tail(proc.stdout, lines=12),
        **video,
    }
    try:
        lossless_video = _probe_video(lossless_out_path)
    except Exception as exc:
        result["lossless_probe_error"] = f"{type(exc).__name__}:{exc}"
    else:
        result["lossless_pixels_sha256"] = str(lossless_video["pixels_sha256"])
        result["lossless_width"] = int(lossless_video["width"])
        result["lossless_height"] = int(lossless_video["height"])
        result["lossless_frames"] = int(lossless_video["frames"])
    if lossless_yuv420_out_path is not None:
        result["lossless_yuv420_output_path"] = str(lossless_yuv420_out_path)
        result["lossless_yuv420_size_bytes"] = int(lossless_yuv420_out_path.stat().st_size)
        result["lossless_yuv420_file_sha256"] = _sha256_file(lossless_yuv420_out_path)
    return result


def _check_reproducibility(runs: List[Dict[str, Any]]) -> List[str]:
    if len(runs) < 2:
        return []
    keys = ["frames", "width", "height", "lossless_file_sha256"]
    baseline = runs[0]
    issues: List[str] = []
    for i in range(1, len(runs)):
        cur = runs[i]
        for k in keys:
            if cur.get(k) != baseline.get(k):
                issues.append(
                    f"run_{i + 1:02d} differs on {k}: baseline={baseline.get(k)} current={cur.get(k)}"
                )
        if (
            baseline.get("lossless_pixels_sha256") is not None
            and cur.get("lossless_pixels_sha256") is not None
            and cur.get("lossless_pixels_sha256") != baseline.get("lossless_pixels_sha256")
        ):
            issues.append(
                "run_{idx:02d} differs on lossless_pixels_sha256: baseline={base} current={cur}".format(
                    idx=i + 1,
                    base=baseline.get("lossless_pixels_sha256"),
                    cur=cur.get("lossless_pixels_sha256"),
                )
            )
        if (
            baseline.get("lossless_yuv420_file_sha256") is not None
            and cur.get("lossless_yuv420_file_sha256") is not None
            and cur.get("lossless_yuv420_file_sha256") != baseline.get("lossless_yuv420_file_sha256")
        ):
            issues.append(
                "run_{idx:02d} differs on lossless_yuv420_file_sha256: baseline={base} current={cur}".format(
                    idx=i + 1,
                    base=baseline.get("lossless_yuv420_file_sha256"),
                    cur=cur.get("lossless_yuv420_file_sha256"),
                )
            )
    return issues


def _effective_decompression_runtime(
    *,
    cfg_path: Path,
    archive_meta: Dict[str, Any],
    max_frames: int,
    no_interpolate: bool,
    amt_batch_size: int | None,
    amt_crop_margin: int | None,
    write_lossless_yuv420: bool,
) -> Dict[str, Any]:
    import run_decompression as decompression_main  # noqa: WPS433
    from decompression import common as rd  # noqa: WPS433

    cfg = rd._load_runtime_cfg(cfg_path)
    dec_cfg = rd._validate_cfg(cfg)
    if bool(no_interpolate):
        interp = (dec_cfg.get("interpolate", {}) or {}).copy()
        interp["enable"] = False
        dec_cfg["interpolate"] = interp

    effective_interp = copy.deepcopy(dec_cfg.get("interpolate", {}) or {})
    if amt_batch_size is not None:
        effective_interp["batch_size"] = int(amt_batch_size)
    if amt_crop_margin is not None:
        effective_interp["crop_margin"] = int(amt_crop_margin)
    dec_cfg["interpolate"] = effective_interp

    meta_copy = copy.deepcopy(archive_meta)
    runtime_device = decompression_main._enforce_strict_gpu_runtime(dec_cfg, meta_copy)
    return {
        "runtime_device": runtime_device,
        "decompression": dec_cfg,
        "archive_dcvc": meta_copy.get("dcvc", {}),
        "requested_overrides": {
            "max_frames": int(max_frames),
            "no_interpolate": bool(no_interpolate),
            "amt_batch_size": (None if amt_batch_size is None else int(amt_batch_size)),
            "amt_crop_margin": (None if amt_crop_margin is None else int(amt_crop_margin)),
            "write_lossless_yuv420": bool(write_lossless_yuv420),
        },
    }


def main() -> None:
    args = _parse_args()
    repeats = max(1, int(args.repeat))

    archive_path = resolve_from_root(args.archive)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    cfg_path = resolve_from_root(args.config)
    _ = load_yaml(cfg_path)  # validate YAML shape early
    meta = _read_archive_meta(archive_path)
    out_root = resolve_out_dir(args.out_dir, "outputs/sanity_checks/decompression")
    effective_runtime = _effective_decompression_runtime(
        cfg_path=cfg_path,
        archive_meta=meta,
        max_frames=int(args.max_frames),
        no_interpolate=bool(args.no_interpolate),
        amt_batch_size=(None if args.amt_batch_size is None else int(args.amt_batch_size)),
        amt_crop_margin=(None if args.amt_crop_margin is None else int(args.amt_crop_margin)),
        write_lossless_yuv420=bool(args.write_lossless_yuv420),
    )

    session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir = out_root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    expected = meta.get("video", {}) or {}
    expected_w = int(expected.get("width", 0) or 0)
    expected_h = int(expected.get("height", 0) or 0)
    expected_frames = int(expected.get("frames_total", 0) or 0)
    if int(args.max_frames) > 0 and expected_frames > 0:
        expected_frames = min(expected_frames, int(args.max_frames))

    runs: List[Dict[str, Any]] = []
    for i in range(1, repeats + 1):
        run_dir = session_dir / f"run_{i:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "reconstructed.mp4"
        lossless_out_path = run_dir / "reconstructed_lossless.avi"
        lossless_yuv420_out_path = (
            run_dir / "reconstructed_lossless_yuv420.mkv" if bool(args.write_lossless_yuv420) else None
        )
        result = _run_decompression_once(
            archive_path=archive_path,
            config_path=cfg_path,
            out_path=out_path,
            lossless_out_path=lossless_out_path,
            lossless_yuv420_out_path=lossless_yuv420_out_path,
            max_frames=int(args.max_frames),
            no_interpolate=bool(args.no_interpolate),
            amt_batch_size=(None if args.amt_batch_size is None else int(args.amt_batch_size)),
            amt_crop_margin=(None if args.amt_crop_margin is None else int(args.amt_crop_margin)),
        )

        if expected_w > 0 and int(result["width"]) != expected_w:
            raise RuntimeError(
                f"Width mismatch in run_{i:02d}: expected={expected_w} actual={result['width']}"
            )
        if expected_h > 0 and int(result["height"]) != expected_h:
            raise RuntimeError(
                f"Height mismatch in run_{i:02d}: expected={expected_h} actual={result['height']}"
            )
        if expected_frames > 0 and int(result["frames"]) != expected_frames:
            raise RuntimeError(
                f"Frame-count mismatch in run_{i:02d}: expected={expected_frames} actual={result['frames']}"
            )

        result["run_index"] = int(i)
        result["run_dir"] = str(run_dir)
        runs.append(result)

    repro_issues = _check_reproducibility(runs)
    summary = {
        "archive_path": str(archive_path),
        "config_path": str(cfg_path),
        "repeat_runs": int(repeats),
        "expected_width": int(expected_w),
        "expected_height": int(expected_h),
        "expected_frames": int(expected_frames),
        "reproducible_pixels": not bool(repro_issues),
        "repro_issues": repro_issues,
        "effective_runtime": effective_runtime,
        "runs": runs,
    }
    summary_path = session_dir / "summary.json"
    write_json(summary_path, summary)

    if repro_issues:
        raise RuntimeError(
            "Decompression reproducibility check failed.\n"
            + "\n".join(repro_issues)
            + f"\nDetails: {summary_path}"
        )

    print("[OK] Decompression sanity check complete")
    print(f"  archive              : {archive_path}")
    print(f"  config               : {cfg_path}")
    print(f"  session_dir          : {session_dir}")
    print(f"  summary_json         : {summary_path}")
    for run in runs:
        print(
            "  run_{idx:02d} output    : {path} ({size} bytes, {human})".format(
                idx=run["run_index"],
                path=run["output_path"],
                size=run["size_bytes"],
                human=human_bytes(int(run["size_bytes"])),
            )
        )
        print(
            "  run_{idx:02d} lossless  : {path} ({size} bytes, {human})".format(
                idx=run["run_index"],
                path=run["lossless_output_path"],
                size=run["lossless_size_bytes"],
                human=human_bytes(int(run["lossless_size_bytes"])),
            )
        )
        if run.get("lossless_yuv420_output_path"):
            print(
                "  run_{idx:02d} yuv420    : {path} ({size} bytes, {human})".format(
                    idx=run["run_index"],
                    path=run["lossless_yuv420_output_path"],
                    size=run["lossless_yuv420_size_bytes"],
                    human=human_bytes(int(run["lossless_yuv420_size_bytes"])),
                )
            )
        print(
            f"  run_{run['run_index']:02d} frames/fps : "
            f"{run['frames']} @ {run['fps']:.3f} "
            f"({run['width']}x{run['height']})"
        )
        print(f"  run_{run['run_index']:02d} pixels_sha : {run['pixels_sha256']}")
    print("  reproducibility       : PASS")


if __name__ == "__main__":
    main()

