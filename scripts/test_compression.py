from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _phase0_utils import (  # noqa: E402
    DEFAULT_COMPRESSION_CONFIG,
    human_bytes,
    load_yaml,
    read_archive_payloads,
    resolve_from_root,
    resolve_out_dir,
    resolve_video_path,
    sha256_bytes,
    sha256_json,
    write_json,
    write_yaml,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 sanity check: compression + reproducibility")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_COMPRESSION_CONFIG),
        help="Compression config YAML path",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional input video override. Defaults to data/test.mp4 when available.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/sanity_checks/compression",
        help="Directory for outputs",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="Number of repeated runs with the same config/video (recommended: 2)",
    )
    parser.add_argument(
        "--allow-larger",
        action="store_true",
        help="Do not fail when compressed archive is not smaller than source video",
    )
    return parser.parse_args()


def _tail(text: str, lines: int = 30) -> str:
    parts = [x for x in text.splitlines() if x.strip()]
    if not parts:
        return "(no output)"
    return "\n".join(parts[-lines:])


def _read_archive_metrics(archive_path: Path) -> Dict[str, Any]:
    payload = read_archive_payloads(archive_path)

    roi_json = json.loads(payload["roi_detections.json"].decode("utf-8"))
    frame_drop_json = json.loads(payload["frame_drop.json"].decode("utf-8"))
    meta = json.loads(payload["meta.json"].decode("utf-8"))
    streams = meta.get("streams", {}) or {}
    roi_stream = streams.get("roi", {}) or {}
    bg_stream = streams.get("bg", {}) or {}
    roi_meta = meta.get("roi_detection", {}) or {}
    model_selection = roi_meta.get("model_selection", {}) or {}
    runtime_meta = meta.get("runtime", {}) or {}
    roi_kept = frame_drop_json.get("roi_kept_frames", frame_drop_json.get("kept_frames", [])) or []
    bg_kept = frame_drop_json.get("bg_kept_frames", frame_drop_json.get("kept_frames", [])) or []
    roi_dropped = frame_drop_json.get("roi_dropped_frames", frame_drop_json.get("dropped_frames", [])) or []
    bg_dropped = frame_drop_json.get("bg_dropped_frames", frame_drop_json.get("dropped_frames", [])) or []

    return {
        "archive_path": str(archive_path),
        "zip_size_bytes": int(archive_path.stat().st_size),
        "roi_bytes": int(len(payload["roi.bin"])),
        "bg_bytes": int(len(payload["bg.bin"])),
        "payload_bytes": int(len(payload["roi.bin"]) + len(payload["bg.bin"])),
        "roi_frames_with_boxes": int(len((roi_json.get("frames", {}) or {}))),
        "roi_kept_frames": int(len(roi_kept)),
        "roi_dropped_frames": int(len(roi_dropped)),
        "bg_kept_frames": int(len(bg_kept)),
        "bg_dropped_frames": int(len(bg_dropped)),
        "roi_stream_frames_encoded": int(roi_stream.get("frames_encoded", 0) or 0),
        "bg_stream_frames_encoded": int(bg_stream.get("frames_encoded", 0) or 0),
        "roi_bin_sha256": sha256_bytes(payload["roi.bin"]),
        "bg_bin_sha256": sha256_bytes(payload["bg.bin"]),
        "roi_json_sha256": sha256_json(roi_json),
        "frame_drop_sha256": sha256_json(frame_drop_json),
        "roi_frame_index_map_sha256": sha256_json(list(roi_stream.get("frame_index_map", []) or [])),
        "bg_frame_index_map_sha256": sha256_json(list(bg_stream.get("frame_index_map", []) or [])),
        "roi_model_format_selected": str(model_selection.get("selected_format", "")),
        "roi_model_path_selected": str(model_selection.get("selected_model_path", "")),
        "roi_model_resolved_path_selected": str(model_selection.get("selected_model_resolved_path", "")),
        "roi_model_fallback_reason": str(model_selection.get("fallback_reason", "")),
        "roi_model_prefer_onnx_strict": bool(model_selection.get("prefer_onnx_strict", False)),
        "codec_backend_selected": str(runtime_meta.get("codec_backend_selected", "")),
        "meta_size_reported": meta.get("sizes", {}),
    }


def _run_compression_once(
    *,
    base_cfg: Dict[str, Any],
    video_path: Path,
    session_dir: Path,
    run_index: int,
) -> Dict[str, Any]:
    run_dir = session_dir / f"run_{run_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    runtime_cfg = copy.deepcopy(base_cfg)
    input_cfg = runtime_cfg.setdefault("input", {})
    if not isinstance(input_cfg, dict):
        runtime_cfg["input"] = {}
        input_cfg = runtime_cfg["input"]
    input_cfg["video_path"] = str(video_path)

    out_cfg = runtime_cfg.setdefault("output", {})
    if not isinstance(out_cfg, dict):
        runtime_cfg["output"] = {}
        out_cfg = runtime_cfg["output"]
    out_cfg["write_outputs"] = True
    out_cfg["out_dir"] = str(run_dir)
    out_cfg["roi_json"] = "roi_detections.json"
    out_cfg["frame_drop_json"] = "frame_drop.json"
    out_cfg["roi_bin"] = "roi.bin"
    out_cfg["bg_bin"] = "bg.bin"
    out_cfg["meta_json"] = "meta.json"

    runtime_cfg_path = run_dir / "compression_config.runtime.yaml"
    write_yaml(runtime_cfg_path, runtime_cfg)

    cmd = [
        sys.executable,
        str(ROOT / "run_compression.py"),
        str(video_path),
        "--config",
        str(runtime_cfg_path),
    ]

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
            "run_compression.py failed.\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout_tail:\n{_tail(proc.stdout)}\n"
            f"stderr_tail:\n{_tail(proc.stderr)}"
        )

    archive_path = run_dir / f"{video_path.stem}.zip"
    if not archive_path.exists():
        zips = sorted(run_dir.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"No zip output found in {run_dir}")
        archive_path = zips[0]

    metrics = _read_archive_metrics(archive_path)
    metrics["run_index"] = int(run_index)
    metrics["run_dir"] = str(run_dir)
    metrics["runtime_config"] = str(runtime_cfg_path)
    metrics["runtime_sec"] = round(elapsed, 3)
    metrics["stdout_tail"] = _tail(proc.stdout, lines=12)
    return metrics


def _check_reproducibility(run_summaries: List[Dict[str, Any]]) -> List[str]:
    if len(run_summaries) < 2:
        return []

    keys = [
        "roi_bytes",
        "bg_bytes",
        "payload_bytes",
        "roi_frames_with_boxes",
        "roi_kept_frames",
        "roi_dropped_frames",
        "bg_kept_frames",
        "bg_dropped_frames",
        "roi_stream_frames_encoded",
        "bg_stream_frames_encoded",
        "roi_bin_sha256",
        "bg_bin_sha256",
        "roi_json_sha256",
        "frame_drop_sha256",
        "roi_frame_index_map_sha256",
        "bg_frame_index_map_sha256",
        "roi_model_format_selected",
        "roi_model_path_selected",
        "roi_model_resolved_path_selected",
        "roi_model_fallback_reason",
        "roi_model_prefer_onnx_strict",
        "codec_backend_selected",
    ]
    baseline = run_summaries[0]
    issues: List[str] = []
    for idx in range(1, len(run_summaries)):
        cur = run_summaries[idx]
        for k in keys:
            if cur.get(k) != baseline.get(k):
                issues.append(
                    f"run_{idx + 1:02d} differs on {k}: "
                    f"baseline={baseline.get(k)} current={cur.get(k)}"
                )
    return issues


def main() -> None:
    args = _parse_args()
    repeats = max(1, int(args.repeat))

    cfg_path = resolve_from_root(args.config)
    cfg = load_yaml(cfg_path)
    video_path = resolve_video_path(cfg, args.video)
    out_root = resolve_out_dir(args.out_dir, "outputs/sanity_checks/compression")

    session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_dir = out_root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    source_size = int(video_path.stat().st_size)
    runs: List[Dict[str, Any]] = []
    for i in range(1, repeats + 1):
        runs.append(
            _run_compression_once(
                base_cfg=cfg,
                video_path=video_path,
                session_dir=session_dir,
                run_index=i,
            )
        )

    size_issues: List[str] = []
    for run in runs:
        if int(run["zip_size_bytes"]) >= source_size:
            size_issues.append(
                f"run_{run['run_index']:02d}: zip={run['zip_size_bytes']} source={source_size}"
            )
    if size_issues and not args.allow_larger:
        raise RuntimeError(
            "Compressed archive is not smaller than source video.\n"
            + "\n".join(size_issues)
            + "\nTip: rerun with --allow-larger only for debugging."
        )

    repro_issues = _check_reproducibility(runs)
    summary = {
        "video_path": str(video_path),
        "config_path": str(cfg_path),
        "source_size_bytes": source_size,
        "source_size_human": human_bytes(source_size),
        "repeat_runs": repeats,
        "size_check_passed": not bool(size_issues),
        "reproducible": not bool(repro_issues),
        "size_issues": size_issues,
        "repro_issues": repro_issues,
        "runs": runs,
    }
    summary_path = session_dir / "summary.json"
    write_json(summary_path, summary)

    if repro_issues:
        raise RuntimeError(
            "Reproducibility check failed for repeated runs.\n"
            + "\n".join(repro_issues)
            + f"\nDetails: {summary_path}"
        )

    print("[OK] Compression sanity check complete")
    print(f"  config               : {cfg_path}")
    print(f"  video                : {video_path}")
    print(f"  source_size          : {source_size} bytes ({human_bytes(source_size)})")
    print(f"  session_dir          : {session_dir}")
    print(f"  summary_json         : {summary_path}")
    for run in runs:
        print(
            "  run_{idx:02d} archive   : {path} ({size} bytes, {human})".format(
                idx=run["run_index"],
                path=run["archive_path"],
                size=run["zip_size_bytes"],
                human=human_bytes(int(run["zip_size_bytes"])),
            )
        )
        print(
            f"  run_{run['run_index']:02d} payload   : {run['payload_bytes']} bytes "
            f"(roi={run['roi_bytes']}, bg={run['bg_bytes']})"
        )
        print(
            f"  run_{run['run_index']:02d} anchors   : "
            f"roi={run['roi_kept_frames']} bg={run['bg_kept_frames']} "
            f"(encoded roi={run['roi_stream_frames_encoded']}, bg={run['bg_stream_frames_encoded']})"
        )
        print(
            f"  run_{run['run_index']:02d} runtime   : "
            f"roi_model={run['roi_model_format_selected']} "
            f"codec={run['codec_backend_selected']}"
        )
    if size_issues:
        print("  size_check            : FAILED (allowed by --allow-larger)")
    else:
        print("  size_check            : PASS")
    print("  reproducibility       : PASS")
    print("  manual_next_step      : Run run_decompression.py on the zip and inspect reconstruction.")


if __name__ == "__main__":
    main()
