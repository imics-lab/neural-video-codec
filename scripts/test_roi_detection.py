from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _phase0_utils import (  # noqa: E402
    DEFAULT_COMPRESSION_CONFIG,
    load_yaml,
    resolve_from_root,
    resolve_out_dir,
    resolve_video_path,
    sha256_json,
    write_json,
)
from roi_detection import run_roi_detection  # noqa: E402


def _estimate_detector_calls(roi_result: Dict[str, Any], keyframe_interval: int) -> int:
    start_raw = roi_result.get("start_frame", 0)
    end_raw = roi_result.get("end_frame", -1)
    start_f = int(0 if start_raw is None else start_raw)
    end_f = int(-1 if end_raw is None else end_raw)
    if keyframe_interval < 1 or end_f < start_f:
        return 0
    first_key = start_f if (start_f % keyframe_interval) == 0 else (start_f + (keyframe_interval - (start_f % keyframe_interval)))
    if first_key > end_f:
        return 0
    return int(((end_f - first_key) // keyframe_interval) + 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 sanity check: ROI detection")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_COMPRESSION_CONFIG),
        help="Compression config YAML path",
    )
    parser.add_argument("--video", type=str, default=None, help="Optional input video override")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/sanity_checks/roi_detection",
        help="Directory for outputs",
    )
    parser.add_argument("--no-overlay", action="store_true", help="Skip ROI overlay video output")
    parser.add_argument(
        "--overlay-max-frames",
        type=int,
        default=0,
        help="If > 0, writes only first N frames in overlay video",
    )
    return parser.parse_args()


def _coerce_frame_map(raw_map: Dict[Any, Any]) -> Dict[int, list[Any]]:
    out: Dict[int, list[Any]] = {}
    for k, v in raw_map.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, list):
            out[idx] = v
    return out


def _draw_roi_overlay(video_path: Path, frame_map: Dict[int, list[Any]], out_path: Path, max_frames: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid input video dimensions")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1e-6, fps),
        (width, height),
        True,
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open overlay writer: {out_path}")

    written = 0
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames > 0 and idx >= max_frames:
                break

            boxes = frame_map.get(idx, [])
            for roi in boxes:
                if not isinstance(roi, dict):
                    continue
                x1 = int(roi.get("x1", 0))
                y1 = int(roi.get("y1", 0))
                x2 = int(roi.get("x2", 0))
                y2 = int(roi.get("y2", 0))
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width - 1, x2))
                y2 = max(0, min(height - 1, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                conf = roi.get("conf", roi.get("confidence", None))
                if isinstance(conf, (int, float)):
                    cv2.putText(
                        frame,
                        f"{float(conf):.2f}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                frame,
                f"frame={idx} rois={len(boxes)}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            written += 1
            idx += 1
    finally:
        cap.release()
        writer.release()

    return written


def main() -> None:
    args = _parse_args()
    cfg_path = resolve_from_root(args.config)
    cfg = load_yaml(cfg_path)
    video_path = resolve_video_path(cfg, args.video)
    out_dir = resolve_out_dir(args.out_dir, "outputs/sanity_checks/roi_detection")

    roi_cfg = cfg.get("roi_detection", {}) or {}
    tracking_cfg = roi_cfg.get("tracking", {}) or {}
    runtime_cfg = roi_cfg.get("runtime", {}) or {}

    t0 = time.perf_counter()
    roi_result = run_roi_detection(str(video_path), roi_cfg)
    dt = time.perf_counter() - t0

    frame_map = _coerce_frame_map(roi_result.get("frames", {}) or {})
    total_boxes = sum(len(v) for v in frame_map.values())
    frames_with_roi = len(frame_map)
    keyframe_interval = int(runtime_cfg.get("keyframe_interval", 1) or 1)
    detector_calls = _estimate_detector_calls(roi_result, keyframe_interval)

    roi_json_path = out_dir / "roi_detections.json"
    summary_path = out_dir / "summary.json"
    overlay_path = out_dir / "roi_overlay.mp4"

    write_json(roi_json_path, roi_result)
    summary = {
        "video_path": str(video_path),
        "config_path": str(cfg_path),
        "frame_count_processed": int(roi_result.get("frame_count", 0)),
        "frames_with_roi": int(frames_with_roi),
        "total_boxes": int(total_boxes),
        "keyframe_interval": int(keyframe_interval),
        "estimated_detector_calls": int(detector_calls),
        "tracking_enable_propagation": bool(tracking_cfg.get("enable_propagation", True)),
        "first_frames_with_roi": sorted(frame_map.keys())[:15],
        "roi_json_sha256": sha256_json(roi_result),
        "runtime_sec": round(dt, 3),
    }

    if not args.no_overlay:
        frames_written = _draw_roi_overlay(
            video_path=video_path,
            frame_map=frame_map,
            out_path=overlay_path,
            max_frames=max(0, int(args.overlay_max_frames)),
        )
        summary["overlay_video_path"] = str(overlay_path)
        summary["overlay_frames_written"] = int(frames_written)

    write_json(summary_path, summary)

    print("[OK] ROI detection sanity check complete")
    print(f"  config               : {cfg_path}")
    print(f"  video                : {video_path}")
    print(f"  roi_json             : {roi_json_path}")
    if not args.no_overlay:
        print(f"  overlay_video        : {overlay_path}")
    print(f"  frames_with_roi      : {frames_with_roi}")
    print(f"  total_boxes          : {total_boxes}")
    print(f"  keyframe_interval    : {summary['keyframe_interval']}")
    print(f"  detector_calls_est   : {summary['estimated_detector_calls']}")
    print(f"  propagation_enabled  : {summary['tracking_enable_propagation']}")
    print(f"  roi_json_sha256      : {summary['roi_json_sha256']}")
    print(f"  runtime_sec          : {summary['runtime_sec']}")
    print("  manual_check         : Confirm animals are detected and boxes are plausible.")


if __name__ == "__main__":
    main()
