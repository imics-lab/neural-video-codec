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
from frame_removal import (  # noqa: E402
    apply_dual_timeline_policy,
    remove_redundant_frames,
    validate_frame_removal_config,
    write_kept_frames_video,
)
from roi_detection import run_roi_detection  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 sanity check: frame removal")
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
        default="outputs/sanity_checks/frame_removal",
        help="Directory for outputs",
    )
    parser.add_argument("--skip-roi", action="store_true", help="Skip ROI detection and use empty ROI map")
    parser.add_argument("--no-viz", action="store_true", help="Skip preview/debug video outputs")
    parser.add_argument(
        "--debug-max-frames",
        type=int,
        default=0,
        help="If > 0, write only first N frames in debug overlay",
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


def _write_debug_overlay(video_path: Path, frame_result: Dict[str, Any], out_path: Path, max_frames: int) -> int:
    per_frame = frame_result.get("per_frame", {}) or {}
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
        raise RuntimeError(f"Failed to open debug writer: {out_path}")

    idx = 0
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames > 0 and idx >= max_frames:
                break

            rec = per_frame.get(str(idx), {})
            keep = bool(rec.get("keep", idx == 0))
            state = str(rec.get("state", "NA"))
            score = rec.get("roi_motion_score", None)
            bbox_move = rec.get("bbox_motion_px", None)
            urgency = rec.get("motion_urgency", None)
            motion_streak = rec.get("motion_streak", None)
            still_streak = rec.get("still_streak", None)
            births = rec.get("track_births", None)

            color = (0, 190, 0) if keep else (0, 0, 220)
            text = f"frame={idx} {'KEEP' if keep else 'DROP'} state={state}"
            if isinstance(score, (int, float)):
                text += f" score={float(score):.2f}"
            if isinstance(bbox_move, (int, float)):
                text += f" bbox_px={float(bbox_move):.2f}"
            if isinstance(urgency, (int, float)):
                text += f" u={float(urgency):.2f}"
            if isinstance(motion_streak, (int, float)) and isinstance(still_streak, (int, float)):
                text += f" m_stk={int(motion_streak)} s_stk={int(still_streak)}"
            if isinstance(births, (int, float)):
                text += f" births={int(births)}"
            cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

            roi_box = rec.get("roi_box", None)
            if isinstance(roi_box, dict):
                x1 = max(0, min(width - 1, int(roi_box.get("x1", 0))))
                y1 = max(0, min(height - 1, int(roi_box.get("y1", 0))))
                x2 = max(0, min(width - 1, int(roi_box.get("x2", 0))))
                y2 = max(0, min(height - 1, int(roi_box.get("y2", 0))))
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            writer.write(frame)
            written += 1
            idx += 1
    finally:
        cap.release()
        writer.release()

    return written


def _state_stats(per_frame: Dict[str, Any]) -> Dict[str, int]:
    if not isinstance(per_frame, dict) or not per_frame:
        return {"motion_frames": 0, "still_frames": 0, "state_transitions": 0}
    keys = sorted((int(k) for k in per_frame.keys() if str(k).isdigit()))
    states = [str((per_frame.get(str(k), {}) or {}).get("state", "STILL")).upper() for k in keys]
    motion = sum(1 for s in states if s == "MOTION")
    still = sum(1 for s in states if s == "STILL")
    transitions = 0
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            transitions += 1
    return {"motion_frames": int(motion), "still_frames": int(still), "state_transitions": int(transitions)}


def _urgency_stats(per_frame: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(per_frame, dict) or not per_frame:
        return {"urgency_mean": 0.0, "urgency_p90": 0.0, "urgency_max": 0.0}
    vals: list[float] = []
    for k in per_frame.keys():
        rec = per_frame.get(str(k), {}) or {}
        u = rec.get("motion_urgency", None)
        if isinstance(u, (int, float)):
            vals.append(float(u))
    if not vals:
        return {"urgency_mean": 0.0, "urgency_p90": 0.0, "urgency_max": 0.0}
    vals.sort()
    n = len(vals)
    p90_idx = int(max(0, min(n - 1, round(0.9 * (n - 1)))))
    return {
        "urgency_mean": float(sum(vals) / float(n)),
        "urgency_p90": float(vals[p90_idx]),
        "urgency_max": float(vals[-1]),
    }


def main() -> None:
    args = _parse_args()
    cfg_path = resolve_from_root(args.config)
    cfg = load_yaml(cfg_path)
    video_path = resolve_video_path(cfg, args.video)
    out_dir = resolve_out_dir(args.out_dir, "outputs/sanity_checks/frame_removal")

    roi_json_path = out_dir / "roi_detections.json"
    frame_drop_path = out_dir / "frame_drop.json"
    summary_path = out_dir / "summary.json"
    roi_keep_preview_path = out_dir / "roi_kept_preview.mp4"
    bg_keep_preview_path = out_dir / "bg_kept_preview.mp4"
    debug_overlay_path = out_dir / "frame_drop_overlay.mp4"

    roi_cfg = cfg.get("roi_detection", {}) or {}
    frame_cfg = cfg.get("frame_removal", {}) or {}
    validate_frame_removal_config(frame_cfg)

    roi_elapsed = 0.0
    if bool(roi_cfg.get("enable", True)) and not args.skip_roi:
        t0 = time.perf_counter()
        roi_result = run_roi_detection(str(video_path), roi_cfg)
        roi_elapsed = time.perf_counter() - t0
        roi_frame_map = _coerce_frame_map(roi_result.get("frames", {}) or {})
        write_json(roi_json_path, roi_result)
    else:
        roi_result = {"frames": {}}
        roi_frame_map = {}

    t1 = time.perf_counter()
    frame_result = remove_redundant_frames(str(video_path), roi_frame_map, frame_cfg)
    frame_elapsed = time.perf_counter() - t1

    frame_result_out = apply_dual_timeline_policy(frame_result, frame_cfg)
    write_json(frame_drop_path, frame_result_out)

    debug_frames_written = 0
    if not args.no_viz:
        write_kept_frames_video(
            video_path=video_path,
            kept_frames=frame_result_out.get("roi_kept_frames", []),
            roi_bbox_map=roi_result.get("frames", {}) or {},
            render_mode="roi",
            output_path=roi_keep_preview_path,
        )
        write_kept_frames_video(
            video_path=video_path,
            kept_frames=frame_result_out.get("bg_kept_frames", []),
            roi_bbox_map=roi_result.get("frames", {}) or {},
            render_mode="bg",
            output_path=bg_keep_preview_path,
        )
        debug_frames_written = _write_debug_overlay(
            video_path=video_path,
            frame_result=frame_result_out,
            out_path=debug_overlay_path,
            max_frames=max(0, int(args.debug_max_frames)),
        )

    kept = int(len(frame_result_out.get("kept_frames", []) or []))
    dropped = int(len(frame_result_out.get("dropped_frames", []) or []))
    total = max(1, kept + dropped)
    roi_kept = int(len(frame_result_out.get("roi_kept_frames", []) or []))
    roi_dropped = int(len(frame_result_out.get("roi_dropped_frames", []) or []))
    bg_kept = int(len(frame_result_out.get("bg_kept_frames", []) or []))
    bg_dropped = int(len(frame_result_out.get("bg_dropped_frames", []) or []))
    roi_total = max(1, roi_kept + roi_dropped)
    bg_total = max(1, bg_kept + bg_dropped)
    st = _state_stats(frame_result_out.get("per_frame", {}) or {})
    urg = _urgency_stats(frame_result_out.get("per_frame", {}) or {})
    state_total = max(1, int(st["motion_frames"]) + int(st["still_frames"]))
    motion_pct = (100.0 * float(st["motion_frames"])) / float(state_total)
    still_pct = (100.0 * float(st["still_frames"])) / float(state_total)
    dual_policy = ((frame_result_out.get("dual_timeline", {}) or {}).get("policy", {}) or {})
    roi_policy_mode = str(dual_policy.get("roi_mode", "fixed_state_interval"))
    summary = {
        "video_path": str(video_path),
        "config_path": str(cfg_path),
        "kept_frames": kept,
        "dropped_frames": dropped,
        "drop_ratio": float(dropped) / float(total),
        "roi_kept_frames": roi_kept,
        "roi_dropped_frames": roi_dropped,
        "roi_drop_ratio": float(roi_dropped) / float(roi_total),
        "bg_kept_frames": bg_kept,
        "bg_dropped_frames": bg_dropped,
        "bg_drop_ratio": float(bg_dropped) / float(bg_total),
        "motion_frames": int(st["motion_frames"]),
        "still_frames": int(st["still_frames"]),
        "motion_pct": float(motion_pct),
        "still_pct": float(still_pct),
        "state_transitions": int(st["state_transitions"]),
        "roi_policy_mode": roi_policy_mode,
        "urgency_mean": float(urg["urgency_mean"]),
        "urgency_p90": float(urg["urgency_p90"]),
        "urgency_max": float(urg["urgency_max"]),
        "first_kept_frames": (frame_result_out.get("kept_frames", []) or [])[:20],
        "frame_drop_sha256": sha256_json(frame_result_out),
        "roi_time_sec": round(roi_elapsed, 3),
        "frame_removal_time_sec": round(frame_elapsed, 3),
    }
    if not args.no_viz:
        summary["roi_keep_preview_path"] = str(roi_keep_preview_path)
        summary["bg_keep_preview_path"] = str(bg_keep_preview_path)
        summary["debug_overlay_path"] = str(debug_overlay_path)
        summary["debug_frames_written"] = int(debug_frames_written)

    write_json(summary_path, summary)

    print("[OK] Frame-removal sanity check complete")
    print(f"  config               : {cfg_path}")
    print(f"  video                : {video_path}")
    if roi_json_path.exists():
        print(f"  roi_json             : {roi_json_path}")
    print(f"  frame_drop_json      : {frame_drop_path}")
    if not args.no_viz:
        print(f"  roi_kept_preview     : {roi_keep_preview_path}")
        print(f"  bg_kept_preview      : {bg_keep_preview_path}")
        print(f"  debug_overlay        : {debug_overlay_path}")
    print(f"  kept_frames          : {kept}")
    print(f"  dropped_frames       : {dropped}")
    print(f"  drop_ratio           : {summary['drop_ratio']:.3f}")
    print(f"  roi_kept_frames      : {roi_kept}")
    print(f"  roi_dropped_frames   : {roi_dropped}")
    print(f"  roi_drop_ratio       : {summary['roi_drop_ratio']:.3f}")
    print(f"  bg_kept_frames       : {bg_kept}")
    print(f"  bg_dropped_frames    : {bg_dropped}")
    print(f"  bg_drop_ratio        : {summary['bg_drop_ratio']:.3f}")
    print(f"  motion_frames        : {summary['motion_frames']}")
    print(f"  still_frames         : {summary['still_frames']}")
    print(f"  motion_pct           : {summary['motion_pct']:.3f}%")
    print(f"  still_pct            : {summary['still_pct']:.3f}%")
    print(f"  state_transitions    : {summary['state_transitions']}")
    print(f"  roi_policy_mode      : {summary['roi_policy_mode']}")
    print(f"  urgency_mean         : {summary['urgency_mean']:.3f}")
    print(f"  urgency_p90          : {summary['urgency_p90']:.3f}")
    print(f"  urgency_max          : {summary['urgency_max']:.3f}")
    print(f"  frame_drop_sha256    : {summary['frame_drop_sha256']}")
    print(f"  roi_time_sec         : {summary['roi_time_sec']}")
    print(f"  frame_removal_time   : {summary['frame_removal_time_sec']}")
    print("  manual_check         : Confirm dropped frames are visually redundant.")


if __name__ == "__main__":
    main()
