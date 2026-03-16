from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _as_bool(value: Any, field_name: str, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    raise ValueError(f"{field_name} must be a boolean")


def _as_int_ge_1(value: Any, field_name: str) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer >= 1") from exc
    if iv < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return iv


def _dedupe_sorted(values: List[int]) -> List[int]:
    return sorted(set(int(v) for v in values)) if values else []


def _entry_for_frame(per_frame: Dict[str, Any], idx: int) -> Dict[str, Any]:
    entry = per_frame.get(str(idx), None)
    return entry if isinstance(entry, dict) else {}


def _entry_has_detection(entry: Dict[str, Any]) -> bool:
    if not isinstance(entry, dict) or not entry:
        return False
    try:
        if float(entry.get("roi_count", 0) or 0) > 0.0:
            return True
    except (TypeError, ValueError, AttributeError):
        pass
    if "bbox_missing" in entry:
        return not bool(entry.get("bbox_missing", False))
    roi_box = entry.get("roi_box", None)
    if isinstance(roi_box, dict):
        try:
            return (
                int(roi_box.get("x2", 0)) > int(roi_box.get("x1", 0))
                and int(roi_box.get("y2", 0)) > int(roi_box.get("y1", 0))
            )
        except (TypeError, ValueError):
            return False
    return False


def _state_for_frame(per_frame: Dict[str, Any], idx: int) -> str:
    entry = _entry_for_frame(per_frame, idx)
    state = str(entry.get("state", "STILL")).strip().upper()
    return state if state in {"MOTION", "STILL"} else "STILL"


def validate_dual_timeline_config(cfg: Dict[str, Any]) -> None:
    frame_cfg = (cfg.get("frame_removal", {}) or {})
    dual_cfg = frame_cfg.get("dual_timeline", {})
    if dual_cfg is None:
        return
    if not isinstance(dual_cfg, dict):
        raise ValueError("frame_removal.dual_timeline must be an object")
    if "enable" in dual_cfg and not isinstance(dual_cfg["enable"], bool):
        raise ValueError("frame_removal.dual_timeline.enable must be a boolean")
    for key in ("roi_motion_interval", "roi_still_interval", "bg_interval", "bg_idle_interval"):
        if key in dual_cfg:
            _as_int_ge_1(dual_cfg[key], f"frame_removal.dual_timeline.{key}")


def _build_fixed_streams(
    *,
    per_frame: Dict[str, Any],
    num_frames: int,
    roi_motion_interval: int,
    roi_still_interval: int,
    bg_interval: int,
    bg_idle_interval: int,
) -> Tuple[List[int], List[int]]:
    roi_kept_frames: List[int] = []
    bg_kept_frames: List[int] = []
    last_roi_keep = -1
    last_bg_keep = -1
    prev_has_detection = False
    prev_bg_mode: str | None = None

    for t in range(max(0, int(num_frames))):
        entry = _entry_for_frame(per_frame, t)
        has_detection = _entry_has_detection(entry)
        bg_mode = "detect" if has_detection else "idle"

        if has_detection:
            state = _state_for_frame(per_frame, t)
            roi_interval = roi_motion_interval if state == "MOTION" else roi_still_interval
            roi_birth = not prev_has_detection
            roi_gap = (t - last_roi_keep) if last_roi_keep >= 0 else (t + 1)
            roi_keep = bool(roi_birth or (roi_gap >= roi_interval))
        else:
            roi_keep = False

        bg_interval_curr = bg_interval if has_detection else bg_idle_interval
        bg_birth = bg_mode != prev_bg_mode
        bg_gap = (t - last_bg_keep) if last_bg_keep >= 0 else (t + 1)
        bg_keep = bool(bg_birth or (bg_gap >= bg_interval_curr))

        if roi_keep:
            roi_kept_frames.append(int(t))
            last_roi_keep = int(t)
        if bg_keep:
            bg_kept_frames.append(int(t))
            last_bg_keep = int(t)

        prev_has_detection = bool(has_detection)
        prev_bg_mode = bg_mode

    return _dedupe_sorted(roi_kept_frames), _dedupe_sorted(bg_kept_frames)


def apply_dual_timeline_metadata(frame_result: Dict[str, Any], dual_meta: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(frame_result)
    out["dual_timeline"] = dict(dual_meta)

    roi_kept = _dedupe_sorted([int(x) for x in (dual_meta.get("roi_kept_frames", []) or [])])
    bg_kept = _dedupe_sorted([int(x) for x in (dual_meta.get("bg_kept_frames", []) or [])])

    stats = dict(out.get("stats", {}) or {})
    n_frames = int(stats.get("num_frames_read", 0) or 0)
    if n_frames <= 0:
        n_frames = max(max(roi_kept or [0]), max(bg_kept or [0])) + 1
    all_frames = set(range(max(0, n_frames)))

    roi_dropped = sorted(all_frames - set(roi_kept))
    bg_dropped = sorted(all_frames - set(bg_kept))
    roi_kept_set = set(roi_kept)
    bg_kept_set = set(bg_kept)

    per_frame_in = out.get("per_frame", {}) or {}
    if isinstance(per_frame_in, dict):
        per_frame_out: Dict[str, Any] = {}
        for key, value in per_frame_in.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                per_frame_out[str(key)] = value
                continue
            rec = dict(value) if isinstance(value, dict) else {}
            rec["keep"] = bool(idx in roi_kept_set)
            rec["bg_keep"] = bool(idx in bg_kept_set)
            per_frame_out[str(idx)] = rec
        out["per_frame"] = per_frame_out

    out["policy_mode"] = "dual_only"
    out["roi_kept_frames"] = roi_kept
    out["roi_dropped_frames"] = roi_dropped
    out["bg_kept_frames"] = bg_kept
    out["bg_dropped_frames"] = bg_dropped
    out["kept_frames"] = roi_kept
    out["dropped_frames"] = roi_dropped

    stats["dual_timeline"] = {
        "enabled": bool(dual_meta.get("enabled", True)),
        "roi_kept_count": int(len(roi_kept)),
        "bg_kept_count": int(len(bg_kept)),
    }
    stats["policy_mode"] = "dual_only"
    out["stats"] = stats
    return out


def apply_dual_timeline_policy(frame_result: Dict[str, Any], frame_cfg: Dict[str, Any]) -> Dict[str, Any]:
    dual_meta = build_dual_timeline_metadata(frame_result, frame_cfg)
    return apply_dual_timeline_metadata(frame_result, dual_meta)


def build_dual_timeline_metadata(frame_result: Dict[str, Any], frame_cfg: Dict[str, Any]) -> Dict[str, Any]:
    dual_cfg = (frame_cfg.get("dual_timeline", {}) or {})
    enabled = _as_bool(
        dual_cfg.get("enable", True),
        "frame_removal.dual_timeline.enable",
        True,
    )
    roi_motion_interval = _as_int_ge_1(
        dual_cfg.get("roi_motion_interval", 2),
        "frame_removal.dual_timeline.roi_motion_interval",
    )
    roi_still_interval = _as_int_ge_1(
        dual_cfg.get("roi_still_interval", 3),
        "frame_removal.dual_timeline.roi_still_interval",
    )
    bg_interval = _as_int_ge_1(
        dual_cfg.get("bg_interval", 6),
        "frame_removal.dual_timeline.bg_interval",
    )
    bg_idle_interval = _as_int_ge_1(
        dual_cfg.get("bg_idle_interval", 10),
        "frame_removal.dual_timeline.bg_idle_interval",
    )

    kept_frames = [int(x) for x in (frame_result.get("kept_frames", []) or [])]
    per_frame = (frame_result.get("per_frame", {}) or {})
    stats = (frame_result.get("stats", {}) or {})

    num_frames = int(stats.get("num_frames_read", 0) or 0)
    if num_frames <= 0 and per_frame:
        frame_ids: List[int] = []
        for key in per_frame.keys():
            try:
                frame_ids.append(int(key))
            except (TypeError, ValueError):
                continue
        if frame_ids:
            num_frames = max(frame_ids) + 1
    if num_frames <= 0 and kept_frames:
        num_frames = int(max(kept_frames)) + 1

    if not enabled:
        roi_kept_frames = _dedupe_sorted(list(kept_frames))
        bg_kept_frames = _dedupe_sorted(list(kept_frames))
        roi_mode = "legacy_single_timeline"
        bg_mode = "legacy_single_timeline"
    else:
        roi_kept_frames, bg_kept_frames = _build_fixed_streams(
            per_frame=per_frame,
            num_frames=num_frames,
            roi_motion_interval=roi_motion_interval,
            roi_still_interval=roi_still_interval,
            bg_interval=bg_interval,
            bg_idle_interval=bg_idle_interval,
        )
        roi_mode = "fixed_state_interval"
        bg_mode = "fixed_segment_interval"

    return {
        "enabled": enabled,
        "roi_kept_frames": roi_kept_frames,
        "bg_kept_frames": bg_kept_frames,
        "start_frame": 0 if num_frames > 0 else (min(kept_frames) if kept_frames else 0),
        "end_frame": (num_frames - 1) if num_frames > 0 else (max(kept_frames) if kept_frames else -1),
        "policy": {
            "roi_mode": roi_mode,
            "bg_mode": bg_mode,
            "roi_motion_interval": int(roi_motion_interval),
            "roi_still_interval": int(roi_still_interval),
            "bg_interval": int(bg_interval),
            "bg_idle_interval": int(bg_idle_interval),
        },
    }
