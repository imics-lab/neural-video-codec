# src/frame_removal/remove_frames.py
"""
Simple ROI motion-state analysis for frame removal.

This module now does one job:
- detect whether ROI is present
- classify ROI as STILL or MOTION
- emit per-frame metadata used by the fixed dual-timeline policy

The actual production keep/drop policy is applied later in
`frame_removal.dual_timeline`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class BoxXYXY:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0
    normalized: bool = False


def _as_section(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _as_bool(value: Any, field_name: str, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    raise ValueError(f"{field_name} must be a boolean")


def _as_int_ge(value: Any, field_name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    try:
        iv = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer >= {minimum}") from exc
    if iv < int(minimum):
        raise ValueError(f"{field_name} must be >= {minimum}")
    return iv


def _as_float_range(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    try:
        fv = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if minimum is not None and fv < float(minimum):
        raise ValueError(f"{field_name} must be >= {minimum}")
    if maximum is not None and fv > float(maximum):
        raise ValueError(f"{field_name} must be <= {maximum}")
    return fv


def _parse_frame_removal_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    params = _as_section(cfg.get("params", {}), "frame_removal.params")

    p_roi = _as_section(params.get("roi", {}), "frame_removal.params.roi")
    halo = _as_float_range(
        p_roi.get("halo_frac", 0.2),
        "frame_removal.params.roi.halo_frac",
        minimum=0.0,
    )
    fixed_size = _as_int_ge(
        p_roi.get("fixed_size", 160),
        "frame_removal.params.roi.fixed_size",
        1,
    )
    blur_ksize = _as_int_ge(
        p_roi.get("blur_ksize", 5),
        "frame_removal.params.roi.blur_ksize",
        0,
    )
    to_gray = _as_bool(
        p_roi.get("gray", True),
        "frame_removal.params.roi.gray",
        True,
    )

    p_smooth = _as_section(
        params.get("bbox_smoothing", {}),
        "frame_removal.params.bbox_smoothing",
    )
    smooth_enable = _as_bool(
        p_smooth.get("enable", True),
        "frame_removal.params.bbox_smoothing.enable",
        True,
    )
    ema_alpha = _as_float_range(
        p_smooth.get("ema_alpha", 0.6),
        "frame_removal.params.bbox_smoothing.ema_alpha",
        minimum=0.0,
        maximum=1.0,
    )

    p_motion = _as_section(params.get("motion", {}), "frame_removal.params.motion")
    pixel_t_low = _as_float_range(
        p_motion.get("t_low", 0.9),
        "frame_removal.params.motion.t_low",
        minimum=0.0,
    )
    pixel_t_high = _as_float_range(
        p_motion.get("t_high", 1.9),
        "frame_removal.params.motion.t_high",
        minimum=0.0,
    )
    if pixel_t_high <= pixel_t_low:
        raise ValueError("motion.t_high must be > motion.t_low for hysteresis")

    state_source = str(p_motion.get("state_source", "hybrid")).strip().lower()
    if state_source not in {"pixel", "bbox", "hybrid"}:
        raise ValueError("motion.state_source must be one of: pixel, bbox, hybrid")

    bbox_t_low_px = _as_float_range(
        p_motion.get("bbox_t_low_px", 0.5),
        "frame_removal.params.motion.bbox_t_low_px",
        minimum=0.0,
    )
    bbox_t_high_px = _as_float_range(
        p_motion.get("bbox_t_high_px", 1.0),
        "frame_removal.params.motion.bbox_t_high_px",
        minimum=0.0,
    )
    if bbox_t_high_px <= bbox_t_low_px:
        raise ValueError("motion.bbox_t_high_px must be > motion.bbox_t_low_px")

    enter_motion_frames = _as_int_ge(
        p_motion.get("enter_motion_frames", 2),
        "frame_removal.params.motion.enter_motion_frames",
        1,
    )
    enter_still_frames = _as_int_ge(
        p_motion.get("enter_still_frames", 4),
        "frame_removal.params.motion.enter_still_frames",
        1,
    )

    dual_cfg = _as_section(cfg.get("dual_timeline", {}), "frame_removal.dual_timeline")
    dual_enabled = _as_bool(
        dual_cfg.get("enable", True),
        "frame_removal.dual_timeline.enable",
        True,
    )
    roi_motion_interval = _as_int_ge(
        dual_cfg.get("roi_motion_interval", 2),
        "frame_removal.dual_timeline.roi_motion_interval",
        1,
    )
    roi_still_interval = _as_int_ge(
        dual_cfg.get("roi_still_interval", 3),
        "frame_removal.dual_timeline.roi_still_interval",
        1,
    )
    bg_interval = _as_int_ge(
        dual_cfg.get("bg_interval", 6),
        "frame_removal.dual_timeline.bg_interval",
        1,
    )
    bg_idle_interval = _as_int_ge(
        dual_cfg.get("bg_idle_interval", 10),
        "frame_removal.dual_timeline.bg_idle_interval",
        1,
    )

    return {
        "halo": halo,
        "fixed_size": fixed_size,
        "blur_ksize": blur_ksize,
        "to_gray": to_gray,
        "smooth_enable": smooth_enable,
        "ema_alpha": ema_alpha,
        "pixel_t_low": pixel_t_low,
        "pixel_t_high": pixel_t_high,
        "state_source": state_source,
        "bbox_t_low_px": bbox_t_low_px,
        "bbox_t_high_px": bbox_t_high_px,
        "enter_motion_frames": enter_motion_frames,
        "enter_still_frames": enter_still_frames,
        "dual_enabled": dual_enabled,
        "roi_motion_interval": roi_motion_interval,
        "roi_still_interval": roi_still_interval,
        "bg_interval": bg_interval,
        "bg_idle_interval": bg_idle_interval,
    }


def validate_frame_removal_config(cfg: Dict[str, Any]) -> None:
    _parse_frame_removal_config(cfg)


def parse_box_any(obj: Any) -> Optional[BoxXYXY]:
    if obj is None:
        return None

    if isinstance(obj, BoxXYXY):
        return obj

    if isinstance(obj, (list, tuple)) and len(obj) >= 4:
        try:
            x1, y1, x2, y2 = map(float, obj[:4])
            return BoxXYXY(x1, y1, x2, y2, conf=1.0, normalized=False)
        except Exception:
            return None

    if isinstance(obj, dict):
        try:
            if "xyxy" in obj and isinstance(obj["xyxy"], (list, tuple)) and len(obj["xyxy"]) >= 4:
                vals = obj["xyxy"][:4]
                x1, y1, x2, y2 = map(float, vals)
            elif "bbox" in obj and isinstance(obj["bbox"], (list, tuple)) and len(obj["bbox"]) >= 4:
                vals = obj["bbox"][:4]
                x1, y1, x2, y2 = map(float, vals)
            elif all(k in obj for k in ("x1", "y1", "x2", "y2")):
                x1 = float(obj["x1"])
                y1 = float(obj["y1"])
                x2 = float(obj["x2"])
                y2 = float(obj["y2"])
            else:
                return None

            conf = float(obj.get("conf", obj.get("confidence", 1.0)))
        except (TypeError, ValueError):
            return None

        normalized = bool(obj.get("normalized", False))
        if "normalized" not in obj:
            mx = max(abs(x1), abs(y1), abs(x2), abs(y2))
            if mx <= 1.5:
                normalized = True
        return BoxXYXY(x1, y1, x2, y2, conf=conf, normalized=normalized)

    return None


def xyxy_to_pixels(b: BoxXYXY, width: int, height: int) -> Tuple[float, float, float, float]:
    if b.normalized:
        return (b.x1 * width, b.y1 * height, b.x2 * width, b.y2 * height)
    return (b.x1, b.y1, b.x2, b.y2)


def ema_smooth(
    prev: Optional[Tuple[float, float, float, float]],
    curr: Tuple[float, float, float, float],
    alpha: float,
) -> Tuple[float, float, float, float]:
    if prev is None:
        return curr
    a = max(0.0, min(1.0, float(alpha)))
    return tuple((a * p + (1.0 - a) * c) for p, c in zip(prev, curr))  # type: ignore[return-value]


def union_boxes_xyxy(boxes: List[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not boxes:
        return None
    x1 = min(float(b[0]) for b in boxes)
    y1 = min(float(b[1]) for b in boxes)
    x2 = max(float(b[2]) for b in boxes)
    y2 = max(float(b[3]) for b in boxes)
    return (x1, y1, x2, y2)


def expand_box_xyxy(x1: float, y1: float, x2: float, y2: float, halo_frac: float) -> Tuple[float, float, float, float]:
    w = max(1.0, abs(x2 - x1))
    h = max(1.0, abs(y2 - y1))
    s = max(w, h)
    pad = float(halo_frac) * s
    return (x1 - pad, y1 - pad, x2 + pad, y2 + pad)


@dataclass(frozen=True)
class ClampedBox:
    x1: int
    y1: int
    x2: int
    y2: int


def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> ClampedBox:
    ix1 = max(0, min(width - 1, int(round(min(x1, x2)))))
    iy1 = max(0, min(height - 1, int(round(min(y1, y2)))))
    ix2 = max(0, min(width - 1, int(round(max(x1, x2)))))
    iy2 = max(0, min(height - 1, int(round(max(y1, y2)))))
    return ClampedBox(ix1, iy1, ix2, iy2)


def _prep(img_bgr: np.ndarray, to_gray: bool, blur_ksize: int) -> np.ndarray:
    out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if to_gray else img_bgr
    k = int(blur_ksize)
    if k >= 3 and k % 2 == 1:
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def roi_motion_score(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    roi: ClampedBox,
    fixed_size: int,
    blur_ksize: int,
    to_gray: bool,
) -> float:
    x1, y1, x2, y2 = roi.x1, roi.y1, roi.x2, roi.y2
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return 0.0

    a = prev_bgr[y1:y2, x1:x2]
    b = curr_bgr[y1:y2, x1:x2]
    if a.size == 0 or b.size == 0:
        return 0.0

    size = max(8, int(fixed_size))
    a = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (size, size), interpolation=cv2.INTER_AREA)

    a = _prep(a, to_gray=to_gray, blur_ksize=blur_ksize)
    b = _prep(b, to_gray=to_gray, blur_ksize=blur_ksize)
    return float(np.mean(cv2.absdiff(a, b)))


def bbox_motion_px(
    prev_xyxy: Optional[Tuple[float, float, float, float]],
    curr_xyxy: Tuple[float, float, float, float],
) -> float:
    if prev_xyxy is None:
        return 0.0
    px1, py1, px2, py2 = prev_xyxy
    cx1 = 0.5 * (px1 + px2)
    cy1 = 0.5 * (py1 + py2)
    cx2 = 0.5 * (curr_xyxy[0] + curr_xyxy[2])
    cy2 = 0.5 * (curr_xyxy[1] + curr_xyxy[3])
    return float(np.hypot(cx2 - cx1, cy2 - cy1))


def _box_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-9))


def _max_pairwise_iou(boxes: List[Tuple[float, float, float, float]]) -> float:
    n = len(boxes)
    if n < 2:
        return 0.0
    best = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            best = max(best, _box_iou_xyxy(boxes[i], boxes[j]))
    return float(best)


def _track_id_from_raw(raw: Any) -> Optional[int]:
    if not isinstance(raw, dict) or "track_id" not in raw:
        return None
    try:
        return int(raw["track_id"])
    except (TypeError, ValueError):
        return None


def _motion_evidence(
    *,
    state_source: str,
    pixel_score: float,
    bbox_move_px: float,
    pixel_t_low: float,
    pixel_t_high: float,
    bbox_t_low_px: float,
    bbox_t_high_px: float,
) -> Tuple[bool, bool]:
    if state_source == "pixel":
        moving_now = pixel_score > pixel_t_high
        still_now = pixel_score < pixel_t_low
    elif state_source == "bbox":
        moving_now = bbox_move_px > bbox_t_high_px
        still_now = bbox_move_px < bbox_t_low_px
    else:
        moving_now = (pixel_score > pixel_t_high) or (bbox_move_px > bbox_t_high_px)
        still_now = (pixel_score < pixel_t_low) and (bbox_move_px < bbox_t_low_px)
    return bool(moving_now), bool(still_now)


def remove_redundant_frames(video_path: str, bbox_map_raw: Dict[int, List[Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    runtime_cfg = _parse_frame_removal_config(cfg)
    halo = float(runtime_cfg["halo"])
    fixed_size = int(runtime_cfg["fixed_size"])
    blur_ksize = int(runtime_cfg["blur_ksize"])
    to_gray = bool(runtime_cfg["to_gray"])
    smooth_enable = bool(runtime_cfg["smooth_enable"])
    ema_alpha = float(runtime_cfg["ema_alpha"])
    pixel_t_low = float(runtime_cfg["pixel_t_low"])
    pixel_t_high = float(runtime_cfg["pixel_t_high"])
    state_source = str(runtime_cfg["state_source"])
    bbox_t_low_px = float(runtime_cfg["bbox_t_low_px"])
    bbox_t_high_px = float(runtime_cfg["bbox_t_high_px"])
    enter_motion_frames = int(runtime_cfg["enter_motion_frames"])
    enter_still_frames = int(runtime_cfg["enter_still_frames"])
    roi_motion_interval = int(runtime_cfg["roi_motion_interval"])
    roi_still_interval = int(runtime_cfg["roi_still_interval"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else -1

    state = "STILL"
    prev_frame: Optional[np.ndarray] = None
    prev_smoothed: Optional[Tuple[float, float, float, float]] = None
    prev_track_ids: set[int] = set()
    prev_has_detection = False
    moving_streak = 0
    still_streak = 0
    last_roi_keep = -1

    kept: List[int] = []
    dropped: List[int] = []
    per_frame: Dict[str, Any] = {}

    t = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            raw_boxes = bbox_map_raw.get(t, []) or []
            parsed_boxes: List[BoxXYXY] = []
            parsed_track_ids: List[int] = []
            parsed_confs: List[float] = []
            for raw in raw_boxes:
                parsed = parse_box_any(raw)
                if parsed is None:
                    continue
                parsed_boxes.append(parsed)
                parsed_confs.append(float(parsed.conf))
                track_id = _track_id_from_raw(raw)
                if track_id is not None:
                    parsed_track_ids.append(int(track_id))

            has_detection = bool(parsed_boxes)
            bbox_move = 0.0
            pixel_score: Optional[float] = None
            motion_evidence = False
            still_evidence = True
            track_births = 0
            track_deaths = 0
            roi_mean_conf = float(np.mean(parsed_confs)) if parsed_confs else 0.0
            roi_max_iou = 0.0
            roi_box_out = None
            keep = False

            if has_detection:
                roi_birth = not prev_has_detection
                current_track_ids = set(parsed_track_ids)
                if current_track_ids:
                    track_births = int(len(current_track_ids - prev_track_ids))
                    track_deaths = int(len(prev_track_ids - current_track_ids))
                    prev_track_ids = set(current_track_ids)
                else:
                    prev_track_ids = set()

                curr_boxes_pixels = [xyxy_to_pixels(box, width, height) for box in parsed_boxes]
                curr_union = union_boxes_xyxy(curr_boxes_pixels)
                if curr_union is None:
                    curr_union = (0.0, 0.0, float(width), float(height))

                prev_for_motion = prev_smoothed
                smoothed_union = ema_smooth(prev_smoothed, curr_union, ema_alpha) if smooth_enable else curr_union
                prev_smoothed = smoothed_union
                bbox_move = bbox_motion_px(prev_for_motion, smoothed_union)

                ex1, ey1, ex2, ey2 = expand_box_xyxy(*curr_union, halo_frac=halo)
                roi_box = clamp_box(ex1, ey1, ex2, ey2, width, height)
                roi_box_out = {
                    "x1": int(roi_box.x1),
                    "y1": int(roi_box.y1),
                    "x2": int(roi_box.x2),
                    "y2": int(roi_box.y2),
                }

                if prev_frame is not None:
                    pixel_score = roi_motion_score(
                        prev_frame,
                        frame,
                        roi_box,
                        fixed_size=fixed_size,
                        blur_ksize=blur_ksize,
                        to_gray=to_gray,
                    )
                else:
                    pixel_score = 0.0

                roi_max_iou = _max_pairwise_iou(curr_boxes_pixels)
                motion_evidence, still_evidence = _motion_evidence(
                    state_source=state_source,
                    pixel_score=float(pixel_score),
                    bbox_move_px=float(bbox_move),
                    pixel_t_low=pixel_t_low,
                    pixel_t_high=pixel_t_high,
                    bbox_t_low_px=bbox_t_low_px,
                    bbox_t_high_px=bbox_t_high_px,
                )

                if motion_evidence:
                    moving_streak += 1
                else:
                    moving_streak = 0
                if still_evidence:
                    still_streak += 1
                else:
                    still_streak = 0

                if state == "STILL" and moving_streak >= enter_motion_frames:
                    state = "MOTION"
                    still_streak = 0
                elif state == "MOTION" and still_streak >= enter_still_frames:
                    state = "STILL"
                    moving_streak = 0

                roi_interval = roi_motion_interval if state == "MOTION" else roi_still_interval
                roi_gap = (t - last_roi_keep) if last_roi_keep >= 0 else (t + 1)
                keep = bool((t == 0) or roi_birth or (roi_gap >= roi_interval))
                if keep:
                    kept.append(int(t))
                    last_roi_keep = int(t)
                else:
                    dropped.append(int(t))
            else:
                prev_track_ids = set()
                moving_streak = 0
                still_streak = 0
                state = "STILL"
                keep = False
                dropped.append(int(t))

            per_frame[str(t)] = {
                "keep": bool(keep),
                "state": state,
                "roi_motion_score": None if pixel_score is None else float(pixel_score),
                "bbox_motion_px": float(bbox_move),
                "motion_urgency": float(1.0 if state == "MOTION" and has_detection else 0.0),
                "state_source": state_source,
                "roi_count": int(len(parsed_boxes)),
                "roi_mean_conf": float(roi_mean_conf),
                "roi_max_iou": float(roi_max_iou),
                "track_births": int(track_births),
                "track_deaths": int(track_deaths),
                "motion_evidence": bool(motion_evidence),
                "still_evidence": bool(still_evidence),
                "motion_streak": int(moving_streak),
                "still_streak": int(still_streak),
                "roi_box": roi_box_out,
                "bbox_missing": bool(not has_detection),
            }

            prev_frame = frame
            prev_has_detection = bool(has_detection)
            t += 1
    finally:
        cap.release()

    stats = {
        "fps_in": fps_in,
        "width": width,
        "height": height,
        "num_frames_read": t,
        "kept": len(kept),
        "dropped": len(dropped),
        "drop_ratio": float(len(dropped)) / max(1, len(kept) + len(dropped)),
        "frame_count_reported": frame_count,
        "state_thresholds": {
            "state_source": state_source,
            "t_low": pixel_t_low,
            "t_high": pixel_t_high,
            "bbox_t_low_px": bbox_t_low_px,
            "bbox_t_high_px": bbox_t_high_px,
            "enter_motion_frames": int(enter_motion_frames),
            "enter_still_frames": int(enter_still_frames),
        },
        "roi": {
            "halo_frac": halo,
            "fixed_size": fixed_size,
            "blur_ksize": blur_ksize,
            "gray": to_gray,
        },
        "bbox_smoothing": {
            "enable": smooth_enable,
            "ema_alpha": ema_alpha,
        },
        "policy": {
            "roi_mode": "simple_state_interval",
            "roi_motion_interval": int(roi_motion_interval),
            "roi_still_interval": int(roi_still_interval),
        },
    }

    return {
        "kept_frames": kept,
        "dropped_frames": dropped,
        "per_frame": per_frame,
        "stats": stats,
    }
