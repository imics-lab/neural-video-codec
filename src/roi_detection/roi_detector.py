# src/roi_detection/roi_detector.py
"""
ROI detection module (pipeline-safe).

Rules:
- No writing files, no creating directories.
- Reads video frames (allowed) and returns ROI results as a dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ----------------------------
# Pure helpers
# ----------------------------
def _clip_xyxy(xyxy: Tuple[float, float, float, float], w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(w - 1), float(x1)))
    y1 = max(0.0, min(float(h - 1), float(y1)))
    x2 = max(0.0, min(float(w - 1), float(x2)))
    y2 = max(0.0, min(float(h - 1), float(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
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
    return inter / (area_a + area_b - inter + 1e-9)


def _union_merge_boxes(
    boxes: List[Tuple[float, float, float, float]],
    scores: Optional[List[float]],
    iou_thr: float,
) -> Tuple[List[Tuple[float, float, float, float]], Optional[List[float]]]:
    """Merge boxes by union if IoU >= iou_thr."""
    if not boxes:
        return [], scores if scores is not None else None

    used = [False] * len(boxes)
    out_boxes: List[Tuple[float, float, float, float]] = []
    out_scores: Optional[List[float]] = [] if scores is not None else None

    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, x2, y2 = boxes[i]
        best_sc = scores[i] if scores is not None else 0.0
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue
                if _box_iou((x1, y1, x2, y2), boxes[j]) >= iou_thr:
                    bx1, by1, bx2, by2 = boxes[j]
                    x1 = min(x1, bx1)
                    y1 = min(y1, by1)
                    x2 = max(x2, bx2)
                    y2 = max(y2, by2)
                    if scores is not None:
                        best_sc = max(best_sc, scores[j])
                    used[j] = True
                    changed = True

        out_boxes.append((x1, y1, x2, y2))
        if out_scores is not None:
            out_scores.append(best_sc)

    return out_boxes, out_scores


def _scale_xyxy(xyxy: Tuple[float, float, float, float], sx: float, sy: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def _shift_bbox(b: Tuple[float, float, float, float], dx: float, dy: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = b
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def _expand_xyxy(
    xyxy: Tuple[float, float, float, float],
    pad_frac: float,
    pad_px: float,
    w: int,
    h: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = (bw * max(0.0, float(pad_frac))) + max(0.0, float(pad_px))
    py = (bh * max(0.0, float(pad_frac))) + max(0.0, float(pad_px))
    return _clip_xyxy((x1 - px, y1 - py, x2 + px, y2 + py), w, h)


def _parse_target_class_ids(raw_value: Any) -> Set[int]:
    """
    Normalize detector class-id config to a non-empty set.
    Defaults to {0}, which is the MegaDetector "animal" class.
    """
    if raw_value is None:
        return {0}
    if isinstance(raw_value, (int, float, str)):
        values = [raw_value]
    elif isinstance(raw_value, (list, tuple, set)):
        values = list(raw_value)
    else:
        return {0}

    class_ids: Set[int] = set()
    for v in values:
        try:
            class_ids.add(int(v))
        except (TypeError, ValueError):
            continue
    return class_ids or {0}


def _seed_klt_points(gray: np.ndarray, bbox: Tuple[float, float, float, float], max_pts: int) -> Optional[np.ndarray]:
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in _clip_xyxy(bbox, w, h)]
    if (x2 - x1) < 12 or (y2 - y1) < 12:
        return None

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_pts,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask,
        blockSize=7,
        useHarrisDetector=False,
    )
    return pts  # (N,1,2) float32 or None


def _klt_translation(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    pts_prev: np.ndarray,
    min_valid: int,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]], int]:
    if pts_prev is None or len(pts_prev) == 0:
        return None, None, 0

    pts_cur, st, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        cur_gray,
        pts_prev,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if pts_cur is None or st is None:
        return None, None, 0

    st = st.reshape(-1)
    good_prev = pts_prev.reshape(-1, 2)[st == 1]
    good_cur = pts_cur.reshape(-1, 2)[st == 1]
    n_valid = int(good_cur.shape[0])
    if n_valid < min_valid:
        return None, None, n_valid

    dxy = good_cur - good_prev
    dx = float(np.median(dxy[:, 0]))
    dy = float(np.median(dxy[:, 1]))
    pts_cur_valid = good_cur.reshape(-1, 1, 2).astype(np.float32)
    return pts_cur_valid, (dx, dy), n_valid


@dataclass
class _Track:
    track_id: int
    bbox_p: Tuple[float, float, float, float]  # processing-space xyxy
    conf: float
    last_det_frame: int
    hits: int = 1
    confirmed: bool = False
    vx: float = 0.0
    vy: float = 0.0
    vw: float = 0.0
    vh: float = 0.0
    pts: Optional[np.ndarray] = None


def _match_dets_to_tracks(
    det_boxes: List[Tuple[float, float, float, float]],
    tracks: List[_Track],
    match_iou: float,
):
    if not det_boxes or not tracks:
        return [], list(range(len(det_boxes))), list(range(len(tracks)))

    iou_mat = np.zeros((len(det_boxes), len(tracks)), dtype=np.float32)
    for i, db in enumerate(det_boxes):
        for j, t in enumerate(tracks):
            iou_mat[i, j] = _box_iou(db, t.bbox_p)

    matches = []
    used_d, used_t = set(), set()

    while True:
        i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        best = float(iou_mat[i, j])
        if best < match_iou:
            break
        if i in used_d or j in used_t:
            iou_mat[i, j] = -1.0
            continue
        matches.append((i, j))
        used_d.add(i)
        used_t.add(j)
        iou_mat[i, :] = -1.0
        iou_mat[:, j] = -1.0

    unmatched_d = [i for i in range(len(det_boxes)) if i not in used_d]
    unmatched_t = [j for j in range(len(tracks)) if j not in used_t]
    return matches, unmatched_d, unmatched_t


def _tracks_within_gap(tracks: List[_Track], frame_idx: int, max_det_gap: int) -> List[_Track]:
    return [t for t in tracks if (frame_idx - t.last_det_frame) <= max_det_gap]


def _detect_proc_boxes(
    model: YOLO,
    frame_p_bgr: np.ndarray,
    imgsz: int,
    conf_thr: float,
    iou_thr: float,
    device: Any,
    half: bool,
    target_class_ids: Set[int],
    bbox_pad_frac: float,
    bbox_pad_px: float,
) -> Tuple[List[Tuple[float, float, float, float]], List[float]]:
    res = model.predict(
        source=[frame_p_bgr],
        imgsz=imgsz,
        conf=conf_thr,
        iou=iou_thr,
        device=device,
        half=half,
        verbose=False,
    )[0]

    if res.boxes is None or len(res.boxes) == 0:
        return [], []

    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy() if res.boxes.cls is not None else np.zeros((len(confs),), dtype=np.float32)

    h, w = frame_p_bgr.shape[:2]
    boxes: List[Tuple[float, float, float, float]] = []
    scores: List[float] = []
    for (x1, y1, x2, y2), sc, c in zip(xyxy, confs, clss):
        if int(c) not in target_class_ids:
            continue
        b = _clip_xyxy((float(x1), float(y1), float(x2), float(y2)), w, h)
        b = _expand_xyxy(b, bbox_pad_frac, bbox_pad_px, w, h)
        boxes.append(b)
        scores.append(float(sc))

    return boxes, scores


def run_roi_detection(
    video_path: str,
    config: Dict[str, Any],
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """
    Pipeline ROI step.

    Returns:
      {
        "video_path": str,
        "frame_count": int,
        "fps": float,
        "width": int,
        "height": int,
        "start_frame": int,
        "end_frame": int,
        "frames": { frame_idx(int): [roi_dict, ...], ... }
      }

    roi_dict:
      {"x1","y1","x2","y2","conf","cls","label","track_id"}
    """
    paths = config.get("paths", {}) or {}
    rt = config.get("runtime", {}) or {}
    tr = config.get("tracking", {}) or {}

    model_path = paths.get("animal_model_path")
    if not model_path:
        raise ValueError("config['paths']['animal_model_path'] is required")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_f = int(rt.get("start_frame", 0))
    end_f = int(rt.get("end_frame", -1))
    if end_f < 0 or end_f >= total:
        end_f = total - 1
    if start_f < 0:
        start_f = 0

    if start_f > end_f:
        cap.release()
        return {
            "video_path": video_path,
            "frame_count": 0,
            "fps": fps,
            "width": w_full,
            "height": h_full,
            "start_frame": start_f,
            "end_frame": end_f,
            "frames": {},
        }

    scale = float(rt.get("processing_scale", 0.5))
    scale = max(0.1, min(1.0, scale))
    w_p = int(round(w_full * scale))
    h_p = int(round(h_full * scale))
    sx_full = w_full / float(w_p)  # proc -> full
    sy_full = h_full / float(h_p)

    model = YOLO(model_path, task="detect", verbose=False)
    device = rt.get("device", 0)
    half = bool(rt.get("half", False))
    imgsz = int(rt.get("imgsz", 640))
    conf_thr = float(rt.get("conf", 0.25))
    iou_thr = float(rt.get("iou_nms", 0.50))
    target_class_ids = _parse_target_class_ids(rt.get("target_class_ids", [0]))
    bbox_pad_frac = float(rt.get("bbox_pad_frac", 0.0))
    bbox_pad_px = float(rt.get("bbox_pad_px", 0.0))

    keyframe_interval = int(rt.get("keyframe_interval", 15))
    model_merge_iou = float(rt.get("model_merge_iou", 0.30))
    if keyframe_interval < 1:
        raise ValueError("config['runtime']['keyframe_interval'] must be >= 1")

    match_iou = float(tr.get("match_iou", 0.30))
    min_hits = int(tr.get("min_hits", 2))
    max_det_gap = int(tr.get("max_det_gap", 90))
    vel_smooth = float(tr.get("vel_smooth", 0.7))
    klt_max_points = int(tr.get("klt_max_points", 80))
    klt_min_points = int(tr.get("klt_min_points", 10))
    if min_hits < 1:
        raise ValueError("config['tracking']['min_hits'] must be >= 1")
    if max_det_gap < 0:
        raise ValueError("config['tracking']['max_det_gap'] must be >= 0")
    if klt_max_points < 1:
        raise ValueError("config['tracking']['klt_max_points'] must be >= 1")
    if klt_min_points < 1:
        raise ValueError("config['tracking']['klt_min_points'] must be >= 1")

    enable_propagation_raw = tr.get("enable_propagation", True)
    if not isinstance(enable_propagation_raw, bool):
        raise TypeError("config['tracking']['enable_propagation'] must be a bool")
    enable_propagation = enable_propagation_raw

    frames_out: Dict[int, List[Dict[str, Any]]] = {}
    tracks: List[_Track] = []
    next_id = 1
    prev_gray_p: Optional[np.ndarray] = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    frame_idx = start_f
    processed = 0
    try:
        while frame_idx <= end_f:
            ok, frame_full = cap.read()
            if not ok:
                break

            frame_p = cv2.resize(frame_full, (w_p, h_p), interpolation=cv2.INTER_AREA)
            gray_p = cv2.cvtColor(frame_p, cv2.COLOR_BGR2GRAY)

            is_key = (frame_idx % keyframe_interval == 0)

            if is_key:
                tracks = _tracks_within_gap(tracks, frame_idx, max_det_gap)
                det_boxes_p, det_scores = _detect_proc_boxes(
                    model=model,
                    frame_p_bgr=frame_p,
                    imgsz=imgsz,
                    conf_thr=conf_thr,
                    iou_thr=iou_thr,
                    device=device,
                    half=half,
                    target_class_ids=target_class_ids,
                    bbox_pad_frac=bbox_pad_frac,
                    bbox_pad_px=bbox_pad_px,
                )
                if det_boxes_p:
                    det_boxes_p, det_scores = _union_merge_boxes(det_boxes_p, det_scores, iou_thr=model_merge_iou)

                matches, unmatched_d, _unmatched_t = _match_dets_to_tracks(det_boxes_p, tracks, match_iou)

                # update matched
                for di, ti in matches:
                    t = tracks[ti]
                    new_box = det_boxes_p[di]
                    new_conf = float(det_scores[di]) if det_scores else t.conf

                    old = t.bbox_p
                    old_cx = (old[0] + old[2]) * 0.5
                    old_cy = (old[1] + old[3]) * 0.5
                    old_w = max(1.0, old[2] - old[0])
                    old_h = max(1.0, old[3] - old[1])

                    ncx = (new_box[0] + new_box[2]) * 0.5
                    ncy = (new_box[1] + new_box[3]) * 0.5
                    nw = max(1.0, new_box[2] - new_box[0])
                    nh = max(1.0, new_box[3] - new_box[1])

                    dt = max(1, frame_idx - t.last_det_frame)
                    vx = (ncx - old_cx) / dt
                    vy = (ncy - old_cy) / dt
                    vw = (nw - old_w) / dt
                    vh = (nh - old_h) / dt

                    t.vx = vel_smooth * t.vx + (1.0 - vel_smooth) * vx
                    t.vy = vel_smooth * t.vy + (1.0 - vel_smooth) * vy
                    t.vw = vel_smooth * t.vw + (1.0 - vel_smooth) * vw
                    t.vh = vel_smooth * t.vh + (1.0 - vel_smooth) * vh

                    t.bbox_p = _clip_xyxy(new_box, w_p, h_p)
                    t.conf = new_conf
                    t.last_det_frame = frame_idx
                    t.hits += 1
                    if t.hits >= min_hits:
                        t.confirmed = True

                    t.pts = _seed_klt_points(gray_p, t.bbox_p, max_pts=klt_max_points)

                # Detector keyframes are authoritative; retire tracks that no longer
                # match a detector box so stale ROI boxes do not persist after exit.
                matched_track_indices = {ti for _, ti in matches}
                tracks = [tracks[ti] for ti in range(len(tracks)) if ti in matched_track_indices]

                # new tracks
                for di in unmatched_d:
                    b = det_boxes_p[di]
                    sc = float(det_scores[di]) if det_scores else 1.0
                    t = _Track(
                        track_id=next_id,
                        bbox_p=_clip_xyxy(b, w_p, h_p),
                        conf=sc,
                        last_det_frame=frame_idx,
                        hits=1,
                        confirmed=(min_hits <= 1),
                    )
                    t.pts = _seed_klt_points(gray_p, t.bbox_p, max_pts=klt_max_points)
                    tracks.append(t)
                    next_id += 1

            # keep tracks alive between keyframes and optionally propagate them.
            if not is_key:
                alive: List[_Track] = []
                for t in tracks:
                    if (frame_idx - t.last_det_frame) > max_det_gap:
                        continue

                    updated = False
                    if enable_propagation and prev_gray_p is not None and t.pts is not None and len(t.pts) >= klt_min_points:
                        pts_cur, dxy, n_valid = _klt_translation(prev_gray_p, gray_p, t.pts, min_valid=klt_min_points)
                        if dxy is not None and pts_cur is not None and n_valid >= klt_min_points:
                            dx, dy = dxy
                            t.bbox_p = _clip_xyxy(_shift_bbox(t.bbox_p, dx, dy), w_p, h_p)
                            t.vx = vel_smooth * t.vx + (1.0 - vel_smooth) * dx
                            t.vy = vel_smooth * t.vy + (1.0 - vel_smooth) * dy
                            t.pts = pts_cur
                            updated = True

                    if enable_propagation and not updated and t.confirmed:
                        x1, y1, x2, y2 = t.bbox_p
                        w_box = max(8.0, (x2 - x1) + t.vw)
                        h_box = max(8.0, (y2 - y1) + t.vh)
                        cx = (x1 + x2) * 0.5 + t.vx
                        cy = (y1 + y2) * 0.5 + t.vy
                        t.bbox_p = _clip_xyxy(
                            (cx - w_box / 2.0, cy - h_box / 2.0, cx + w_box / 2.0, cy + h_box / 2.0),
                            w_p,
                            h_p,
                        )
                        t.pts = _seed_klt_points(gray_p, t.bbox_p, max_pts=klt_max_points)

                    alive.append(t)
                tracks = alive

            # Emit ROI tracks immediately from the first detector hit so sparse
            # sampling does not miss the leading frames of an animal track.
            rois: List[Dict[str, Any]] = []
            emit_tracks_this_frame = bool(is_key or enable_propagation)
            if emit_tracks_this_frame:
                for t in tracks:
                    bbox_full = _scale_xyxy(t.bbox_p, sx_full, sy_full)
                    bbox_full = _clip_xyxy(bbox_full, w_full, h_full)
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox_full]
                    rois.append(
                        {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "conf": float(t.conf) if t.conf > 0 else 1.0,
                            "cls": 0,
                            "label": "animal",
                            "track_id": int(t.track_id),
                        }
                    )
            if rois:
                frames_out[int(frame_idx)] = rois

            prev_gray_p = gray_p
            frame_idx += 1
            processed += 1
            if progress_cb is not None:
                progress_cb(1)
    finally:
        cap.release()

    return {
        "video_path": video_path,
        "frame_count": int(processed),
        "fps": float(fps),
        "width": int(w_full),
        "height": int(h_full),
        "start_frame": int(start_f),
        "end_frame": int(end_f),
        "frames": frames_out,
    }
