"""
Animal detector with KLT-based inter-frame tracking (pure module).

Supports two backends:
  - yolov9c  : ultralytics YOLO (COCO classes, default)
  - megadetector : MegaDetector v5a (Microsoft CameraTraps, class 0=animal)

Rules:
- No filesystem writes.
- Returns detection results as a plain dict (JSON-serialisable).

Public API:
    run_detection(video_path, config) -> DetectionResult
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


# ── COCO animal class IDs (default) ──────────────────────────────────────────
# bird=14, cat=15, dog=16, horse=17, sheep=18, cow=19,
# elephant=20, bear=21, zebra=22, giraffe=23
_DEFAULT_ANIMAL_CLASSES: Set[int] = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}

_MD_V5A_URL = (
    "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt"
)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _clip_xyxy(box: Tuple[float, ...], w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(w - 1), float(x1)))
    y1 = max(0.0, min(float(h - 1), float(y1)))
    x2 = max(0.0, min(float(w - 1), float(x2)))
    y2 = max(0.0, min(float(h - 1), float(y2)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2


def _iou(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def _expand_box(
    box: Tuple[float, float, float, float],
    pad_frac: float,
    pad_px: float,
    w: int,
    h: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    px = bw * max(0.0, pad_frac) + max(0.0, pad_px)
    py = bh * max(0.0, pad_frac) + max(0.0, pad_px)
    return _clip_xyxy((x1 - px, y1 - py, x2 + px, y2 + py), w, h)


# ── Simple IoU tracker ────────────────────────────────────────────────────────

@dataclass
class _Track:
    track_id: int
    bbox: Tuple[float, float, float, float]  # processing-space xyxy
    conf: float
    cls: int
    label: str
    last_det_frame: int
    hits: int = 1
    confirmed: bool = False
    vx: float = 0.0
    vy: float = 0.0
    last_confirmed_bbox: Tuple[float, float, float, float] = None  # bbox at last confirmed detection


def _match_greedy(
    dets: List[Tuple[float, float, float, float]],
    tracks: List[_Track],
    iou_thr: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy max-IoU matching. Returns (matches, unmatched_dets, unmatched_tracks)."""
    if not dets or not tracks:
        return [], list(range(len(dets))), list(range(len(tracks)))
    mat = np.array([[_iou(d, t.bbox) for t in tracks] for d in dets], dtype=np.float32)
    matches, used_d, used_t = [], set(), set()
    while True:
        i, j = divmod(int(np.argmax(mat)), mat.shape[1])
        if mat[i, j] < iou_thr:
            break
        if i in used_d or j in used_t:
            mat[i, j] = -1.0
            continue
        matches.append((i, j))
        used_d.add(i)
        used_t.add(j)
        mat[i, :] = -1.0
        mat[:, j] = -1.0
    unmatched_d = [i for i in range(len(dets)) if i not in used_d]
    unmatched_t = [j for j in range(len(tracks)) if j not in used_t]
    return matches, unmatched_d, unmatched_t


# ── KLT inter-frame propagation ───────────────────────────────────────────────

def _seed_klt(gray: np.ndarray, box: Tuple[float, ...], max_pts: int) -> Optional[np.ndarray]:
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in _clip_xyxy(box, w, h)]
    if (x2 - x1) < 12 or (y2 - y1) < 12:
        return None
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_pts, qualityLevel=0.01,
                                  minDistance=5, mask=mask, blockSize=7)
    return pts  # (N,1,2) float32 or None


def _klt_shift(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    pts: np.ndarray,
    min_valid: int,
) -> Optional[Tuple[float, float]]:
    cur_pts, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, cur_gray, pts, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if cur_pts is None or st is None:
        return None
    st = st.reshape(-1)
    good_prev = pts.reshape(-1, 2)[st == 1]
    good_cur  = cur_pts.reshape(-1, 2)[st == 1]
    if good_cur.shape[0] < min_valid:
        return None
    dxy = good_cur - good_prev
    return float(np.median(dxy[:, 0])), float(np.median(dxy[:, 1]))


# ── Detection call ────────────────────────────────────────────────────────────

def _detect_boxes(
    model,
    frame_bgr: np.ndarray,
    imgsz: int,
    conf_thr: float,
    iou_thr: float,
    device: Any,
    half: bool,
    animal_classes: Set[int],
    pad_frac: float,
    pad_px: float,
) -> Tuple[List[Tuple[float, float, float, float]], List[float], List[int]]:
    """Run YOLO11, filter animal classes, expand boxes. Returns (boxes, confs, class_ids)."""
    results = model.predict(
        source=[frame_bgr], imgsz=imgsz, conf=conf_thr, iou=iou_thr,
        device=device, half=half, verbose=False,
    )[0]

    if results.boxes is None or len(results.boxes) == 0:
        return [], [], []

    xyxy  = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    clss  = results.boxes.cls.cpu().numpy() if results.boxes.cls is not None \
            else np.zeros(len(confs), dtype=np.float32)

    h, w = frame_bgr.shape[:2]
    out_boxes, out_confs, out_cls = [], [], []
    for (x1, y1, x2, y2), sc, c in zip(xyxy, confs, clss):
        cid = int(c)
        if cid not in animal_classes:
            continue
        box = _clip_xyxy((float(x1), float(y1), float(x2), float(y2)), w, h)
        box = _expand_box(box, pad_frac, pad_px, w, h)
        out_boxes.append(box)
        out_confs.append(float(sc))
        out_cls.append(cid)

    return out_boxes, out_confs, out_cls


# ── MegaDetector helpers ──────────────────────────────────────────────────────

def _ensure_megadetector(model_path: str) -> str:
    """Download md_v5a.0.pt if not present."""
    p = Path(model_path)
    if not p.exists():
        import urllib.request
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MegaDetector v5a → {p} ...")
        urllib.request.urlretrieve(_MD_V5A_URL, str(p))
    return str(p)


def _detect_boxes_md(
    model,
    frame_bgr: np.ndarray,
    conf_thr: float,
    pad_frac: float,
    pad_px: float,
) -> Tuple[List[Tuple[float, float, float, float]], List[float], List[int]]:
    """Run MegaDetector v5 (torch.hub YOLOv5), keep class 0 (animal) only."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    preds = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,cls]
    h, w = frame_bgr.shape[:2]
    out_boxes, out_confs, out_cls = [], [], []
    for x1, y1, x2, y2, sc, c in preds:
        if float(sc) < conf_thr:
            continue
        if int(c) != 0:          # 0=animal, 1=person, 2=vehicle
            continue
        box = _clip_xyxy((float(x1), float(y1), float(x2), float(y2)), w, h)
        box = _expand_box(box, pad_frac, pad_px, w, h)
        out_boxes.append(box)
        out_confs.append(float(sc))
        out_cls.append(0)
    return out_boxes, out_confs, out_cls


# ── Public API ────────────────────────────────────────────────────────────────

def run_detection(
    video_path: str,
    config: Dict[str, Any],
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """
    Run YOLO11 detection + IoU tracker on every frame of a video.

    Returns dict:
        {
            "video_path": str,
            "frame_count": int,
            "fps": float,
            "width": int,
            "height": int,
            "frames": {
                frame_idx (int): [
                    {"x1","y1","x2","y2","conf","cls","label","track_id"}
                ]
            }
        }
    """
    model_path = config.get("model_path")
    if not model_path:
        raise ValueError("config['model_path'] is required")

    rt = config
    variant   = str(rt.get("variant", "yolo")).lower()
    device    = rt.get("device", 0)
    half      = bool(rt.get("half", False))
    imgsz     = int(rt.get("imgsz", 640))
    conf_thr  = float(rt.get("conf", 0.25))
    iou_thr   = float(rt.get("iou_nms", 0.50))
    scale     = float(rt.get("processing_scale", 0.5))
    pad_frac  = float(rt.get("bbox_pad_frac", 0.10))
    pad_px    = float(rt.get("bbox_pad_px", 8.0))
    kf_int    = int(rt.get("keyframe_interval", 15))

    raw_classes = rt.get("animal_classes", None)
    animal_classes: Set[int] = (
        set(int(c) for c in raw_classes)
        if raw_classes is not None
        else _DEFAULT_ANIMAL_CLASSES
    )

    tr         = rt.get("tracking", {}) or {}
    track_en   = bool(tr.get("enabled", True))
    match_iou  = float(tr.get("match_iou", 0.30))
    min_hits   = int(tr.get("min_hits", 2))
    max_gap    = int(tr.get("max_det_gap", 90))
    vel_smooth = float(tr.get("vel_smooth", 0.70))
    klt_max      = int(tr.get("klt_max_points", 80))
    klt_min      = int(tr.get("klt_min_points", 10))
    mask_hold    = int(tr.get("mask_hold_frames", 0))

    use_md = (variant == "megadetector")
    if use_md:
        import torch, warnings
        model_path = _ensure_megadetector(model_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = torch.hub.load("ultralytics/yolov5", "custom",
                                   path=model_path, force_reload=False, verbose=False)
        model.conf = conf_thr
        dev = device if isinstance(device, str) else f"cuda:{device}"
        model.to(dev)
        if half:
            model.half()
    else:
        from ultralytics import YOLO  # deferred to keep module import fast
        model = YOLO(model_path, task="detect", verbose=False)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps     = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w_full  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_full  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale   = max(0.1, min(1.0, scale))
    w_p     = int(round(w_full * scale))
    h_p     = int(round(h_full * scale))
    sx      = w_full / float(w_p)
    sy      = h_full / float(h_p)

    # COCO class id → name (subset we care about)
    _COCO_NAMES: Dict[int, str] = {
        14: "bird", 15: "cat", 16: "dog", 17: "horse",
        18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
        22: "zebra", 23: "giraffe",
    }

    frames_out: Dict[int, List[Dict[str, Any]]] = {}
    tracks:    List[_Track] = []
    next_id    = 1
    prev_gray: Optional[np.ndarray] = None
    frame_idx  = 0

    try:
        while True:
            ok, frame_full = cap.read()
            if not ok:
                break

            frame_p = cv2.resize(frame_full, (w_p, h_p), interpolation=cv2.INTER_AREA)
            gray_p  = cv2.cvtColor(frame_p, cv2.COLOR_BGR2GRAY)
            is_key  = (frame_idx % kf_int == 0)

            if is_key:
                # Retire stale tracks
                tracks = [t for t in tracks if (frame_idx - t.last_det_frame) <= max_gap]

                if use_md:
                    det_boxes, det_confs, det_cls = _detect_boxes_md(
                        model, frame_p, conf_thr, pad_frac, pad_px,
                    )
                else:
                    det_boxes, det_confs, det_cls = _detect_boxes(
                        model, frame_p, imgsz, conf_thr, iou_thr,
                        device, half, animal_classes, pad_frac, pad_px,
                    )

                if track_en:
                    matches, unmatched_d, unmatched_t = _match_greedy(det_boxes, tracks, match_iou)

                    # Update matched tracks first (ti indexes the original tracks list)
                    matched_t_idx = {j for _, j in matches}
                    for di, ti in matches:
                        t = tracks[ti]
                        nb = det_boxes[di]
                        old_cx = (t.bbox[0] + t.bbox[2]) * 0.5
                        old_cy = (t.bbox[1] + t.bbox[3]) * 0.5
                        ncx = (nb[0] + nb[2]) * 0.5
                        ncy = (nb[1] + nb[3]) * 0.5
                        dt = max(1, frame_idx - t.last_det_frame)
                        t.vx = vel_smooth * t.vx + (1 - vel_smooth) * (ncx - old_cx) / dt
                        t.vy = vel_smooth * t.vy + (1 - vel_smooth) * (ncy - old_cy) / dt
                        t.bbox = nb
                        t.conf = det_confs[di]
                        t.last_det_frame = frame_idx
                        t.last_confirmed_bbox = nb
                        t.hits += 1
                        if t.hits >= min_hits:
                            t.confirmed = True

                    # Prune unmatched tracks after updating (detector is authoritative at keyframes)
                    tracks = [tracks[j] for j in range(len(tracks)) if j in matched_t_idx]

                    for di in unmatched_d:
                        cid = det_cls[di]
                        label = "animal" if use_md else _COCO_NAMES.get(cid, str(cid))
                        t = _Track(
                            track_id=next_id,
                            bbox=det_boxes[di],
                            conf=det_confs[di],
                            cls=cid,
                            label=label,
                            last_det_frame=frame_idx,
                            confirmed=(min_hits <= 1),
                        )
                        tracks.append(t)
                        next_id += 1
                else:
                    # No tracking — create fresh single-hit tracks each keyframe
                    tracks = []
                    for i, (box, sc, cid) in enumerate(zip(det_boxes, det_confs, det_cls)):
                        tracks.append(_Track(
                            track_id=next_id + i, bbox=box, conf=sc, cls=cid,
                            label=_COCO_NAMES.get(cid, str(cid)),
                            last_det_frame=frame_idx, confirmed=True,
                        ))
                    next_id += len(det_boxes)

            else:
                # Inter-keyframe: propagate confirmed tracks with KLT
                alive = []
                for t in tracks:
                    if (frame_idx - t.last_det_frame) > max_gap:
                        continue
                    if track_en and t.confirmed and prev_gray is not None:
                        pts = _seed_klt(prev_gray, t.bbox, klt_max)
                        if pts is not None and len(pts) >= klt_min:
                            shift = _klt_shift(prev_gray, gray_p, pts, klt_min)
                            if shift is not None:
                                dx, dy = shift
                                x1, y1, x2, y2 = t.bbox
                                t.bbox = _clip_xyxy((x1 + dx, y1 + dy, x2 + dx, y2 + dy), w_p, h_p)
                    alive.append(t)
                tracks = alive

            # Emit confirmed tracks for this frame
            # Also emit recently-lost tracks using last known bbox (temporal hold)
            rois = []
            for t in tracks:
                if not t.confirmed:
                    continue
                frames_since_det = frame_idx - t.last_det_frame
                if frames_since_det > mask_hold and t.last_confirmed_bbox is not None:
                    # Detection was lost — use last confirmed bbox during hold window
                    bx1, by1, bx2, by2 = t.last_confirmed_bbox
                elif frames_since_det > mask_hold:
                    continue
                else:
                    bx1, by1, bx2, by2 = t.bbox
                # Scale back to full resolution
                fx1 = int(round(bx1 * sx)); fy1 = int(round(by1 * sy))
                fx2 = int(round(bx2 * sx)); fy2 = int(round(by2 * sy))
                fx1, fy1, fx2, fy2 = [
                    max(0, min(v, lim - 1))
                    for v, lim in zip([fx1, fy1, fx2, fy2], [w_full, h_full, w_full, h_full])
                ]
                rois.append({
                    "x1": fx1, "y1": fy1, "x2": fx2, "y2": fy2,
                    "conf": float(t.conf),
                    "cls": int(t.cls),
                    "label": t.label,
                    "track_id": int(t.track_id),
                })

            if rois:
                frames_out[int(frame_idx)] = rois

            prev_gray = gray_p
            frame_idx += 1
            if progress_cb is not None:
                progress_cb(1)

    finally:
        cap.release()

    return {
        "video_path": str(video_path),
        "frame_count": int(frame_idx),
        "fps": float(fps),
        "width": int(w_full),
        "height": int(h_full),
        "frames": frames_out,
    }
