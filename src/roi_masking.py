from __future__ import annotations

from typing import Any, Dict, List, Mapping

import cv2
import numpy as np


def boxes_for_frame(roi_boxes_map: Mapping[Any, Any], frame_idx: int) -> List[Any]:
    boxes = roi_boxes_map.get(frame_idx, None)
    if boxes is None:
        boxes = roi_boxes_map.get(str(frame_idx), None)
    return boxes if isinstance(boxes, list) else []


def build_boxes_mask(
    *,
    width: int,
    height: int,
    boxes: List[Any],
    roi_min_conf: float = 0.0,
    roi_dilate_px: int = 0,
) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    for box in boxes:
        if not isinstance(box, dict):
            continue
        try:
            conf = float(box.get("conf", box.get("confidence", 1.0)))
        except (TypeError, ValueError):
            conf = 1.0
        if conf < float(roi_min_conf):
            continue
        try:
            x1 = max(0, min(int(width - 1), int(box.get("x1", 0))))
            y1 = max(0, min(int(height - 1), int(box.get("y1", 0))))
            x2 = max(0, min(int(width - 1), int(box.get("x2", 0))))
            y2 = max(0, min(int(height - 1), int(box.get("y2", 0))))
        except (TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    dpx = max(0, int(roi_dilate_px))
    if dpx > 0:
        k = max(3, 2 * dpx + 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, ker, iterations=1)
    return mask


def build_frame_mask(
    *,
    frame_idx: int,
    width: int,
    height: int,
    mask_source: str,
    roi_boxes_map: Dict[str, Any],
    frame_drop_json: Dict[str, Any],
    roi_min_conf: float,
    roi_dilate_px: int,
) -> np.ndarray:
    src = str(mask_source or "roi_detection").strip().lower()
    if src == "frame_drop_roi_box":
        mask = np.zeros((int(height), int(width)), dtype=np.uint8)
        pf = (frame_drop_json.get("per_frame", {}) or {}).get(str(int(frame_idx)), {}) or {}
        if bool(pf.get("bbox_missing", False)):
            return mask
        box = pf.get("roi_box", {}) or {}
        try:
            x1 = max(0, min(int(width - 1), int(box.get("x1", 0))))
            y1 = max(0, min(int(height - 1), int(box.get("y1", 0))))
            x2 = max(0, min(int(width - 1), int(box.get("x2", 0))))
            y2 = max(0, min(int(height - 1), int(box.get("y2", 0))))
        except (TypeError, ValueError):
            return mask
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        return mask

    return build_boxes_mask(
        width=int(width),
        height=int(height),
        boxes=boxes_for_frame(roi_boxes_map, int(frame_idx)),
        roi_min_conf=float(roi_min_conf),
        roi_dilate_px=int(roi_dilate_px),
    )


def mask_to_alpha(mask_u8: np.ndarray, feather_px: int) -> np.ndarray:
    binary = (mask_u8 > 0).astype(np.uint8)
    if int(feather_px) <= 0:
        return binary.astype(np.float32)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    alpha = np.clip(dist / float(max(1, int(feather_px))), 0.0, 1.0)
    alpha[binary == 0] = 0.0
    return alpha.astype(np.float32)


def compose_soft(roi_frame: np.ndarray, bg_frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if roi_frame.shape != bg_frame.shape:
        raise ValueError("ROI and background frames must have identical shape for compositing")
    if alpha.shape[:2] != roi_frame.shape[:2]:
        raise ValueError("Alpha mask must match frame height and width")
    if alpha.ndim == 2:
        alpha3 = alpha[..., None]
    else:
        alpha3 = alpha
    alpha3 = np.clip(alpha3.astype(np.float32), 0.0, 1.0)
    roi_f = roi_frame.astype(np.float32)
    bg_f = bg_frame.astype(np.float32)
    out = np.clip((roi_f * alpha3) + (bg_f * (1.0 - alpha3)), 0.0, 255.0)
    return out.astype(np.uint8)
