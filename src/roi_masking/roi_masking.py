"""
ROI mask building and compositing helpers (pure module — no I/O).
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

import cv2
import numpy as np


# ── Box lookup ─────────────────────────────────────────────────────────────────

def boxes_for_frame(roi_map: Mapping[Any, Any], frame_idx: int) -> List[Dict[str, Any]]:
    """Return detection list for a frame, tolerating int or str keys."""
    boxes = roi_map.get(frame_idx) or roi_map.get(str(frame_idx))
    return boxes if isinstance(boxes, list) else []


# ── Mask construction ──────────────────────────────────────────────────────────

def build_boxes_mask(
    *,
    width: int,
    height: int,
    boxes: List[Dict[str, Any]],
    min_conf: float = 0.0,
    dilate_px: int = 0,
) -> np.ndarray:
    """
    Paint all detection bounding boxes onto a binary uint8 mask.

    Args:
        width, height: Frame dimensions.
        boxes:         List of dicts with keys x1, y1, x2, y2, conf.
        min_conf:      Minimum confidence to include a box.
        dilate_px:     Morphological dilation in pixels.

    Returns:
        (H, W) uint8 mask: 255 = ROI, 0 = background.
    """
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    for box in boxes:
        if not isinstance(box, dict):
            continue
        try:
            conf = float(box.get("conf", box.get("confidence", 1.0)))
        except (TypeError, ValueError):
            conf = 1.0
        if conf < float(min_conf):
            continue
        try:
            x1 = max(0, min(int(width - 1),  int(box["x1"])))
            y1 = max(0, min(int(height - 1), int(box["y1"])))
            x2 = max(0, min(int(width - 1),  int(box["x2"])))
            y2 = max(0, min(int(height - 1), int(box["y2"])))
        except (KeyError, TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    if int(dilate_px) > 0:
        k = max(3, 2 * int(dilate_px) + 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, ker, iterations=1)

    return mask


def build_frame_mask(
    *,
    frame_idx: int,
    width: int,
    height: int,
    roi_map: Mapping[Any, Any],
    min_conf: float = 0.0,
    dilate_px: int = 0,
) -> np.ndarray:
    """Build ROI mask for a single frame from the detection map."""
    boxes = boxes_for_frame(roi_map, frame_idx)
    return build_boxes_mask(
        width=int(width),
        height=int(height),
        boxes=boxes,
        min_conf=float(min_conf),
        dilate_px=int(dilate_px),
    )


# ── Alpha helpers ──────────────────────────────────────────────────────────────

def mask_to_alpha(mask_u8: np.ndarray, feather_px: int = 0) -> np.ndarray:
    """
    Convert a binary uint8 mask to a float32 alpha map.

    If feather_px > 0, the edges are softened using distance transform.
    """
    binary = (mask_u8 > 0).astype(np.uint8)
    if int(feather_px) <= 0:
        return binary.astype(np.float32)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    alpha = np.clip(dist / float(max(1, int(feather_px))), 0.0, 1.0)
    alpha[binary == 0] = 0.0
    return alpha.astype(np.float32)


# ── Frame compositing ──────────────────────────────────────────────────────────

def compose_soft(
    roi_frame: np.ndarray,
    bg_frame: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Alpha-composite ROI frame over background frame.

        out = roi * alpha + bg * (1 - alpha)

    Args:
        roi_frame: BGR uint8 (H, W, 3) — high-quality ROI stream decoded frame.
        bg_frame:  BGR uint8 (H, W, 3) — background stream decoded frame.
        alpha:     Float32 (H, W) or (H, W, 1) blending weight in [0, 1].

    Returns:
        BGR uint8 (H, W, 3) composited frame.
    """
    if roi_frame.shape != bg_frame.shape:
        raise ValueError("ROI and BG frames must have identical shape")
    if alpha.shape[:2] != roi_frame.shape[:2]:
        raise ValueError("Alpha must match frame H×W")

    alpha3 = alpha[..., None] if alpha.ndim == 2 else alpha
    alpha3 = np.clip(alpha3.astype(np.float32), 0.0, 1.0)
    out = np.clip(
        roi_frame.astype(np.float32) * alpha3
        + bg_frame.astype(np.float32) * (1.0 - alpha3),
        0.0,
        255.0,
    )
    return out.astype(np.uint8)
