"""
FrameExtractor — context-manager iterator over video frames.
load_roi_data  — auto-detecting ROI JSON parser (Format A and Format B).
create_weight_mask — float32 (H, W) weight mask from ROI data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# ROI JSON loading
# ---------------------------------------------------------------------------

def _normalise_region(r: dict) -> dict:
    """Convert any supported region format to {x, y, w, h, weight, conf, label}."""
    if "x1" in r:
        # Format B: x1/y1/x2/y2 corners
        x = int(r["x1"])
        y = int(r["y1"])
        w = int(r["x2"]) - x
        h = int(r["y2"]) - y
    else:
        # Format A: x/y/w/h
        x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])

    return {
        "x": x,
        "y": y,
        "w": max(w, 1),
        "h": max(h, 1),
        "weight": float(r.get("weight", r.get("conf", 1.0))),
        "conf":   float(r.get("conf", 1.0)),
        "label":  r.get("label", ""),
    }


def load_roi_data(json_path: str) -> Dict[int, List[dict]]:
    """
    Load an ROI JSON file and return a normalised dict:
        {frame_idx (int) -> list of region dicts}

    Auto-detects:
      Format A (flat):  {"0": [...], "1": [...]}
      Format B (nested): {"video_path": ..., "frames": {"15": [...], ...}}

    This mirrors the logic from distil_clean/utils/roi.py.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Format B: has a 'frames' sub-dict alongside metadata keys
    if "frames" in raw and isinstance(raw["frames"], dict):
        frame_dict = raw["frames"]
    else:
        frame_dict = raw

    normalised: Dict[int, List[dict]] = {}
    for key, regions in frame_dict.items():
        if not isinstance(regions, list):
            continue  # skip metadata fields that aren't frame entries
        try:
            idx = int(key)
        except ValueError:
            continue
        normalised[idx] = [_normalise_region(r) for r in regions]

    return normalised


# ---------------------------------------------------------------------------
# Weight mask creation
# ---------------------------------------------------------------------------

def create_weight_mask(
    frame_idx: int,
    roi_data: Dict[int, List[dict]],
    h: int,
    w: int,
    default_weight: float = 3.0,
    padding: int = 16,
) -> np.ndarray:
    """
    Create a float32 weight mask (H, W):
      - Background pixels = 1.0
      - ROI pixels = default_weight

    Args:
        frame_idx:      Current frame index.
        roi_data:       Normalised ROI data from load_roi_data().
        h:              Frame height in pixels.
        w:              Frame width in pixels.
        default_weight: Weight value applied to ROI regions.
        padding:        Extra pixels added around each ROI box.

    Returns:
        (H, W) float32 mask.
    """
    mask = np.ones((h, w), dtype=np.float32)
    regions = roi_data.get(frame_idx, [])

    for region in regions:
        x  = max(0, region["x"] - padding)
        y  = max(0, region["y"] - padding)
        x2 = min(w, region["x"] + region["w"] + padding)
        y2 = min(h, region["y"] + region["h"] + padding)
        mask[y:y2, x:x2] = default_weight

    return mask


# ---------------------------------------------------------------------------
# FrameExtractor
# ---------------------------------------------------------------------------

class FrameExtractor:
    """
    Context-manager iterator over video frames (BGR uint8).

    Usage::

        with FrameExtractor(video_path) as extractor:
            for frame_idx, frame_bgr in extractor:
                process(frame_idx, frame_bgr)

    Properties: fps, num_frames, width, height
    """

    def __init__(self, video_path: str) -> None:
        self.video_path = str(video_path)
        self._cap: Optional[cv2.VideoCapture] = None

    # -- context manager --

    def __enter__(self) -> "FrameExtractor":
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
        return self

    def __exit__(self, *_) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # -- iterator --

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        if self._cap is None:
            raise RuntimeError("FrameExtractor must be used as a context manager.")
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1

    # -- properties --

    @property
    def fps(self) -> float:
        self._require_open()
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    @property
    def num_frames(self) -> int:
        self._require_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        self._require_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        self._require_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _require_open(self) -> None:
        if self._cap is None:
            raise RuntimeError("FrameExtractor is not open. Use as a context manager.")
