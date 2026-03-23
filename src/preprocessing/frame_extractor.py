"""
Frame extraction utilities (pure module — no filesystem writes).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def probe_video(video_path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoInfo(width=w, height=h, fps=fps, frame_count=n)


def extract_frames(
    video_path: str | Path,
    downscale: float = 1.0,
    start_frame: int = 0,
    end_frame: int = -1,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Yield (frame_index, BGR_frame) for each frame in the video.

    Args:
        video_path:  Path to the source video file.
        downscale:   Resize factor applied before yielding (1.0 = no resize).
        start_frame: First frame index (inclusive).
        end_frame:   Last frame index (inclusive); -1 = until end of video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame < 0 or end_frame >= total:
        end_frame = total - 1
    start_frame = max(0, start_frame)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    scale = float(downscale)
    apply_scale = abs(scale - 1.0) > 1e-4

    idx = start_frame
    try:
        while idx <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if apply_scale:
                new_w = max(1, int(round(frame.shape[1] * scale)))
                new_h = max(1, int(round(frame.shape[0] * scale)))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            yield idx, frame
            idx += 1
    finally:
        cap.release()
