"""
Video assembly — writes a list of BGR frames to an MP4 file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def assemble_video(
    frames: Iterable[np.ndarray],
    output_path: str | Path,
    fps: float,
    codec: str = "avc1",
) -> Path:
    """
    Write BGR frames to an MP4.

    Args:
        frames:      Iterable of BGR uint8 arrays, all the same spatial size.
        output_path: Destination file path (created if absent, parent must exist).
        fps:         Output frame rate.
        codec:       FourCC codec string (default "mp4v" — H.264 via "avc1" also works).

    Returns:
        Resolved path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer: cv2.VideoWriter | None = None
    fourcc = cv2.VideoWriter_fourcc(*codec)

    try:
        for frame in frames:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(
                        f"VideoWriter failed to open: {output_path} "
                        f"(codec={codec}, fps={fps}, size={w}x{h})"
                    )
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()

    if not output_path.exists():
        raise RuntimeError(f"VideoWriter did not produce output file: {output_path}")

    return output_path.resolve()
