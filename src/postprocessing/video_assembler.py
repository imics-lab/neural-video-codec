"""
VideoAssembler — context-manager wrapper around cv2.VideoWriter.

Writes BGR uint8 frames to an MP4 file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class VideoAssembler:
    """
    Context manager that writes BGR frames to an MP4 output file.

    Usage::

        with VideoAssembler(output_path, fps, width, height) as assembler:
            for frame in frames:
                assembler.write(frame)

    Args:
        output_path: Destination MP4 file path.
        fps:         Output frame rate.
        width:       Frame width in pixels.
        height:      Frame height in pixels.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
    ) -> None:
        self.output_path = str(output_path)
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self._writer: Optional[cv2.VideoWriter] = None

    # -- context manager --

    def __enter__(self) -> "VideoAssembler":
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            max(self.fps, 1e-6),
            (self.width, self.height),
        )
        if not self._writer.isOpened():
            raise IOError(f"Cannot open output video for writing: {self.output_path}")
        return self

    def __exit__(self, *_) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    # -- write --

    def write(self, frame_bgr: np.ndarray) -> None:
        """Write one BGR uint8 frame to the output video."""
        if self._writer is None:
            raise RuntimeError("VideoAssembler is not open. Use as a context manager.")
        self._writer.write(frame_bgr)
