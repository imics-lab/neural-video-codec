"""
Video assembly — writes a list of BGR frames to an MP4 file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# Codec preference order: mp4v first (universally supported), then H.264 variants.
_CODEC_FALLBACKS = ["mp4v", "X264", "avc1", "XVID"]


def assemble_video(
    frames: Iterable[np.ndarray],
    output_path: str | Path,
    fps: float,
    codec: str | None = None,
) -> Path:
    """
    Write BGR frames to an MP4.

    Args:
        frames:      Iterable of BGR uint8 arrays, all the same spatial size.
        output_path: Destination file path (created if absent, parent must exist).
        fps:         Output frame rate.
        codec:       FourCC codec string. If None, tries X264 → avc1 → mp4v → XVID.

    Returns:
        Resolved path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    codecs_to_try = [codec] if codec else _CODEC_FALLBACKS

    writer: cv2.VideoWriter | None = None
    out_w: int | None = None
    out_h: int | None = None

    try:
        for frame in frames:
            if writer is None:
                h, w = frame.shape[:2]
                out_w, out_h = w, h

                for c in codecs_to_try:
                    fourcc = cv2.VideoWriter_fourcc(*c)
                    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
                    if writer.isOpened():
                        break
                    writer.release()
                    writer = None

                if writer is None or not writer.isOpened():
                    raise RuntimeError(
                        f"VideoWriter failed to open {output_path} with any codec "
                        f"({codecs_to_try}), fps={fps}, size={out_w}x{out_h}"
                    )

            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()

    if not output_path.exists():
        raise RuntimeError(f"VideoWriter did not produce output file: {output_path}")

    return output_path.resolve()
