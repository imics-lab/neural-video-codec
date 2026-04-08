"""
Video assembly — writes a list of BGR frames to an MP4 file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# Max output resolution — frames are downscaled to fit within this box while
# preserving aspect ratio.
MAX_WIDTH  = 2160
MAX_HEIGHT = 1440

# Codec preference order: software H.264, then MPEG-4 as universal fallback.
_CODEC_FALLBACKS = ["X264", "avc1", "mp4v", "XVID"]


def _fit_size(w: int, h: int) -> tuple[int, int]:
    """Downscale (w, h) to fit within MAX_WIDTH × MAX_HEIGHT, preserving AR."""
    scale = min(MAX_WIDTH / w, MAX_HEIGHT / h, 1.0)
    return int(w * scale), int(h * scale)


def assemble_video(
    frames: Iterable[np.ndarray],
    output_path: str | Path,
    fps: float,
    codec: str | None = None,
) -> Path:
    """
    Write BGR frames to an MP4, capped at 2160×1440.

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
    chosen_codec: str | None = None

    try:
        for frame in frames:
            if writer is None:
                h, w = frame.shape[:2]
                out_w, out_h = _fit_size(w, h)

                for c in codecs_to_try:
                    fourcc = cv2.VideoWriter_fourcc(*c)
                    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
                    if writer.isOpened():
                        chosen_codec = c
                        break
                    writer.release()
                    writer = None

                if writer is None or not writer.isOpened():
                    raise RuntimeError(
                        f"VideoWriter failed to open {output_path} with any codec "
                        f"({codecs_to_try}), fps={fps}, size={out_w}x{out_h}"
                    )

            if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()

    if not output_path.exists():
        raise RuntimeError(f"VideoWriter did not produce output file: {output_path}")

    return output_path.resolve()
