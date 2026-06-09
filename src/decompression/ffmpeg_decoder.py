"""
src/decompression/ffmpeg_decoder.py

Decode ffmpeg-encoded stream bytes (AV1, HEVC, H.264) back to BGR frame arrays.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Callable, List, Optional

import numpy as np


def decode_ffmpeg_stream_bytes(
    stream_bytes: bytes,
    *,
    width: int,
    height: int,
    frame_count_hint: Optional[int] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> List[np.ndarray]:
    """
    Decode an ffmpeg-encoded stream (mp4 container, any codec) to BGR frames.

    Parameters
    ----------
    stream_bytes    : raw bytes of the encoded mp4 file
    width, height   : expected frame dimensions
    frame_count_hint: used only for progress reporting
    progress_cb     : optional callable(frame_index) called each decoded frame
    """
    with tempfile.TemporaryDirectory(prefix="ffdec_") as td:
        in_path = os.path.join(td, "stream.mp4")
        with open(in_path, "wb") as f:
            f.write(stream_bytes)

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", in_path,
            "-vf", f"scale={int(width)}:{int(height)}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        frame_size = int(width) * int(height) * 3
        frames: List[np.ndarray] = []
        idx = 0
        while True:
            chunk = proc.stdout.read(frame_size)
            if len(chunk) < frame_size:
                break
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape(int(height), int(width), 3).copy()
            frames.append(frame)
            if progress_cb is not None:
                progress_cb(idx)
            idx += 1

        proc.stdout.close()
        proc.wait()

    return frames
