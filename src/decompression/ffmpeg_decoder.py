"""
src/decompression/ffmpeg_decoder.py

Decode ffmpeg-encoded stream bytes (AV1, HEVC, H.264) back to BGR frame arrays.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional

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
        err_path = os.path.join(td, "ffmpeg_stderr.txt")
        with open(in_path, "wb") as f:
            f.write(stream_bytes)

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-i", in_path,
            "-vf", f"scale={int(width)}:{int(height)}",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "pipe:1",
        ]
        with open(err_path, "wb") as err_f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_f)

        frame_size = int(width) * int(height) * 3
        frames: List[np.ndarray] = []
        idx = 0
        total_bytes = 0
        while True:
            chunk = proc.stdout.read(frame_size)
            total_bytes += len(chunk)
            if len(chunk) < frame_size:
                break
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape(int(height), int(width), 3).copy()
            frames.append(frame)
            if progress_cb is not None:
                progress_cb(idx)
            idx += 1

        proc.stdout.close()
        rc = proc.wait()

        if frame_count_hint is not None and len(frames) != frame_count_hint:
            err_text = ""
            try:
                with open(err_path, "r", errors="replace") as ef:
                    err_text = ef.read(2000).strip()
            except Exception:
                pass
            print(f"[ffdec] decoded={len(frames)} expected={frame_count_hint} "
                  f"total_bytes={total_bytes} rc={rc} "
                  f"stream_bytes={len(stream_bytes)} {width}x{height}", flush=True)
            if err_text:
                print(f"[ffdec] ffmpeg stderr: {err_text[:500]}", flush=True)

    return frames
