"""
ROI/BG decompression helpers (pure module).

Given:
  - roi.bin (ROI stream encoded with ffmpeg AV1)
  - bg.bin  (BG stream encoded with ffmpeg HEVC)
  - meta.json produced by compression step

This module decodes ROI/BG streams into frame arrays.
Final timeline interpolation and ROI/BG compositing are handled by the decompression runner.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _resolve_hint(meta_hint: Any, user_limit: Optional[int]) -> Optional[int]:
    hint: Optional[int] = None
    if meta_hint is not None:
        try:
            hint = int(meta_hint)
        except (TypeError, ValueError):
            hint = None
    if user_limit is None:
        return hint
    try:
        lim = int(user_limit)
    except (TypeError, ValueError):
        lim = -1
    if lim <= 0:
        return hint
    if hint is None:
        return lim
    return min(hint, lim)


def _codec_suffix(stream_meta: Dict[str, Any]) -> str:
    container = str(stream_meta.get("container", "mkv") or "mkv").strip().lower().lstrip(".")
    if not container:
        container = "mkv"
    return f".{container}"


def _ffmpeg_bin(meta: Dict[str, Any]) -> str:
    codec_cfg = meta.get("codec", {}) or {}
    if not isinstance(codec_cfg, dict):
        codec_cfg = {}
    return str(codec_cfg.get("ffmpeg_bin", "ffmpeg") or "ffmpeg").strip() or "ffmpeg"


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: List[bytes] = []
    remaining = int(size)
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decode_stream_bytes_ffmpeg(
    stream_bytes: bytes,
    *,
    stream_meta: Dict[str, Any],
    meta: Dict[str, Any],
    video_info: Dict[str, Any],
    frame_count_hint: Optional[int] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    frame_consumer: Optional[Callable[[np.ndarray], None]] = None,
) -> List[np.ndarray]:
    width = int(video_info.get("width", 0) or 0)
    height = int(video_info.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid video width/height in archive metadata")

    ffmpeg_bin = _ffmpeg_bin(meta)
    frame_size = int(width * height * 3)
    if frame_size <= 0:
        raise RuntimeError("Invalid raw frame size derived from archive metadata")

    frames: List[np.ndarray] = []
    output_limit = None if frame_count_hint is None or int(frame_count_hint) < 0 else int(frame_count_hint)

    with tempfile.TemporaryDirectory(prefix="wildroi_decode_stream_") as td:
        in_path = Path(td) / f"stream{_codec_suffix(stream_meta)}"
        in_path.write_bytes(stream_bytes)
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(in_path),
        ]
        if output_limit is not None:
            cmd.extend(["-frames:v", str(output_limit)])
        cmd.extend(["-f", "rawvideo", "-pix_fmt", "bgr24", "-vsync", "0", "-"])

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.stdout is None:
            raise RuntimeError("ffmpeg stdout pipe is unavailable for decoding")

        frames_decoded = 0
        while output_limit is None or frames_decoded < output_limit:
            raw = _read_exact(proc.stdout, frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                proc.kill()
                raise RuntimeError(
                    f"ffmpeg returned a partial decoded frame for stream {stream_meta.get('codec', 'unknown')}"
                )
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
            if frame_consumer is None:
                frames.append(frame)
            else:
                frame_consumer(frame)
            frames_decoded += 1
            if progress_cb is not None:
                progress_cb(1)

        stderr_tail = ""
        if proc.stderr is not None:
            try:
                stderr_tail = proc.stderr.read().decode("utf-8", errors="replace").strip()
            except Exception:
                stderr_tail = ""
        rc = proc.wait()
        if rc != 0:
            tail = "\n".join(stderr_tail.splitlines()[-20:]) if stderr_tail else "(no ffmpeg stderr)"
            raise RuntimeError(f"ffmpeg decode failed with exit code {rc}.\n{tail}")

    return frames


def decode_roi_bg_streams(
    roi_bin_bytes: bytes,
    bg_bin_bytes: bytes,
    meta: Dict[str, Any],
    progress_cb_roi: Optional[Callable[[int], None]] = None,
    progress_cb_bg: Optional[Callable[[int], None]] = None,
    max_frames_roi: Optional[int] = None,
    max_frames_bg: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    video_info = meta.get("video", {}) or {}
    streams = meta.get("streams", {}) or {}
    roi_stream_meta = streams.get("roi", {}) or {}
    bg_stream_meta = streams.get("bg", {}) or {}

    roi_count_hint = roi_stream_meta.get("frames_encoded", None)
    if roi_count_hint is None and isinstance(roi_stream_meta.get("frame_index_map", None), list):
        roi_count_hint = len(roi_stream_meta.get("frame_index_map", []))
    bg_count_hint = bg_stream_meta.get("frames_encoded", None)
    if bg_count_hint is None and isinstance(bg_stream_meta.get("frame_index_map", None), list):
        bg_count_hint = len(bg_stream_meta.get("frame_index_map", []))

    roi_hint = _resolve_hint(roi_count_hint, max_frames_roi)
    bg_hint = _resolve_hint(bg_count_hint, max_frames_bg)

    roi_frames = _decode_stream_bytes_ffmpeg(
        roi_bin_bytes,
        stream_meta=roi_stream_meta,
        meta=meta,
        video_info=video_info,
        frame_count_hint=roi_hint,
        progress_cb=progress_cb_roi,
    )
    bg_frames = _decode_stream_bytes_ffmpeg(
        bg_bin_bytes,
        stream_meta=bg_stream_meta,
        meta=meta,
        video_info=video_info,
        frame_count_hint=bg_hint,
        progress_cb=progress_cb_bg,
    )
    return roi_frames, bg_frames


def decode_roi_bg_streams_to_memmap(
    roi_bin_bytes: bytes,
    bg_bin_bytes: bytes,
    meta: Dict[str, Any],
    work_dir: Path,
    progress_cb_roi: Optional[Callable[[int], None]] = None,
    progress_cb_bg: Optional[Callable[[int], None]] = None,
    max_frames_roi: Optional[int] = None,
    max_frames_bg: Optional[int] = None,
) -> Tuple[np.memmap, int, np.memmap, int]:
    video_info = meta.get("video", {}) or {}
    streams = meta.get("streams", {}) or {}
    roi_stream_meta = streams.get("roi", {}) or {}
    bg_stream_meta = streams.get("bg", {}) or {}

    width = int(video_info.get("width", 0) or 0)
    height = int(video_info.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid video width/height in archive metadata")

    roi_count_hint = roi_stream_meta.get("frames_encoded", None)
    if roi_count_hint is None and isinstance(roi_stream_meta.get("frame_index_map", None), list):
        roi_count_hint = len(roi_stream_meta.get("frame_index_map", []))
    bg_count_hint = bg_stream_meta.get("frames_encoded", None)
    if bg_count_hint is None and isinstance(bg_stream_meta.get("frame_index_map", None), list):
        bg_count_hint = len(bg_stream_meta.get("frame_index_map", []))

    roi_hint = _resolve_hint(roi_count_hint, max_frames_roi)
    bg_hint = _resolve_hint(bg_count_hint, max_frames_bg)
    if roi_hint is None or bg_hint is None:
        raise RuntimeError("Archive is missing ROI/BG frame count hints required for low-memory decode.")

    work_dir.mkdir(parents=True, exist_ok=True)
    roi_path = work_dir / "roi_frames.npy"
    bg_path = work_dir / "bg_frames.npy"
    roi_map = np.lib.format.open_memmap(
        roi_path,
        mode="w+",
        dtype=np.uint8,
        shape=(int(roi_hint), int(height), int(width), 3),
    )
    bg_map = np.lib.format.open_memmap(
        bg_path,
        mode="w+",
        dtype=np.uint8,
        shape=(int(bg_hint), int(height), int(width), 3),
    )

    roi_written = 0
    bg_written = 0

    def _consume_roi(frame: np.ndarray) -> None:
        nonlocal roi_written
        if roi_written >= int(roi_map.shape[0]):
            raise RuntimeError("Decoded more ROI frames than expected from archive metadata.")
        roi_map[roi_written] = frame
        roi_written += 1

    def _consume_bg(frame: np.ndarray) -> None:
        nonlocal bg_written
        if bg_written >= int(bg_map.shape[0]):
            raise RuntimeError("Decoded more BG frames than expected from archive metadata.")
        bg_map[bg_written] = frame
        bg_written += 1

    _decode_stream_bytes_ffmpeg(
        roi_bin_bytes,
        stream_meta=roi_stream_meta,
        meta=meta,
        video_info=video_info,
        frame_count_hint=roi_hint,
        progress_cb=progress_cb_roi,
        frame_consumer=_consume_roi,
    )
    _decode_stream_bytes_ffmpeg(
        bg_bin_bytes,
        stream_meta=bg_stream_meta,
        meta=meta,
        video_info=video_info,
        frame_count_hint=bg_hint,
        progress_cb=progress_cb_bg,
        frame_consumer=_consume_bg,
    )
    roi_map.flush()
    bg_map.flush()
    return roi_map, roi_written, bg_map, bg_written


def decode_roi_bg_streams_to_cache(
    roi_bin_bytes: bytes,
    bg_bin_bytes: bytes,
    meta: Dict[str, Any],
    work_dir: Path,
    progress_cb_roi: Optional[Callable[[int], None]] = None,
    progress_cb_bg: Optional[Callable[[int], None]] = None,
    max_frames_roi: Optional[int] = None,
    max_frames_bg: Optional[int] = None,
) -> Tuple[List[Path], int, List[Path], int]:
    streams = meta.get("streams", {}) or {}
    roi_stream_meta = streams.get("roi", {}) or {}
    bg_stream_meta = streams.get("bg", {}) or {}

    roi_count_hint = roi_stream_meta.get("frames_encoded", None)
    if roi_count_hint is None and isinstance(roi_stream_meta.get("frame_index_map", None), list):
        roi_count_hint = len(roi_stream_meta.get("frame_index_map", []))
    bg_count_hint = bg_stream_meta.get("frames_encoded", None)
    if bg_count_hint is None and isinstance(bg_stream_meta.get("frame_index_map", None), list):
        bg_count_hint = len(bg_stream_meta.get("frame_index_map", []))

    roi_hint = _resolve_hint(roi_count_hint, max_frames_roi)
    bg_hint = _resolve_hint(bg_count_hint, max_frames_bg)
    if roi_hint is None or bg_hint is None:
        raise RuntimeError("Archive is missing ROI/BG frame count hints required for low-memory decode.")

    roi_dir = work_dir / "roi"
    bg_dir = work_dir / "bg"
    roi_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)

    roi_paths: List[Path] = []
    bg_paths: List[Path] = []
    roi_written = 0
    bg_written = 0

    def _write_frame(dst_dir: Path, idx: int, frame: np.ndarray, suffix: str, params: List[int], out: List[Path]) -> None:
        path = dst_dir / f"{idx:06d}{suffix}"
        ok = cv2.imwrite(str(path), frame, params)
        if not ok:
            raise RuntimeError(f"Failed to write decoded temp frame: {path}")
        out.append(path)

    def _consume_roi(frame: np.ndarray) -> None:
        nonlocal roi_written
        _write_frame(roi_dir, roi_written, frame, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 97], roi_paths)
        roi_written += 1

    def _consume_bg(frame: np.ndarray) -> None:
        nonlocal bg_written
        _write_frame(bg_dir, bg_written, frame, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95], bg_paths)
        bg_written += 1

    _decode_stream_bytes_ffmpeg(
        roi_bin_bytes,
        stream_meta=roi_stream_meta,
        meta=meta,
        video_info=meta.get("video", {}) or {},
        frame_count_hint=roi_hint,
        progress_cb=progress_cb_roi,
        frame_consumer=_consume_roi,
    )
    _decode_stream_bytes_ffmpeg(
        bg_bin_bytes,
        stream_meta=bg_stream_meta,
        meta=meta,
        video_info=meta.get("video", {}) or {},
        frame_count_hint=bg_hint,
        progress_cb=progress_cb_bg,
        frame_consumer=_consume_bg,
    )
    return roi_paths, roi_written, bg_paths, bg_written
