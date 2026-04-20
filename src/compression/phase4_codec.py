from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import cv2
import numpy as np

from .ffmpeg_codec import VideoInfo, encode_ffmpeg_frames_to_bytes, probe_video
try:
    from roi_masking import boxes_for_frame as _boxes_for_frame_shared, build_boxes_mask
except ImportError:  # pragma: no cover - test import path fallback
    from src.roi_masking import boxes_for_frame as _boxes_for_frame_shared, build_boxes_mask


def _resolve_path(raw: str | Path, root_dir: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (root_dir / p).resolve()


def _pick_indices(frame_drop_result: Dict[str, Any], key: str, fallback_key: str) -> List[int]:
    vals = frame_drop_result.get(key, None)
    if vals is None:
        vals = frame_drop_result.get(fallback_key, None)
    if vals is None:
        vals = []
    return [int(x) for x in vals]


def _resolve_frame_index_map(
    *,
    stream_name: str,
    encoded_meta: Dict[str, Any],
    requested_indices: List[int],
) -> List[int]:
    target_len = int(encoded_meta.get("frames_encoded", 0) or 0)
    if target_len <= 0:
        return []

    authoritative = encoded_meta.get("frame_index_map", None)
    if isinstance(authoritative, list) and authoritative:
        resolved = [int(x) for x in authoritative]
        source_name = "encoder metadata"
    else:
        resolved = [int(x) for x in requested_indices[:target_len]]
        source_name = "requested indices"

    if len(resolved) != target_len:
        raise RuntimeError(
            f"{stream_name} frame_index_map length mismatch: expected {target_len}, "
            f"got {len(resolved)} from {source_name}."
        )
    if any(resolved[i] >= resolved[i + 1] for i in range(len(resolved) - 1)):
        raise RuntimeError(f"{stream_name} frame_index_map must be strictly increasing.")
    return resolved


def _boxes_for_frame(roi_bbox_map: Mapping[Any, Any], frame_idx: int) -> List[Any]:
    return _boxes_for_frame_shared(roi_bbox_map, frame_idx)


def _apply_roi_mask(frame: np.ndarray, boxes: List[Any], *, roi_min_conf: float = 0.0) -> np.ndarray:
    out = frame.copy()
    if not boxes:
        out[:] = 0
        return out

    h, w = out.shape[:2]
    mask = build_boxes_mask(
        width=int(w),
        height=int(h),
        boxes=boxes,
        roi_min_conf=float(roi_min_conf),
        roi_dilate_px=0,
    )
    out[mask == 0] = 0
    return out


def _iter_kept_roi_frames(
    *,
    video_path: Path,
    roi_kept_frames: List[int],
    roi_bbox_map: Mapping[Any, Any],
    roi_min_conf: float,
) -> Iterable[Tuple[int, np.ndarray]]:
    roi_keep_set = set(int(x) for x in roi_kept_frames)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for ROI keep streaming: {video_path}")

    src_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if src_idx in roi_keep_set:
                yield int(src_idx), _apply_roi_mask(
                    frame,
                    _boxes_for_frame(roi_bbox_map, src_idx),
                    roi_min_conf=float(roi_min_conf),
                )
            src_idx += 1
    finally:
        cap.release()


def _iter_kept_bg_frames(
    *,
    video_path: Path,
    bg_kept_frames: List[int],
) -> Iterable[Tuple[int, np.ndarray]]:
    bg_keep_set = set(int(x) for x in bg_kept_frames)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for BG keep streaming: {video_path}")

    src_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if src_idx in bg_keep_set:
                yield int(src_idx), frame
            src_idx += 1
    finally:
        cap.release()


def _open_frame_store(path: Path, frame_count: int, info: VideoInfo) -> np.memmap:
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.uint8,
        shape=(int(frame_count), int(info.height), int(info.width), 3),
    )


def _close_memmap(arr: Any) -> None:
    if arr is None:
        return
    flush = getattr(arr, "flush", None)
    if callable(flush):
        flush()
    mmap_obj = getattr(arr, "_mmap", None)
    if mmap_obj is not None:
        try:
            mmap_obj.close()
        except Exception:
            pass


def _capture_rendered_kept_frames_single_pass(
    *,
    video_path: Path,
    info: VideoInfo,
    roi_kept_frames: List[int],
    bg_kept_frames: List[int],
    roi_bbox_map: Mapping[Any, Any],
    roi_min_conf: float = 0.0,
    work_dir: Path,
) -> Dict[str, Any]:
    roi_keep_set = set(int(x) for x in roi_kept_frames)
    bg_keep_set = set(int(x) for x in bg_kept_frames)
    roi_store = _open_frame_store(work_dir / "roi_frames.npy", len(roi_keep_set), info) if roi_keep_set else None
    bg_store = _open_frame_store(work_dir / "bg_frames.npy", len(bg_keep_set), info) if bg_keep_set else None

    roi_indices_actual: List[int] = []
    bg_indices_actual: List[int] = []
    roi_pos = 0
    bg_pos = 0
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video for rendered keep stream cache: {video_path}")

        src_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if roi_store is not None and src_idx in roi_keep_set:
                    roi_store[roi_pos] = _apply_roi_mask(
                        frame,
                        _boxes_for_frame(roi_bbox_map, src_idx),
                        roi_min_conf=float(roi_min_conf),
                    )
                    roi_indices_actual.append(int(src_idx))
                    roi_pos += 1
                if bg_store is not None and src_idx in bg_keep_set:
                    bg_store[bg_pos] = frame
                    bg_indices_actual.append(int(src_idx))
                    bg_pos += 1
                src_idx += 1
        finally:
            cap.release()

        if roi_store is not None:
            roi_store.flush()
        if bg_store is not None:
            bg_store.flush()
        return {
            "roi_store": roi_store,
            "roi_indices": roi_indices_actual,
            "roi_count": int(roi_pos),
            "bg_store": bg_store,
            "bg_indices": bg_indices_actual,
            "bg_count": int(bg_pos),
        }
    except Exception:
        _close_memmap(roi_store)
        _close_memmap(bg_store)
        raise


def _iter_cached_frames(store: Any, frame_indices: List[int], frame_count: int) -> Iterable[Tuple[int, np.ndarray]]:
    for pos in range(int(frame_count)):
        yield int(frame_indices[pos]), np.array(store[pos], copy=True)


def _stream_codec_meta(encoding_meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "codec": str(encoding_meta.get("codec", "")),
        "encoder": str(encoding_meta.get("encoder", "")),
        "container": str(encoding_meta.get("container", "mkv")),
        "pix_fmt": str(encoding_meta.get("pix_fmt", "yuv420p")),
        "preset": encoding_meta.get("preset", None),
        "qp": int(encoding_meta.get("qp", 0) or 0),
    }


def _ffmpeg_bins_from_cfg(compression_cfg: Dict[str, Any]) -> Dict[str, str]:
    codec_cfg = compression_cfg.get("codec", {}) or {}
    if not isinstance(codec_cfg, dict):
        codec_cfg = {}
    ffmpeg_bin = str(codec_cfg.get("ffmpeg_bin", "ffmpeg") or "ffmpeg").strip() or "ffmpeg"
    ffprobe_bin = str(codec_cfg.get("ffprobe_bin", "ffprobe") or "ffprobe").strip() or "ffprobe"
    return {"ffmpeg_bin": ffmpeg_bin, "ffprobe_bin": ffprobe_bin}


def compress_keep_streams(
    *,
    source_video_path: str | Path,
    roi_bbox_map: Mapping[Any, Any],
    frame_drop_result: Dict[str, Any],
    compression_cfg: Dict[str, Any],
    root_dir: str | Path,
) -> Dict[str, Any]:
    root = Path(root_dir).expanduser().resolve()
    source_video = _resolve_path(source_video_path, root)

    if not source_video.exists():
        raise FileNotFoundError(f"Source video not found: {source_video}")

    quality_cfg = (compression_cfg.get("quality", {}) or {})
    roi_cfg = (compression_cfg.get("roi", {}) or {})
    ffmpeg_bins = _ffmpeg_bins_from_cfg(compression_cfg)

    roi_qp = int(quality_cfg.get("roi_qp_i", quality_cfg.get("roi_qp_p", 10)))
    bg_qp = int(quality_cfg.get("bg_qp_i", quality_cfg.get("bg_qp_p", 35)))
    min_conf = float(roi_cfg.get("min_conf", 0.25))
    requested_dilate_px = int(roi_cfg.get("dilate_px", 0))
    low_memory = bool(compression_cfg.get("low_memory", False))

    src = probe_video(str(source_video))
    info = VideoInfo(width=int(src.width), height=int(src.height), fps=float(src.fps), frames=int(src.frames))
    roi_indices = _pick_indices(frame_drop_result, "roi_kept_frames", "kept_frames")
    bg_indices = _pick_indices(frame_drop_result, "bg_kept_frames", "kept_frames")

    if low_memory:
        roi_encoded = encode_ffmpeg_frames_to_bytes(
            _iter_kept_roi_frames(
                video_path=source_video,
                roi_kept_frames=roi_indices,
                roi_bbox_map=roi_bbox_map,
                roi_min_conf=float(min_conf),
            ),
            info=info,
            cfg=compression_cfg,
            stream_name="roi",
            video_path=f"{source_video}#roi_stream",
        )
        bg_encoded = encode_ffmpeg_frames_to_bytes(
            _iter_kept_bg_frames(
                video_path=source_video,
                bg_kept_frames=bg_indices,
            ),
            info=info,
            cfg=compression_cfg,
            stream_name="bg",
            video_path=f"{source_video}#bg_stream",
        )
    else:
        with tempfile.TemporaryDirectory(prefix="wildroi_phase4_") as td:
            cached_streams = _capture_rendered_kept_frames_single_pass(
                video_path=source_video,
                info=info,
                roi_kept_frames=roi_indices,
                bg_kept_frames=bg_indices,
                roi_bbox_map=roi_bbox_map,
                roi_min_conf=float(min_conf),
                work_dir=Path(td),
            )
            try:
                roi_encoded = encode_ffmpeg_frames_to_bytes(
                    _iter_cached_frames(
                        cached_streams.get("roi_store", None),
                        cached_streams.get("roi_indices", []),
                        int(cached_streams.get("roi_count", 0) or 0),
                    ),
                    info=info,
                    cfg=compression_cfg,
                    stream_name="roi",
                    video_path=f"{source_video}#roi_stream",
                )
                bg_encoded = encode_ffmpeg_frames_to_bytes(
                    _iter_cached_frames(
                        cached_streams.get("bg_store", None),
                        cached_streams.get("bg_indices", []),
                        int(cached_streams.get("bg_count", 0) or 0),
                    ),
                    info=info,
                    cfg=compression_cfg,
                    stream_name="bg",
                    video_path=f"{source_video}#bg_stream",
                )
            finally:
                _close_memmap(cached_streams.get("roi_store", None))
                _close_memmap(cached_streams.get("bg_store", None))

    roi_bytes = bytes(roi_encoded["bitstream_bytes"])
    bg_bytes = bytes(bg_encoded["bitstream_bytes"])
    roi_meta = dict(roi_encoded.get("meta", {}) or {})
    bg_meta = dict(bg_encoded.get("meta", {}) or {})

    roi_indices = _resolve_frame_index_map(
        stream_name="ROI",
        encoded_meta=roi_meta,
        requested_indices=roi_indices,
    )
    bg_indices = _resolve_frame_index_map(
        stream_name="BG",
        encoded_meta=bg_meta,
        requested_indices=bg_indices,
    )

    meta: Dict[str, Any] = {
        "video": {
            "path": str(source_video),
            "width": int(src.width),
            "height": int(src.height),
            "fps": float(src.fps),
            "frames_total": int(src.frames),
            "start_frame": 0,
            "end_frame": int(max(0, src.frames - 1)),
        },
        "roi": {
            "min_conf": float(min_conf),
            "dilate_px": 0,
            "visible_dilate_px": 0,
            "requested_dilate_px": int(requested_dilate_px),
        },
        "quality": {
            "roi_qp_i": int(roi_qp),
            "roi_qp_p": int(roi_qp),
            "bg_qp_i": int(bg_qp),
            "bg_qp_p": int(bg_qp),
        },
        "codec": {
            "implementation": "ffmpeg",
            "ffmpeg_bin": str(ffmpeg_bins["ffmpeg_bin"]),
            "ffprobe_bin": str(ffmpeg_bins["ffprobe_bin"]),
        },
        "streams": {
            "roi": {
                "source_keep_render": "direct_frame_stream",
                "frames_encoded": int(roi_meta.get("frames_encoded", 0)),
                "frame_index_map": roi_indices,
                "qp_i": int(roi_qp),
                "qp_p": int(roi_qp),
                "compressed_bytes": int(len(roi_bytes)),
                **_stream_codec_meta(roi_meta),
            },
            "bg": {
                "source_keep_render": "direct_frame_stream",
                "frames_encoded": int(bg_meta.get("frames_encoded", 0)),
                "frame_index_map": bg_indices,
                "qp_i": int(bg_qp),
                "qp_p": int(bg_qp),
                "compressed_bytes": int(len(bg_bytes)),
                **_stream_codec_meta(bg_meta),
            },
        },
        "sizes": {
            "roi_bytes": int(len(roi_bytes)),
            "bg_bytes": int(len(bg_bytes)),
        },
        "kept_frames_used": {
            "roi": roi_indices,
            "bg": bg_indices,
        },
    }

    return {
        "roi_bin_bytes": roi_bytes,
        "bg_bin_bytes": bg_bytes,
        "meta": meta,
    }
