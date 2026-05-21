from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import cv2
import numpy as np
import torch

from .dcvc_encoder import VideoInfo, encode_dcvc_frames_to_bytes, probe_video
try:
    from roi_masking import boxes_for_frame as _boxes_for_frame_shared, build_boxes_mask
except ImportError:  # pragma: no cover - test import path fallback
    from src.roi_masking import boxes_for_frame as _boxes_for_frame_shared, build_boxes_mask


def _resolve_path(raw: str | Path, root_dir: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (root_dir / p).resolve()


def _parse_use_cuda(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "cuda"}:
        return True
    if s in {"0", "false", "no", "cpu"}:
        return False
    if s == "auto":
        return bool(torch.cuda.is_available())
    return True


def _parse_cuda_index(raw: Any) -> int | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return int(raw) if raw >= 0 else None
    s = str(raw).strip()
    if s.isdigit():
        return int(s)
    return None


def _resolve_dcvc_device(raw_device: Any, raw_use_cuda: Any, raw_cuda_idx: Any) -> Dict[str, Any]:
    cuda_ok = bool(torch.cuda.is_available())
    req_idx = _parse_cuda_index(raw_cuda_idx)

    if raw_device is None:
        use_cuda_req = _parse_use_cuda(raw_use_cuda)
        req_label = "cuda" if use_cuda_req else "cpu"
    elif isinstance(raw_device, int):
        req_label = "cuda"
        req_idx = int(raw_device) if int(raw_device) >= 0 else req_idx
        use_cuda_req = True
    else:
        s = str(raw_device).strip().lower()
        if not s or s == "auto":
            req_label = "auto"
        elif s == "cpu":
            req_label = "cpu"
        elif s == "cuda":
            req_label = "cuda"
        elif s.startswith("cuda:"):
            idx = s.split(":", 1)[1].strip()
            if idx.isdigit():
                req_idx = int(idx)
                req_label = "cuda"
            else:
                req_label = "auto"
        elif s.isdigit():
            req_idx = int(s)
            req_label = "cuda"
        else:
            req_label = "auto"
        use_cuda_req = req_label != "cpu"

    if req_label == "cpu":
        return {
            "device_requested": req_label,
            "device_selected": "cpu",
            "use_cuda_requested": bool(use_cuda_req),
            "use_cuda_selected": False,
            "cuda_idx_selected": None,
        }
    if req_label == "cuda":
        if cuda_ok:
            return {
                "device_requested": req_label,
                "device_selected": "cuda",
                "use_cuda_requested": bool(use_cuda_req),
                "use_cuda_selected": True,
                "cuda_idx_selected": (req_idx if req_idx is not None else 0),
            }
        return {
            "device_requested": req_label,
            "device_selected": "cpu",
            "use_cuda_requested": bool(use_cuda_req),
            "use_cuda_selected": False,
            "cuda_idx_selected": None,
        }
    if cuda_ok:
        return {
            "device_requested": req_label,
            "device_selected": "cuda",
            "use_cuda_requested": bool(use_cuda_req),
            "use_cuda_selected": True,
            "cuda_idx_selected": (req_idx if req_idx is not None else 0),
        }
    return {
        "device_requested": req_label,
        "device_selected": "cpu",
        "use_cuda_requested": bool(use_cuda_req),
        "use_cuda_selected": False,
        "cuda_idx_selected": None,
    }


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
            f"{stream_name} frame_index_map length mismatch: expected {target_len}, got {len(resolved)} from {source_name}."
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
    # Exact visible-mask semantics: confidence filtering is allowed, but visible
    # dilation is not. Extra context belongs to AMT crop selection later.
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
    """
    Stream ROI frames (masked) directly from the video without caching all frames in RAM.
    """
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
    """
    Stream BG frames directly from the video without caching all frames in RAM.
    """
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


def compress_keep_streams_dcvc(
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

    dcvc_cfg_raw = (compression_cfg.get("dcvc", {}) or {})
    quality_cfg = (compression_cfg.get("quality", {}) or {})
    roi_cfg = (compression_cfg.get("roi", {}) or {})

    repo_dir = _resolve_path(str(dcvc_cfg_raw.get("repo_dir", "DCVC")), root)
    model_i = _resolve_path(str(dcvc_cfg_raw.get("model_i", "")), root)
    model_p = _resolve_path(str(dcvc_cfg_raw.get("model_p", "")), root)
    if not model_i.exists():
        raise FileNotFoundError(f"DCVC image model not found: {model_i}")
    if not model_p.exists():
        raise FileNotFoundError(f"DCVC video model not found: {model_p}")
    if not repo_dir.exists():
        raise FileNotFoundError(f"DCVC repo_dir not found: {repo_dir}")

    dcvc_cfg: Dict[str, Any] = dict(dcvc_cfg_raw)
    dcvc_cfg["repo_dir"] = str(repo_dir)
    dcvc_cfg["model_i"] = str(model_i)
    dcvc_cfg["model_p"] = str(model_p)
    dcvc_device = _resolve_dcvc_device(
        dcvc_cfg_raw.get("device", None),
        dcvc_cfg_raw.get("use_cuda", True),
        dcvc_cfg_raw.get("cuda_idx", None),
    )
    dcvc_cfg["device"] = str(dcvc_device["device_selected"])
    dcvc_cfg["use_cuda"] = bool(dcvc_device["use_cuda_selected"])
    dcvc_cfg["cuda_idx"] = dcvc_device["cuda_idx_selected"]
    if "reset_interval" in dcvc_cfg_raw:
        dcvc_cfg["reset_interval"] = int(dcvc_cfg_raw["reset_interval"])
    if "intra_period" in dcvc_cfg_raw:
        dcvc_cfg["intra_period"] = int(dcvc_cfg_raw["intra_period"])

    roi_qp_i = int(quality_cfg.get("roi_qp_i", 50))
    roi_qp_p = int(quality_cfg.get("roi_qp_p", roi_qp_i))
    bg_qp_i = int(quality_cfg.get("bg_qp_i", 20))
    bg_qp_p = int(quality_cfg.get("bg_qp_p", bg_qp_i))
    min_conf = float(roi_cfg.get("min_conf", 0.25))
    requested_dilate_px = int(roi_cfg.get("dilate_px", 0))
    low_memory = bool(compression_cfg.get("low_memory", False))

    src = probe_video(str(source_video))
    info = VideoInfo(width=int(src.width), height=int(src.height), fps=float(src.fps), frames=int(src.frames))
    roi_indices = _pick_indices(frame_drop_result, "roi_kept_frames", "kept_frames")
    bg_indices = _pick_indices(frame_drop_result, "bg_kept_frames", "kept_frames")
    if low_memory:
        # Stream frames directly into the encoder to avoid caching all kept frames
        # as full-resolution arrays in RAM (the source of multi-GB RSS spikes).
        roi_encoded = encode_dcvc_frames_to_bytes(
            _iter_kept_roi_frames(
                video_path=source_video,
                roi_kept_frames=roi_indices,
                roi_bbox_map=roi_bbox_map,
                roi_min_conf=float(min_conf),
            ),
            info=info,
            cfg={"dcvc": dcvc_cfg, "quality": {"qp_i": roi_qp_i, "qp_p": roi_qp_p}},
            video_path=f"{source_video}#roi_stream",
        )
        bg_encoded = encode_dcvc_frames_to_bytes(
            _iter_kept_bg_frames(
                video_path=source_video,
                bg_kept_frames=bg_indices,
            ),
            info=info,
            cfg={"dcvc": dcvc_cfg, "quality": {"qp_i": bg_qp_i, "qp_p": bg_qp_p}},
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
                roi_encoded = encode_dcvc_frames_to_bytes(
                    _iter_cached_frames(
                        cached_streams.get("roi_store", None),
                        cached_streams.get("roi_indices", []),
                        int(cached_streams.get("roi_count", 0) or 0),
                    ),
                    info=info,
                    cfg={"dcvc": dcvc_cfg, "quality": {"qp_i": roi_qp_i, "qp_p": roi_qp_p}},
                    video_path=f"{source_video}#roi_stream",
                )
                bg_encoded = encode_dcvc_frames_to_bytes(
                    _iter_cached_frames(
                        cached_streams.get("bg_store", None),
                        cached_streams.get("bg_indices", []),
                        int(cached_streams.get("bg_count", 0) or 0),
                    ),
                    info=info,
                    cfg={"dcvc": dcvc_cfg, "quality": {"qp_i": bg_qp_i, "qp_p": bg_qp_p}},
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
            "roi_qp_i": int(roi_qp_i),
            "roi_qp_p": int(roi_qp_p),
            "bg_qp_i": int(bg_qp_i),
            "bg_qp_p": int(bg_qp_p),
        },
        "dcvc": {
            "repo_dir": str(repo_dir),
            "model_i": str(model_i),
            "model_p": str(model_p),
            "device_requested": str(dcvc_device["device_requested"]),
            "device_selected": str(dcvc_device["device_selected"]),
            "force_zero_thres": dcvc_cfg.get("force_zero_thres", None),
            "use_cuda": bool(dcvc_device["use_cuda_selected"]),
            "cuda_idx": dcvc_cfg.get("cuda_idx", None),
            "reset_interval": int(dcvc_cfg.get("reset_interval", 32)),
            "intra_period": int(dcvc_cfg.get("intra_period", -1)),
            "device": str(roi_meta.get("device", bg_meta.get("device", "cpu"))),
        },
        "streams": {
            "roi": {
                "source_keep_render": "direct_frame_stream",
                "frames_encoded": int(roi_meta.get("frames_encoded", 0)),
                "frame_index_map": roi_indices,
                "qp_i": int(roi_qp_i),
                "qp_p": int(roi_qp_p),
                "compressed_bytes": int(len(roi_bytes)),
            },
            "bg": {
                "source_keep_render": "direct_frame_stream",
                "frames_encoded": int(bg_meta.get("frames_encoded", 0)),
                "frame_index_map": bg_indices,
                "qp_i": int(bg_qp_i),
                "qp_p": int(bg_qp_p),
                "compressed_bytes": int(len(bg_bytes)),
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
