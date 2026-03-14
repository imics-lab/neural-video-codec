from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import cv2
import numpy as np
import torch

from .dcvc_encoder import VideoInfo, encode_dcvc_frames_to_bytes, probe_video


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


def _trim_or_fill_indices(indices: List[int], target_len: int) -> List[int]:
    if target_len <= 0:
        return []
    if not indices:
        return list(range(target_len))
    if len(indices) >= target_len:
        return [int(x) for x in indices[:target_len]]
    out = [int(x) for x in indices]
    last = out[-1]
    while len(out) < target_len:
        last += 1
        out.append(last)
    return out


def _boxes_for_frame(roi_bbox_map: Mapping[Any, Any], frame_idx: int) -> List[Any]:
    boxes = roi_bbox_map.get(frame_idx, None)
    if boxes is None:
        boxes = roi_bbox_map.get(str(frame_idx), None)
    return boxes if isinstance(boxes, list) else []


def _apply_roi_mask(frame: np.ndarray, boxes: List[Any]) -> np.ndarray:
    out = frame.copy()
    if not boxes:
        out[:] = 0
        return out

    h, w = out.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for roi in boxes:
        if not isinstance(roi, dict):
            continue
        x1 = max(0, min(w - 1, int(roi.get("x1", 0))))
        y1 = max(0, min(h - 1, int(roi.get("y1", 0))))
        x2 = max(0, min(w, int(roi.get("x2", 0))))
        y2 = max(0, min(h, int(roi.get("y2", 0))))
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = 255
    out[mask == 0] = 0
    return out


def _iter_rendered_kept_frames(
    *,
    video_path: Path,
    kept_frames: List[int],
    roi_bbox_map: Mapping[Any, Any],
    render_mode: str,
) -> Iterable[Tuple[int, np.ndarray]]:
    mode = str(render_mode).strip().lower()
    if mode not in {"roi", "bg"}:
        raise ValueError("render_mode must be one of: roi, bg")

    keep_set = set(int(x) for x in kept_frames)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for rendered keep stream: {video_path}")

    src_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if src_idx not in keep_set:
                src_idx += 1
                continue

            rendered = frame
            if mode == "roi":
                rendered = _apply_roi_mask(frame, _boxes_for_frame(roi_bbox_map, src_idx))
            yield int(src_idx), rendered
            src_idx += 1
    finally:
        cap.release()


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

    src = probe_video(str(source_video))
    info = VideoInfo(width=int(src.width), height=int(src.height), fps=float(src.fps), frames=int(src.frames))
    roi_indices = _pick_indices(frame_drop_result, "roi_kept_frames", "kept_frames")
    bg_indices = _pick_indices(frame_drop_result, "bg_kept_frames", "kept_frames")

    roi_encoded = encode_dcvc_frames_to_bytes(
        _iter_rendered_kept_frames(
            video_path=source_video,
            kept_frames=roi_indices,
            roi_bbox_map=roi_bbox_map,
            render_mode="roi",
        ),
        info=info,
        cfg={"dcvc": dcvc_cfg, "quality": {"qp_i": roi_qp_i, "qp_p": roi_qp_p}},
        video_path=f"{source_video}#roi_stream",
    )
    bg_encoded = encode_dcvc_frames_to_bytes(
        _iter_rendered_kept_frames(
            video_path=source_video,
            kept_frames=bg_indices,
            roi_bbox_map=roi_bbox_map,
            render_mode="bg",
        ),
        info=info,
        cfg={"dcvc": dcvc_cfg, "quality": {"qp_i": bg_qp_i, "qp_p": bg_qp_p}},
        video_path=f"{source_video}#bg_stream",
    )

    roi_bytes = bytes(roi_encoded["bitstream_bytes"])
    bg_bytes = bytes(bg_encoded["bitstream_bytes"])
    roi_meta = dict(roi_encoded.get("meta", {}) or {})
    bg_meta = dict(bg_encoded.get("meta", {}) or {})

    roi_indices = _trim_or_fill_indices(roi_indices, int(roi_meta.get("frames_encoded", 0) or 0))
    bg_indices = _trim_or_fill_indices(bg_indices, int(bg_meta.get("frames_encoded", 0) or 0))
    min_conf = float(roi_cfg.get("min_conf", 0.25))
    dilate_px = int(roi_cfg.get("dilate_px", 0))

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
            "dilate_px": int(dilate_px),
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
