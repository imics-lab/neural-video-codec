"""
src/decompression/reconstruct.py

Full archive-to-video reconstruction with ROI compositing, temporal stabilization,
and optional AMT interpolation.

Public API:
    decompress_archive(archive_path, dec_cfg_raw, output_path=None, ...)
        -> (frames, fps, width, height, roi_detections)
    decompress_archive_bytes(archive_bytes, dec_cfg_raw, ...)
        -> same
"""
from __future__ import annotations

import gc
import json
import shutil
import subprocess
import tempfile
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from . import common as rd
from .roi_bg_decompress import decode_roi_bg_streams_to_memmap
from .interpolation_amt import AmtInterpolator, _pad_to_divisor, _unpad

try:
    from roi_masking import mask_to_alpha
except ImportError:
    from src.roi_masking import mask_to_alpha


# ── GPU utilities ──────────────────────────────────────────────────────────────

def _normalize_cuda_device(raw_device: Any, *, field_name: str) -> Tuple[str, Optional[int]]:
    if raw_device is None:
        return "cuda", None
    if isinstance(raw_device, bool):
        raise ValueError(f"{field_name} must be: auto, cuda, cuda:<index>, or integer GPU index.")
    if isinstance(raw_device, int):
        if int(raw_device) < 0:
            raise ValueError(f"{field_name} integer GPU index must be >= 0.")
        return f"cuda:{int(raw_device)}", int(raw_device)
    s = str(raw_device).strip().lower()
    if s in {"", "auto", "cuda"}:
        return "cuda", None
    if s.startswith("cuda:"):
        idx = s.split(":", 1)[1].strip()
        if idx.isdigit():
            return f"cuda:{int(idx)}", int(idx)
        raise ValueError(f"{field_name} must be cuda:<index> for an explicit CUDA index.")
    if s.isdigit():
        return f"cuda:{int(s)}", int(s)
    if s in {"cpu", "mps"}:
        raise ValueError(f"Strict GPU runtime forbids {field_name} set to CPU/MPS.")
    raise ValueError(f"{field_name} must be: auto, cuda, cuda:<index>, or integer GPU index.")


def _enforce_strict_gpu_runtime(dec_cfg: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("Strict GPU runtime requires CUDA, but torch.cuda.is_available() is false.")
    dcvc_cfg = (meta.get("dcvc", {}) or {}).copy()
    if not rd._coerce_bool(dcvc_cfg.get("use_cuda", True), default=True):
        raise ValueError("Strict GPU runtime forbids dcvc.use_cuda=false.")
    dcvc_device = str(dcvc_cfg.get("device", "cuda")).strip().lower()
    if dcvc_device in {"cpu", "mps"}:
        raise ValueError("Strict GPU runtime forbids dcvc.device set to CPU/MPS.")
    dcvc_selected_device, dcvc_selected_idx = _normalize_cuda_device(
        dcvc_cfg.get("cuda_idx", dcvc_cfg.get("device", "cuda")),
        field_name="decompression.dcvc.device",
    )
    dcvc_cfg["use_cuda"] = True
    dcvc_cfg["device"] = str(dcvc_selected_device)
    dcvc_cfg["cuda_idx"] = dcvc_selected_idx
    meta["dcvc"] = dcvc_cfg
    interp_cfg = (dec_cfg.get("interpolate", {}) or {}).copy()
    interp_selected_device = None
    interp_selected_idx = None
    if bool(interp_cfg.get("enable", True)):
        interp_selected_device, interp_selected_idx = _normalize_cuda_device(
            interp_cfg.get("device", "cuda"),
            field_name="decompression.interpolate.device",
        )
        interp_cfg["device"] = str(interp_selected_device)
        dec_cfg["interpolate"] = interp_cfg
    return {
        "runtime_mode": "strict_gpu_only",
        "dcvc_device_selected": str(dcvc_cfg.get("device", "cuda")),
        "dcvc_cuda_idx_selected": dcvc_selected_idx,
        "dcvc_use_cuda_selected": bool(dcvc_cfg.get("use_cuda", True)),
        "interpolate_enabled": bool((dec_cfg.get("interpolate", {}) or {}).get("enable", True)),
        "interpolate_device_selected": interp_selected_device,
        "interpolate_cuda_idx_selected": interp_selected_idx,
    }


def _apply_dcvc_overrides(meta: Dict[str, Any], dec_cfg: Dict[str, Any]) -> None:
    dcvc_cfg = (meta.get("dcvc", {}) or {}).copy()
    dcvc_override = dec_cfg.get("dcvc", {}) or {}
    if not isinstance(dcvc_override, dict):
        meta["dcvc"] = dcvc_cfg
        return
    device_overridden = False
    cuda_idx_overridden = False
    for key in ("repo_dir", "model_i", "model_p", "device", "use_cuda", "cuda_idx", "force_zero_thres"):
        value = dcvc_override.get(key, None)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        dcvc_cfg[key] = value
        if key == "device":
            device_overridden = True
        elif key == "cuda_idx":
            cuda_idx_overridden = True
    if device_overridden and not cuda_idx_overridden:
        dcvc_cfg.pop("cuda_idx", None)
    meta["dcvc"] = dcvc_cfg


def _pick_fps(meta: Dict[str, Any], frame_drop_json: Dict[str, Any]) -> float:
    v = meta.get("video", {}) or {}
    if v.get("fps"):
        return float(v["fps"])
    stats = frame_drop_json.get("stats", {}) or {}
    if stats.get("fps_in"):
        return float(stats["fps_in"])
    return 30.0


# ── Lossless video writers (used for evaluation export) ───────────────────────

class _LosslessEvalWriter:
    def __init__(self, *, out_path: Path, width: int, height: int, fps: float,
                 ffmpeg_bin: str = "ffmpeg") -> None:
        self.out_path = out_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(max(1e-6, fps))
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s:v", f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "-",
            "-an", "-c:v", "ffv1", "-level", "3", str(self.out_path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[:2] != (self.height, self.width):
            raise ValueError("LosslessEvalWriter: wrong frame size")
        if self._proc.stdin is None:
            raise RuntimeError("LosslessEvalWriter stdin not available")
        self._proc.stdin.write(frame_bgr.tobytes())

    def close(self) -> None:
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        stderr_tail = ""
        if self._proc.stderr is not None:
            try:
                stderr_tail = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
            except Exception:
                pass
        rc = self._proc.wait()
        if rc != 0:
            tail = "\n".join(stderr_tail.splitlines()[-20:]) if stderr_tail else "(no stderr)"
            raise RuntimeError(f"LosslessEvalWriter failed (exit {rc}).\n{tail}")


class _LosslessYuv420Writer:
    def __init__(self, *, out_path: Path, width: int, height: int, fps: float,
                 ffmpeg_bin: str = "ffmpeg") -> None:
        self.out_path = out_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(max(1e-6, fps))
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s:v", f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "-",
            "-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p", str(self.out_path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[:2] != (self.height, self.width):
            raise ValueError("LosslessYuv420Writer: wrong frame size")
        if self._proc.stdin is None:
            raise RuntimeError("LosslessYuv420Writer stdin not available")
        self._proc.stdin.write(frame_bgr.tobytes())

    def close(self) -> None:
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        stderr_tail = ""
        if self._proc.stderr is not None:
            try:
                stderr_tail = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
            except Exception:
                pass
        rc = self._proc.wait()
        if rc != 0:
            tail = "\n".join(stderr_tail.splitlines()[-20:]) if stderr_tail else "(no stderr)"
            raise RuntimeError(f"LosslessYuv420Writer failed (exit {rc}).\n{tail}")


# ── AMT frame interpolation ────────────────────────────────────────────────────

_AMT_MIN_SIDE = 128


def _interpolate_many_batched(
    interpolator: AmtInterpolator,
    frame0_bgr: np.ndarray,
    frame1_bgr: np.ndarray,
    count: int,
    batch_size: int,
    max_crop_side: int,
) -> List[np.ndarray]:
    n = max(0, int(count))
    if n <= 0:
        return []
    bsz = max(1, int(batch_size))
    if frame0_bgr.shape != frame1_bgr.shape:
        raise ValueError("AMT interpolation requires equal frame shapes")
    orig_h, orig_w = frame0_bgr.shape[:2]
    work0, work1 = frame0_bgr, frame1_bgr
    max_side = max(0, int(max_crop_side))
    if max_side > 0:
        current_max = max(orig_h, orig_w)
        if current_max > max_side:
            scale = float(max_side) / float(current_max)
            resized_w = max(1, int(round(orig_w * scale)))
            resized_h = max(1, int(round(orig_h * scale)))
            work0 = cv2.resize(frame0_bgr, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
            work1 = cv2.resize(frame1_bgr, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    work_h, work_w = work0.shape[:2]
    min_pad_h = max(0, _AMT_MIN_SIDE - work_h)
    min_pad_w = max(0, _AMT_MIN_SIDE - work_w)
    if min_pad_h > 0 or min_pad_w > 0:
        top = min_pad_h // 2
        bottom = min_pad_h - top
        left = min_pad_w // 2
        right = min_pad_w - left
        work0 = cv2.copyMakeBorder(work0, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        work1 = cv2.copyMakeBorder(work1, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    else:
        top = left = 0
    in0 = interpolator._bgr_to_tensor(work0).to(interpolator.device, non_blocking=True)
    in1 = interpolator._bgr_to_tensor(work1).to(interpolator.device, non_blocking=True)
    in0, pad = _pad_to_divisor(in0, interpolator.pad_to)
    in1, _ = _pad_to_divisor(in1, interpolator.pad_to)
    ts = [float(i) / float(n + 1) for i in range(1, n + 1)]
    out: List[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, n, bsz):
            t_batch = ts[s: s + bsz]
            b = len(t_batch)
            embt = torch.tensor(t_batch, dtype=torch.float32,
                                device=interpolator.device).view(b, 1, 1, 1)
            in0_b = in0.repeat(b, 1, 1, 1)
            in1_b = in1.repeat(b, 1, 1, 1)
            if interpolator.fp16 and interpolator.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = interpolator.model(in0_b, in1_b, embt,
                                              scale_factor=1.0, eval=True)["imgt_pred"]
            else:
                pred = interpolator.model(in0_b, in1_b, embt,
                                          scale_factor=1.0, eval=True)["imgt_pred"]
            pred = _unpad(pred, pad)
            for j in range(b):
                mid = interpolator._tensor_to_bgr(pred[j: j + 1])
                if min_pad_h > 0 or min_pad_w > 0:
                    mid = mid[top: top + work_h, left: left + work_w]
                if mid.shape[0] != orig_h or mid.shape[1] != orig_w:
                    mid = cv2.resize(mid, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                out.append(mid)
            del embt, in0_b, in1_b, pred
    del in0, in1
    return out


# ── Anchor/frame helpers ───────────────────────────────────────────────────────

def _normalize_anchor_positions(
    indices: Sequence[int], frame_count: int
) -> Tuple[List[int], List[int]]:
    n = min(len(indices), int(frame_count))
    if n <= 0:
        return [], []
    lookup: Dict[int, int] = {}
    for pos in range(n):
        lookup[int(indices[pos])] = int(pos)
    keys = sorted(lookup.keys())
    return keys, [lookup[k] for k in keys]


def _filter_anchor_positions(
    indices: Sequence[int], positions: Sequence[int], total_frames: int
) -> Tuple[List[int], List[int]]:
    out_i: List[int] = []
    out_p: List[int] = []
    for idx, pos in zip(indices, positions):
        i = int(idx)
        if 0 <= i < int(total_frames):
            out_i.append(i)
            out_p.append(int(pos))
    return out_i, out_p


def _cache_put(cache: Dict[int, np.ndarray], key: int, value: np.ndarray,
               max_items: int) -> np.ndarray:
    cache[int(key)] = value
    if len(cache) > int(max_items):
        cache.pop(next(iter(cache)))
    return value


def _load_store_frame(
    store: Sequence[Any], positions: Sequence[int], slot: int,
    width: int, height: int, cache: Dict[int, np.ndarray],
    max_cache_items: int = 4,
) -> np.ndarray:
    cached = cache.get(int(slot), None)
    if cached is not None:
        return cached
    raw = store[int(positions[int(slot)])]
    if isinstance(raw, (str, Path)):
        frame = cv2.imread(str(raw), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read cached decoded frame: {raw}")
    else:
        frame = raw
    frame = rd._resize_if_needed(frame, width=width, height=height)
    return _cache_put(cache, int(slot), frame, max_cache_items)


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


def _frame_linear_at(
    frame_idx: int, anchor_indices: Sequence[int], anchor_positions: Sequence[int],
    store: Sequence[Any], width: int, height: int, cache: Dict[int, np.ndarray],
) -> np.ndarray:
    if not anchor_indices:
        raise RuntimeError("Cannot interpolate from an empty anchor set")
    t = int(frame_idx)
    right = bisect_right(anchor_indices, t)
    if right <= 0:
        return _load_store_frame(store, anchor_positions, 0, width, height, cache)
    if right >= len(anchor_indices):
        return _load_store_frame(store, anchor_positions, len(anchor_indices) - 1,
                                 width, height, cache)
    left_slot = int(right - 1)
    right_slot = int(right)
    li = int(anchor_indices[left_slot])
    ri = int(anchor_indices[right_slot])
    left = _load_store_frame(store, anchor_positions, left_slot, width, height, cache)
    if t <= li or ri <= li:
        return left
    right_frame = _load_store_frame(store, anchor_positions, right_slot, width, height, cache)
    alpha = float(t - li) / float(ri - li)
    return np.clip(
        left.astype(np.float32) * (1.0 - alpha) + right_frame.astype(np.float32) * alpha,
        0, 255,
    ).astype(np.uint8)


def _bbox_from_mask(
    mask: np.ndarray, margin: int, width: int, height: int
) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x0 = max(0, int(xs.min()) - int(margin))
    y0 = max(0, int(ys.min()) - int(margin))
    x1 = min(int(width), int(xs.max()) + 1 + int(margin))
    y1 = min(int(height), int(ys.max()) + 1 + int(margin))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, float(v))))


def _frame_motion_urgency(frame_drop_json: Dict[str, Any], frame_idx: int) -> float:
    pf_all = frame_drop_json.get("per_frame", {}) or {}
    pf = pf_all.get(str(int(frame_idx)), {}) or {}
    if not isinstance(pf, dict):
        return 0.0
    state = str(pf.get("state", "STILL")).strip().upper()
    default_u = 1.0 if state == "MOTION" else 0.0
    try:
        u = float(pf.get("motion_urgency", default_u))
    except (TypeError, ValueError):
        u = default_u
    return _clamp01(u)


def _mask_at(
    frame_idx: int, *, width: int, height: int, mask_source: str,
    roi_boxes_map: Dict[str, Any], frame_drop_json: Dict[str, Any],
    roi_min_conf: float, roi_dilate_px: int,
    cache: Dict[int, np.ndarray], max_cache_items: int = 32,
) -> np.ndarray:
    cached = cache.get(int(frame_idx), None)
    if cached is not None:
        return cached
    mask = rd._frame_mask(
        frame_idx=int(frame_idx), width=width, height=height,
        mask_source=mask_source, roi_boxes_map=roi_boxes_map,
        frame_drop_json=frame_drop_json, roi_min_conf=roi_min_conf,
        roi_dilate_px=roi_dilate_px,
    )
    return _cache_put(cache, int(frame_idx), mask, max_cache_items)


def _alpha_at(
    frame_idx: int, *, width: int, height: int, mask_source: str,
    roi_boxes_map: Dict[str, Any], frame_drop_json: Dict[str, Any],
    roi_min_conf: float, roi_dilate_px: int, blend_edge_px: int,
    mask_cache: Dict[int, np.ndarray], alpha_cache: Dict[int, np.ndarray],
    max_cache_items: int = 32,
) -> np.ndarray:
    cached = alpha_cache.get(int(frame_idx), None)
    if cached is not None:
        return cached
    mask = _mask_at(
        frame_idx, width=width, height=height, mask_source=mask_source,
        roi_boxes_map=roi_boxes_map, frame_drop_json=frame_drop_json,
        roi_min_conf=roi_min_conf, roi_dilate_px=roi_dilate_px, cache=mask_cache,
    )
    alpha = mask_to_alpha(mask, int(blend_edge_px))
    return _cache_put(alpha_cache, int(frame_idx), alpha, max_cache_items)


def _anchor_context_at(
    *, slot: int,
    roi_indices: Sequence[int], roi_positions: Sequence[int], roi_store: Sequence[Any],
    bg_indices: Sequence[int], bg_positions: Sequence[int], bg_store: Sequence[Any],
    width: int, height: int, mask_source: str,
    roi_boxes_map: Dict[str, Any], frame_drop_json: Dict[str, Any],
    roi_min_conf: float, roi_dilate_px: int, roi_blend_edge_px: int,
    bg_cache: Dict[int, np.ndarray], roi_cache: Dict[int, np.ndarray],
    context_cache: Dict[int, np.ndarray],
    mask_cache: Dict[int, np.ndarray], alpha_cache: Dict[int, np.ndarray],
) -> np.ndarray:
    cached = context_cache.get(int(slot), None)
    if cached is not None:
        return cached
    frame_idx = int(roi_indices[int(slot)])
    roi_frame = _load_store_frame(roi_store, roi_positions, int(slot), width, height, roi_cache)
    bg_frame = _frame_linear_at(frame_idx, bg_indices, bg_positions, bg_store, width, height,
                                 bg_cache)
    alpha = _alpha_at(
        frame_idx, width=width, height=height, mask_source=mask_source,
        roi_boxes_map=roi_boxes_map, frame_drop_json=frame_drop_json,
        roi_min_conf=roi_min_conf, roi_dilate_px=roi_dilate_px,
        blend_edge_px=int(roi_blend_edge_px),
        mask_cache=mask_cache, alpha_cache=alpha_cache,
    )
    anchor = rd._compose_soft(roi_frame, bg_frame, alpha)
    return _cache_put(context_cache, int(slot), anchor, 4)


def _build_roi_segment(
    *, left_slot: int, right_slot: int,
    roi_indices: Sequence[int], roi_positions: Sequence[int], roi_store: Sequence[Any],
    bg_indices: Sequence[int], bg_positions: Sequence[int], bg_store: Sequence[Any],
    width: int, height: int,
    interpolator: Optional[AmtInterpolator],
    batch_size: int, crop_margin: int, max_crop_side: int,
    mask_source: str, roi_boxes_map: Dict[str, Any], frame_drop_json: Dict[str, Any],
    roi_min_conf: float, roi_dilate_px: int, roi_blend_edge_px: int,
    bg_cache: Dict[int, np.ndarray], roi_cache: Dict[int, np.ndarray],
    context_cache: Dict[int, np.ndarray],
    mask_cache: Dict[int, np.ndarray], alpha_cache: Dict[int, np.ndarray],
) -> Dict[str, Any]:
    li = int(roi_indices[int(left_slot)])
    ri = int(roi_indices[int(right_slot)])
    gap = int(ri - li - 1)
    segment: Dict[str, Any] = {"li": li, "ri": ri, "gap": gap, "bbox": None, "mids": []}
    if gap <= 0:
        return segment
    union = np.zeros((height, width), dtype=np.uint8)
    for t in range(li, ri + 1):
        union = cv2.bitwise_or(
            union,
            _mask_at(t, width=width, height=height, mask_source=mask_source,
                     roi_boxes_map=roi_boxes_map, frame_drop_json=frame_drop_json,
                     roi_min_conf=roi_min_conf, roi_dilate_px=roi_dilate_px,
                     cache=mask_cache),
        )
    bbox = _bbox_from_mask(union, margin=int(crop_margin), width=width, height=height)
    segment["bbox"] = bbox
    if bbox is None:
        return segment
    x0, y0, x1, y1 = bbox
    ctx_kwargs = dict(
        roi_indices=roi_indices, roi_positions=roi_positions, roi_store=roi_store,
        bg_indices=bg_indices, bg_positions=bg_positions, bg_store=bg_store,
        width=width, height=height, mask_source=mask_source,
        roi_boxes_map=roi_boxes_map, frame_drop_json=frame_drop_json,
        roi_min_conf=roi_min_conf, roi_dilate_px=roi_dilate_px,
        roi_blend_edge_px=int(roi_blend_edge_px),
        bg_cache=bg_cache, roi_cache=roi_cache, context_cache=context_cache,
        mask_cache=mask_cache, alpha_cache=alpha_cache,
    )
    left_ctx = _anchor_context_at(slot=int(left_slot), **ctx_kwargs)
    right_ctx = _anchor_context_at(slot=int(right_slot), **ctx_kwargs)
    left_crop = left_ctx[y0:y1, x0:x1]
    right_crop = right_ctx[y0:y1, x0:x1]
    if interpolator is None:
        mids: List[np.ndarray] = []
        for j in range(1, gap + 1):
            a = float(j) / float(gap + 1)
            mids.append(cv2.addWeighted(left_crop, 1.0 - a, right_crop, a, 0.0))
        segment["mids"] = mids
        return segment
    segment["mids"] = _interpolate_many_batched(
        interpolator, left_crop, right_crop, gap,
        batch_size=batch_size, max_crop_side=max_crop_side,
    )
    return segment


# ── Public API ─────────────────────────────────────────────────────────────────

def decompress_archive(
    archive_path: Path,
    dec_cfg_raw: Dict[str, Any],
    output_path: Optional[Path] = None,
    *,
    no_interpolate: bool = False,
    no_roi_stabilize: bool = False,
    roi_alpha_still: Optional[float] = None,
    roi_alpha_motion: Optional[float] = None,
    roi_mask_dilate: Optional[int] = None,
    roi_stabilize_overlap_only: bool = False,
    amt_batch_size: Optional[int] = None,
    amt_crop_margin: Optional[int] = None,
    max_frames: int = 0,
    lossless_output: Optional[Path] = None,
    lossless_yuv420_output: Optional[Path] = None,
    strict_gpu: bool = True,
) -> Tuple[List[np.ndarray], float, int, int, Dict[str, Any]]:
    """
    Decompress a codec archive, returning (frames, fps, width, height, roi_detections).
    If output_path is given, also writes a reconstructed video to disk.

    dec_cfg_raw: the inner 'decompression' config dict, or an outer dict with a
                 'decompression' key (the full YAML content).
    strict_gpu: require CUDA to be available; set False when calling from within
                the pipeline where GPU context is already established.
    """
    archive_path = Path(archive_path).expanduser().resolve()

    if "decompression" in dec_cfg_raw:
        dec_cfg = rd._validate_cfg(dec_cfg_raw)
    else:
        dec_cfg = rd._validate_cfg({"decompression": dec_cfg_raw})

    if no_interpolate:
        interp = (dec_cfg.get("interpolate", {}) or {}).copy()
        interp["enable"] = False
        dec_cfg["interpolate"] = interp
    if no_roi_stabilize:
        dec_cfg["roi_temporal_stabilize"] = False
    if roi_alpha_still is not None:
        dec_cfg["roi_temporal_alpha_still"] = float(roi_alpha_still)
    if roi_alpha_motion is not None:
        dec_cfg["roi_temporal_alpha_motion"] = float(roi_alpha_motion)
    if roi_mask_dilate is not None:
        dec_cfg["roi_temporal_mask_dilate"] = int(roi_mask_dilate)
    if roi_stabilize_overlap_only:
        dec_cfg["roi_temporal_overlap_only"] = True

    roi_temporal_stabilize = bool(dec_cfg.get("roi_temporal_stabilize", True))
    roi_alpha_still_v = _clamp01(float(dec_cfg.get("roi_temporal_alpha_still", 0.70)))
    roi_alpha_motion_v = _clamp01(float(dec_cfg.get("roi_temporal_alpha_motion", 0.92)))
    if roi_alpha_motion_v < roi_alpha_still_v:
        roi_alpha_still_v, roi_alpha_motion_v = roi_alpha_motion_v, roi_alpha_still_v
    roi_blend_edge_px = max(0, int(dec_cfg.get("roi_blend_edge_px", 2)))
    roi_temporal_mask_dilate = max(0, int(dec_cfg.get("roi_temporal_mask_dilate", 1)))
    roi_temporal_overlap_only = bool(dec_cfg.get("roi_temporal_overlap_only", True))

    interp_cfg = dec_cfg.get("interpolate", {}) or {}
    amt_batch_size_v = int(amt_batch_size) if amt_batch_size is not None \
        else int(interp_cfg.get("batch_size", 1))
    amt_crop_margin_v = int(amt_crop_margin) if amt_crop_margin is not None \
        else int(interp_cfg.get("crop_margin", 8))
    amt_max_crop_side = int(interp_cfg.get("max_crop_side", 768))

    payloads = rd._load_archive_payloads(archive_path)
    meta = json.loads(payloads["meta.json"].decode("utf-8"))
    _apply_dcvc_overrides(meta, dec_cfg)

    if strict_gpu:
        _enforce_strict_gpu_runtime(dec_cfg, meta)

    roi_json = json.loads(payloads["roi_detections.json"].decode("utf-8"))
    frame_drop_json = json.loads(payloads["frame_drop.json"].decode("utf-8"))

    roi_indices_raw = rd._pick_stream_indices(frame_drop_json, meta, "roi")
    bg_indices_raw = rd._pick_stream_indices(frame_drop_json, meta, "bg")
    _codec_dbg = str(meta.get("codec", "dcvc")).lower()
    if _codec_dbg != "dcvc":
        print(f"[decomp] {_codec_dbg}: roi_indices len={len(roi_indices_raw)}"
              f" first5={roi_indices_raw[:5]} last5={roi_indices_raw[-5:]}", flush=True)
        print(f"[decomp] {_codec_dbg}: bg_indices  len={len(bg_indices_raw)}"
              f" first5={bg_indices_raw[:5]} last5={bg_indices_raw[-5:]}", flush=True)
    roi_decode_limit: Optional[int] = None
    bg_decode_limit: Optional[int] = None
    if int(max_frames) > 0:
        roi_decode_limit = max(1, sum(1 for idx in roi_indices_raw if int(idx) < max_frames))
        bg_decode_limit = max(1, sum(1 for idx in bg_indices_raw if int(idx) < max_frames))

    writer = None
    lossless_writer: Optional[_LosslessEvalWriter] = None
    lossless_yuv420_writer: Optional[_LosslessYuv420Writer] = None
    decode_tmp_dir = Path(tempfile.mkdtemp(prefix="decomp_decode_"))
    try:
        roi_store, roi_frame_count, bg_store, bg_frame_count = decode_roi_bg_streams_to_memmap(
            payloads["roi.bin"], payloads["bg.bin"], meta,
            work_dir=decode_tmp_dir,
            max_frames_roi=roi_decode_limit,
            max_frames_bg=bg_decode_limit,
        )

        if int(roi_frame_count) <= 0 or int(bg_frame_count) <= 0:
            raise RuntimeError("Decoded zero ROI/BG frames from archive")

        roi_indices, roi_positions = _normalize_anchor_positions(roi_indices_raw, roi_frame_count)
        bg_indices, bg_positions = _normalize_anchor_positions(bg_indices_raw, bg_frame_count)
        if not roi_indices or not bg_indices:
            raise RuntimeError("Missing ROI/BG anchors after alignment")

        video_info = meta.get("video", {}) or {}
        ref_w = int(video_info.get("width", 0) or
                    (roi_store[0].shape[1] if roi_frame_count > 0 else 0))
        ref_h = int(video_info.get("height", 0) or
                    (roi_store[0].shape[0] if roi_frame_count > 0 else 0))
        if ref_w <= 0 or ref_h <= 0:
            raise RuntimeError("Invalid width/height in archive metadata")

        total_frames = rd._infer_total_frames(meta, frame_drop_json, roi_indices, bg_indices)
        total_frames = max(total_frames, max(max(roi_indices), max(bg_indices)) + 1)
        if int(max_frames) > 0:
            total_frames = min(total_frames, int(max_frames))
        if total_frames <= 0:
            raise RuntimeError("No frames to reconstruct")

        roi_indices, roi_positions = _filter_anchor_positions(roi_indices, roi_positions,
                                                               total_frames)
        bg_indices, bg_positions = _filter_anchor_positions(bg_indices, bg_positions, total_frames)
        if not roi_indices or not bg_indices:
            raise RuntimeError("No valid ROI/BG anchors in selected frame window")

        raw_mask_source = str(dec_cfg.get("mask_source", "roi_detection")).strip().lower()
        mask_source = "frame_drop_roi_box" if raw_mask_source == "frame_drop_roi_box" \
            else "roi_detection"

        roi_boxes_map = roi_json.get("frames", {}) or {}
        roi_cfg = meta.get("roi", {}) or {}
        roi_min_conf = float(roi_cfg.get("min_conf", 0.25))
        roi_dilate_px = int(roi_cfg.get("visible_dilate_px", roi_cfg.get("dilate_px", 0)))
        fps = _pick_fps(meta, frame_drop_json)

        interpolation_requested = bool(interp_cfg.get("enable", True))
        roi_interpolator: Optional[AmtInterpolator] = None
        roi_mode = "linear"
        if interpolation_requested:
            roi_interpolator, _ = rd._init_amt_interpolator(interp_cfg)
            roi_mode = "amt"

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*str(dec_cfg["codec"])),
                max(1e-6, float(fps)),
                (int(ref_w), int(ref_h)),
                True,
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter: {output_path}")
        if lossless_output is not None:
            lossless_writer = _LosslessEvalWriter(
                out_path=Path(lossless_output), width=ref_w, height=ref_h, fps=fps)
        if lossless_yuv420_output is not None:
            lossless_yuv420_writer = _LosslessYuv420Writer(
                out_path=Path(lossless_yuv420_output), width=ref_w, height=ref_h, fps=fps)

        bg_frame_cache: Dict[int, np.ndarray] = {}
        roi_frame_cache: Dict[int, np.ndarray] = {}
        roi_context_cache: Dict[int, np.ndarray] = {}
        mask_cache: Dict[int, np.ndarray] = {}
        alpha_cache: Dict[int, np.ndarray] = {}
        roi_slot_by_frame = {int(idx): slot for slot, idx in enumerate(roi_indices)}
        current_segment: Optional[Dict[str, Any]] = None
        stab_kernel = None
        if roi_temporal_mask_dilate > 0:
            k = max(3, 2 * int(roi_temporal_mask_dilate) + 1)
            stab_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        frames_out: List[np.ndarray] = []
        prev_out: Optional[np.ndarray] = None
        prev_mask: Optional[np.ndarray] = None

        for t in range(total_frames):
            bg_f = _frame_linear_at(t, bg_indices, bg_positions, bg_store,
                                    ref_w, ref_h, bg_frame_cache)
            mask = _mask_at(t, width=ref_w, height=ref_h, mask_source=mask_source,
                            roi_boxes_map=roi_boxes_map, frame_drop_json=frame_drop_json,
                            roi_min_conf=roi_min_conf, roi_dilate_px=roi_dilate_px,
                            cache=mask_cache)

            if int(np.count_nonzero(mask)) == 0:
                out = bg_f.copy()
            else:
                _ctx_kwargs = dict(
                    roi_indices=roi_indices, roi_positions=roi_positions, roi_store=roi_store,
                    bg_indices=bg_indices, bg_positions=bg_positions, bg_store=bg_store,
                    width=ref_w, height=ref_h, mask_source=mask_source,
                    roi_boxes_map=roi_boxes_map, frame_drop_json=frame_drop_json,
                    roi_min_conf=roi_min_conf, roi_dilate_px=roi_dilate_px,
                    roi_blend_edge_px=int(roi_blend_edge_px),
                    bg_cache=bg_frame_cache, roi_cache=roi_frame_cache,
                    context_cache=roi_context_cache,
                    mask_cache=mask_cache, alpha_cache=alpha_cache,
                )
                slot = roi_slot_by_frame.get(int(t), None)
                if slot is not None:
                    out = _anchor_context_at(slot=int(slot), **_ctx_kwargs)
                elif t < int(roi_indices[0]):
                    out = _anchor_context_at(slot=0, **_ctx_kwargs)
                elif t > int(roi_indices[-1]):
                    out = _anchor_context_at(slot=len(roi_indices) - 1, **_ctx_kwargs)
                else:
                    right = bisect_right(roi_indices, int(t))
                    left_slot = max(0, int(right - 1))
                    right_slot = min(len(roi_indices) - 1, int(right))
                    li = int(roi_indices[left_slot])
                    ri = int(roi_indices[right_slot])
                    if right_slot == left_slot or t <= li or t >= ri:
                        out = _anchor_context_at(
                            slot=int(left_slot if t <= li else right_slot), **_ctx_kwargs)
                    else:
                        if (current_segment is None
                                or int(current_segment.get("li", -1)) != li
                                or int(current_segment.get("ri", -1)) != ri):
                            current_segment = _build_roi_segment(
                                left_slot=int(left_slot), right_slot=int(right_slot),
                                roi_indices=roi_indices, roi_positions=roi_positions,
                                roi_store=roi_store, bg_indices=bg_indices,
                                bg_positions=bg_positions, bg_store=bg_store,
                                width=ref_w, height=ref_h,
                                interpolator=(roi_interpolator if roi_mode == "amt" else None),
                                batch_size=int(amt_batch_size_v),
                                crop_margin=int(amt_crop_margin_v),
                                max_crop_side=int(amt_max_crop_side),
                                mask_source=mask_source, roi_boxes_map=roi_boxes_map,
                                frame_drop_json=frame_drop_json,
                                roi_min_conf=roi_min_conf, roi_dilate_px=roi_dilate_px,
                                roi_blend_edge_px=int(roi_blend_edge_px),
                                bg_cache=bg_frame_cache, roi_cache=roi_frame_cache,
                                context_cache=roi_context_cache,
                                mask_cache=mask_cache, alpha_cache=alpha_cache,
                            )
                        roi_source = bg_f.copy()
                        bbox = current_segment.get("bbox") if current_segment is not None else None
                        mids = current_segment.get("mids", []) if current_segment is not None \
                            else []
                        mid_idx = int(t - li - 1)
                        if bbox is not None and 0 <= mid_idx < len(mids):
                            x0, y0, x1, y1 = bbox
                            roi_source[y0:y1, x0:x1] = mids[mid_idx]
                        alpha = _alpha_at(
                            t, width=ref_w, height=ref_h, mask_source=mask_source,
                            roi_boxes_map=roi_boxes_map, frame_drop_json=frame_drop_json,
                            roi_min_conf=roi_min_conf, roi_dilate_px=roi_dilate_px,
                            blend_edge_px=int(roi_blend_edge_px),
                            mask_cache=mask_cache, alpha_cache=alpha_cache,
                        )
                        out = rd._compose_soft(roi_source, bg_f, alpha)

            if roi_temporal_stabilize and prev_out is not None:
                stab_mask = cv2.dilate(mask, stab_kernel, iterations=1) \
                    if stab_kernel is not None else mask
                if roi_temporal_overlap_only and prev_mask is not None:
                    stab_bool = (stab_mask > 0) & (prev_mask > 0)
                else:
                    stab_bool = stab_mask > 0
                if bool(np.any(stab_bool)):
                    urg = _frame_motion_urgency(frame_drop_json, t)
                    a = _clamp01(roi_alpha_still_v + (roi_alpha_motion_v - roi_alpha_still_v) * urg)
                    cur_px = out[stab_bool].astype(np.float32)
                    prv_px = prev_out[stab_bool].astype(np.float32)
                    out[stab_bool] = np.clip(
                        a * cur_px + (1.0 - a) * prv_px, 0, 255).astype(np.uint8)

            frames_out.append(out)
            if writer is not None:
                writer.write(out)
            if lossless_writer is not None:
                lossless_writer.write(out)
            if lossless_yuv420_writer is not None:
                lossless_yuv420_writer.write(out)
            prev_out = out
            prev_mask = cv2.dilate(mask, stab_kernel, iterations=1) \
                if stab_kernel is not None else mask
            if current_segment is not None and t >= int(current_segment.get("ri", -1)):
                current_segment = None

    finally:
        if writer is not None:
            writer.release()
        if lossless_writer is not None:
            lossless_writer.close()
        if lossless_yuv420_writer is not None:
            lossless_yuv420_writer.close()
        _close_memmap(locals().get("roi_store"))
        _close_memmap(locals().get("bg_store"))
        if decode_tmp_dir.exists():
            shutil.rmtree(decode_tmp_dir, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return frames_out, fps, ref_w, ref_h, roi_json


def decompress_archive_bytes(
    archive_bytes: bytes,
    dec_cfg_raw: Dict[str, Any],
    output_path: Optional[Path] = None,
    **kwargs,
) -> Tuple[List[np.ndarray], float, int, int, Dict[str, Any]]:
    """Convenience wrapper: write archive bytes to a temp file, then decompress."""
    kwargs.setdefault("strict_gpu", False)
    with tempfile.TemporaryDirectory(prefix="decomp_arch_") as td:
        archive_path = Path(td) / "archive.zip"
        archive_path.write_bytes(archive_bytes)
        return decompress_archive(archive_path, dec_cfg_raw, output_path, **kwargs)
