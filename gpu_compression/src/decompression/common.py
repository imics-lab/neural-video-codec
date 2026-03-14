from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .interpolation_amt import AmtInterpolator

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("wildroi.decompression")
VERBOSE_LOGS = False


def _setup_logging(verbose: bool = False) -> None:
    global VERBOSE_LOGS
    VERBOSE_LOGS = bool(verbose)
    level = logging.INFO if VERBOSE_LOGS else logging.ERROR
    logging.basicConfig(level=level, format="%(message)s", force=True)
    LOGGER.setLevel(level)


def _log(event: str, **kwargs: Any) -> None:
    if not VERBOSE_LOGS:
        return
    LOGGER.info(json.dumps({"event": event, **kwargs}, separators=(",", ":")))


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return bool(default)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on", "cuda"}:
        return True
    if s in {"0", "false", "no", "n", "off", "cpu", "none", "null", ""}:
        return False
    return bool(value)


def _load_runtime_cfg(cfg_path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format: {cfg_path}")
    return data


def _validate_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dec = cfg.get("decompression", {}) or {}
    if not isinstance(dec, dict):
        raise ValueError("decompression config must be an object")
    out = dict(dec)
    out.setdefault("codec", "mp4v")
    out.setdefault("mask_source", "roi_detection")
    # Temporal ROI stabilization to reduce patch flicker/jitter.
    out.setdefault("roi_temporal_stabilize", True)
    out.setdefault("roi_temporal_alpha_still", 0.70)
    out.setdefault("roi_temporal_alpha_motion", 0.92)
    out.setdefault("roi_temporal_mask_dilate", 1)
    out.setdefault("roi_temporal_overlap_only", True)
    interp = out.get("interpolate", {}) or {}
    if not isinstance(interp, dict):
        interp = {}
    interp.setdefault("enable", True)
    interp.setdefault("model", "amt-s")
    interp.setdefault("weights_path", None)
    interp.setdefault("repo_dir", "_third_party_amt")
    interp.setdefault("device", "cuda")
    interp.setdefault("fp16", True)
    interp.setdefault("pad_to", 16)
    interp.setdefault("batch_size", 1)
    interp.setdefault("crop_margin", 8)
    interp.setdefault("max_crop_side", 768)
    interp_dev = str(interp.get("device", "cuda")).strip().lower()
    if interp_dev in {"cpu", "mps"}:
        raise ValueError("Strict GPU runtime forbids decompression.interpolate.device set to CPU/MPS.")
    if interp_dev not in {"", "auto", "cuda"} and not interp_dev.startswith("cuda:"):
        raise ValueError(
            "decompression.interpolate.device must be one of: auto, cuda, cuda:<index> for strict GPU runtime."
        )
    try:
        interp_batch_size = int(interp.get("batch_size", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("decompression.interpolate.batch_size must be an integer >= 1") from exc
    if interp_batch_size < 1:
        raise ValueError("decompression.interpolate.batch_size must be >= 1")
    interp["batch_size"] = int(interp_batch_size)
    try:
        interp_crop_margin = int(interp.get("crop_margin", 8))
    except (TypeError, ValueError) as exc:
        raise ValueError("decompression.interpolate.crop_margin must be an integer >= 0") from exc
    if interp_crop_margin < 0:
        raise ValueError("decompression.interpolate.crop_margin must be >= 0")
    interp["crop_margin"] = int(interp_crop_margin)
    try:
        interp_max_crop_side = int(interp.get("max_crop_side", 768))
    except (TypeError, ValueError) as exc:
        raise ValueError("decompression.interpolate.max_crop_side must be an integer >= 0") from exc
    if interp_max_crop_side < 0:
        raise ValueError("decompression.interpolate.max_crop_side must be >= 0")
    interp["max_crop_side"] = int(interp_max_crop_side)
    out["interpolate"] = interp

    dcvc = out.get("dcvc", {}) or {}
    if not isinstance(dcvc, dict):
        dcvc = {}
    if "use_cuda" in dcvc and _coerce_bool(dcvc.get("use_cuda"), default=True) is False:
        raise ValueError("Strict GPU runtime forbids decompression.dcvc.use_cuda=false.")
    dcvc_dev = str(dcvc.get("device", "cuda")).strip().lower()
    if dcvc_dev in {"cpu", "mps"}:
        raise ValueError("Strict GPU runtime forbids decompression.dcvc.device set to CPU/MPS.")
    out["dcvc"] = dcvc
    return out


def _load_archive_payloads(archive_path: Path) -> Dict[str, bytes]:
    required = ["meta.json", "roi_detections.json", "frame_drop.json", "roi.bin", "bg.bin"]
    out: Dict[str, bytes] = {}
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = set(zf.namelist())
        missing = [n for n in required if n not in names]
        if missing:
            raise FileNotFoundError(f"Missing archive entries: {missing}")
        for name in required:
            out[name] = zf.read(name)
    return out


def _default_output_name(archive_path: Path, meta: Optional[Dict[str, Any]] = None) -> str:
    name = f"{archive_path.stem}.mp4"
    if isinstance(meta, dict):
        v = meta.get("video", {}) or {}
        src = str(v.get("path", "") or "").strip()
        if src:
            # Handle Windows-style paths stored in metadata even when running on Linux containers.
            if "\\" in src or (len(src) >= 2 and src[1] == ":"):
                stem = PureWindowsPath(src).stem
            else:
                stem = Path(src).stem
            if stem:
                name = f"{stem}.mp4"
    return name


def _resolve_output_path(archive_path: Path, dec_cfg: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Path:
    override = dec_cfg.get("output_path", None)
    name = _default_output_name(archive_path, meta=meta)
    if override:
        raw = str(override)
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if not p.suffix:
            raise ValueError(
                "decompression output_path must include a video filename, not just a directory. "
                f"Example: outputs/decompression/{name}"
            )
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    p = archive_path.parent / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _pick_stream_indices(frame_drop_json: Dict[str, Any], meta: Dict[str, Any], stream: str) -> List[int]:
    key = "roi_kept_frames" if stream == "roi" else "bg_kept_frames"
    arr = frame_drop_json.get(key, None)
    if isinstance(arr, list) and arr:
        return [int(x) for x in arr]
    stream_meta = (meta.get("streams", {}) or {}).get(stream, {}) or {}
    m = stream_meta.get("frame_index_map", None)
    if isinstance(m, list) and m:
        return [int(x) for x in m]
    return []


def _align_indices_and_frames(indices: Sequence[int], frames: Sequence[np.ndarray]) -> Tuple[List[int], List[np.ndarray]]:
    n = min(len(indices), len(frames))
    if n <= 0:
        return [], []
    ii = [int(indices[i]) for i in range(n)]
    ff = [frames[i] for i in range(n)]
    pairs = sorted(zip(ii, ff), key=lambda x: x[0])
    dedup: Dict[int, np.ndarray] = {}
    for idx, fr in pairs:
        dedup[idx] = fr
    keys = sorted(dedup.keys())
    return keys, [dedup[k] for k in keys]


def _infer_total_frames(meta: Dict[str, Any], frame_drop_json: Dict[str, Any], roi_indices: Sequence[int], bg_indices: Sequence[int]) -> int:
    v = meta.get("video", {}) or {}
    n = int(v.get("frames_total", 0) or 0)
    if n > 0:
        return n
    stats = frame_drop_json.get("stats", {}) or {}
    n = int(stats.get("num_frames_read", 0) or 0)
    if n > 0:
        return n
    return max(max(roi_indices or [0]), max(bg_indices or [0])) + 1


def _filter_anchors_in_range(indices: Sequence[int], frames: Sequence[np.ndarray], total_frames: int) -> Tuple[List[int], List[np.ndarray]]:
    out_i: List[int] = []
    out_f: List[np.ndarray] = []
    for idx, fr in zip(indices, frames):
        i = int(idx)
        if 0 <= i < int(total_frames):
            out_i.append(i)
            out_f.append(fr)
    return _align_indices_and_frames(out_i, out_f)


def _resize_if_needed(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == int(width) and frame.shape[0] == int(height):
        return frame
    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def _interpolate_timeline_linear(anchor_indices: Sequence[int], anchor_frames: Sequence[np.ndarray], total_frames: int, progress_desc: str = "") -> List[np.ndarray]:
    _ = progress_desc
    if not anchor_indices or not anchor_frames:
        return []
    out: Dict[int, np.ndarray] = {int(i): f for i, f in zip(anchor_indices, anchor_frames)}
    n = min(len(anchor_indices), len(anchor_frames))
    for i in range(n - 1):
        li = int(anchor_indices[i])
        ri = int(anchor_indices[i + 1])
        lf = anchor_frames[i].astype(np.float32)
        rf = anchor_frames[i + 1].astype(np.float32)
        gap = max(0, ri - li - 1)
        if gap <= 0:
            continue
        denom = float(gap + 1)
        for j in range(1, gap + 1):
            a = float(j) / denom
            out[li + j] = np.clip(lf * (1.0 - a) + rf * a, 0, 255).astype(np.uint8)
    first_i = int(anchor_indices[0])
    last_i = int(anchor_indices[n - 1])
    for t in range(0, first_i):
        out[t] = anchor_frames[0]
    for t in range(last_i + 1, int(total_frames)):
        out[t] = anchor_frames[n - 1]
    return [out[t] for t in range(int(total_frames))]


def _frame_mask(
    *,
    frame_idx: int,
    width: int,
    height: int,
    mask_source: str,
    roi_boxes_map: Dict[str, Any],
    frame_drop_json: Dict[str, Any],
    roi_min_conf: float,
    roi_dilate_px: int,
) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    src = str(mask_source or "roi_detection").strip().lower()
    if src == "frame_drop_roi_box":
        pf = (frame_drop_json.get("per_frame", {}) or {}).get(str(int(frame_idx)), {}) or {}
        if bool(pf.get("bbox_missing", False)):
            return mask
        b = pf.get("roi_box", {}) or {}
        x1 = max(0, min(int(width - 1), int(b.get("x1", 0))))
        y1 = max(0, min(int(height - 1), int(b.get("y1", 0))))
        x2 = max(0, min(int(width - 1), int(b.get("x2", 0))))
        y2 = max(0, min(int(height - 1), int(b.get("y2", 0))))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    else:
        boxes = roi_boxes_map.get(str(int(frame_idx)), roi_boxes_map.get(int(frame_idx), [])) or []
        for b in boxes:
            if not isinstance(b, dict):
                continue
            conf = float(b.get("conf", b.get("confidence", 1.0)))
            if conf < float(roi_min_conf):
                continue
            x1 = max(0, min(int(width - 1), int(b.get("x1", 0))))
            y1 = max(0, min(int(height - 1), int(b.get("y1", 0))))
            x2 = max(0, min(int(width - 1), int(b.get("x2", 0))))
            y2 = max(0, min(int(height - 1), int(b.get("y2", 0))))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    dpx = max(0, int(roi_dilate_px))
    if dpx > 0:
        k = max(3, 2 * dpx + 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, ker, iterations=1)
    return mask


def _compose_hard(roi_frame: np.ndarray, bg_frame: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    out = bg_frame.copy()
    m = mask_u8 > 0
    out[m] = roi_frame[m]
    return out


def _init_amt_interpolator(interp_cfg: Dict[str, Any]) -> Tuple[Optional[AmtInterpolator], bool]:
    if not bool(interp_cfg.get("enable", True)):
        return None, False
    variant = str(interp_cfg.get("model", "amt-s"))
    repo_dir = str((ROOT / str(interp_cfg.get("repo_dir", "_third_party_amt"))).resolve())
    weights_path_raw = interp_cfg.get("weights_path", None)
    if weights_path_raw:
        weights_path = str(Path(str(weights_path_raw)).expanduser().resolve())
    else:
        weights_name = "amt-l.pth" if variant == "amt-l" else "amt-s.pth"
        weights_path = str((ROOT / "models" / weights_name).resolve())
    device = str(interp_cfg.get("device", "auto"))
    fp16 = bool(interp_cfg.get("fp16", True))
    pad_to = int(interp_cfg.get("pad_to", 16))
    try:
        return AmtInterpolator(
            amt_repo_dir=repo_dir,
            variant=variant,
            weights_path=weights_path,
            device=device,
            fp16=fp16,
            pad_to=pad_to,
        ), False
    except Exception:
        if not fp16:
            raise
        return AmtInterpolator(
            amt_repo_dir=repo_dir,
            variant=variant,
            weights_path=weights_path,
            device=device,
            fp16=False,
            pad_to=pad_to,
        ), True
