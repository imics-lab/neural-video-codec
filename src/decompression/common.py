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
try:
    from roi_masking import build_frame_mask, compose_soft
except ImportError:  # pragma: no cover - test import path fallback
    from src.roi_masking import build_frame_mask, compose_soft

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("wildroi.decompression")
VERBOSE_LOGS = False
ARCHIVE_MANIFEST_NAME = "archive_manifest.json"
ARCHIVE_REQUIRED_ENTRIES = (
    "meta.json",
    "roi_detections.json",
    "frame_drop.json",
    "roi.bin",
    "bg.bin",
)


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
    out.setdefault("roi_blend_edge_px", 2)
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
    try:
        roi_blend_edge_px = int(out.get("roi_blend_edge_px", 2))
    except (TypeError, ValueError) as exc:
        raise ValueError("decompression.roi_blend_edge_px must be an integer >= 0") from exc
    if roi_blend_edge_px < 0:
        raise ValueError("decompression.roi_blend_edge_px must be >= 0")
    out["roi_blend_edge_px"] = int(roi_blend_edge_px)

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
    out: Dict[str, bytes] = {}
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = set(zf.namelist())
        entry_map = {name: name for name in ARCHIVE_REQUIRED_ENTRIES if name in names}
        missing = [name for name in ARCHIVE_REQUIRED_ENTRIES if name not in entry_map]
        if missing:
            if ARCHIVE_MANIFEST_NAME not in names:
                raise FileNotFoundError(f"Missing archive entries: {missing}")
            try:
                manifest = json.loads(zf.read(ARCHIVE_MANIFEST_NAME).decode("utf-8"))
            except Exception as exc:
                raise RuntimeError(f"Invalid archive manifest: {ARCHIVE_MANIFEST_NAME}") from exc
            entries = manifest.get("entries", {}) if isinstance(manifest, dict) else {}
            if not isinstance(entries, dict):
                raise RuntimeError(f"Invalid archive manifest: {ARCHIVE_MANIFEST_NAME}")
            entry_map = {}
            manifest_missing: List[str] = []
            for canonical_name in ARCHIVE_REQUIRED_ENTRIES:
                actual_name = entries.get(canonical_name, canonical_name)
                if not isinstance(actual_name, str) or not actual_name.strip():
                    manifest_missing.append(canonical_name)
                    continue
                if actual_name not in names:
                    manifest_missing.append(canonical_name)
                    continue
                entry_map[canonical_name] = actual_name
            if manifest_missing:
                raise FileNotFoundError(f"Missing archive entries: {manifest_missing}")
        for canonical_name in ARCHIVE_REQUIRED_ENTRIES:
            out[canonical_name] = zf.read(entry_map[canonical_name])
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
    stream_meta = (meta.get("streams", {}) or {}).get(stream, {}) or {}
    m = stream_meta.get("frame_index_map", None)
    if isinstance(m, list) and m:
        return [int(x) for x in m]
    key = "roi_kept_frames" if stream == "roi" else "bg_kept_frames"
    arr = frame_drop_json.get(key, None)
    if isinstance(arr, list) and arr:
        return [int(x) for x in arr]
    return []


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


def _resize_if_needed(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == int(width) and frame.shape[0] == int(height):
        return frame
    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)


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
    return build_frame_mask(
        frame_idx=int(frame_idx),
        width=int(width),
        height=int(height),
        mask_source=str(mask_source),
        roi_boxes_map=roi_boxes_map,
        frame_drop_json=frame_drop_json,
        roi_min_conf=float(roi_min_conf),
        roi_dilate_px=int(roi_dilate_px),
    )


def _compose_soft(roi_frame: np.ndarray, bg_frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return compose_soft(roi_frame, bg_frame, alpha)


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
