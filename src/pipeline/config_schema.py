"""
Config validation for all pipeline stages.

Each validate_* function accepts a plain dict and raises ValueError listing all
problems found.  This mirrors the reference project's approach so errors surface
early and clearly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Helpers ───────────────────────────────────────────────────────────────────

def _as_dict(v: Any, name: str, errors: List[str]) -> Dict[str, Any]:
    if not isinstance(v, dict):
        errors.append(f"{name} must be an object/dict")
        return {}
    return v


def _check_range(v: Any, name: str, errors: List[str],
                 lo: Optional[float] = None, hi: Optional[float] = None) -> None:
    if not isinstance(v, (int, float)):
        errors.append(f"{name} must be a number")
        return
    if lo is not None and float(v) < lo:
        errors.append(f"{name} must be >= {lo}, got {v}")
    if hi is not None and float(v) > hi:
        errors.append(f"{name} must be <= {hi}, got {v}")


def _must_exist(path_str: str, field: str, root: Path, errors: List[str]) -> None:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    if not p.exists():
        errors.append(f"{field} does not exist: {p}")


def _check_device(val: Any, name: str, errors: List[str]) -> None:
    if isinstance(val, bool):
        errors.append(f"{name}: use a string like 'cuda' or 'cuda:0', not a bool")
    elif isinstance(val, int):
        if val < 0:
            errors.append(f"{name} integer index must be >= 0")
    elif isinstance(val, str):
        s = val.strip().lower()
        if s in {"cpu", "mps"}:
            errors.append(f"{name}: CPU/MPS not supported in GPU runtime")
        elif not (s in {"auto", "cuda"} or s.isdigit()
                  or (s.startswith("cuda:") and s.split(":", 1)[1].isdigit())):
            errors.append(f"{name} must be 'auto', 'cuda', 'cuda:<N>', or integer index")
    else:
        errors.append(f"{name} must be a string or integer")


# ── Compression config ────────────────────────────────────────────────────────

def validate_compression_config(cfg: Dict[str, Any], root_dir: Optional[Path] = None) -> None:
    errors: List[str] = []
    root = (root_dir or Path.cwd()).resolve()

    det = _as_dict(cfg.get("detection", {}), "detection", errors)
    if not det.get("model_path"):
        errors.append("detection.model_path is required")
    elif isinstance(det["model_path"], str):
        _must_exist(det["model_path"], "detection.model_path", root, errors)

    if "conf" in det:
        _check_range(det["conf"], "detection.conf", errors, 0.0, 1.0)
    if "iou_nms" in det:
        _check_range(det["iou_nms"], "detection.iou_nms", errors, 0.0, 1.0)
    if "processing_scale" in det:
        _check_range(det["processing_scale"], "detection.processing_scale", errors, 0.1, 1.0)

    comp = _as_dict(cfg.get("compression", {}), "compression", errors)
    dcvc = _as_dict(comp.get("dcvc", {}), "compression.dcvc", errors)
    qual = _as_dict(comp.get("quality", {}), "compression.quality", errors)

    for key in ("model_i", "model_p", "repo_dir"):
        if not dcvc.get(key):
            errors.append(f"compression.dcvc.{key} is required")
    if dcvc.get("model_i"):
        _must_exist(str(dcvc["model_i"]), "compression.dcvc.model_i", root, errors)
    if dcvc.get("model_p"):
        _must_exist(str(dcvc["model_p"]), "compression.dcvc.model_p", root, errors)
    if dcvc.get("repo_dir"):
        _must_exist(str(dcvc["repo_dir"]), "compression.dcvc.repo_dir", root, errors)
    if "device" in dcvc:
        _check_device(dcvc["device"], "compression.dcvc.device", errors)

    for qp_key in ("roi_qp_i", "roi_qp_p", "bg_qp_i", "bg_qp_p"):
        if qp_key in qual:
            _check_range(qual[qp_key], f"compression.quality.{qp_key}", errors, 0, 63)

    if errors:
        raise ValueError("Invalid compression config:\n- " + "\n- ".join(errors))


# ── Decompression config ──────────────────────────────────────────────────────

def validate_decompression_config(cfg: Dict[str, Any], root_dir: Optional[Path] = None) -> None:
    errors: List[str] = []
    root = (root_dir or Path.cwd()).resolve()

    decomp = _as_dict(cfg.get("decompression", {}), "decompression", errors)
    dcvc   = _as_dict(decomp.get("dcvc", {}), "decompression.dcvc", errors)

    for key in ("model_i", "model_p", "repo_dir"):
        if not dcvc.get(key):
            errors.append(f"decompression.dcvc.{key} is required")
    if dcvc.get("model_i"):
        _must_exist(str(dcvc["model_i"]), "decompression.dcvc.model_i", root, errors)
    if dcvc.get("model_p"):
        _must_exist(str(dcvc["model_p"]), "decompression.dcvc.model_p", root, errors)
    if dcvc.get("repo_dir"):
        _must_exist(str(dcvc["repo_dir"]), "decompression.dcvc.repo_dir", root, errors)

    if errors:
        raise ValueError("Invalid decompression config:\n- " + "\n- ".join(errors))


# ── Restoration config ────────────────────────────────────────────────────────

def validate_restoration_config(cfg: Dict[str, Any], root_dir: Optional[Path] = None) -> None:
    errors: List[str] = []
    root = (root_dir or Path.cwd()).resolve()

    if "device" in cfg:
        _check_device(cfg["device"], "device", errors)

    model = _as_dict(cfg.get("model", {}), "model", errors)
    ckpt  = model.get("checkpoint")
    if not ckpt:
        errors.append("model.checkpoint is required")
    elif isinstance(ckpt, str) and ckpt.strip():
        _must_exist(ckpt.strip(), "model.checkpoint", root, errors)

    T = model.get("temporal_window", 3)
    if isinstance(T, int) and T % 2 == 0:
        errors.append("model.temporal_window must be odd")

    infer = _as_dict(cfg.get("inference", {}), "inference", errors)
    if "ddim_steps" in infer:
        _check_range(infer["ddim_steps"], "inference.ddim_steps", errors, lo=1)
    if "t_start" in infer:
        _check_range(infer["t_start"], "inference.t_start", errors, lo=0, hi=999)
    if "tile_size" in infer:
        _check_range(infer["tile_size"], "inference.tile_size", errors, lo=0)

    if errors:
        raise ValueError("Invalid restoration config:\n- " + "\n- ".join(errors))


# ── Upscaling config ──────────────────────────────────────────────────────────

def validate_upscaling_config(cfg: Dict[str, Any], root_dir: Optional[Path] = None) -> None:
    errors: List[str] = []
    root = (root_dir or Path.cwd()).resolve()

    if "device" in cfg:
        _check_device(cfg["device"], "device", errors)

    scale = cfg.get("scale")
    if scale is None:
        errors.append("scale is required")
    elif not isinstance(scale, int) or scale not in {1, 2, 4}:
        errors.append("scale must be an integer in {1, 2, 4}")

    model = _as_dict(cfg.get("model", {}), "model", errors)
    ckpt  = model.get("checkpoint")
    if not ckpt:
        errors.append("model.checkpoint is required")
    elif isinstance(ckpt, str) and ckpt.strip():
        _must_exist(ckpt.strip(), "model.checkpoint", root, errors)

    infer = _as_dict(cfg.get("inference", {}), "inference", errors)
    if "ddim_steps" in infer:
        _check_range(infer["ddim_steps"], "inference.ddim_steps", errors, lo=1)
    if "t_start" in infer:
        _check_range(infer["t_start"], "inference.t_start", errors, lo=0, hi=999)
    if "tile_size" in infer:
        _check_range(infer["tile_size"], "inference.tile_size", errors, lo=0)

    if errors:
        raise ValueError("Invalid upscaling config:\n- " + "\n- ".join(errors))
