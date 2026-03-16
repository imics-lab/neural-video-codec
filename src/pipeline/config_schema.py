from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _as_dict(v: Any, name: str, errors: list[str]) -> Dict[str, Any]:
    if not isinstance(v, dict):
        errors.append(f"{name} must be an object")
        return {}
    return v


def _ensure_type(v: Any, expected: type | tuple[type, ...], name: str, errors: list[str]) -> None:
    if not isinstance(v, expected):
        errors.append(f"{name} must be of type {expected}, got {type(v)}")


def _ensure_number_range(v: Any, name: str, errors: list[str], min_v: Optional[float] = None, max_v: Optional[float] = None) -> None:
    if not isinstance(v, (int, float)):
        errors.append(f"{name} must be a number")
        return
    fv = float(v)
    if min_v is not None and fv < min_v:
        errors.append(f"{name} must be >= {min_v}, got {fv}")
    if max_v is not None and fv > max_v:
        errors.append(f"{name} must be <= {max_v}, got {fv}")


def _resolve_path(path_value: str, root_dir: Path) -> Path:
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p
    return (root_dir / p).resolve()


def _must_exist(path_value: str, field_name: str, root_dir: Path, errors: list[str]) -> None:
    path = _resolve_path(path_value, root_dir)
    if not path.exists():
        errors.append(f"{field_name} does not exist: {path}")


def validate_pipeline_config(cfg: Dict[str, Any], video_path: Optional[str] = None, root_dir: Optional[Path] = None) -> None:
    """
    Validate runtime-critical config fields. Raises ValueError on any issue.
    """
    errors: list[str] = []
    if not isinstance(cfg, dict):
        raise TypeError("Config root must be an object")

    root = (root_dir or Path.cwd()).resolve()

    input_cfg = _as_dict(cfg.get("input", {}), "input", errors)
    roi_cfg = _as_dict(cfg.get("roi_detection", {}), "roi_detection", errors)
    frame_cfg = _as_dict(cfg.get("frame_removal", {}), "frame_removal", errors)
    comp_cfg = _as_dict(cfg.get("compression", {}), "compression", errors)
    out_cfg = _as_dict(cfg.get("output", {}), "output", errors)

    # input
    cfg_video = input_cfg.get("video_path")
    if video_path is None and not cfg_video:
        errors.append("video_path is required: pass a CLI video path or set input.video_path in a custom config")
    chosen_video = str(video_path) if video_path else str(cfg_video)
    if chosen_video:
        _must_exist(chosen_video, "video_path", root, errors)

    # roi_detection
    paths_cfg = _as_dict(roi_cfg.get("paths", {}), "roi_detection.paths", errors)
    runtime_cfg = _as_dict(roi_cfg.get("runtime", {}), "roi_detection.runtime", errors)
    tracking_cfg = _as_dict(roi_cfg.get("tracking", {}), "roi_detection.tracking", errors)

    roi_enabled = bool(roi_cfg.get("enable", True))
    model_path = paths_cfg.get("animal_model_path")
    onnx_model_path = paths_cfg.get("animal_model_path_onnx")
    if roi_enabled:
        if not model_path:
            errors.append("roi_detection.paths.animal_model_path is required when roi_detection.enable=true")
        else:
            _must_exist(str(model_path), "roi_detection.paths.animal_model_path", root, errors)
    if onnx_model_path:
        _must_exist(str(onnx_model_path), "roi_detection.paths.animal_model_path_onnx", root, errors)

    if "processing_scale" in runtime_cfg:
        _ensure_number_range(runtime_cfg["processing_scale"], "roi_detection.runtime.processing_scale", errors, 0.1, 1.0)
    if "imgsz" in runtime_cfg:
        _ensure_number_range(runtime_cfg["imgsz"], "roi_detection.runtime.imgsz", errors, 64, None)
    if "conf" in runtime_cfg:
        _ensure_number_range(runtime_cfg["conf"], "roi_detection.runtime.conf", errors, 0.0, 1.0)
    if "iou_nms" in runtime_cfg:
        _ensure_number_range(runtime_cfg["iou_nms"], "roi_detection.runtime.iou_nms", errors, 0.0, 1.0)
    if "bbox_pad_frac" in runtime_cfg:
        _ensure_number_range(runtime_cfg["bbox_pad_frac"], "roi_detection.runtime.bbox_pad_frac", errors, 0.0, None)
    if "bbox_pad_px" in runtime_cfg:
        _ensure_number_range(runtime_cfg["bbox_pad_px"], "roi_detection.runtime.bbox_pad_px", errors, 0.0, None)
    if "keyframe_interval" in runtime_cfg:
        _ensure_number_range(runtime_cfg["keyframe_interval"], "roi_detection.runtime.keyframe_interval", errors, 1, None)
    if "device" in runtime_cfg:
        dev = runtime_cfg.get("device")
        if isinstance(dev, bool):
            errors.append(
                "roi_detection.runtime.device must be one of: auto, cuda, cuda:<index>, or integer GPU index"
            )
        elif isinstance(dev, int):
            if int(dev) < 0:
                errors.append("roi_detection.runtime.device integer index must be >= 0")
        elif isinstance(dev, str):
            s = dev.strip().lower()
            if s in {"cpu", "mps"}:
                errors.append("Strict GPU runtime forbids roi_detection.runtime.device set to CPU/MPS")
            valid = (
                s in {"auto", "cuda"}
                or s.isdigit()
                or (s.startswith("cuda:") and s.split(":", 1)[1].strip().isdigit())
            )
            if not valid:
                errors.append(
                    "roi_detection.runtime.device must be one of: auto, cuda, cuda:<index>, or integer GPU index"
                )
        else:
            errors.append(
                "roi_detection.runtime.device must be one of: auto, cuda, cuda:<index>, or integer GPU index"
            )
    if "prefer_onnx" in runtime_cfg:
        _ensure_type(runtime_cfg["prefer_onnx"], bool, "roi_detection.runtime.prefer_onnx", errors)
    if "prefer_onnx_strict" in runtime_cfg:
        _ensure_type(runtime_cfg["prefer_onnx_strict"], bool, "roi_detection.runtime.prefer_onnx_strict", errors)
    if bool(runtime_cfg.get("prefer_onnx_strict", False)) and not bool(runtime_cfg.get("prefer_onnx", False)):
        errors.append("roi_detection.runtime.prefer_onnx_strict=true requires roi_detection.runtime.prefer_onnx=true")
    if bool(runtime_cfg.get("prefer_onnx", False)):
        if not onnx_model_path:
            errors.append(
                "roi_detection.runtime.prefer_onnx=true requires roi_detection.paths.animal_model_path_onnx"
            )
        elif roi_enabled:
            _must_exist(str(onnx_model_path), "roi_detection.paths.animal_model_path_onnx", root, errors)

    if "match_iou" in tracking_cfg:
        _ensure_number_range(tracking_cfg["match_iou"], "roi_detection.tracking.match_iou", errors, 0.0, 1.0)
    if "min_hits" in tracking_cfg:
        _ensure_number_range(tracking_cfg["min_hits"], "roi_detection.tracking.min_hits", errors, 1, None)
    if "enable_propagation" in tracking_cfg:
        _ensure_type(tracking_cfg["enable_propagation"], bool, "roi_detection.tracking.enable_propagation", errors)

    # frame_removal
    params_cfg = _as_dict(frame_cfg.get("params", {}), "frame_removal.params", errors)
    frame_roi_cfg = _as_dict(params_cfg.get("roi", {}), "frame_removal.params.roi", errors)
    motion_cfg = _as_dict(params_cfg.get("motion", {}), "frame_removal.params.motion", errors)
    smooth_cfg = _as_dict(params_cfg.get("bbox_smoothing", {}), "frame_removal.params.bbox_smoothing", errors)
    dual_cfg = _as_dict(frame_cfg.get("dual_timeline", {}), "frame_removal.dual_timeline", errors)

    if "halo_frac" in frame_roi_cfg:
        _ensure_number_range(
            frame_roi_cfg["halo_frac"],
            "frame_removal.params.roi.halo_frac",
            errors,
            0.0,
            None,
        )
    if "fixed_size" in frame_roi_cfg:
        _ensure_number_range(
            frame_roi_cfg["fixed_size"],
            "frame_removal.params.roi.fixed_size",
            errors,
            1,
            None,
        )
    if "blur_ksize" in frame_roi_cfg:
        _ensure_number_range(
            frame_roi_cfg["blur_ksize"],
            "frame_removal.params.roi.blur_ksize",
            errors,
            0,
            None,
        )
    if "gray" in frame_roi_cfg:
        _ensure_type(frame_roi_cfg["gray"], bool, "frame_removal.params.roi.gray", errors)

    t_low = motion_cfg.get("t_low")
    t_high = motion_cfg.get("t_high")
    if t_low is not None:
        _ensure_number_range(t_low, "frame_removal.params.motion.t_low", errors, 0.0, None)
    if t_high is not None:
        _ensure_number_range(t_high, "frame_removal.params.motion.t_high", errors, 0.0, None)
    if isinstance(t_low, (int, float)) and isinstance(t_high, (int, float)) and float(t_high) <= float(t_low):
        errors.append("frame_removal.params.motion.t_high must be > t_low")
    if "state_source" in motion_cfg:
        src = str(motion_cfg.get("state_source", "")).strip().lower()
        if src not in {"pixel", "bbox", "hybrid"}:
            errors.append("frame_removal.params.motion.state_source must be one of: pixel, bbox, hybrid")
    bbox_t_low_px = motion_cfg.get("bbox_t_low_px", None)
    bbox_t_high_px = motion_cfg.get("bbox_t_high_px", None)
    if bbox_t_low_px is not None:
        _ensure_number_range(
            bbox_t_low_px,
            "frame_removal.params.motion.bbox_t_low_px",
            errors,
            0.0,
            None,
        )
    if bbox_t_high_px is not None:
        _ensure_number_range(
            bbox_t_high_px,
            "frame_removal.params.motion.bbox_t_high_px",
            errors,
            0.0,
            None,
        )
    if isinstance(bbox_t_low_px, (int, float)) and isinstance(bbox_t_high_px, (int, float)):
        if float(bbox_t_high_px) <= float(bbox_t_low_px):
            errors.append("frame_removal.params.motion.bbox_t_high_px must be > bbox_t_low_px")
    if "enter_motion_frames" in motion_cfg:
        _ensure_number_range(
            motion_cfg["enter_motion_frames"],
            "frame_removal.params.motion.enter_motion_frames",
            errors,
            1,
            None,
        )
    if "enter_still_frames" in motion_cfg:
        _ensure_number_range(
            motion_cfg["enter_still_frames"],
            "frame_removal.params.motion.enter_still_frames",
            errors,
            1,
            None,
        )
    if "enable" in smooth_cfg:
        _ensure_type(smooth_cfg["enable"], bool, "frame_removal.params.bbox_smoothing.enable", errors)
    if "ema_alpha" in smooth_cfg:
        _ensure_number_range(
            smooth_cfg["ema_alpha"],
            "frame_removal.params.bbox_smoothing.ema_alpha",
            errors,
            0.0,
            1.0,
        )
    if "enable" in dual_cfg:
        _ensure_type(dual_cfg["enable"], bool, "frame_removal.dual_timeline.enable", errors)
    for key in ("roi_motion_interval", "roi_still_interval", "bg_interval", "bg_idle_interval"):
        if key in dual_cfg:
            _ensure_number_range(
                dual_cfg[key],
                f"frame_removal.dual_timeline.{key}",
                errors,
                1,
                None,
            )

    # compression
    dcvc_cfg = _as_dict(comp_cfg.get("dcvc", {}), "compression.dcvc", errors)
    quality_cfg = _as_dict(comp_cfg.get("quality", {}), "compression.quality", errors)
    roi_comp_cfg = _as_dict(comp_cfg.get("roi", {}), "compression.roi", errors)

    for key in ("model_i", "model_p", "repo_dir"):
        if not dcvc_cfg.get(key):
            errors.append(f"compression.dcvc.{key} is required")
    if dcvc_cfg.get("model_i"):
        _must_exist(str(dcvc_cfg["model_i"]), "compression.dcvc.model_i", root, errors)
    if dcvc_cfg.get("model_p"):
        _must_exist(str(dcvc_cfg["model_p"]), "compression.dcvc.model_p", root, errors)
    if dcvc_cfg.get("repo_dir"):
        _must_exist(str(dcvc_cfg["repo_dir"]), "compression.dcvc.repo_dir", root, errors)
    if "reset_interval" in dcvc_cfg:
        _ensure_number_range(dcvc_cfg["reset_interval"], "compression.dcvc.reset_interval", errors, 1, None)
    if "device" in dcvc_cfg:
        dev = dcvc_cfg.get("device")
        if isinstance(dev, bool):
            errors.append(
                "compression.dcvc.device must be one of: auto, cuda, cuda:<index>, or integer GPU index"
            )
        elif isinstance(dev, int):
            if int(dev) < 0:
                errors.append("compression.dcvc.device integer index must be >= 0")
        elif isinstance(dev, str):
            s = dev.strip().lower()
            if s in {"cpu", "mps"}:
                errors.append("Strict GPU runtime forbids compression.dcvc.device set to CPU/MPS")
            valid = (
                s in {"auto", "cuda"}
                or s.isdigit()
                or (s.startswith("cuda:") and s.split(":", 1)[1].strip().isdigit())
            )
            if not valid:
                errors.append(
                    "compression.dcvc.device must be one of: auto, cuda, cuda:<index>, or integer GPU index"
                )
        else:
            errors.append(
                "compression.dcvc.device must be one of: auto, cuda, cuda:<index>, or integer GPU index"
            )
    if "use_cuda" in dcvc_cfg:
        _ensure_type(dcvc_cfg["use_cuda"], bool, "compression.dcvc.use_cuda", errors)
        if isinstance(dcvc_cfg.get("use_cuda"), bool) and not bool(dcvc_cfg.get("use_cuda")):
            errors.append("Strict GPU runtime forbids compression.dcvc.use_cuda=false")

    for key in ("roi_qp_i", "roi_qp_p", "bg_qp_i", "bg_qp_p"):
        if key in quality_cfg:
            _ensure_number_range(quality_cfg[key], f"compression.quality.{key}", errors, 0, 63)
    if "min_conf" in roi_comp_cfg:
        _ensure_number_range(roi_comp_cfg["min_conf"], "compression.roi.min_conf", errors, 0.0, 1.0)

    # output
    if "write_outputs" in out_cfg:
        _ensure_type(out_cfg["write_outputs"], bool, "output.write_outputs", errors)
    if "out_dir" in out_cfg:
        _ensure_type(out_cfg["out_dir"], str, "output.out_dir", errors)

    if errors:
        raise ValueError("Invalid pipeline config:\n- " + "\n- ".join(errors))
