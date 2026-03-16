from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover - handled by runtime fallbacks for import-time safety
    torch = None  # type: ignore[assignment]

try:
    import yaml
except Exception:  # pragma: no cover - JSON fallback remains available
    yaml = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Lazy-loaded runtime components so module import stays lightweight for tests/tools.
compress_keep_streams_dcvc: Optional[Callable[..., Dict[str, Any]]] = None
apply_dual_timeline_policy: Optional[Callable[..., Dict[str, Any]]] = None
build_dual_timeline_metadata: Optional[Callable[..., Dict[str, Any]]] = None
remove_redundant_frames: Optional[Callable[..., Dict[str, Any]]] = None
validate_frame_removal_config: Optional[Callable[..., None]] = None
validate_dual_timeline_config: Optional[Callable[..., None]] = None
validate_pipeline_config: Optional[Callable[..., None]] = None
run_roi_detection: Optional[Callable[..., Dict[str, Any]]] = None


LOGGER = logging.getLogger("wildroi.compression")
VERBOSE_LOGS = False
ARCHIVE_MANIFEST_NAME = "archive_manifest.json"
RUNTIME_CONFIG_ENTRY_NAME = "compression.runtime_config.json"


def _setup_logging(verbose: bool = False) -> None:
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(level=level, format="%(message)s", force=True)
    LOGGER.setLevel(level)


def _log(event: str, **kwargs: Any) -> None:
    if not VERBOSE_LOGS:
        return
    payload = {"event": event, **kwargs}
    LOGGER.info(json.dumps(payload, separators=(",", ":")))


def _status(message: str) -> None:
    if VERBOSE_LOGS:
        return
    print(message)


def _format_bytes(num_bytes: int) -> str:
    value = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def _configure_runtime_logging(verbose: bool) -> None:
    global VERBOSE_LOGS
    VERBOSE_LOGS = bool(verbose)
    os.environ["YOLO_VERBOSE"] = "true" if VERBOSE_LOGS else "false"
    if VERBOSE_LOGS:
        return
    for logger_name in ("ultralytics", "onnxruntime"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def _load_runtime_components() -> None:
    global compress_keep_streams_dcvc
    global apply_dual_timeline_policy
    global build_dual_timeline_metadata
    global remove_redundant_frames
    global validate_frame_removal_config
    global validate_dual_timeline_config
    global validate_pipeline_config
    global run_roi_detection

    if compress_keep_streams_dcvc is None:
        try:
            from compression import compress_keep_streams_dcvc as _compress_keep_streams_dcvc

            compress_keep_streams_dcvc = _compress_keep_streams_dcvc
        except Exception:
            pass

    if apply_dual_timeline_policy is None:
        try:
            from frame_removal import apply_dual_timeline_policy as _apply_dual_timeline_policy

            apply_dual_timeline_policy = _apply_dual_timeline_policy
        except Exception:
            pass
    if build_dual_timeline_metadata is None:
        try:
            from frame_removal import build_dual_timeline_metadata as _build_dual_timeline_metadata

            build_dual_timeline_metadata = _build_dual_timeline_metadata
        except Exception:
            pass
    if remove_redundant_frames is None:
        try:
            from frame_removal import remove_redundant_frames as _remove_redundant_frames

            remove_redundant_frames = _remove_redundant_frames
        except Exception:
            pass
    if validate_frame_removal_config is None:
        try:
            from frame_removal import validate_frame_removal_config as _validate_frame_removal_config

            validate_frame_removal_config = _validate_frame_removal_config
        except Exception:
            pass
    if validate_dual_timeline_config is None:
        try:
            from frame_removal import validate_dual_timeline_config as _validate_dual_timeline_config

            validate_dual_timeline_config = _validate_dual_timeline_config
        except Exception:
            pass
    if validate_pipeline_config is None:
        try:
            from pipeline.config_schema import validate_pipeline_config as _validate_pipeline_config

            validate_pipeline_config = _validate_pipeline_config
        except Exception:
            pass
    if run_roi_detection is None:
        try:
            from roi_detection import run_roi_detection as _run_roi_detection

            run_roi_detection = _run_roi_detection
        except Exception:
            pass


def _cuda_is_available() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase1 + phase2 + phase4 compression pipeline")
    parser.add_argument(
        "video_path",
        help="Input video path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "gpu" / "compression.yaml"),
        help="Path to pipeline YAML config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output archive path override (.zip)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed diagnostic logs instead of production-style progress output",
    )
    return parser.parse_args()


def _load_config_file(cfg_path: Path) -> Dict[str, Any]:
    raw_text = cfg_path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(raw_text) or {}
    else:
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Config parsing requires PyYAML for .yaml/.yml files. "
                "Install dependency `pyyaml` or provide JSON config content."
            ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {cfg_path}: expected object at root.")
    return data


def _resolve_config_path(raw_path: str) -> Path:
    p = Path(str(raw_path)).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (ROOT / p).resolve()


def _resolve_input_video_path(raw_path: str) -> Path:
    return _resolve_config_path(raw_path)


def _detect_onnx_gpu_provider() -> Tuple[bool, Optional[str]]:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        return False, f"onnxruntime_import_error:{type(exc).__name__}"

    if not VERBOSE_LOGS:
        try:
            ort.set_default_logger_severity(3)
        except Exception:
            pass

    try:
        providers = [str(p) for p in (ort.get_available_providers() or [])]
    except Exception as exc:
        return False, f"onnxruntime_provider_query_error:{type(exc).__name__}"

    for provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
        if provider in providers:
            return True, provider
    return False, "onnxruntime_no_gpu_provider"


def _select_roi_model_for_runtime(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select the ROI detector backend deterministically from config.

    Default behavior is to keep the configured `animal_model_path`. ONNX is only
    selected when `roi_detection.runtime.prefer_onnx=true`. If
    `prefer_onnx_strict=true`, the run fails instead of silently falling back.

    Config keys (roi_detection):
      paths.animal_model_path         -> default/fallback model path
      paths.animal_model_path_onnx    -> ONNX model path candidate
      runtime.prefer_onnx            -> opt into ONNX backend selection
      runtime.prefer_onnx_strict     -> fail if requested ONNX backend is unavailable
    """
    roi_cfg = (cfg.get("roi_detection", {}) or {}).copy()
    paths = (roi_cfg.get("paths", {}) or {}).copy()
    runtime_cfg = (roi_cfg.get("runtime", {}) or {}).copy()

    default_model = paths.get("animal_model_path", None)
    onnx_model = paths.get("animal_model_path_onnx", None)
    prefer_onnx = bool(runtime_cfg.get("prefer_onnx", False)) if isinstance(runtime_cfg.get("prefer_onnx", False), bool) else False
    prefer_onnx_strict = (
        bool(runtime_cfg.get("prefer_onnx_strict", False))
        if isinstance(runtime_cfg.get("prefer_onnx_strict", False), bool)
        else False
    )

    selected_format = "default"
    selected_model = default_model
    fallback_reason = None
    onnx_gpu_runtime_ready = False
    onnx_runtime_status: Optional[str] = None

    if prefer_onnx:
        if onnx_model:
            onnx_exists = _resolve_config_path(str(onnx_model)).exists()
            if onnx_exists:
                onnx_gpu_runtime_ready, onnx_runtime_status = _detect_onnx_gpu_provider()
                if onnx_gpu_runtime_ready:
                    selected_model = onnx_model
                    selected_format = "onnx"
                elif prefer_onnx_strict:
                    raise RuntimeError(
                        "roi_detection.runtime.prefer_onnx=true and prefer_onnx_strict=true, "
                        "but no GPU ONNX Runtime provider is available."
                    )
                else:
                    fallback_reason = "onnx_runtime_unavailable_fallback_default"
            else:
                if prefer_onnx_strict:
                    raise FileNotFoundError(
                        f"roi_detection.paths.animal_model_path_onnx does not exist: {_resolve_config_path(str(onnx_model))}"
                    )
                fallback_reason = "onnx_missing_fallback_default"
        else:
            if prefer_onnx_strict:
                raise ValueError(
                    "roi_detection.runtime.prefer_onnx=true and prefer_onnx_strict=true require "
                    "roi_detection.paths.animal_model_path_onnx to be configured."
                )
            fallback_reason = "onnx_not_configured_fallback_default"

    if selected_model:
        paths["animal_model_path"] = str(selected_model)
    roi_cfg["paths"] = paths
    roi_cfg["runtime"] = runtime_cfg
    cfg["roi_detection"] = roi_cfg

    return {
        "selected_format": selected_format,
        "selected_model_path": str(selected_model) if selected_model else None,
        "onnx_model_path": (str(onnx_model) if onnx_model else None),
        "onnx_gpu_runtime_ready": bool(onnx_gpu_runtime_ready) if onnx_model else None,
        "onnx_runtime_status": onnx_runtime_status,
        "prefer_onnx": bool(prefer_onnx),
        "prefer_onnx_strict": bool(prefer_onnx_strict),
        "fallback_reason": fallback_reason,
    }


def _safe_write_zip(zip_path: Path, payloads: Dict[str, bytes]) -> None:
    tmp_zip = zip_path.with_suffix(zip_path.suffix + ".tmp")
    if tmp_zip.exists():
        tmp_zip.unlink()
    with zipfile.ZipFile(tmp_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in payloads.items():
            zf.writestr(name, content)
    tmp_zip.replace(zip_path)


def _package_version(distribution: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:
        return None
    try:
        return str(version(distribution))
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _git_commit(root: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except Exception:
        return None
    commit = str(proc.stdout).strip()
    return commit or None


def _collect_runtime_provenance(cfg_path: Path, video_path: str) -> Dict[str, Any]:
    provenance: Dict[str, Any] = {
        "config_path": str(cfg_path),
        "video_path": str(video_path),
        "python_version": str(sys.version.split()[0]),
        "platform": platform.platform(),
        "git_commit": _git_commit(ROOT),
        "torch_version": (str(getattr(torch, "__version__", "")) if torch is not None else None),
        "torch_cuda_version": (
            str(getattr(getattr(torch, "version", None), "cuda", "")) if torch is not None else None
        ),
        "onnxruntime_version": _package_version("onnxruntime-gpu") or _package_version("onnxruntime"),
        "ultralytics_version": _package_version("ultralytics"),
    }
    if torch is not None and _cuda_is_available():
        try:
            provenance["cuda_device_count"] = int(torch.cuda.device_count())
        except Exception:
            provenance["cuda_device_count"] = None
    else:
        provenance["cuda_device_count"] = 0
    return provenance


def _apply_dual_timeline_policy(frame_result: Dict[str, Any], frame_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if apply_dual_timeline_policy is not None:
        return apply_dual_timeline_policy(frame_result, frame_cfg)

    if build_dual_timeline_metadata is None:
        kept = sorted(set(int(x) for x in (frame_result.get("kept_frames", []) or [])))
        dual = {
            "enabled": False,
            "roi_kept_frames": kept,
            "bg_kept_frames": kept,
            "policy": {"roi_mode": "legacy_single_timeline"},
        }
    else:
        dual = build_dual_timeline_metadata(frame_result, frame_cfg)
    frame_result["dual_timeline"] = dual

    roi_kept = sorted(set(int(x) for x in (dual.get("roi_kept_frames", []) or [])))
    bg_kept = sorted(set(int(x) for x in (dual.get("bg_kept_frames", []) or [])))

    stats = dict(frame_result.get("stats", {}) or {})
    n_frames = int(stats.get("num_frames_read", 0) or 0)
    if n_frames <= 0:
        n_frames = max(max(roi_kept or [0]), max(bg_kept or [0])) + 1
    all_frames = set(range(max(0, n_frames)))

    roi_dropped = sorted(all_frames - set(roi_kept))
    bg_dropped = sorted(all_frames - set(bg_kept))

    frame_result["policy_mode"] = "dual_only"
    frame_result["roi_kept_frames"] = roi_kept
    frame_result["roi_dropped_frames"] = roi_dropped
    frame_result["bg_kept_frames"] = bg_kept
    frame_result["bg_dropped_frames"] = bg_dropped
    frame_result["kept_frames"] = roi_kept
    frame_result["dropped_frames"] = roi_dropped

    stats["dual_timeline"] = {
        "enabled": bool(dual.get("enabled", True)),
        "roi_kept_count": int(len(roi_kept)),
        "bg_kept_count": int(len(bg_kept)),
    }
    stats["policy_mode"] = "dual_only"
    frame_result["stats"] = stats
    return frame_result


def _require_cuda_runtime() -> None:
    if not _cuda_is_available():
        raise RuntimeError("Strict GPU runtime requires CUDA, but torch.cuda.is_available() is false.")


def _pick_best_device_for_roi(raw_device: Any) -> Tuple[Any, str]:
    _require_cuda_runtime()
    if isinstance(raw_device, int):
        if int(raw_device) < 0:
            raise ValueError("roi_detection.runtime.device CUDA index must be >= 0 for strict GPU runtime.")
        return int(raw_device), "cuda"

    s = str(raw_device).strip().lower()
    if not s or s in {"auto", "cuda"}:
        return 0, "cuda"
    if s.startswith("cuda:"):
        idx = s.split(":", 1)[1].strip()
        if idx.isdigit():
            return int(idx), "cuda"
        raise ValueError("roi_detection.runtime.device must be cuda:<index> when using explicit CUDA index.")
    if s.isdigit():
        return int(s), "cuda"
    if s in {"cpu", "mps"}:
        raise ValueError("Strict GPU runtime forbids roi_detection.runtime.device set to CPU/MPS.")
    raise ValueError(
        "Invalid roi_detection.runtime.device for strict GPU runtime. "
        "Use auto, cuda, cuda:<index>, or integer GPU index."
    )


def _parse_cuda_index(raw: Any) -> Optional[int]:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return int(raw) if raw >= 0 else None
    s = str(raw).strip()
    if s.isdigit():
        return int(s)
    return None


def _parse_use_cuda_compat(raw: Any) -> bool:
    if isinstance(raw, bool):
        return bool(raw)
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "cuda"}:
        return True
    if s in {"0", "false", "no", "cpu"}:
        return False
    if s == "auto":
        return _cuda_is_available()
    return bool(raw)


def _pick_dcvc_device(raw_device: Any, raw_use_cuda: Any, raw_cuda_idx: Any) -> Tuple[str, str, bool, Optional[int], bool]:
    _require_cuda_runtime()
    req_idx = _parse_cuda_index(raw_cuda_idx)

    if raw_device is None:
        use_cuda_req = _parse_use_cuda_compat(raw_use_cuda)
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
        elif s == "mps":
            raise ValueError("Strict GPU runtime forbids compression.dcvc.device set to MPS.")
        elif s == "cuda":
            req_label = "cuda"
        elif s.startswith("cuda:"):
            idx = s.split(":", 1)[1].strip()
            if idx.isdigit():
                req_idx = int(idx)
                req_label = "cuda"
            else:
                raise ValueError("compression.dcvc.device must be cuda:<index> when using explicit CUDA index.")
        elif s.isdigit():
            req_idx = int(s)
            req_label = "cuda"
        else:
            raise ValueError(
                "Invalid compression.dcvc.device for strict GPU runtime. "
                "Use auto, cuda, cuda:<index>, or integer GPU index."
            )
        use_cuda_req = req_label != "cpu"

    if req_label == "cpu":
        raise ValueError("Strict GPU runtime forbids compression.dcvc.device=cpu or use_cuda=false.")
    return req_label, "cuda", True, (req_idx if req_idx is not None else 0), bool(use_cuda_req)


def _apply_runtime_device_fallbacks(cfg: Dict[str, Any]) -> Dict[str, Any]:
    _require_cuda_runtime()
    report: Dict[str, Any] = {
        "cuda_available": True,
        "runtime_mode": "strict_gpu_only",
    }

    roi_cfg = cfg.get("roi_detection", {}) or {}
    roi_rt = roi_cfg.get("runtime", {}) or {}
    roi_enable = bool(roi_cfg.get("enable", True))

    comp_cfg = cfg.get("compression", {}) or {}
    dcvc_cfg = comp_cfg.get("dcvc", {}) or {}
    req_label, sel_label, dcvc_use_cuda_sel, dcvc_cuda_idx_sel, dcvc_use_cuda_req = _pick_dcvc_device(
        dcvc_cfg.get("device", None),
        dcvc_cfg.get("use_cuda", True),
        dcvc_cfg.get("cuda_idx", None),
    )
    dcvc_device_selected = str(sel_label)
    if dcvc_use_cuda_sel and dcvc_cuda_idx_sel is not None:
        dcvc_device_selected = f"cuda:{int(dcvc_cuda_idx_sel)}"
    dcvc_cfg["device"] = dcvc_device_selected
    dcvc_cfg["use_cuda"] = bool(dcvc_use_cuda_sel)
    dcvc_cfg["cuda_idx"] = dcvc_cuda_idx_sel if dcvc_use_cuda_sel else None
    comp_cfg["dcvc"] = dcvc_cfg
    cfg["compression"] = comp_cfg

    report["dcvc_device_requested"] = str(req_label)
    report["dcvc_device_selected"] = str(dcvc_device_selected)
    report["dcvc_use_cuda_requested"] = bool(dcvc_use_cuda_req)
    report["dcvc_use_cuda_selected"] = bool(dcvc_use_cuda_sel)
    report["dcvc_cuda_idx_selected"] = dcvc_cuda_idx_sel if dcvc_use_cuda_sel else None

    if roi_enable:
        requested = roi_rt.get("device", "auto")
        selected_device, backend = _pick_best_device_for_roi(requested)
        roi_rt["device"] = selected_device
        if bool(roi_rt.get("half", False)) and backend != "cuda":
            roi_rt["half"] = False
        roi_cfg["runtime"] = roi_rt
        cfg["roi_detection"] = roi_cfg

        report["roi_device_requested"] = str(requested)
        report["roi_device_selected"] = str(selected_device)
        report["roi_backend_selected"] = backend
        report["roi_half_selected"] = bool(roi_rt.get("half", False))

    return report


def _resolve_archive_output_path(
    *,
    video_path: str,
    out_cfg: Dict[str, Any],
    output_override: Optional[str],
) -> Path:
    if output_override:
        out = Path(output_override).expanduser()
        if not out.is_absolute():
            out = (ROOT / out).resolve()
        if out.suffix.lower() != ".zip":
            out = out.with_suffix(".zip")
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    video_file = _resolve_input_video_path(video_path)
    zip_name = str(out_cfg.get("zip_name", f"{video_file.stem}.zip")).strip()
    if not zip_name:
        zip_name = f"{video_file.stem}.zip"
    if not zip_name.lower().endswith(".zip"):
        zip_name += ".zip"

    out_dir_raw = str(out_cfg.get("out_dir", "") or "").strip()
    normalized = out_dir_raw.replace("\\", "/").strip().strip(".").strip("/")
    if not out_dir_raw or normalized == "outputs/zip":
        out_dir = video_file.parent
    else:
        out_dir = _resolve_config_path(out_dir_raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / zip_name


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)
    _configure_runtime_logging(args.verbose)
    started = time.time()
    run_id = str(uuid.uuid4())
    _load_runtime_components()
    if (
        validate_pipeline_config is None
        or remove_redundant_frames is None
        or compress_keep_streams_dcvc is None
    ):
        raise RuntimeError("Failed to load one or more runtime pipeline components")

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    cfg = _load_config_file(cfg_path)
    video_path = str(_resolve_input_video_path(str(args.video_path)))
    _status(f"Compressing {Path(video_path).name}")
    validate_pipeline_config(cfg, video_path=video_path, root_dir=ROOT)
    model_selection = _select_roi_model_for_runtime(cfg)
    frame_cfg = cfg.get("frame_removal", {}) or {}
    if validate_frame_removal_config is not None:
        validate_frame_removal_config(frame_cfg)
    if validate_dual_timeline_config is not None:
        validate_dual_timeline_config(cfg)
    runtime_device = _apply_runtime_device_fallbacks(cfg)

    _log("compression.start", run_id=run_id, config=str(cfg_path), video=str(video_path))
    _log("phase1.model_select", run_id=run_id, **model_selection)
    _log("runtime.device_select", run_id=run_id, **runtime_device)

    roi_cfg = cfg.get("roi_detection", {}) or {}
    _status("ROI detection...")
    if bool(roi_cfg.get("enable", True)):
        if run_roi_detection is None:
            raise RuntimeError("ROI detection is enabled but ROI runtime component is unavailable")
        roi_result = run_roi_detection(video_path, roi_cfg)
    else:
        roi_result = {"video_path": video_path, "frames": {}}
    roi_frames = (roi_result.get("frames", {}) or {})
    bbox_map = {int(k): v for k, v in roi_frames.items()}
    _log("phase1.complete", run_id=run_id, roi_frames_with_boxes=int(len(bbox_map)))

    _status("Selecting ROI and background anchor frames...")
    frame_result = remove_redundant_frames(video_path, bbox_map, frame_cfg)
    frame_result = _apply_dual_timeline_policy(frame_result, frame_cfg)
    _log(
        "phase2.complete",
        run_id=run_id,
        roi_kept=int(len(frame_result.get("roi_kept_frames", []) or [])),
        bg_kept=int(len(frame_result.get("bg_kept_frames", []) or [])),
    )

    _status("Encoding ROI and background streams...")
    comp_cfg = cfg.get("compression", {}) or {}
    compression_result = compress_keep_streams_dcvc(
        source_video_path=video_path,
        roi_bbox_map=roi_frames,
        frame_drop_result=frame_result,
        compression_cfg=comp_cfg,
        root_dir=ROOT,
    )
    compression_meta = dict(compression_result.get("meta", {}) or {})
    compression_meta["roi_detection"] = {
        "enabled": bool(roi_cfg.get("enable", True)),
        "model_selection": model_selection,
        "runtime_device": {
            "requested": runtime_device.get("roi_device_requested", None),
            "selected": runtime_device.get("roi_device_selected", None),
            "backend": runtime_device.get("roi_backend_selected", None),
            "half": runtime_device.get("roi_half_selected", None),
        },
    }
    compression_meta["runtime"] = {
        "dcvc_device_selected": runtime_device.get("dcvc_device_selected", None),
        "dcvc_cuda_idx_selected": runtime_device.get("dcvc_cuda_idx_selected", None),
        "runtime_mode": runtime_device.get("runtime_mode", None),
    }
    compression_meta["provenance"] = _collect_runtime_provenance(cfg_path, video_path)
    compression_result["meta"] = compression_meta

    _log(
        "phase4.complete",
        run_id=run_id,
        roi_bytes=int(len(compression_result.get("roi_bin_bytes", b""))),
        bg_bytes=int(len(compression_result.get("bg_bin_bytes", b""))),
    )

    out_cfg = cfg.get("output", {}) or {}
    write_outputs = bool(out_cfg.get("write_outputs", True)) or (args.output is not None)
    if not write_outputs:
        duration_sec = round(time.time() - started, 3)
        _log("compression.complete", run_id=run_id, duration_sec=round(time.time() - started, 3))
        _status(f"[OK] compression completed in {duration_sec:.3f}s (archive writing disabled)")
        return

    _status("Writing archive...")
    zip_path = _resolve_archive_output_path(
        video_path=video_path,
        out_cfg=out_cfg,
        output_override=args.output,
    )
    archive_entries = {
        "roi_detections.json": str(out_cfg.get("roi_json", "roi_detections.json")),
        "frame_drop.json": str(out_cfg.get("frame_drop_json", "frame_drop.json")),
        "roi.bin": str(out_cfg.get("roi_bin", "roi.bin")),
        "bg.bin": str(out_cfg.get("bg_bin", "bg.bin")),
        "meta.json": str(out_cfg.get("dcvc_meta", "meta.json")),
    }
    runtime_cfg_bytes = json.dumps(cfg, indent=2, sort_keys=True).encode("utf-8")
    archive_payloads = {
        archive_entries["roi_detections.json"]: json.dumps(roi_result, indent=2).encode("utf-8"),
        archive_entries["frame_drop.json"]: json.dumps(frame_result, indent=2).encode("utf-8"),
        archive_entries["roi.bin"]: bytes(compression_result["roi_bin_bytes"]),
        archive_entries["bg.bin"]: bytes(compression_result["bg_bin_bytes"]),
        archive_entries["meta.json"]: json.dumps(compression_result["meta"], indent=2).encode("utf-8"),
        RUNTIME_CONFIG_ENTRY_NAME: runtime_cfg_bytes,
        ARCHIVE_MANIFEST_NAME: json.dumps(
            {
                "version": 2,
                "entries": archive_entries,
                "extras": {
                    "runtime_config": RUNTIME_CONFIG_ENTRY_NAME,
                },
            },
            indent=2,
        ).encode("utf-8"),
    }
    _safe_write_zip(zip_path, archive_payloads)

    _log(
        "compression.complete",
        run_id=run_id,
        zip=str(zip_path),
        duration_sec=round(time.time() - started, 3),
    )
    duration_sec = round(time.time() - started, 3)
    zip_size = zip_path.stat().st_size if zip_path.exists() else 0
    print(f"[OK] wrote compressed archive: {zip_path} ({_format_bytes(zip_size)}, {duration_sec:.3f}s)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _setup_logging(VERBOSE_LOGS)
        _log("compression.cancelled")
        print("[ERROR] compression cancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        _setup_logging(VERBOSE_LOGS)
        _log("compression.failed", error_type=type(exc).__name__, error=str(exc))
        print(f"[ERROR] compression failed: {exc}", file=sys.stderr)
        sys.exit(1)
