from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

from .dcvc_encoder import VideoInfo, encode_dcvc_frames_to_bytes, probe_video
from .phase4_dcvc import (
    _capture_rendered_kept_frames_single_pass,
    _close_memmap,
    _iter_cached_frames,
    _iter_kept_bg_frames,
    _iter_kept_roi_frames,
    _pick_indices,
    _resolve_frame_index_map,
    _resolve_path,
)
from .ffmpeg_encoder import _crf_for, _resolve_encoder, encode_frames_ffmpeg_to_bytes


_INT16_CODECS = {"dcvc_int16", "int16", "dcvc_rt_int16", "dcvc_rt", "dcvc-rt"}
_MICROSOFT_DCVC_CODECS = {"dcvc", "microsoft_dcvc"}
_FFMPEG_CODEC_ALIASES = {"h265": "hevc", "hevc": "hevc", "h264": "h264", "av1": "av1"}


def _normalize_stream_codec(raw: Any) -> str:
    codec = str(raw or "dcvc_int16").strip().lower()
    if codec == "microsoft_dcvc":
        return "dcvc"
    return _FFMPEG_CODEC_ALIASES.get(codec, codec)


def _stream_codec(compression_cfg: Dict[str, Any], stream_name: str) -> str:
    stream_cfg = compression_cfg.get("stream_codecs", {}) or {}
    fallback = compression_cfg.get("codec", "dcvc_int16")
    return _normalize_stream_codec(stream_cfg.get(stream_name, fallback))


def _dcvc_int16_root(dcvc_cfg: Dict[str, Any], root_dir: Path) -> Path:
    return _resolve_path(str(dcvc_cfg.get("repo_dir", "dcvc_int16")), root_dir)


def _bundle_path(dcvc_cfg: Dict[str, Any], codec_root: Path) -> Path:
    raw = str(dcvc_cfg.get("bundle_path", "models/int16_bundle_v1.0.0.pt"))
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (codec_root / p).resolve()


def _device_label(dcvc_cfg: Dict[str, Any]) -> str:
    raw = dcvc_cfg.get("device", None)
    if raw is None:
        idx = dcvc_cfg.get("cuda_idx", 0)
        return f"cuda:{int(idx)}"
    s = str(raw).strip().lower()
    if s in {"", "auto", "cuda"}:
        idx = dcvc_cfg.get("cuda_idx", 0)
        return f"cuda:{int(idx)}"
    if s.isdigit():
        return f"cuda:{int(s)}"
    return str(raw)


def _write_lossless_video(
    frames_iter: Iterable[Tuple[int, np.ndarray]],
    *,
    info: VideoInfo,
    out_path: Path,
) -> List[int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{int(info.width)}x{int(info.height)}",
        "-r",
        str(float(info.fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    frame_indices: List[int] = []
    try:
        if proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        for src_idx, frame in frames_iter:
            if frame.shape[:2] != (int(info.height), int(info.width)):
                raise ValueError("INT16 adapter received a frame with unexpected dimensions")
            proc.stdin.write(frame.tobytes())
            frame_indices.append(int(src_idx))
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if rc != 0:
        raise RuntimeError(f"Failed to write temporary lossless video for dcvc_int16.\n{stderr}")
    return frame_indices


def _run_int16_encode(
    input_video: Path,
    *,
    output_dir: Path,
    codec_root: Path,
    bundle_path: Path,
    qp_i: int,
    qp_p: int,
    frames: int,
    device: str,
    config: str | Path | None,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(codec_root / "encode_mp4_to_bin.py"),
        "--input_mp4",
        str(input_video),
        "--bundle_path",
        str(bundle_path),
        "--output_dir",
        str(output_dir),
        "--frames",
        str(int(frames)),
        "--qp_i",
        str(int(qp_i)),
        "--qp_p",
        str(int(qp_p)),
        "--device",
        str(device),
    ]
    if config:
        cmd.extend(["--config", str(config)])
    completed = subprocess.run(cmd, cwd=str(codec_root), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"dcvc_int16 encode failed ({completed.returncode}).\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    sidecars = [
        p
        for p in sorted(output_dir.glob("*_q*.json"))
        if not p.name.endswith("_decode.json") and "_pframe_profile" not in p.name
    ]
    if not sidecars:
        raise FileNotFoundError(f"dcvc_int16 encode did not write a metrics JSON in {output_dir}")
    metrics_path = sidecars[-1]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    bitstream_path = Path(str(metrics["bitstream_path"]))
    if not bitstream_path.is_absolute():
        bitstream_path = (codec_root / bitstream_path).resolve()
    return {
        "bitstream_bytes": bitstream_path.read_bytes(),
        "meta": {
            "codec": "dcvc_int16",
            "frames_encoded": int(metrics.get("frames_encoded", frames)),
            "frame_index_map": [],
            "width": int(metrics.get("width", 0) or 0),
            "height": int(metrics.get("height", 0) or 0),
            "fps": float(metrics.get("fps", 0.0) or 0.0),
            "qp_i": int(qp_i),
            "qp_p": int(qp_p),
            "compressed_bytes": int(bitstream_path.stat().st_size),
            "metrics": metrics,
        },
    }


def _encode_rendered_frames_to_bytes(
    frames_iter: Iterable[Tuple[int, np.ndarray]],
    *,
    info: VideoInfo,
    codec: str,
    compression_cfg: Dict[str, Any],
    codec_root: Path,
    bundle_path: Path,
    qp_i: int,
    qp_p: int,
    device: str,
    config: str | Path | None,
    stream_name: str,
    work_dir: Path,
) -> Dict[str, Any]:
    codec = _normalize_stream_codec(codec)
    if codec in _MICROSOFT_DCVC_CODECS:
        dcvc_cfg = compression_cfg.get("dcvc", {}) or {}
        encoded = encode_dcvc_frames_to_bytes(
            frames_iter,
            info=info,
            cfg={"dcvc": dcvc_cfg, "quality": {"qp_i": qp_i, "qp_p": qp_p}},
            video_path=f"<{stream_name}_rendered_frame_stream>",
        )
        meta = dict(encoded.get("meta", {}) or {})
        meta["codec"] = "dcvc"
        encoded["meta"] = meta
        return encoded

    if codec not in _INT16_CODECS:
        if codec not in {"h264", "hevc", "av1"}:
            raise ValueError(
                f"Unsupported {stream_name} stream codec: {codec}. "
                "Use dcvc, dcvc_int16, h264, h265/hevc, or av1."
            )
        ffmpeg_cfg = compression_cfg.get("ffmpeg", {}) or {}
        quality_cfg = compression_cfg.get("quality", {}) or {}
        encoder, preset = _resolve_encoder(codec, ffmpeg_cfg)
        crf = _crf_for(codec, stream_name, quality_cfg, ffmpeg_cfg)
        return encode_frames_ffmpeg_to_bytes(
            frames_iter,
            info=info,
            codec=codec,
            crf=crf,
            encoder=encoder,
            preset=preset,
        )

    stream_dir = work_dir / stream_name
    stream_dir.mkdir(parents=True, exist_ok=True)
    temp_video = stream_dir / f"{stream_name}.mkv"
    frame_indices = _write_lossless_video(frames_iter, info=info, out_path=temp_video)
    encoded = _run_int16_encode(
        temp_video,
        output_dir=stream_dir / "encoded",
        codec_root=codec_root,
        bundle_path=bundle_path,
        qp_i=qp_i,
        qp_p=qp_p,
        frames=len(frame_indices),
        device=device,
        config=config,
    )
    meta = dict(encoded.get("meta", {}) or {})
    meta["codec"] = "dcvc_int16"
    meta["frame_index_map"] = frame_indices
    meta["frames_encoded"] = len(frame_indices)
    meta["source_video_format"] = "temporary_lossless_ffv1_mkv"
    encoded["meta"] = meta
    return encoded


def compress_keep_streams_dcvc_int16(
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

    dcvc_cfg = dict(compression_cfg.get("dcvc_int16", {}) or compression_cfg.get("dcvc", {}) or {})
    quality_cfg = compression_cfg.get("quality", {}) or {}
    roi_cfg = compression_cfg.get("roi", {}) or {}

    roi_codec = _stream_codec(compression_cfg, "roi")
    bg_codec = _stream_codec(compression_cfg, "bg")
    needs_int16 = roi_codec in _INT16_CODECS or bg_codec in _INT16_CODECS
    needs_microsoft_dcvc = roi_codec in _MICROSOFT_DCVC_CODECS or bg_codec in _MICROSOFT_DCVC_CODECS

    codec_root = _dcvc_int16_root(dcvc_cfg, root)
    bundle = _bundle_path(dcvc_cfg, codec_root)
    if needs_int16:
        if not codec_root.exists():
            raise FileNotFoundError(f"dcvc_int16 repo_dir not found: {codec_root}")
        if not bundle.exists():
            raise FileNotFoundError(f"dcvc_int16 bundle not found: {bundle}")
    if needs_microsoft_dcvc:
        repo_dir = _resolve_path(str(dcvc_cfg.get("repo_dir", "DCVC")), root)
        model_i = _resolve_path(str(dcvc_cfg.get("model_i", "")), root)
        model_p = _resolve_path(str(dcvc_cfg.get("model_p", "")), root)
        if not repo_dir.exists():
            raise FileNotFoundError(f"DCVC repo_dir not found: {repo_dir}")
        if not model_i.exists():
            raise FileNotFoundError(f"DCVC image model not found: {model_i}")
        if not model_p.exists():
            raise FileNotFoundError(f"DCVC video model not found: {model_p}")
        dcvc_cfg["repo_dir"] = str(repo_dir)
        dcvc_cfg["model_i"] = str(model_i)
        dcvc_cfg["model_p"] = str(model_p)
        compression_cfg = dict(compression_cfg)
        compression_cfg["dcvc"] = dcvc_cfg
    config = dcvc_cfg.get("config", None)
    if config:
        config_path = Path(str(config)).expanduser()
        config = str(config_path if config_path.is_absolute() else (codec_root / config_path).resolve())
    device = _device_label(dcvc_cfg)

    roi_qp_i = int(quality_cfg.get("roi_qp_i", 63))
    roi_qp_p = int(quality_cfg.get("roi_qp_p", roi_qp_i))
    bg_qp_i = int(quality_cfg.get("bg_qp_i", 25))
    bg_qp_p = int(quality_cfg.get("bg_qp_p", bg_qp_i))
    min_conf = float(roi_cfg.get("min_conf", 0.25))
    requested_dilate_px = int(roi_cfg.get("dilate_px", 0))

    src = probe_video(str(source_video))
    info = VideoInfo(width=int(src.width), height=int(src.height), fps=float(src.fps), frames=int(src.frames))
    roi_indices = _pick_indices(frame_drop_result, "roi_kept_frames", "kept_frames")
    bg_indices = _pick_indices(frame_drop_result, "bg_kept_frames", "kept_frames")

    with tempfile.TemporaryDirectory(prefix="wildroi_dcvc_int16_") as td:
        work_dir = Path(td)
        if bool(compression_cfg.get("low_memory", False)):
            roi_frames = _iter_kept_roi_frames(
                video_path=source_video,
                roi_kept_frames=roi_indices,
                roi_bbox_map=roi_bbox_map,
                roi_min_conf=min_conf,
            )
            bg_frames = _iter_kept_bg_frames(video_path=source_video, bg_kept_frames=bg_indices)
            roi_encoded = _encode_rendered_frames_to_bytes(
                roi_frames,
                info=info,
                codec=roi_codec,
                compression_cfg=compression_cfg,
                codec_root=codec_root,
                bundle_path=bundle,
                qp_i=roi_qp_i,
                qp_p=roi_qp_p,
                device=device,
                config=config,
                stream_name="roi",
                work_dir=work_dir,
            )
            bg_encoded = _encode_rendered_frames_to_bytes(
                bg_frames,
                info=info,
                codec=bg_codec,
                compression_cfg=compression_cfg,
                codec_root=codec_root,
                bundle_path=bundle,
                qp_i=bg_qp_i,
                qp_p=bg_qp_p,
                device=device,
                config=config,
                stream_name="bg",
                work_dir=work_dir,
            )
        else:
            cached = _capture_rendered_kept_frames_single_pass(
                video_path=source_video,
                info=info,
                roi_kept_frames=roi_indices,
                bg_kept_frames=bg_indices,
                roi_bbox_map=roi_bbox_map,
                roi_min_conf=min_conf,
                work_dir=work_dir,
            )
            try:
                roi_encoded = _encode_rendered_frames_to_bytes(
                    _iter_cached_frames(cached.get("roi_store"), cached.get("roi_indices", []), int(cached.get("roi_count", 0) or 0)),
                    info=info,
                    codec=roi_codec,
                    compression_cfg=compression_cfg,
                    codec_root=codec_root,
                    bundle_path=bundle,
                    qp_i=roi_qp_i,
                    qp_p=roi_qp_p,
                    device=device,
                    config=config,
                    stream_name="roi",
                    work_dir=work_dir,
                )
                bg_encoded = _encode_rendered_frames_to_bytes(
                    _iter_cached_frames(cached.get("bg_store"), cached.get("bg_indices", []), int(cached.get("bg_count", 0) or 0)),
                    info=info,
                    codec=bg_codec,
                    compression_cfg=compression_cfg,
                    codec_root=codec_root,
                    bundle_path=bundle,
                    qp_i=bg_qp_i,
                    qp_p=bg_qp_p,
                    device=device,
                    config=config,
                    stream_name="bg",
                    work_dir=work_dir,
                )
            finally:
                _close_memmap(cached.get("roi_store"))
                _close_memmap(cached.get("bg_store"))

    roi_bytes = bytes(roi_encoded["bitstream_bytes"])
    bg_bytes = bytes(bg_encoded["bitstream_bytes"])
    roi_meta = dict(roi_encoded.get("meta", {}) or {})
    bg_meta = dict(bg_encoded.get("meta", {}) or {})
    roi_indices = _resolve_frame_index_map(stream_name="ROI", encoded_meta=roi_meta, requested_indices=roi_indices)
    bg_indices = _resolve_frame_index_map(stream_name="BG", encoded_meta=bg_meta, requested_indices=bg_indices)

    meta: Dict[str, Any] = {
        "codec": roi_codec if roi_codec == bg_codec else "mixed",
        "stream_codecs": {
            "roi": roi_codec,
            "bg": bg_codec,
        },
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
            "backend": "dcvc_int16",
            "repo_dir": str(codec_root),
            "bundle_path": str(bundle),
            "device": str(device),
            "use_cuda": str(device).startswith("cuda"),
            "cuda_idx": int(str(device).split(":", 1)[1]) if str(device).startswith("cuda:") else None,
            "config": str(config) if config else None,
        },
        "streams": {
            "roi": {
                **roi_meta,
                "frame_index_map": roi_indices,
                "compressed_bytes": int(len(roi_bytes)),
            },
            "bg": {
                **bg_meta,
                "frame_index_map": bg_indices,
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
    return {"roi_bin_bytes": roi_bytes, "bg_bin_bytes": bg_bytes, "meta": meta}
