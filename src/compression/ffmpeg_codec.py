from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int


def probe_video(video_path: str) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return VideoInfo(width=width, height=height, fps=fps, frames=frames)


def _coerce_positive_fps(raw_fps: Any) -> float:
    try:
        fps = float(raw_fps)
    except (TypeError, ValueError):
        fps = 30.0
    return fps if fps > 0.0 else 30.0


def _codec_file_suffix(container: str) -> str:
    normalized = str(container or "mkv").strip().lower().lstrip(".")
    if not normalized:
        normalized = "mkv"
    return f".{normalized}"


@lru_cache(maxsize=8)
def _available_ffmpeg_encoders(ffmpeg_bin: str) -> set[str]:
    proc = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"Failed to query ffmpeg encoders from {ffmpeg_bin!r}.\n{tail}")
    encoders: set[str] = set()
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Encoders:"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1].strip())
    return encoders


def _default_encoder_candidates(codec_name: str) -> List[str]:
    codec = str(codec_name).strip().lower()
    if codec == "av1":
        return ["libsvtav1", "libaom-av1", "librav1e"]
    if codec == "hevc":
        return ["libx265"]
    return []


def _resolve_encoder(stream_name: str, ffmpeg_bin: str, codec_name: str, raw_encoder: str, raw_candidates: Any) -> str:
    available = _available_ffmpeg_encoders(ffmpeg_bin)
    encoder = str(raw_encoder or "").strip()
    requested_auto = not encoder or encoder.lower() == "auto"

    if requested_auto:
        if isinstance(raw_candidates, list) and raw_candidates:
            candidates = [str(item).strip() for item in raw_candidates if str(item).strip()]
        else:
            candidates = _default_encoder_candidates(codec_name)
        for candidate in candidates:
            if candidate in available:
                return candidate
        raise RuntimeError(
            f"No supported ffmpeg encoder found for {stream_name} codec={codec_name!r}. "
            f"Tried {candidates}. Available encoders include: "
            f"{sorted(name for name in available if codec_name in name or name.startswith('lib'))[:40]}"
        )

    if encoder not in available:
        raise RuntimeError(
            f"Configured ffmpeg encoder {encoder!r} for {stream_name} is not available in {ffmpeg_bin!r}. "
            f"For portability, set compression.codec.{stream_name}.encoder=auto. "
            f"Available encoders include: "
            f"{sorted(name for name in available if codec_name in name or name.startswith('lib'))[:40]}"
        )
    return encoder


def _normalize_preset_value(preset: Any) -> str | int | None:
    if preset is None:
        return None
    if isinstance(preset, int):
        return int(preset)
    s = str(preset).strip()
    if not s:
        return None
    if s.lstrip("-").isdigit():
        return int(s)
    return s.lower()


def _stream_codec_config(cfg: Dict[str, Any], stream_name: str) -> Dict[str, Any]:
    codec_root = cfg.get("codec", {}) or {}
    if not isinstance(codec_root, dict):
        raise ValueError("compression.codec must be an object")
    stream_cfg = codec_root.get(stream_name, {}) or {}
    if not isinstance(stream_cfg, dict):
        raise ValueError(f"compression.codec.{stream_name} must be an object")
    ffmpeg_bin = str(codec_root.get("ffmpeg_bin", "ffmpeg") or "ffmpeg").strip() or "ffmpeg"
    ffprobe_bin = str(codec_root.get("ffprobe_bin", "ffprobe") or "ffprobe").strip() or "ffprobe"
    codec_name = str(stream_cfg.get("codec", stream_name)).strip().lower()
    encoder = _resolve_encoder(
        stream_name,
        ffmpeg_bin,
        codec_name,
        str(stream_cfg.get("encoder", "")).strip(),
        stream_cfg.get("encoder_candidates", None),
    )
    container = str(stream_cfg.get("container", "mkv")).strip().lower() or "mkv"
    pix_fmt = str(stream_cfg.get("pix_fmt", "yuv420p")).strip().lower() or "yuv420p"
    preset = _normalize_preset_value(stream_cfg.get("preset", None))
    qp = stream_cfg.get("qp", None)

    if qp is None:
        raise ValueError(f"compression.codec.{stream_name}.qp is required")

    try:
        qp_value = int(qp)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"compression.codec.{stream_name}.qp must be an integer") from exc

    if qp_value < 0 or qp_value > 63:
        raise ValueError(f"compression.codec.{stream_name}.qp must be in [0, 63], got {qp_value}")

    return {
        "ffmpeg_bin": ffmpeg_bin,
        "ffprobe_bin": ffprobe_bin,
        "codec": codec_name,
        "encoder": encoder,
        "container": container,
        "pix_fmt": pix_fmt,
        "preset": preset,
        "qp": qp_value,
    }


def _encoder_args(stream_cfg: Dict[str, Any]) -> List[str]:
    encoder = str(stream_cfg.get("encoder", "")).strip().lower()
    preset = stream_cfg.get("preset", None)
    qp = int(stream_cfg.get("qp", 0))
    pix_fmt = str(stream_cfg.get("pix_fmt", "yuv420p"))

    args: List[str] = ["-an", "-c:v", str(stream_cfg["encoder"]), "-pix_fmt", pix_fmt]
    if encoder == "libsvtav1":
        if preset is not None:
            if isinstance(preset, str):
                svt_map = {
                    "slowest": 0,
                    "slow": 2,
                    "medium": 6,
                    "fast": 10,
                    "fastest": 13,
                }
                preset = svt_map.get(preset, preset)
            args.extend(["-preset", str(preset)])
        args.extend(["-qp", str(qp)])
        return args
    if encoder == "libaom-av1":
        cpu_used = preset if isinstance(preset, int) else {
            None: 0,
            "slowest": 0,
            "slow": 2,
            "medium": 4,
            "fast": 6,
            "fastest": 8,
        }.get(preset, 0)
        args.extend(["-cpu-used", str(int(cpu_used)), "-crf", str(qp), "-usage", "good", "-row-mt", "1"])
        return args
    if encoder == "librav1e":
        speed = preset if isinstance(preset, int) else {
            None: 0,
            "slowest": 0,
            "slow": 2,
            "medium": 5,
            "fast": 8,
            "fastest": 10,
        }.get(preset, 0)
        args.extend(["-speed", str(int(speed)), "-qp", str(qp)])
        return args
    if encoder == "libx265":
        if preset is not None and str(preset).strip():
            args.extend(["-preset", str(preset)])
        args.extend(["-qp", str(qp), "-x265-params", "log-level=error"])
        return args
    raise ValueError(f"Unsupported ffmpeg encoder configured: {stream_cfg['encoder']}")


def _read_stderr_tail(proc: subprocess.Popen[bytes]) -> str:
    if proc.stderr is None:
        return ""
    try:
        stderr_text = proc.stderr.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
    if not stderr_text:
        return ""
    return "\n".join(stderr_text.splitlines()[-20:])


def encode_ffmpeg_frames_to_bytes(
    frames: Iterable[Tuple[int, np.ndarray]],
    *,
    info: VideoInfo,
    cfg: Dict[str, Any],
    stream_name: str,
    video_path: str = "<rendered_frame_stream>",
) -> Dict[str, Any]:
    stream_cfg = _stream_codec_config(cfg, stream_name)
    ffmpeg_bin = str(stream_cfg["ffmpeg_bin"])
    output_suffix = _codec_file_suffix(str(stream_cfg["container"]))
    fps = _coerce_positive_fps(info.fps)
    width = int(info.width)
    height = int(info.height)

    frame_index_map: List[int] = []

    with tempfile.TemporaryDirectory(prefix=f"wildroi_encode_{stream_name}_") as td:
        out_path = Path(td) / f"{stream_name}{output_suffix}"
        cmd = [
            ffmpeg_bin,
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
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
        ]
        cmd.extend(_encoder_args(stream_cfg))
        cmd.append(str(out_path))

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        frames_encoded = 0
        try:
            if proc.stdin is None:
                raise RuntimeError("ffmpeg stdin pipe is unavailable for encoding")
            for src_idx, frame in frames:
                if frame.dtype != np.uint8:
                    raise ValueError(f"{stream_name} encoder requires uint8 frames")
                if frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError(f"{stream_name} encoder requires HxWx3 frames")
                if frame.shape[0] != height or frame.shape[1] != width:
                    raise ValueError(
                        f"{stream_name} frame shape mismatch: expected {height}x{width}, "
                        f"got {frame.shape[0]}x{frame.shape[1]}"
                    )
                try:
                    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                except BrokenPipeError as exc:
                    rc = proc.wait(timeout=5)
                    stderr_tail = _read_stderr_tail(proc)
                    tail = stderr_tail or "(no ffmpeg stderr)"
                    raise RuntimeError(
                        f"ffmpeg {stream_name} encode terminated early with exit code {rc} while writing frames.\n"
                        f"command: {' '.join(str(part) for part in cmd)}\n{tail}"
                    ) from exc
                frame_index_map.append(int(src_idx))
                frames_encoded += 1
        finally:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        rc = proc.wait()
        stderr_tail = _read_stderr_tail(proc)
        if rc != 0:
            tail = stderr_tail or "(no ffmpeg stderr)"
            raise RuntimeError(f"ffmpeg {stream_name} encode failed with exit code {rc}.\n{tail}")
        if frames_encoded <= 0:
            raise RuntimeError(f"{stream_name} encoder received zero frames")
        if not out_path.exists():
            raise RuntimeError(f"ffmpeg {stream_name} encode did not create output: {out_path}")

        bitstream = out_path.read_bytes()

    meta = {
        "video_path": str(video_path),
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "frames_total": int(info.frames),
        "frames_encoded": int(frames_encoded),
        "frame_index_map": frame_index_map,
        "codec": str(stream_cfg["codec"]),
        "encoder": str(stream_cfg["encoder"]),
        "container": str(stream_cfg["container"]),
        "pix_fmt": str(stream_cfg["pix_fmt"]),
        "preset": stream_cfg.get("preset", None),
        "qp": int(stream_cfg["qp"]),
        "compressed_bytes": int(len(bitstream)),
        "ffmpeg_bin": ffmpeg_bin,
        "ffprobe_bin": str(stream_cfg["ffprobe_bin"]),
        "execution_backend": "ffmpeg",
    }
    return {"bitstream_bytes": bitstream, "meta": meta}
