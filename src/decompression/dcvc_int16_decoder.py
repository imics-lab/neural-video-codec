from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _remap_container_path(raw_path: str) -> Optional[Path]:
    p = Path(str(raw_path)).expanduser()
    if not p.is_absolute():
        return None
    parts = p.parts
    if len(parts) >= 3 and str(parts[1]).lower() == "app":
        return (ROOT / Path(*parts[2:])).resolve()
    return None


def _resolve_runtime_path(raw_path: str, *, kind: str) -> Path:
    p = Path(str(raw_path)).expanduser()
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
        mapped = _remap_container_path(str(p))
        if mapped is not None:
            candidates.append(mapped)
    else:
        base = ROOT / "dcvc_int16" if kind == "bundle" else ROOT
        candidates.append((base / p).resolve())
    if kind == "repo":
        candidates.append((ROOT / "dcvc_int16").resolve())
    elif kind == "bundle":
        candidates.append((ROOT / "models" / p.name).resolve())
        candidates.append((ROOT / "dcvc_int16" / "models" / p.name).resolve())
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0].resolve() if candidates else p.resolve()


def _read_bgr_frames_from_yuv420(yuv_path: Path, *, width: int, height: int, limit: Optional[int]) -> List[np.ndarray]:
    frame_size = int(width) * int(height) * 3 // 2
    frames: List[np.ndarray] = []
    with open(yuv_path, "rb") as fp:
        while True:
            if limit is not None and len(frames) >= int(limit):
                break
            raw = fp.read(frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                raise RuntimeError(f"Truncated dcvc_int16 YUV frame in {yuv_path}")
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((int(height) * 3 // 2, int(width)))
            frames.append(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420))
    return frames


def decode_stream_bytes_dcvc_int16(
    stream_bytes: bytes,
    meta: Dict[str, Any],
    stream: str,
    *,
    frame_count_hint: Optional[int] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> List[np.ndarray]:
    dcvc_cfg = meta.get("dcvc", {}) or {}
    video_info = meta.get("video", {}) or {}
    stream_meta = (meta.get("streams", {}) or {}).get(stream, {}) or {}

    width = int(video_info.get("width", stream_meta.get("width", 0)) or 0)
    height = int(video_info.get("height", stream_meta.get("height", 0)) or 0)
    fps = float(video_info.get("fps", stream_meta.get("fps", 30.0)) or 30.0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid width/height in dcvc_int16 archive metadata")

    codec_root = _resolve_runtime_path(str(dcvc_cfg.get("repo_dir", "dcvc_int16")), kind="repo")
    bundle_path = _resolve_runtime_path(
        str(dcvc_cfg.get("bundle_path", "models/int16_bundle_v1.0.0.pt")),
        kind="bundle",
    )
    if not codec_root.exists():
        raise FileNotFoundError(f"dcvc_int16 repo_dir not found: {codec_root}")
    if not bundle_path.exists():
        raise FileNotFoundError(f"dcvc_int16 bundle not found: {bundle_path}")

    device = str(dcvc_cfg.get("device", "cuda:0"))
    if device == "cuda" and dcvc_cfg.get("cuda_idx", None) is not None:
        device = f"cuda:{int(dcvc_cfg['cuda_idx'])}"
    config = dcvc_cfg.get("config", None)

    with tempfile.TemporaryDirectory(prefix=f"wildroi_dcvc_int16_decode_{stream}_") as td:
        work = Path(td)
        input_bin = work / f"{stream}.bin"
        output_mp4 = work / f"{stream}_decoded.mp4"
        input_bin.write_bytes(stream_bytes)
        sidecar = {
            "fps": fps,
            "width": width,
            "height": height,
            "frames_encoded": int(frame_count_hint or stream_meta.get("frames_encoded", 0) or 0),
        }
        input_bin.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
        cmd = [
            sys.executable,
            str(codec_root / "decode_bin_to_mp4.py"),
            "--input_bin",
            str(input_bin),
            "--bundle_path",
            str(bundle_path),
            "--output_mp4",
            str(output_mp4),
            "--fps",
            str(fps),
            "--device",
            device,
            "--keep_yuv",
        ]
        if config:
            cfg_path = Path(str(config)).expanduser()
            if not cfg_path.is_absolute():
                cfg_path = (codec_root / cfg_path).resolve()
            cmd.extend(["--config", str(cfg_path)])
        completed = subprocess.run(cmd, cwd=str(codec_root), capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"dcvc_int16 decode failed ({completed.returncode}).\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        yuv_path = output_mp4.with_name(f"{output_mp4.stem}_temp.yuv")
        if not yuv_path.exists():
            raise FileNotFoundError(f"dcvc_int16 decode did not keep expected YUV output: {yuv_path}")
        frames = _read_bgr_frames_from_yuv420(
            yuv_path,
            width=width,
            height=height,
            limit=frame_count_hint,
        )
    if progress_cb is not None:
        for _ in frames:
            progress_cb(1)
    return frames
