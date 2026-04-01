#!/usr/bin/env python3
"""
run_upscale.py — Super-resolution using Real-ESRGAN (pretrained, no training required).

Uses RealESRGAN_x4plus (4× upscaling) by default, or x2plus for 2× upscaling.

Usage:
    python run_upscale.py --input FRAMES_DIR_OR_VIDEO
                          [--output OUTPUT.mp4]
                          [--scale 4]
                          [--model-path weights/RealESRGAN_x4plus.pth]
                          [--tile 512]
                          [--fps 30]
                          [--verbose]
"""
from __future__ import annotations

import argparse
import logging
import sys
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
import time
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger("codec.upscale")

_MODEL_URLS = {
    4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}
_DEFAULT_WEIGHTS = {
    4: "weights/RealESRGAN_x4plus.pth",
    2: "weights/RealESRGAN_x2plus.pth",
}


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", force=True)


def _status(msg: str) -> None:
    print(msg, flush=True)


def _load_frames(source: Path) -> tuple[list[np.ndarray], float]:
    if source.is_dir():
        paths = sorted(source.glob("*.png")) + sorted(source.glob("*.jpg"))
        if not paths:
            raise ValueError(f"No PNG/JPG files in {source}")
        return [cv2.imread(str(p)) for p in paths], 30.0
    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while True:
        ok, frm = cap.read()
        if not ok:
            break
        frames.append(frm)
    cap.release()
    return frames, fps


def _download_weights(scale: int, dest: Path) -> None:
    import urllib.request
    url = _MODEL_URLS[scale]
    dest.parent.mkdir(parents=True, exist_ok=True)
    _status(f"Downloading Real-ESRGAN x{scale} weights from {url} ...")
    urllib.request.urlretrieve(url, dest)
    _status(f"Saved to {dest}")


def _build_upsampler(scale: int, model_path: Path, tile: int, half: bool):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    if scale == 4:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)

    upsampler = RealESRGANer(
        scale=scale,
        model_path=str(model_path),
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=half,
    )
    return upsampler


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Super-resolution with Real-ESRGAN (pretrained).")
    p.add_argument("--input",       required=True,  help="Input video or frames directory")
    p.add_argument("--output",      default=None,   help="Output video path")
    p.add_argument("--scale",       type=int, default=4, choices=[2, 4],
                   help="Upscale factor (default: 4)")
    p.add_argument("--model-path",  default=None,
                   help="Path to .pth weights (auto-downloaded if missing)")
    p.add_argument("--tile",        type=int, default=512,
                   help="Tile size for large frames (0=no tiling, default: 512)")
    p.add_argument("--half",        action="store_true",
                   help="Use FP16 inference (faster, slightly lower quality)")
    p.add_argument("--fps",         type=float, default=None)
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    input_path = Path(args.input)
    if not input_path.exists():
        _status(f"ERROR: Input not found: {input_path}")
        return 1

    scale = args.scale
    model_path = Path(args.model_path) if args.model_path else Path(_DEFAULT_WEIGHTS[scale])

    if not model_path.exists():
        _download_weights(scale, model_path)

    out_path = Path(args.output) if args.output else (
        Path("outputs/upscaling") / (input_path.stem + f"_x{scale}.mp4")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _status(f"Loading frames from {input_path} ...")
    frames, src_fps = _load_frames(input_path)
    fps = args.fps or src_fps
    _status(f"Loaded {len(frames)} frames at {fps:.1f} fps")

    _status(f"Loading Real-ESRGAN x{scale} from {model_path} ...")
    upsampler = _build_upsampler(scale, model_path, tile=args.tile, half=args.half)

    _status(f"Upscaling {len(frames)} frames ...")
    t0 = time.perf_counter()
    upscaled = []
    for i, frame in enumerate(frames):
        out_frame, _ = upsampler.enhance(frame, outscale=scale)
        upscaled.append(out_frame)
        if (i + 1) % 50 == 0 or (i + 1) == len(frames):
            _status(f"  {i+1}/{len(frames)} frames done")

    elapsed = time.perf_counter() - t0
    _status(f"Upscaling done in {elapsed:.1f}s ({len(frames)/elapsed:.1f} fps)")

    _status(f"Assembling video -> {out_path} ...")
    h, w = upscaled[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for frame in upscaled:
        writer.write(frame)
    writer.release()
    _status(f"Saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
