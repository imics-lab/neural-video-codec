#!/usr/bin/env python3
"""
run_upscale.py — Model S (2× diffusion super-resolution) entry point.

Usage:
    python run_upscale.py --input FRAMES_DIR_OR_VIDEO
                          [--config configs/gpu/upscaling.yaml]
                          [--output OUTPUT.mp4] [--fps 30] [--verbose]
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

LOGGER = logging.getLogger("codec.upscale")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", force=True)


def _status(msg: str) -> None:
    print(msg, flush=True)


def _load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    if yaml is not None:
        with open(p) as f:
            return yaml.safe_load(f) or {}
    with open(p) as f:
        import json
        return json.load(f)


def _load_frames(source: str) -> tuple[List[np.ndarray], float]:
    p = Path(source)
    if p.is_dir():
        paths = sorted(glob.glob(str(p / "*.png")) + glob.glob(str(p / "*.jpg")))
        if not paths:
            raise ValueError(f"No PNG/JPG files in {p}")
        return [cv2.imread(fp) for fp in paths], 30.0
    elif p.is_file():
        cap = cv2.VideoCapture(str(p))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frames = []
        while True:
            ok, frm = cap.read()
            if not ok: break
            frames.append(frm)
        cap.release()
        return frames, fps
    else:
        raise FileNotFoundError(f"Not found: {p}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model S: 2× diffusion super-resolution.")
    p.add_argument("--input",      required=True)
    p.add_argument("--config",     default="configs/gpu/upscaling.yaml")
    p.add_argument("--output",     default=None)
    p.add_argument("--fps",        type=float, default=None)
    p.add_argument("--ddim-steps", type=int,   default=None,
                   help="Override inference.ddim_steps")
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    cfg = _load_config(args.config)
    if args.ddim_steps is not None:
        cfg.setdefault("inference", {})["ddim_steps"] = args.ddim_steps

    input_path = Path(args.input)
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(cfg.get("output", {}).get("out_dir", "outputs/upscaling"))
        out_path = out_dir / (input_path.stem + "_upscaled.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _status(f"Loading from {input_path} ...")
    frames, src_fps = _load_frames(str(input_path))
    fps = args.fps or src_fps
    _status(f"Loaded {len(frames)} frames")

    _status(f"Running Model S (2× SR, {cfg.get('inference',{}).get('ddim_steps',50)} DDIM steps) ...")
    t0 = time.perf_counter()

    done = [0]
    def _cb(n: int) -> None:
        done[0] += n
        if args.verbose:
            LOGGER.debug(f"Upscaled {done[0]}/{len(frames)}")

    from src.upscaling.phase_upscale import upscale_frames
    upscaled = upscale_frames(frames, cfg, progress_cb=_cb)

    elapsed = time.perf_counter() - t0
    _status(f"Upscaling done in {elapsed:.1f}s")

    from src.postprocessing.video_assembler import assemble_video
    assemble_video(iter(upscaled), out_path, fps=fps)
    _status(f"Saved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
