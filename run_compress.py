#!/usr/bin/env python3
"""
run_compress.py — Edge-side compression entry point.

Usage:
    python run_compress.py --video INPUT.mp4 [--config configs/gpu/compression.yaml]
                           [--output OUTPUT.zip] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

LOGGER = logging.getLogger("codec.compress")
_VERBOSE = False


def _setup_logging(verbose: bool) -> None:
    global _VERBOSE
    _VERBOSE = verbose
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", force=True)
    LOGGER.setLevel(level)
    if not verbose:
        for name in ("ultralytics", "onnxruntime"):
            logging.getLogger(name).setLevel(logging.ERROR)
        os.environ["YOLO_VERBOSE"] = "false"


def _status(msg: str) -> None:
    print(msg, flush=True)


def _load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if yaml is not None:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    with open(path) as f:
        import json as _json
        return _json.load(f)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DCVC dual-stream compression with YOLO11 ROI detection."
    )
    p.add_argument("--video",   required=True, help="Input video file path")
    p.add_argument("--config",  default="configs/gpu/compression.yaml",
                   help="Compression config YAML (default: configs/gpu/compression.yaml)")
    p.add_argument("--output",  default=None,
                   help="Output ZIP path (default: outputs/compression/<stem>.zip)")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    p.add_argument("--downscale", type=float, default=1.0,
                   help="Downscale input by this factor before compression (eval mode)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}", file=sys.stderr)
        return 1

    cfg = _load_config(args.config)

    # Resolve output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(cfg.get("output", {}).get("out_dir", "outputs/compression"))
        out_path = out_dir / (video_path.stem + ".zip")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply eval-mode downscale
    if abs(args.downscale - 1.0) > 1e-4:
        _status(f"Eval mode: downscaling input by {args.downscale}×")
        # Write downscaled temp file
        import tempfile, cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  * args.downscale)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.downscale)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        while True:
            ok, frm = cap.read()
            if not ok:
                break
            writer.write(cv2.resize(frm, (W, H), interpolation=cv2.INTER_AREA))
        cap.release(); writer.release()
        video_path = Path(tmp.name)

    _status(f"Compressing {video_path} ...")
    t0 = time.perf_counter()

    def _cb(stage: str, n: int) -> None:
        if _VERBOSE:
            LOGGER.debug(f"[{stage}] +{n}")

    from src.compression.phase_compress import compress_video
    archive_bytes = compress_video(str(video_path), cfg, progress_cb=_cb)

    with open(out_path, "wb") as f:
        f.write(archive_bytes)

    elapsed = time.perf_counter() - t0
    size_mb = len(archive_bytes) / 1e6
    _status(f"Done in {elapsed:.1f}s — {size_mb:.2f} MB → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
