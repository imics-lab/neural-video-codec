#!/usr/bin/env python3
"""
run_decompress.py — Server-side decompression entry point.

Usage:
    python run_decompress.py --archive INPUT.zip [--config configs/gpu/decompression.yaml]
                             [--output OUTPUT.mp4] [--verbose]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

LOGGER = logging.getLogger("codec.decompress")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", force=True)
    LOGGER.setLevel(level)


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DCVC dual-stream decompression.")
    p.add_argument("--archive", required=True, help="Compressed ZIP archive")
    p.add_argument("--config",  default="configs/gpu/decompression.yaml")
    p.add_argument("--output",  default=None, help="Output MP4 path")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    archive_path = Path(args.archive)
    if not archive_path.exists():
        print(f"ERROR: Archive not found: {archive_path}", file=sys.stderr)
        return 1

    cfg = _load_config(args.config)

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(cfg.get("output", {}).get("out_dir", "outputs/decompression"))
        out_path = out_dir / (archive_path.stem + "_decompressed.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _status(f"Decompressing {archive_path} ...")
    t0 = time.perf_counter()

    def _cb(stage: str, n: int) -> None:
        if args.verbose:
            LOGGER.debug(f"[{stage}] +{n}")

    from src.decompression.phase_decompress import decompress_archive
    result = decompress_archive(archive_path, cfg, progress_cb=_cb)

    frames = result["frames"]
    fps    = result["fps"]

    from src.postprocessing.video_assembler import assemble_video
    assemble_video(iter(frames), out_path, fps=fps)

    elapsed = time.perf_counter() - t0
    _status(f"Done in {elapsed:.1f}s -- {len(frames)} frames -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
