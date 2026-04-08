#!/usr/bin/env python3
"""
Prepare training pairs for Model R (restoration).

For each video in the input folder:
  1. Extract frames at original quality
  2. DCVC compress → decompress the same video
  3. Save paired PNGs:
       <output_dir>/original/<video_stem>/<frame_idx:06d>.png
       <output_dir>/degraded/<video_stem>/<frame_idx:06d>.png

Usage:
    python data_prep/prepare_restoration.py
        --videos /path/to/videos
        --output /path/to/pairs
        --config configs/gpu/compression.yaml
        [--workers 1]
        [--max-videos N]
        [--verbose]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Also add DCVC repo so its src/ merges with ours as a namespace package
_dcvc = ROOT / "DCVC"
if _dcvc.exists() and str(_dcvc) not in sys.path:
    sys.path.insert(1, str(_dcvc))

LOGGER = logging.getLogger("data_prep.restoration")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", force=True)
    LOGGER.setLevel(level)


def _load_config(path: str) -> Dict[str, Any]:
    from pathlib import Path as P
    import yaml
    with open(P(path)) as f:
        return yaml.safe_load(f) or {}


def _find_videos(folder: Path, extensions: tuple = (".mp4", ".avi", ".mov", ".mkv")) -> List[Path]:
    videos = []
    for ext in extensions:
        videos.extend(folder.rglob(f"*{ext}"))
    return sorted(videos)


def _compress_decompress(video_path: Path, cfg: Dict[str, Any]) -> List[np.ndarray]:
    """Compress then decompress a video, returning degraded BGR frames."""
    from src.compression.phase_compress import compress_video
    from src.decompression.phase_decompress import decompress_archive

    LOGGER.info(f"  Compressing {video_path.name} ...")
    archive = compress_video(str(video_path), cfg)

    LOGGER.info(f"  Decompressing ...")
    decomp_cfg = {"decompression": cfg.get("compression", {}).copy()}
    # Reuse the DCVC paths from compression config
    dcvc = cfg.get("compression", {}).get("dcvc", {})
    decomp_cfg["decompression"]["dcvc"] = dcvc

    result = decompress_archive(archive, decomp_cfg)
    return result["frames"]


def _extract_original_frames(video_path: Path, resolution: Optional[tuple] = None) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frm = cap.read()
        if not ok:
            break
        if resolution is not None:
            frm = cv2.resize(frm, resolution, interpolation=cv2.INTER_AREA)
        frames.append(frm)
    cap.release()
    return frames


def _resize_video_temp(video_path: Path, resolution: Optional[tuple], max_frames: Optional[int] = None) -> Path:
    """Write a resized/trimmed copy of the video to a temp file and return its path."""
    import tempfile
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if resolution is None:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = (w, h)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    wtr = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, resolution)
    count = 0
    while True:
        ok, frm = cap.read()
        if not ok:
            break
        wtr.write(cv2.resize(frm, resolution, interpolation=cv2.INTER_AREA))
        count += 1
        if max_frames and count >= max_frames:
            break
    cap.release()
    wtr.release()
    return Path(tmp.name)


def process_video(
    video_path: Path,
    output_dir: Path,
    cfg: Dict[str, Any],
    patch_size: int = 256,
    patches_per_frame: int = 4,
    resolution: Optional[tuple] = None,
    max_frames: Optional[int] = None,
) -> int:
    """Process one video. Returns number of pairs written."""
    stem = video_path.stem
    orig_dir = output_dir / "original" / stem
    deg_dir  = output_dir / "degraded" / stem

    # Skip if already processed
    if orig_dir.exists() and any(orig_dir.iterdir()):
        existing = sum(1 for _ in orig_dir.glob("*.png"))
        LOGGER.info(f"Skipping {stem} — already has {existing} pairs")
        return existing

    orig_dir.mkdir(parents=True, exist_ok=True)
    deg_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Processing {stem} ...")
    t0 = time.perf_counter()

    # Resize video before compression so DCVC artifacts match inference resolution
    compress_path = video_path
    if resolution is not None or max_frames is not None:
        LOGGER.info(f"  Preparing input (resolution={resolution}, max_frames={max_frames}) ...")
        compress_path = _resize_video_temp(video_path, resolution, max_frames=max_frames)

    original_frames = _extract_original_frames(compress_path, resolution=None)
    degraded_frames = _compress_decompress(compress_path, cfg)

    import os
    if compress_path != video_path:
        os.unlink(compress_path)

    n = min(len(original_frames), len(degraded_frames))
    if n == 0:
        LOGGER.warning(f"  No frames decoded for {stem}")
        return 0

    pairs_written = 0
    for fi in range(n):
        orig = original_frames[fi]
        deg  = degraded_frames[fi]

        if orig.shape != deg.shape:
            deg = cv2.resize(deg, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_AREA)

        # Extract random patches to increase effective dataset size
        H, W = orig.shape[:2]
        if H >= patch_size and W >= patch_size:
            for _ in range(patches_per_frame):
                y0 = np.random.randint(0, H - patch_size + 1)
                x0 = np.random.randint(0, W - patch_size + 1)
                orig_patch = orig[y0:y0 + patch_size, x0:x0 + patch_size]
                deg_patch  = deg[y0:y0 + patch_size, x0:x0 + patch_size]
                name = f"{fi:06d}_{y0}_{x0}.png"
                cv2.imwrite(str(orig_dir / name), orig_patch)
                cv2.imwrite(str(deg_dir  / name), deg_patch)
                pairs_written += 1
        else:
            # Frame smaller than patch — save full frame
            name = f"{fi:06d}.png"
            cv2.imwrite(str(orig_dir / name), orig)
            cv2.imwrite(str(deg_dir  / name), deg)
            pairs_written += 1

    elapsed = time.perf_counter() - t0
    LOGGER.info(f"  {stem}: {pairs_written} pairs in {elapsed:.1f}s")
    return pairs_written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--videos",     required=True, help="Folder of source videos")
    p.add_argument("--output",     required=True, help="Output pairs folder")
    p.add_argument("--config",     default="configs/gpu/compression.yaml")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--patches-per-frame", type=int, default=4)
    p.add_argument("--resolution", default="1080x720",
                   help="Resize frames to WxH before compression (match inference resolution)")
    p.add_argument("--max-frames-per-video", type=int, default=150,
                   help="Limit frames per video fed to DCVC (default 150 = 5s at 30fps)")
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    cfg = _load_config(args.config)
    videos_dir = Path(args.videos)
    output_dir = Path(args.output)

    if not videos_dir.exists():
        print(f"ERROR: Videos folder not found: {videos_dir}", file=sys.stderr)
        return 1

    videos = _find_videos(videos_dir)
    if not videos:
        print(f"No videos found in {videos_dir}", file=sys.stderr)
        return 1

    if args.max_videos:
        videos = videos[:args.max_videos]

    resolution = None
    if args.resolution:
        w, h = (int(x) for x in args.resolution.lower().split("x"))
        resolution = (w, h)
        print(f"Resizing all frames to {w}x{h} to match inference resolution")

    print(f"Found {len(videos)} videos. Output -> {output_dir}")
    total_pairs = 0
    for i, vp in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {vp.name}")
        try:
            n = process_video(vp, output_dir, cfg,
                              patch_size=args.patch_size,
                              patches_per_frame=args.patches_per_frame,
                              resolution=resolution,
                              max_frames=args.max_frames_per_video)
            total_pairs += n
        except Exception as e:
            LOGGER.error(f"Failed on {vp.name}: {e}")

    print(f"\nDone. Total pairs written: {total_pairs}")
    print(f"  Originals: {output_dir}/original/")
    print(f"  Degraded:  {output_dir}/degraded/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
