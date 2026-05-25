from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from decompression import common as rd
from decompression.reconstruct import decompress_archive


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompress a codec archive to a reconstructed video."
    )
    parser.add_argument("archive_path", type=str,
                        help="Path to compressed .zip from the compression runner")
    parser.add_argument(
        "--config", type=str,
        default=str(ROOT / "configs" / "gpu" / "decompression.yaml"),
        help="Path to pipeline YAML config",
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: next to the archive)")
    parser.add_argument("--lossless-output", type=str, default=None,
                        help="Optional lossless FFV1 output for evaluation")
    parser.add_argument("--lossless-yuv420-output", type=str, default=None,
                        help="Optional lossless FFV1 yuv420p output")
    parser.add_argument("--amt-workers", type=int, default=1,
                        help="Deprecated compatibility flag (ignored)")
    parser.add_argument("--amt-batch-size", type=int, default=None)
    parser.add_argument("--amt-crop-margin", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Reconstruct only first N frames (0 = all)")
    parser.add_argument("--no-interpolate", action="store_true",
                        help="Disable ROI AMT interpolation")
    parser.add_argument("--no-roi-stabilize", action="store_true",
                        help="Disable temporal ROI stabilization")
    parser.add_argument("--roi-alpha-still", type=float, default=None)
    parser.add_argument("--roi-alpha-motion", type=float, default=None)
    parser.add_argument("--roi-mask-dilate", type=int, default=None)
    parser.add_argument("--roi-stabilize-overlap-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def main() -> None:
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass

    args = _parse_args()
    rd._setup_logging(args.verbose)

    archive_path = Path(args.archive_path).expanduser().resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = rd._load_runtime_cfg(cfg_path)

    # Resolve default output path from archive meta if not specified
    if args.output is None:
        payloads = rd._load_archive_payloads(archive_path)
        import json
        meta = json.loads(payloads["meta.json"].decode("utf-8"))
        out_path = rd._resolve_output_path(archive_path, {"output_path": None}, meta=meta)
    else:
        out_path = Path(args.output).expanduser().resolve()

    lossless_out = Path(args.lossless_output).expanduser().resolve() \
        if args.lossless_output else None
    lossless_yuv420_out = Path(args.lossless_yuv420_output).expanduser().resolve() \
        if args.lossless_yuv420_output else None

    started = time.time()
    frames, fps, w, h, _ = decompress_archive(
        archive_path, cfg,
        output_path=out_path,
        no_interpolate=bool(args.no_interpolate),
        no_roi_stabilize=bool(args.no_roi_stabilize),
        roi_alpha_still=args.roi_alpha_still,
        roi_alpha_motion=args.roi_alpha_motion,
        roi_mask_dilate=args.roi_mask_dilate,
        roi_stabilize_overlap_only=bool(args.roi_stabilize_overlap_only),
        amt_batch_size=args.amt_batch_size,
        amt_crop_margin=args.amt_crop_margin,
        max_frames=int(args.max_frames),
        lossless_output=lossless_out,
        lossless_yuv420_output=lossless_yuv420_out,
        strict_gpu=True,
    )
    elapsed = round(time.time() - started, 3)
    out_size = out_path.stat().st_size if out_path.exists() else 0
    print(f"[OK] {out_path} ({_format_bytes(out_size)}, {len(frames)} frames, {elapsed:.3f}s)")
    if lossless_out and lossless_out.exists():
        print(f"[OK] lossless: {lossless_out} ({_format_bytes(lossless_out.stat().st_size)})")
    if lossless_yuv420_out and lossless_yuv420_out.exists():
        print(f"[OK] yuv420:   {lossless_yuv420_out} "
              f"({_format_bytes(lossless_yuv420_out.stat().st_size)})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rd._log("decompression.cancelled")
        print("[ERROR] decompression cancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        rd._log("decompression.failed", error_type=type(exc).__name__, error=str(exc))
        print(f"[ERROR] decompression failed: {exc}", file=sys.stderr)
        sys.exit(1)
