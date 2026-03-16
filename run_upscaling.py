"""
run_upscaling.py — Main CLI entry point for the diffusion upscaling pipeline.

Usage:
    python run_upscaling.py INPUT_VIDEO [options]

Options:
    --output PATH       Output video path (default: output/upscaled.mp4)
    --config PATH       Config file (default: configs/gpu/upscaling.yaml)
    --roi_json PATH     ROI detections JSON (optional; uniform mask if omitted)
    --ddim_steps INT    Override number of DDIM denoising steps
    --checkpoint PATH   Override model checkpoint path
    --scale INT         Override scale factor from config
    --verbose           Show detailed diagnostic logs
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

LOGGER = logging.getLogger("neural_video_codec.upscaling")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", force=True)
    LOGGER.setLevel(level)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diffusion-based video super-resolution inference",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_video",
        help="Path to the LR input video",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/upscaled.mp4",
        help="Output video path (default: output/upscaled.mp4)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "gpu" / "upscaling.yaml"),
        help="Config YAML file (default: configs/gpu/upscaling.yaml)",
    )
    parser.add_argument(
        "--roi_json",
        type=str,
        default=None,
        help="ROI detections JSON (optional; uniform mask used if omitted)",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=None,
        help="Override number of DDIM denoising steps",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override model checkpoint path",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=None,
        help="Override scale factor from config",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed diagnostic logs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    # 1. Hard-fail if CUDA unavailable
    try:
        import torch
    except ImportError:
        print(
            "[ERROR] PyTorch is not installed. Install it with:\n"
            "  pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu126",
            file=sys.stderr,
        )
        sys.exit(1)

    if not torch.cuda.is_available():
        print(
            "[ERROR] CUDA is not available. This pipeline requires a CUDA-capable GPU.\n"
            "  - Ensure NVIDIA drivers (535+) and CUDA 12.x are installed.\n"
            "  - Verify with: python -c \"import torch; print(torch.cuda.is_available())\"",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Load config via OmegaConf
    try:
        from omegaconf import OmegaConf
    except ImportError:
        print(
            "[ERROR] omegaconf is not installed. Install it with: pip install omegaconf",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    if not cfg_path.exists():
        print(f"[ERROR] Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = OmegaConf.load(str(cfg_path))

    # 3. Apply CLI overrides
    if args.scale is not None:
        if args.scale < 1:
            print(f"[ERROR] --scale must be >= 1, got {args.scale}", file=sys.stderr)
            sys.exit(1)
        cfg.scale = args.scale

    if args.ddim_steps is not None:
        if args.ddim_steps < 1:
            print(f"[ERROR] --ddim_steps must be >= 1, got {args.ddim_steps}", file=sys.stderr)
            sys.exit(1)
        cfg.inference.ddim_steps = args.ddim_steps

    if args.checkpoint is not None:
        cfg.model.checkpoint = args.checkpoint

    # 4. Validate and resolve checkpoint path
    ckpt_path = Path(cfg.model.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = (ROOT / ckpt_path).resolve()
    if not ckpt_path.exists():
        print(
            f"[ERROR] Model checkpoint not found: {ckpt_path}\n\n"
            "To download the model:\n"
            "  1. See models/README.md for download instructions.\n"
            "  2. Download diffusion_4x.pth from the GitHub Releases page.\n"
            "  3. Place it in the models/ directory.\n"
            "  4. Verify SHA256 against models/models.manifest.json.",
            file=sys.stderr,
        )
        sys.exit(1)
    cfg.model.checkpoint = str(ckpt_path)

    # 5. Validate input video exists
    input_video = Path(args.input_video)
    if not input_video.is_absolute():
        input_video = (ROOT / input_video).resolve()
    if not input_video.exists():
        print(f"[ERROR] Input video not found: {input_video}", file=sys.stderr)
        sys.exit(1)

    # 6. Resolve output path and create parent directory
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 7. Validate config fields
    try:
        from pipeline.config_schema import validate_upscaling_config
        validate_upscaling_config(OmegaConf.to_container(cfg, resolve=True), root_dir=ROOT)
    except Exception as exc:
        print(f"[ERROR] Config validation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # 8. Build typed config and run upscaling phase
    from upscaling.phase_upscale import UpscalingPhase

    # Convert OmegaConf to a simple namespace-style object via a wrapper
    # that UpscalingPhase/Upscaler accesses by attribute
    cfg_obj = OmegaConf.to_object(cfg)

    started = time.time()
    LOGGER.info(f"Starting upscaling: {input_video}")

    phase = UpscalingPhase(cfg_obj)
    phase.run(
        input_video=str(input_video),
        roi_json=args.roi_json,
        output_video=str(output_path),
    )

    duration_sec = round(time.time() - started, 3)
    print(f"[OK] Upscaled video written to: {output_path} ({duration_sec:.3f}s)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ERROR] Cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
