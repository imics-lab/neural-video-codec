#!/usr/bin/env python3
"""
generate_wan_bookends.py — Generate videos from first and last frames using Wan2.1 I2V.

Extracts the first and last frame of an input video, runs Wan2.1 I2V on each,
and saves the results as separate videos for comparison.

Usage:
    python generate_wan_bookends.py --input video.mp4 --config configs/gpu/compression.yaml
    python generate_wan_bookends.py --input video.mp4 --num-frames 49 --steps 30
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _frame_to_bgr(f) -> np.ndarray:
    if isinstance(f, np.ndarray):
        arr = f if f.dtype == np.uint8 else (f * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)


def load_pipe(model_id: str, cpu_offload: bool):
    from diffusers import WanImageToVideoPipeline, FlowMatchEulerDiscreteScheduler
    print(f"[wan] Loading {model_id} ...", flush=True)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.vae.enable_slicing()
    print("[wan] Model loaded.", flush=True)
    return pipe


def generate(pipe, frame_bgr: np.ndarray, out_w: int, out_h: int,
             num_frames: int, steps: int, guidance: float,
             prompt: str, neg_prompt: str, seed: int) -> list:
    pil = _bgr_to_pil(cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output = pipe(
        image               = pil,
        prompt              = prompt,
        negative_prompt     = neg_prompt,
        height              = out_h,
        width               = out_w,
        num_frames          = num_frames,
        num_inference_steps = steps,
        guidance_scale      = guidance,
        generator           = generator,
    )
    raw = output.frames
    if isinstance(raw, np.ndarray):
        while raw.ndim > 4:
            raw = raw[0]
        if raw.ndim == 3:
            raw = raw[np.newaxis]
        frames = [_frame_to_bgr(raw[i]) for i in range(len(raw))]
    else:
        flat = raw
        while flat and isinstance(flat[0], (list, tuple)):
            flat = flat[0]
        frames = [_frame_to_bgr(f) for f in flat]
    return frames


def save_video(frames: list, path: Path, fps: float):
    from src.postprocessing.video_assembler import assemble_video
    path.parent.mkdir(parents=True, exist_ok=True)
    assemble_video(iter(frames), path, fps=fps)
    print(f"Saved {len(frames)} frames → {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       required=True, help="Input video")
    p.add_argument("--config",      default="configs/gpu/compression.yaml")
    p.add_argument("--output-dir",  default="outputs/wan_bookends")
    p.add_argument("--num-frames",  type=int,   default=49)
    p.add_argument("--steps",       type=int,   default=20)
    p.add_argument("--guidance",    type=float, default=5.0)
    p.add_argument("--out-w",       type=int,   default=None)
    p.add_argument("--out-h",       type=int,   default=None)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--no-cpu-offload", action="store_true")
    args = p.parse_args()

    try:
        import yaml
        cfg = yaml.safe_load(open(args.config)) or {}
    except Exception:
        cfg = {}

    wan_cfg    = cfg.get("wan_upscaling", {}) or {}
    model_id   = wan_cfg.get("model_id",  "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    out_w      = args.out_w   or wan_cfg.get("out_w",  960)
    out_h      = args.out_h   or wan_cfg.get("out_h",  720)
    cpu_offload = not args.no_cpu_offload

    prompt = (
        "high quality, sharp details, no compression artifacts, "
        "professional wildlife video, natural motion, natural colors"
    )
    neg_prompt = "blurry, low resolution, compression artifacts, noise, flickering"

    # ── Extract first and last frames ─────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, first_frame = cap.read()
    if not ok:
        print("ERROR: could not read input video"); return 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ok, last_frame = cap.read()
    if not ok:
        last_frame = first_frame
    cap.release()

    print(f"Input: {args.input}  ({total} frames @ {fps:.1f} fps)")
    print(f"Generating {args.num_frames} frames at {out_w}×{out_h} from first and last frames")

    stem     = Path(args.input).stem
    out_dir  = Path(args.output_dir)
    out_from_first = out_dir / f"{stem}_wan_from_first.mp4"
    out_from_last  = out_dir / f"{stem}_wan_from_last.mp4"

    # ── Load model once ───────────────────────────────────────────────────────
    pipe = load_pipe(model_id, cpu_offload)

    gen_kwargs = dict(
        out_w=out_w, out_h=out_h,
        num_frames=args.num_frames, steps=args.steps,
        guidance=args.guidance, prompt=prompt, neg_prompt=neg_prompt,
        seed=args.seed,
    )

    # ── Generate from first frame ─────────────────────────────────────────────
    print("\n[wan] Generating from first frame ...", flush=True)
    frames_from_first = generate(pipe, first_frame, **gen_kwargs)
    save_video(frames_from_first, out_from_first, fps)

    # ── Generate from last frame ──────────────────────────────────────────────
    print("\n[wan] Generating from last frame ...", flush=True)
    frames_from_last = generate(pipe, last_frame, **gen_kwargs)
    save_video(frames_from_last, out_from_last, fps)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
