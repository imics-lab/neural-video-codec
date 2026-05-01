#!/usr/bin/env python3
"""
generate_wan_bookends.py — Generate full video using Wan2.1 I2V autoregressively
                           from the first frame (forward) and from the last frame
                           (backward, then reversed).

Each chunk is conditioned on the last frame of the previous chunk, so the
generation flows naturally rather than resetting to the same endpoint every time.
Both outputs match the original video length exactly.

Usage:
    python generate_wan_bookends.py --input video.mp4 --config configs/gpu/compression.yaml
    python generate_wan_bookends.py --input video.mp4 --steps 30 --chunk-frames 17
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


def generate_chunk(pipe, cond_bgr: np.ndarray, out_w: int, out_h: int,
                   num_frames: int, steps: int, guidance: float,
                   prompt: str, neg_prompt: str, seed: int) -> list:
    """Generate num_frames from a single conditioning BGR frame."""
    pil = _bgr_to_pil(cv2.resize(cond_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4))
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
        return [_frame_to_bgr(raw[i]) for i in range(len(raw))]
    else:
        flat = raw
        while flat and isinstance(flat[0], (list, tuple)):
            flat = flat[0]
        return [_frame_to_bgr(f) for f in flat]


def generate_full_video(pipe, cond_bgr: np.ndarray, total_frames: int,
                        out_w: int, out_h: int, chunk_frames: int, overlap: int,
                        steps: int, guidance: float,
                        prompt: str, neg_prompt: str, seed: int,
                        reverse: bool = False) -> list:
    """
    Generate a video of total_frames length, autoregressively conditioned:
    each chunk is conditioned on the last frame of the previous chunk.
    When reverse=True the sequence is flipped at the end (for from-last mode).
    """
    step         = max(1, chunk_frames - overlap)
    chunk_starts = list(range(0, total_frames, step))
    if chunk_starts[-1] + chunk_frames < total_frames:
        chunk_starts.append(total_frames - chunk_frames)

    acc   = np.zeros((total_frames, out_h, out_w, 3), dtype=np.float32)
    count = np.zeros((total_frames,), dtype=np.float32)

    cur_cond = cond_bgr
    for ci, start in enumerate(chunk_starts):
        end    = min(start + chunk_frames, total_frames)
        n      = end - start
        print(f"[wan] chunk {ci+1}/{len(chunk_starts)}  frames {start}–{end-1}", flush=True)
        result = generate_chunk(pipe, cur_cond, out_w, out_h, n,
                                steps, guidance, prompt, neg_prompt, seed)
        while len(result) < n:
            result.append(result[-1].copy())

        cur_cond = result[-1]   # condition next chunk on last generated frame

        for li, fi in enumerate(range(start, end)):
            weight = 1.0
            if li < overlap and ci > 0:
                weight = (li + 1) / (overlap + 1)
            elif li >= n - overlap and ci < len(chunk_starts) - 1:
                weight = (n - li) / (overlap + 1)
            acc[fi]   += result[li].astype(np.float32) * weight
            count[fi] += weight

    frames = [np.clip(acc[fi] / max(count[fi], 1e-6), 0, 255).astype(np.uint8)
              for fi in range(total_frames)]
    return frames[::-1] if reverse else frames


def save_video(frames: list, path: Path, fps: float):
    from src.postprocessing.video_assembler import assemble_video
    path.parent.mkdir(parents=True, exist_ok=True)
    assemble_video(iter(frames), path, fps=fps)
    print(f"Saved {len(frames)} frames → {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",          required=True, help="Input video")
    p.add_argument("--config",         default="configs/gpu/compression.yaml")
    p.add_argument("--output-dir",     default="outputs/wan_bookends")
    p.add_argument("--chunk-frames",   type=int,   default=17,
                   help="Frames per Wan2.1 call (default 17)")
    p.add_argument("--overlap-frames", type=int,   default=4)
    p.add_argument("--steps",          type=int,   default=20)
    p.add_argument("--guidance",       type=float, default=5.0)
    p.add_argument("--out-w",          type=int,   default=None)
    p.add_argument("--out-h",          type=int,   default=None)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--no-cpu-offload", action="store_true")
    args = p.parse_args()

    try:
        import yaml
        cfg = yaml.safe_load(open(args.config)) or {}
    except Exception:
        cfg = {}

    wan_cfg     = cfg.get("wan_upscaling", {}) or {}
    model_id    = wan_cfg.get("model_id", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    out_w       = args.out_w  or wan_cfg.get("out_w", 960)
    out_h       = args.out_h  or wan_cfg.get("out_h", 720)
    cpu_offload = not args.no_cpu_offload

    prompt = (
        "high quality, sharp details, no compression artifacts, "
        "professional wildlife video, natural motion, natural colors"
    )
    neg_prompt = "blurry, low resolution, compression artifacts, noise, flickering"

    # ── Extract first and last frames, get total length ───────────────────────
    cap   = cv2.VideoCapture(args.input)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, first_frame = cap.read()
    assert ok, "Could not read input video"

    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ok, last_frame = cap.read()
    if not ok:
        last_frame = first_frame
    cap.release()

    print(f"Input: {args.input}  ({total} frames @ {fps:.1f} fps)")
    print(f"Output resolution: {out_w}×{out_h}  chunk={args.chunk_frames}  overlap={args.overlap_frames}")

    stem          = Path(args.input).stem
    out_dir       = Path(args.output_dir)
    out_from_first = out_dir / f"{stem}_wan_from_first.mp4"
    out_from_last  = out_dir / f"{stem}_wan_from_last.mp4"

    gen_kwargs = dict(
        total_frames=total, out_w=out_w, out_h=out_h,
        chunk_frames=args.chunk_frames, overlap=args.overlap_frames,
        steps=args.steps, guidance=args.guidance,
        prompt=prompt, neg_prompt=neg_prompt, seed=args.seed,
    )

    # ── Load model once ───────────────────────────────────────────────────────
    pipe = load_pipe(model_id, cpu_offload)

    print(f"\n[wan] Generating {total} frames from first frame (autoregressive forward) ...", flush=True)
    frames_first = generate_full_video(pipe, first_frame, **gen_kwargs, reverse=False)
    save_video(frames_first, out_from_first, fps)

    print(f"\n[wan] Generating {total} frames from last frame (autoregressive backward) ...", flush=True)
    frames_last = generate_full_video(pipe, last_frame, **gen_kwargs, reverse=True)
    save_video(frames_last, out_from_last, fps)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
