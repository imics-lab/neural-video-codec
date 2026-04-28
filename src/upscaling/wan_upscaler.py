"""
wan_upscaler.py — Video upscaling via Wan2.1 Image-to-Video.

Wan2.1 I2V takes the first frame of each chunk as a conditioning image and
generates a temporally-coherent high-resolution video segment.  We process
the input video in overlapping chunks; the first frame of each chunk is used
as the conditioning image so content is preserved.

Requires: pip install diffusers transformers accelerate
Model:     Wan-AI/Wan2.1-I2V-14B-480P  (or -720P for 960×720 output)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image


_DEFAULT_MODEL = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
_DEFAULT_PROMPT = (
    "high quality, sharp details, no compression artifacts, "
    "professional video, natural motion"
)
_NEG_PROMPT = (
    "blurry, low resolution, compression artifacts, noise, "
    "oversaturated, cartoon"
)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


class WanUpscaler:
    """
    Upscale a list of BGR frames using Wan2.1 I2V in temporal chunks.

    Config keys (all optional):
        model_id        (str)   HuggingFace model ID
        out_w / out_h   (int)   output resolution (default 960×720)
        chunk_frames    (int)   frames per Wan2.1 call, must be odd (default 17)
        overlap_frames  (int)   frames shared between adjacent chunks (default 4)
        num_steps       (int)   diffusion steps (default 20)
        guidance_scale  (float) CFG guidance (default 5.0)
        prompt          (str)   positive text prompt
        neg_prompt      (str)   negative text prompt
        device          (str)   cuda device (default "cuda")
        cpu_offload     (bool)  enable model CPU offload to save VRAM (default True)
        seed            (int)   RNG seed for reproducibility (default 42)
    """

    def __init__(self, cfg: dict) -> None:
        self.out_w          = int(cfg.get("out_w",         960))
        self.out_h          = int(cfg.get("out_h",         720))
        self.chunk_frames   = int(cfg.get("chunk_frames",   17))
        self.overlap_frames = int(cfg.get("overlap_frames",  4))
        self.num_steps      = int(cfg.get("num_steps",      20))
        self.guidance       = float(cfg.get("guidance_scale", 5.0))
        self.prompt         = str(cfg.get("prompt",     _DEFAULT_PROMPT))
        self.neg_prompt     = str(cfg.get("neg_prompt", _NEG_PROMPT))
        self.seed           = int(cfg.get("seed", 42))
        device_str          = str(cfg.get("device", "cuda"))
        self.device         = torch.device(device_str)
        cpu_offload         = bool(cfg.get("cpu_offload", True))
        model_id            = str(cfg.get("model_id", _DEFAULT_MODEL))

        from diffusers import WanImageToVideoPipeline, FlowMatchEulerDiscreteScheduler
        print(f"[wan] Loading {model_id} ...", flush=True)
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        # Replace UniPC (multi-step, accumulates cached outputs → device mismatch with offload)
        # with FlowMatchEuler which is single-step and matches Wan2.1's training objective.
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        if cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to(self.device)
        self.pipe.vae.enable_slicing()
        print("[wan] Model loaded.", flush=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def upscale_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Upscale a list of BGR uint8 frames.
        Returns a list of BGR uint8 frames at (out_h, out_w).
        """
        N = len(frames)
        if N == 0:
            return []

        step        = max(1, self.chunk_frames - self.overlap_frames)
        chunk_starts = list(range(0, N, step))
        # Ensure last chunk covers the end
        if chunk_starts[-1] + self.chunk_frames < N:
            chunk_starts.append(N - self.chunk_frames)

        # Accumulate results with overlap averaging
        acc   = np.zeros((N, self.out_h, self.out_w, 3), dtype=np.float32)
        count = np.zeros((N,), dtype=np.float32)

        for ci, start in enumerate(chunk_starts):
            end    = min(start + self.chunk_frames, N)
            chunk  = frames[start:end]
            print(f"[wan] chunk {ci+1}/{len(chunk_starts)}  frames {start}–{end-1}", flush=True)
            result = self._upscale_chunk(chunk)

            for li, fi in enumerate(range(start, end)):
                # Blend with linear ramp at chunk edges for smooth transitions
                weight = 1.0
                if li < self.overlap_frames and ci > 0:
                    weight = (li + 1) / (self.overlap_frames + 1)
                elif li >= (end - start) - self.overlap_frames and ci < len(chunk_starts) - 1:
                    tail = (end - start) - li
                    weight = tail / (self.overlap_frames + 1)
                acc[fi]   += result[li].astype(np.float32) * weight
                count[fi] += weight

        # Normalise
        out_frames = []
        for fi in range(N):
            w = max(count[fi], 1e-6)
            out_frames.append(np.clip(acc[fi] / w, 0, 255).astype(np.uint8))
        return out_frames

    # ── Internal ──────────────────────────────────────────────────────────────

    def _upscale_chunk(self, chunk: List[np.ndarray]) -> List[np.ndarray]:
        """Run Wan2.1 I2V on one chunk. Returns BGR uint8 frames at (out_h, out_w)."""
        cond_pil = _bgr_to_pil(chunk[0])
        cond_pil = cond_pil.resize((self.out_w, self.out_h), Image.LANCZOS)

        n_frames = len(chunk)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        output = self.pipe(
            image              = cond_pil,
            prompt             = self.prompt,
            negative_prompt    = self.neg_prompt,
            height             = self.out_h,
            width              = self.out_w,
            num_frames         = n_frames,
            num_inference_steps= self.num_steps,
            guidance_scale     = self.guidance,
            generator          = generator,
        )

        # output.frames shape varies by diffusers version:
        #   np.ndarray (B, F, H, W, 3) — batched numpy  ← most common with new diffusers
        #   np.ndarray (F, H, W, 3)    — unbatched numpy
        #   List[List[PIL.Image]]      — batched PIL
        #   List[PIL.Image]            — flat PIL
        raw = output.frames
        if isinstance(raw, np.ndarray):
            # Strip all leading batch dims until we have exactly (F, H, W, 3)
            while raw.ndim > 4:
                raw = raw[0]
            if raw.ndim == 3:          # single frame (H,W,3)
                raw = raw[np.newaxis]  # → (1,H,W,3)
            result = []
            for i in range(min(len(raw), n_frames)):
                f = raw[i]
                # diffusers returns float32 in [0,1]; scale to [0,255] before uint8 cast
                if f.dtype != np.uint8:
                    f = (f * 255).clip(0, 255).astype(np.uint8)
                result.append(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        else:
            # List — flatten any nesting (batch wrapper)
            flat = raw
            while flat and isinstance(flat[0], (list, tuple)):
                flat = flat[0]
            result = []
            for f in flat[:n_frames]:
                if isinstance(f, np.ndarray):
                    if f.dtype != np.uint8:
                        f = (f * 255).clip(0, 255).astype(np.uint8)
                    result.append(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                else:
                    result.append(_pil_to_bgr(f))
        # Pad if Wan returned fewer frames than requested
        while len(result) < n_frames:
            result.append(result[-1].copy())
        return result
