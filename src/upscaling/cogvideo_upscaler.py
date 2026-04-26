"""
cogvideo_upscaler.py — Video enhancement via CogVideoX Video-to-Video.

CogVideoX V2V encodes the full input video, adds a controlled amount of noise
(strength), then denoises with 3D spatiotemporal attention — so the whole video
is processed coherently with no per-chunk jumping artefacts.

Requires: pip install diffusers transformers accelerate
Model:     THUDM/CogVideoX-5b  (or CogVideoX1.5-5b for higher resolution)
"""
from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image


_DEFAULT_MODEL  = "THUDM/CogVideoX-5b"
_DEFAULT_PROMPT = (
    "high quality, sharp details, no compression artifacts, "
    "professional wildlife video, natural colors"
)
_NEG_PROMPT = "blurry, low resolution, compression artifacts, noise, flickering"

# CogVideoX requires (num_frames - 1) % 8 == 0  →  valid: 1,9,17,25,33,41,49
_COGVIDEO_FRAME_STEP = 8


def _nearest_valid(n: int) -> int:
    """Round n up to nearest valid CogVideoX frame count."""
    if n <= 1:
        return 1
    rem = (n - 1) % _COGVIDEO_FRAME_STEP
    return n if rem == 0 else n + (_COGVIDEO_FRAME_STEP - rem)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _frame_to_bgr(f) -> np.ndarray:
    if isinstance(f, np.ndarray):
        arr = f
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)


class CogVideoUpscaler:
    """
    Enhance a list of BGR frames using CogVideoX V2V in overlapping chunks.

    Config keys (all optional):
        model_id        (str)   HuggingFace model ID
        out_w / out_h   (int)   output resolution (default 720×480)
        strength        (float) noise strength 0–1; lower = closer to input (default 0.55)
        chunk_frames    (int)   frames per CogVideoX call, rounded to valid count (default 49)
        overlap_frames  (int)   overlap between adjacent chunks (default 8)
        num_steps       (int)   diffusion steps (default 50)
        guidance_scale  (float) CFG scale (default 6.0)
        prompt          (str)   positive text prompt
        device          (str)   cuda device (default "cuda")
        cpu_offload     (bool)  sequential CPU offload to save VRAM (default True)
        seed            (int)   RNG seed (default 42)
    """

    def __init__(self, cfg: dict) -> None:
        self.out_w          = int(cfg.get("out_w",          720))
        self.out_h          = int(cfg.get("out_h",          480))
        self.strength       = float(cfg.get("strength",      0.55))
        self.chunk_frames   = _nearest_valid(int(cfg.get("chunk_frames",  49)))
        self.overlap_frames = int(cfg.get("overlap_frames",   8))
        self.num_steps      = int(cfg.get("num_steps",        50))
        self.guidance       = float(cfg.get("guidance_scale", 6.0))
        self.prompt         = str(cfg.get("prompt",     _DEFAULT_PROMPT))
        self.neg_prompt     = str(cfg.get("neg_prompt", _NEG_PROMPT))
        self.seed           = int(cfg.get("seed", 42))
        device_str          = str(cfg.get("device", "cuda"))
        self.device         = torch.device(device_str)
        cpu_offload         = bool(cfg.get("cpu_offload", True))
        model_id            = str(cfg.get("model_id", _DEFAULT_MODEL))

        from diffusers import CogVideoXVideoToVideoPipeline
        from transformers import T5Tokenizer
        from huggingface_hub import hf_hub_download
        print(f"[cogvideo] Loading {model_id} ...", flush=True)
        # Load T5Tokenizer directly from the spiece.model file, bypassing
        # the from_pretrained fast-tokenizer conversion that calls load_tiktoken_bpe
        spiece_path = hf_hub_download(model_id, "tokenizer/spiece.model")
        tokenizer = T5Tokenizer(vocab_file=spiece_path, model_max_length=226, legacy=True)
        self.pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
            model_id,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
        )
        if cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to(self.device)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        print("[cogvideo] Model loaded.", flush=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def upscale_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Enhance a list of BGR uint8 frames. Returns BGR uint8 at (out_h, out_w)."""
        N = len(frames)
        if N == 0:
            return []

        step         = max(1, self.chunk_frames - self.overlap_frames)
        chunk_starts = list(range(0, N, step))
        if chunk_starts[-1] + self.chunk_frames < N:
            chunk_starts.append(N - self.chunk_frames)

        acc   = np.zeros((N, self.out_h, self.out_w, 3), dtype=np.float32)
        count = np.zeros((N,), dtype=np.float32)

        for ci, start in enumerate(chunk_starts):
            end   = min(start + self.chunk_frames, N)
            chunk = frames[start:end]
            print(f"[cogvideo] chunk {ci+1}/{len(chunk_starts)}  frames {start}–{end-1}", flush=True)
            result = self._process_chunk(chunk)

            for li, fi in enumerate(range(start, end)):
                weight = 1.0
                if li < self.overlap_frames and ci > 0:
                    weight = (li + 1) / (self.overlap_frames + 1)
                elif li >= (end - start) - self.overlap_frames and ci < len(chunk_starts) - 1:
                    tail = (end - start) - li
                    weight = tail / (self.overlap_frames + 1)
                acc[fi]   += result[li].astype(np.float32) * weight
                count[fi] += weight

        return [np.clip(acc[fi] / max(count[fi], 1e-6), 0, 255).astype(np.uint8)
                for fi in range(N)]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_chunk(self, chunk: List[np.ndarray]) -> List[np.ndarray]:
        """Run CogVideoX V2V on one chunk. Returns BGR uint8 at (out_h, out_w)."""
        pil_frames = [_bgr_to_pil(cv2.resize(f, (self.out_w, self.out_h),
                                              interpolation=cv2.INTER_LANCZOS4))
                      for f in chunk]

        # Pad to valid frame count if needed
        n_valid = _nearest_valid(len(pil_frames))
        while len(pil_frames) < n_valid:
            pil_frames.append(pil_frames[-1])

        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        output = self.pipe(
            video              = pil_frames,
            prompt             = self.prompt,
            negative_prompt    = self.neg_prompt,
            height             = self.out_h,
            width              = self.out_w,
            strength           = self.strength,
            num_inference_steps= self.num_steps,
            guidance_scale     = self.guidance,
            generator          = generator,
        )

        raw = output.frames
        # Normalise output to list of BGR uint8, stripping any batch/nesting
        if isinstance(raw, np.ndarray):
            while raw.ndim > 4:
                raw = raw[0]
            if raw.ndim == 3:
                raw = raw[np.newaxis]
            result = [_frame_to_bgr(raw[i]) for i in range(min(len(raw), len(chunk)))]
        else:
            flat = raw
            while flat and isinstance(flat[0], (list, tuple)):
                flat = flat[0]
            result = [_frame_to_bgr(f) for f in flat[:len(chunk)]]

        while len(result) < len(chunk):
            result.append(result[-1].copy())
        return result
