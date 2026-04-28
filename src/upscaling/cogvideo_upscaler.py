"""
cogvideo_upscaler.py — Video enhancement via CogVideoX Video-to-Video.

CogVideoX V2V encodes the full input video, adds a controlled amount of noise
(strength), then denoises with 3D spatiotemporal attention — so the whole video
is processed coherently with no per-chunk jumping artefacts.

Optional ROI/BG separate mode: background is processed at a lower strength
(subtle clean-up), detected ROI regions are processed at a higher strength
(aggressive sharpening), then composited back with a feathered soft mask.

Requires: pip install diffusers transformers accelerate
Model:     THUDM/CogVideoX-5b  (or CogVideoX1.5-5b for higher resolution)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
    if n <= 1:
        return 1
    rem = (n - 1) % _COGVIDEO_FRAME_STEP
    return n if rem == 0 else n + (_COGVIDEO_FRAME_STEP - rem)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _frame_to_bgr(f) -> np.ndarray:
    if isinstance(f, np.ndarray):
        arr = f if f.dtype == np.uint8 else (f * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)


def _soft_mask(h: int, w: int, bboxes: List[Tuple[int, int, int, int]],
               feather: int) -> np.ndarray:
    """
    Build a float32 (h, w, 1) soft mask with 1 inside ROI bboxes, 0 outside,
    and a smooth feathered transition of `feather` pixels.
    bboxes: list of (x1, y1, x2, y2) in output-space pixel coordinates.
    """
    mask = np.zeros((h, w), dtype=np.float32)
    for x1, y1, x2, y2 in bboxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
    if feather > 0:
        k = feather | 1  # ensure odd kernel
        mask = cv2.GaussianBlur(mask, (k * 4 + 1, k * 4 + 1), k)
    return mask[:, :, np.newaxis]


def _scale_bboxes(bboxes: List, src_h: int, src_w: int,
                  dst_h: int, dst_w: int) -> List[Tuple[int, int, int, int]]:
    """Scale bbox coordinates from src to dst resolution.
    Accepts dicts {"x1","y1","x2","y2"} or sequences (x1,y1,x2,y2).
    """
    sx, sy = dst_w / src_w, dst_h / src_h
    out = []
    for b in bboxes:
        if isinstance(b, dict):
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        else:
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        out.append((int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)))
    return out


class CogVideoUpscaler:
    """
    Enhance a list of BGR frames using CogVideoX V2V in overlapping chunks.

    Config keys (all optional):
        model_id          (str)   HuggingFace model ID
        out_w / out_h     (int)   output resolution (default 720×480)
        strength          (float) noise strength for full-frame / BG mode (default 0.55)
        chunk_frames      (int)   frames per call, rounded to valid count (default 49)
        overlap_frames    (int)   overlap between adjacent chunks (default 8)
        num_steps         (int)   diffusion steps (default 50)
        guidance_scale    (float) CFG scale (default 6.0)
        prompt            (str)   positive text prompt
        device            (str)   cuda device (default "cuda")
        cpu_offload       (bool)  sequential CPU offload (default True)
        seed              (int)   RNG seed (default 42)

        roi_bg_separate   (bool)  enable separate ROI/BG processing (default False)
        bg_strength       (float) strength for background pass (default 0.35)
        roi_strength      (float) strength for ROI pass (default 0.70)
        roi_feather_px    (int)   feather pixels for ROI mask blending (default 24)
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

        self.roi_feather_px  = int(cfg.get("roi_feather_px",  24))

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

    def upscale_sequence(self, frames: List[np.ndarray],
                         detections: Optional[Dict] = None) -> List[np.ndarray]:
        """
        Enhance a list of BGR uint8 frames. Returns BGR uint8 at (out_h, out_w).

        detections: optional dict {frame_idx: [bbox, ...]} where each bbox is
                    {"x1","y1","x2","y2"} in source-frame pixel coordinates.
                    When provided, original ROI regions are composited on top of
                    the CogVideoX output to prevent hallucination in preserved areas.
        """
        if not frames:
            return []

        enhanced = self._upscale_chunks(frames, self.strength)

        if detections:
            enhanced = self._paste_original_roi(frames, enhanced, detections)

        return enhanced

    # ── ROI overlay ───────────────────────────────────────────────────────────

    def _paste_original_roi(self, src_frames: List[np.ndarray],
                            enhanced: List[np.ndarray],
                            detections: Dict) -> List[np.ndarray]:
        """Paste bicubic-resized original frames over CogVideoX output in ROI regions."""
        src_h, src_w = src_frames[0].shape[:2]
        orig_resized = [cv2.resize(f, (self.out_w, self.out_h), interpolation=cv2.INTER_LANCZOS4)
                        for f in src_frames]
        result = []
        for fi, enh in enumerate(enhanced):
            bboxes_src = detections.get(fi) or detections.get(str(fi)) or []
            if bboxes_src:
                bboxes_out = _scale_bboxes(bboxes_src, src_h, src_w,
                                           self.out_h, self.out_w)
                mask = _soft_mask(self.out_h, self.out_w, bboxes_out, self.roi_feather_px)
                comp = (enh.astype(np.float32) * (1.0 - mask)
                        + orig_resized[fi].astype(np.float32) * mask)
                result.append(np.clip(comp, 0, 255).astype(np.uint8))
            else:
                result.append(enh)
        return result

    # ── Chunked processing ────────────────────────────────────────────────────

    def _upscale_chunks(self, frames: List[np.ndarray],
                        strength: float) -> List[np.ndarray]:
        N = len(frames)
        step         = max(1, self.chunk_frames - self.overlap_frames)
        chunk_starts = list(range(0, N, step))
        if chunk_starts[-1] + self.chunk_frames < N:
            chunk_starts.append(N - self.chunk_frames)

        acc   = np.zeros((N, self.out_h, self.out_w, 3), dtype=np.float32)
        count = np.zeros((N,), dtype=np.float32)

        for ci, start in enumerate(chunk_starts):
            end   = min(start + self.chunk_frames, N)
            chunk = frames[start:end]
            print(f"[cogvideo] chunk {ci+1}/{len(chunk_starts)}  frames {start}–{end-1}"
                  f"  strength={strength}", flush=True)
            result = self._process_chunk(chunk, strength)

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

    def _process_chunk(self, chunk: List[np.ndarray],
                       strength: float) -> List[np.ndarray]:
        pil_frames = [_bgr_to_pil(cv2.resize(f, (self.out_w, self.out_h),
                                              interpolation=cv2.INTER_LANCZOS4))
                      for f in chunk]
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
            strength           = strength,
            num_inference_steps= self.num_steps,
            guidance_scale     = self.guidance,
            generator          = generator,
        )

        raw = output.frames
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
