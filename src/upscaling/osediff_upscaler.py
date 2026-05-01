"""
osediff_upscaler.py — Per-frame super-resolution via OSEDiff (one-step diffusion SR).

OSEDiff is built on SD 2.1 with LoRA adapters fine-tuned for real-world image
restoration + 4× upscaling in a single diffusion step.

Requires a working OSEDiff installation (set osediff_repo in config).
Checkpoints must already be downloaded (preset/models/osediff.pkl, models/sd21-base/).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


_DEFAULT_PROMPT = (
    "high quality, sharp details, professional wildlife photograph, natural colors"
)
_TO_TENSOR = T.ToTensor()
_TO_PIL    = T.ToPILImage()


def _bgr_to_tensor(bgr: np.ndarray, out_w: int, out_h: int) -> torch.Tensor:
    """Resize BGR frame → RGB PIL → float tensor in [-1, 1], shape (1,3,H,W)."""
    rgb = cv2.cvtColor(cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4),
                       cv2.COLOR_BGR2RGB)
    t = _TO_TENSOR(rgb).unsqueeze(0)   # [0, 1]
    return t * 2.0 - 1.0               # [-1, 1]


def _tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    """Float tensor in [-1, 1], shape (1,3,H,W) → BGR uint8."""
    arr = (t[0].cpu().float() * 0.5 + 0.5).clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


class OSEDiffUpscaler:
    """
    Upscale a list of BGR frames using OSEDiff one-step diffusion SR.

    Config keys (all optional except osediff_repo):
        osediff_repo      (str)   Path to OSEDiff checkout (e.g. /home/user/OSEDiff)
        sd21_path         (str)   Path to SD 2.1 base model dir inside repo
                                  (default: <osediff_repo>/models/sd21-base)
        osediff_path      (str)   Path to osediff.pkl checkpoint
                                  (default: <osediff_repo>/preset/models/osediff.pkl)
        out_w / out_h     (int)   Output resolution (default 960×720)
        mixed_precision   (str)   "fp16" or "fp32" (default "fp16")
        batch_size        (int)   Frames per forward pass (default 1)
        prompt            (str)   Text prompt for all frames
        color_fix         (str)   "adain", "wavelet", or "" / None (default "adain")
        seed              (int)   RNG seed (default 42)
        vae_encoder_tiled_size  (int)  default 1024
        vae_decoder_tiled_size  (int)  default 224
        latent_tiled_size       (int)  default 96
        latent_tiled_overlap    (int)  default 32
    """

    def __init__(self, cfg: dict) -> None:
        repo = Path(cfg["osediff_repo"])
        if not repo.exists():
            raise FileNotFoundError(f"OSEDiff repo not found: {repo}")

        for p in [str(repo), str(repo / "src")]:
            if p not in sys.path:
                sys.path.insert(0, p)

        sd21_path    = str(cfg.get("sd21_path",    repo / "models"  / "sd21-base"))
        osediff_path = str(cfg.get("osediff_path", repo / "preset"  / "models" / "osediff.pkl"))

        self.out_w   = int(cfg.get("out_w",  960))
        self.out_h   = int(cfg.get("out_h",  720))
        self.batch   = int(cfg.get("batch_size", 1))
        self.prompt  = str(cfg.get("prompt", _DEFAULT_PROMPT))
        self.color_fix = str(cfg.get("color_fix", "adain"))
        self.seed    = int(cfg.get("seed", 42))

        args = argparse.Namespace(
            pretrained_model_name_or_path = sd21_path,
            osediff_path                  = osediff_path,
            mixed_precision               = str(cfg.get("mixed_precision", "fp16")),
            merge_and_unload_lora         = False,
            vae_encoder_tiled_size        = int(cfg.get("vae_encoder_tiled_size", 1024)),
            vae_decoder_tiled_size        = int(cfg.get("vae_decoder_tiled_size", 224)),
            latent_tiled_size             = int(cfg.get("latent_tiled_size",       96)),
            latent_tiled_overlap          = int(cfg.get("latent_tiled_overlap",    32)),
        )

        from osediff import OSEDiff_test
        print("[osediff] Loading model ...", flush=True)
        self.model = OSEDiff_test(args)
        print("[osediff] Model loaded.", flush=True)

        if self.color_fix:
            from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
            self._adain_fix   = adain_color_fix
            self._wavelet_fix = wavelet_color_fix

    # ── Public API ────────────────────────────────────────────────────────────

    def upscale_sequence(self, frames: List[np.ndarray],
                         detections: Optional[Dict] = None) -> List[np.ndarray]:
        """
        Upscale a list of BGR uint8 frames. Returns BGR uint8 at (out_h, out_w).
        detections is accepted for API compatibility but not used (OSEDiff is
        per-frame; no hallucination risk on subject detail).
        """
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        result = []
        for i in range(0, len(frames), self.batch):
            batch = frames[i : i + self.batch]
            for j, bgr in enumerate(batch):
                print(f"[osediff] frame {i+j+1}/{len(frames)}", flush=True)
                result.append(self._upscale_frame(bgr))
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _upscale_frame(self, bgr: np.ndarray) -> np.ndarray:
        lq = _bgr_to_tensor(bgr, self.out_w, self.out_h)

        with torch.no_grad():
            sr = self.model(lq, self.prompt)

        sr_bgr = _tensor_to_bgr(sr)

        if self.color_fix == "adain":
            ref = cv2.resize(bgr, (self.out_w, self.out_h), interpolation=cv2.INTER_LANCZOS4)
            sr_bgr = self._apply_color_fix(sr_bgr, ref, mode="adain")
        elif self.color_fix == "wavelet":
            ref = cv2.resize(bgr, (self.out_w, self.out_h), interpolation=cv2.INTER_LANCZOS4)
            sr_bgr = self._apply_color_fix(sr_bgr, ref, mode="wavelet")

        return sr_bgr

    def _apply_color_fix(self, sr_bgr: np.ndarray, ref_bgr: np.ndarray,
                         mode: str) -> np.ndarray:
        sr_pil  = Image.fromarray(cv2.cvtColor(sr_bgr,  cv2.COLOR_BGR2RGB))
        ref_pil = Image.fromarray(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB))
        if mode == "adain":
            out_pil = self._adain_fix(target=sr_pil, source=ref_pil)
        else:
            out_pil = self._wavelet_fix(target=sr_pil, source=ref_pil)
        return cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
