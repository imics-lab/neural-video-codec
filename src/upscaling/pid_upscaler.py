"""
pid_upscaler.py - Per-frame super-resolution via PiD (Pixel Diffusion Decoder).

PiD replaces the standard VAE decoder with a diffusion-based pixel decoder,
enabling high-fidelity upscaling in a single 4-step distilled pass. Uses the
from_clean path: BGR frame -> VAE encode -> PiD pixel-decode at target resolution.

Requires a PiD checkout (set pid_repo in config) or pip-installed pid package.
Checkpoints must be downloaded separately (see https://github.com/nv-tlabs/PiD).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch


_DEFAULT_PROMPT = (
    "high quality, sharp details, professional wildlife photograph, natural colors"
)


def _adain_color_fix(target_bgr: np.ndarray, source_bgr: np.ndarray) -> np.ndarray:
    """Match per-channel mean+std of source onto target (adaptive instance norm)."""
    t = target_bgr.astype(np.float32)
    s = source_bgr.astype(np.float32)
    for c in range(3):
        t_std = t[:, :, c].std() + 1e-5
        s_std = s[:, :, c].std() + 1e-5
        t[:, :, c] = (t[:, :, c] - t[:, :, c].mean()) / t_std * s_std + s[:, :, c].mean()
    return t.clip(0, 255).astype(np.uint8)


def _bgr_to_input_tensor(bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 -> bfloat16 CUDA tensor in [-1, 1], shape (1, 3, H, W).

    Input is cropped to the nearest 16-multiple as required by PiD's VAE encoder.
    """
    h, w = bgr.shape[:2]
    h16 = (h // 16) * 16
    w16 = (w // 16) * 16
    if h16 != h or w16 != w:
        bgr = bgr[:h16, :w16]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W] in [0, 255]
    return (t / 127.5 - 1.0).to(dtype=torch.bfloat16, device="cuda")


def _output_tensor_to_bgr(t: torch.Tensor, out_h: int, out_w: int) -> np.ndarray:
    """PiD output [3, 1, H, W] in [-1, 1] -> BGR uint8 resized to (out_h, out_w)."""
    arr = t[:, 0].float().cpu().clamp(-1, 1)           # [3, H, W]
    arr = arr.permute(1, 2, 0).numpy()                  # [H, W, 3] in [-1, 1]
    arr = ((arr * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if bgr.shape[1] != out_w or bgr.shape[0] != out_h:
        bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    return bgr


class PiDUpscaler:
    """
    Upscale a list of BGR frames using PiD (Pixel Diffusion Decoder).

    Config keys:
        pid_repo         (str)   Path to PiD checkout (default: third_party/PiD).
                                 Omit if pid is installed as a package.
        checkpoint_path  (str)   Path to the PiD .pth checkpoint (required).
        backbone         (str)   Decoder backbone: flux, flux2, sd3, sdxl, ...
                                 (default "flux2")
        pid_ckpt_type    (str)   "2k" or "2kto4k" (default "2k")
        load_ema_to_reg  (bool)  Load EMA weights into the regular slot (default True)
        out_w / out_h    (int)   Output resolution (default 960x720)
        pid_steps        (int)   Diffusion steps; 4 is standard for distilled ckpts
                                 (default 4)
        cfg_scale        (float) Classifier-free guidance scale; 1.0 for distilled
                                 (default 1.0)
        shift            (float) Flow-matching shift parameter (default 1.0)
        prompt           (str)   Text prompt applied to every frame
        seed             (int)   Base RNG seed; incremented per frame (default 42)
        color_fix        (str)   "adain" to match source chroma, "" to disable
                                 (default "adain")
    """

    def __init__(self, cfg: dict) -> None:
        _default_repo = Path(__file__).resolve().parents[2] / "third_party" / "PiD"
        repo_raw = cfg.get("pid_repo", None)
        if repo_raw is not None:
            repo = Path(str(repo_raw))
            if not repo.exists():
                raise FileNotFoundError(f"PiD repo not found: {repo}")
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
        elif _default_repo.exists():
            if str(_default_repo) not in sys.path:
                sys.path.insert(0, str(_default_repo))

        checkpoint_path = cfg.get("checkpoint_path", None)
        if not checkpoint_path:
            raise ValueError(
                "pid_upscaling.checkpoint_path is required. "
                "Download from https://huggingface.co/nvidia/PiD and set the path."
            )

        self.out_w     = int(cfg.get("out_w",    960))
        self.out_h     = int(cfg.get("out_h",    720))
        self.pid_steps = int(cfg.get("pid_steps",  4))
        self.cfg_scale = float(cfg.get("cfg_scale", 1.0))
        self.shift     = float(cfg.get("shift",     1.0))
        self.prompt    = str(cfg.get("prompt", _DEFAULT_PROMPT))
        self.seed      = int(cfg.get("seed", 42))
        self.color_fix = str(cfg.get("color_fix", "adain"))

        load_args = argparse.Namespace(
            backbone              = str(cfg.get("backbone", "flux2")),
            checkpoint_path       = str(checkpoint_path),
            pid_ckpt_type         = str(cfg.get("pid_ckpt_type", "2k")),
            load_ema_to_reg       = bool(cfg.get("load_ema_to_reg", True)),
            experiment            = None,
            extra_experiment_opts = None,
        )

        from pid._src.inference.decoder import load_our_decoder
        print("[pid] Loading model ...", flush=True)
        self.model = load_our_decoder(load_args, [], True)
        self._caption_key = self.model.config.input_caption_key
        print("[pid] Model loaded.", flush=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def upscale_sequence(self, frames: List[np.ndarray],
                         detections: Optional[Dict] = None) -> List[np.ndarray]:
        """
        Upscale a list of BGR uint8 frames. Returns BGR uint8 at (out_h, out_w).
        detections is accepted for API compatibility but not used.
        """
        result = []
        for i, bgr in enumerate(frames):
            print(f"[pid] frame {i + 1}/{len(frames)}", flush=True)
            result.append(self._upscale_frame(bgr, seed=self.seed + i))
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _upscale_frame(self, bgr: np.ndarray, seed: int = 42) -> np.ndarray:
        input_tensor = _bgr_to_input_tensor(bgr)

        with torch.no_grad():
            clean_latent = self.model.encode_lq_latent(input_tensor)
            data_batch = {
                self._caption_key: [self.prompt],
                "LQ_latent": clean_latent.to(dtype=torch.bfloat16, device="cuda"),
                "degrade_sigma": torch.tensor([0.0], device="cuda", dtype=torch.float32),
            }
            samples_out = self.model.generate_samples_from_batch(
                data_batch,
                cfg_scale=self.cfg_scale,
                num_steps=self.pid_steps,
                seed=seed,
                shift=self.shift,
                image_size=(self.out_h, self.out_w),
            )

        out_bgr = _output_tensor_to_bgr(samples_out[0], self.out_h, self.out_w)

        if self.color_fix == "adain":
            ref = cv2.resize(bgr, (self.out_w, self.out_h), interpolation=cv2.INTER_LANCZOS4)
            out_bgr = _adain_color_fix(out_bgr, ref)

        return out_bgr
