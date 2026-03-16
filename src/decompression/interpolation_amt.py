from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


_AMT_IMPORT_LOCK = threading.Lock()


def _ensure_amt_repo_on_path(amt_repo_dir: Path) -> None:
    repo = amt_repo_dir.expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"AMT repo folder not found: {repo}")
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _build_network_cfg(variant: str) -> Dict[str, Any]:
    v = str(variant).strip().lower()
    if v == "amt-s":
        return {"name": "networks.AMT-S.Model", "params": {"corr_radius": 3, "corr_lvls": 4, "num_flows": 3}}
    if v == "amt-l":
        return {"name": "networks.AMT-L.Model", "params": {"corr_radius": 3, "corr_lvls": 4, "num_flows": 5}}
    raise ValueError("AMT variant must be one of: amt-s, amt-l")


def _resolve_device(device: str) -> torch.device:
    d = str(device).strip().lower()
    if d in {"cpu", "mps"}:
        raise ValueError("Strict GPU runtime forbids AMT interpolation device=cpu/mps.")
    if d not in {"", "auto", "cuda"} and not d.startswith("cuda:"):
        raise ValueError("AMT interpolation device must be one of: auto, cuda, cuda:<index> for strict GPU runtime.")
    if not torch.cuda.is_available():
        raise RuntimeError("Strict GPU runtime requires CUDA for AMT interpolation.")
    if d in {"", "auto", "cuda"}:
        selected_idx = 0
    else:
        idx = d.split(":", 1)[1].strip()
        if not idx.isdigit():
            raise ValueError("AMT interpolation device must be cuda:<index> when using an explicit CUDA index.")
        selected_idx = int(idx)
    device_count = int(torch.cuda.device_count() or 0)
    if device_count > 0 and selected_idx >= device_count:
        raise ValueError(
            f"AMT interpolation CUDA index {selected_idx} is out of range for {device_count} visible CUDA device(s)."
        )
    torch.cuda.set_device(int(selected_idx))
    return torch.device(f"cuda:{int(selected_idx)}")


def _pad_to_divisor(x: torch.Tensor, divisor: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    h, w = x.shape[-2:]
    d = max(1, int(divisor))
    pad_h = (d - (h % d)) % d
    pad_w = (d - (w % d)) % d
    pad = (pad_w // 2, pad_w - (pad_w // 2), pad_h // 2, pad_h - (pad_h // 2))
    if pad_h == 0 and pad_w == 0:
        return x, pad
    return F.pad(x, pad, mode="replicate"), pad


def _unpad(x: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = pad
    if left == 0 and right == 0 and top == 0 and bottom == 0:
        return x
    h, w = x.shape[-2:]
    return x[..., top : h - bottom, left : w - right]


class AmtInterpolator:
    def __init__(
        self,
        *,
        amt_repo_dir: str,
        variant: str,
        weights_path: str,
        device: str = "auto",
        fp16: bool = True,
        pad_to: int = 16,
    ) -> None:
        repo_dir = Path(amt_repo_dir).expanduser().resolve()
        weights = Path(weights_path).expanduser().resolve()
        if not weights.exists():
            raise FileNotFoundError(f"AMT checkpoint not found: {weights}")

        with _AMT_IMPORT_LOCK:
            _ensure_amt_repo_on_path(repo_dir)
            from utils.build_utils import build_from_cfg

            cfg = _build_network_cfg(variant)
            net = build_from_cfg(cfg)
            ckpt = torch.load(str(weights), map_location="cpu")
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            net.load_state_dict(state, strict=True)

        self.device = _resolve_device(device)
        self.model = net.to(self.device).eval()
        self.pad_to = max(1, int(pad_to))
        self.fp16 = bool(fp16) and self.device.type == "cuda"

    @staticmethod
    def _bgr_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return t

    @staticmethod
    def _tensor_to_bgr(frame_t: torch.Tensor) -> np.ndarray:
        arr = (
            frame_t.clamp(0.0, 1.0)
            .mul(255.0)
            .byte()
            .squeeze(0)
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def interpolate(self, frame0_bgr: np.ndarray, frame1_bgr: np.ndarray, t: float) -> np.ndarray:
        if frame0_bgr.shape != frame1_bgr.shape:
            raise ValueError("AMT interpolation requires both frames to have identical shape")
        if not (0.0 < float(t) < 1.0):
            raise ValueError("t must be strictly between 0 and 1")

        in0 = self._bgr_to_tensor(frame0_bgr).to(self.device, non_blocking=True)
        in1 = self._bgr_to_tensor(frame1_bgr).to(self.device, non_blocking=True)
        in0, pad = _pad_to_divisor(in0, self.pad_to)
        in1, _ = _pad_to_divisor(in1, self.pad_to)
        embt = torch.tensor(float(t), dtype=torch.float32, device=self.device).view(1, 1, 1, 1)

        with torch.no_grad():
            if self.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = self.model(in0, in1, embt, scale_factor=1.0, eval=True)["imgt_pred"]
            else:
                pred = self.model(in0, in1, embt, scale_factor=1.0, eval=True)["imgt_pred"]
        pred = _unpad(pred, pad)
        return self._tensor_to_bgr(pred)
