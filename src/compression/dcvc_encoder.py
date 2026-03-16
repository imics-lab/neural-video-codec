# src/compression/dcvc_encoder.py
"""
DCVC encoding (pure module).

Refactored from: refs/compression.py
- No YAML loading
- No filesystem writes (returns bytes)
- No tqdm
- Can encode either:
    * full video frames, or
    * only selected frame indices (e.g., kept_frames from frame_removal)
  while keeping an index map to reconstruct the timeline later.

Public API:
  encode_dcvc_to_bytes(video_path: str, cfg: dict, keep_frames: Optional[list[int]] = None) -> dict
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return bool(default)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on", "cuda"}:
        return True
    if s in {"0", "false", "no", "n", "off", "cpu", "none", "null", ""}:
        return False
    return bool(value)


def probe_video(video_path: str) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return VideoInfo(w, h, fps, n)


def _setup_dcvc_imports(dcvc_repo_dir: str) -> Path:
    """
    Ensure DCVC repo is importable and prevent namespace collision with this project's `src/` directory.
    """
    repo = Path(dcvc_repo_dir).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"DCVC repo not found: {repo}")

    # If this project's folder created a namespace package called "src", remove it so DCVC can import its own.
    if "src" in sys.modules:
        del sys.modules["src"]

    if str(repo) in sys.path:
        # move it to front
        sys.path.remove(str(repo))
    sys.path.insert(0, str(repo))
    return repo


def _normalize_cuda_index(cuda_idx: Optional[object]) -> Optional[int]:
    if cuda_idx is None or isinstance(cuda_idx, bool):
        return None
    raw = cuda_idx[0] if isinstance(cuda_idx, (list, tuple)) and cuda_idx else cuda_idx
    try:
        idx = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid compression.dcvc.cuda_idx: {cuda_idx!r}") from exc
    if idx < 0:
        raise ValueError("compression.dcvc.cuda_idx must be >= 0")
    return idx


def _configure_cuda(use_cuda: bool, cuda_idx: Optional[object]) -> torch.device:
    if not bool(use_cuda):
        raise ValueError("Strict GPU runtime forbids compression.dcvc.use_cuda=false.")
    if not torch.cuda.is_available():
        raise RuntimeError("Strict GPU runtime requires CUDA, but torch.cuda.is_available() is false.")
    selected_idx = _normalize_cuda_index(cuda_idx)
    if selected_idx is None:
        selected_idx = 0
    device_count = int(torch.cuda.device_count() or 0)
    if device_count > 0 and selected_idx >= device_count:
        raise ValueError(
            f"compression.dcvc.cuda_idx={selected_idx} is out of range for {device_count} visible CUDA device(s)."
        )
    torch.cuda.set_device(int(selected_idx))
    return torch.device(f"cuda:{int(selected_idx)}")


def bgr_to_yuv420_planes(frame_bgr: np.ndarray):
    """
    Same conversion as refs/compression.py.
    Returns:
      y: (1,H,W)
      uv: (2,H/2,W/2)
    """
    h, w = frame_bgr.shape[:2]
    yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420)
    flat = yuv.reshape(-1)
    y_size = w * h
    uv_size = (w * h) // 4
    y = flat[:y_size].reshape(1, h, w)
    u = flat[y_size:y_size + uv_size].reshape(h // 2, w // 2)
    v = flat[y_size + uv_size:y_size + 2 * uv_size].reshape(h // 2, w // 2)
    uv = np.stack([u, v], axis=0)
    return y, uv


def _encode_dcvc_core(
    *,
    frame_iter: Iterable[Tuple[int, np.ndarray]],
    info: VideoInfo,
    cfg: Dict[str, Any],
    video_path: str,
    use_kept_frames: bool,
) -> Dict[str, Any]:
    dcvc_cfg = cfg.get("dcvc", {}) or {}
    q_cfg = cfg.get("quality", {}) or {}

    repo_dir = str(dcvc_cfg.get("repo_dir", "DCVC"))
    model_i = str(dcvc_cfg.get("model_i", ""))
    model_p = str(dcvc_cfg.get("model_p", ""))
    if not model_i or not model_p:
        raise ValueError("compression.dcvc.model_i and compression.dcvc.model_p must be set")

    force_zero_thres = dcvc_cfg.get("force_zero_thres", None)
    use_cuda = _coerce_bool(dcvc_cfg.get("use_cuda", True), default=True)
    cuda_idx = dcvc_cfg.get("cuda_idx", None)
    reset_interval = int(dcvc_cfg.get("reset_interval", 32))
    intra_period = int(dcvc_cfg.get("intra_period", -1))
    max_frames = int(dcvc_cfg.get("max_frames", -1))

    qp_i = int(q_cfg.get("qp_i", 3))
    qp_p = int(q_cfg.get("qp_p", 3))

    _setup_dcvc_imports(repo_dir)
    device = _configure_cuda(use_cuda, cuda_idx)

    from src.layers.cuda_inference import replicate_pad  # noqa: E402
    from src.models.image_model import DMCI  # noqa: E402
    from src.utils.stream_helper import SPSHelper, write_sps, write_ip  # noqa: E402
    from src.utils.transforms import ycbcr420_to_444_np  # noqa: E402

    i_frame_net, p_frame_net = _init_models(
        model_i=model_i,
        model_p=model_p,
        force_zero_thres=force_zero_thres,
        device=device,
    )

    padding_r, padding_b = DMCI.get_padding_size(info.height, info.width, 16)
    use_two_entropy_coders = info.width * info.height > 1280 * 720
    i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
    p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)

    output_buff = io.BytesIO()
    sps_helper = SPSHelper()
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    p_frame_net.set_curr_poc(0)
    last_qp = 0
    frame_index_map: List[int] = []
    encoded_frames = 0
    enc_idx = 0

    try:
        with torch.no_grad():
            for src_idx, frame in frame_iter:
                if max_frames >= 0 and encoded_frames >= max_frames:
                    break

                y, uv = bgr_to_yuv420_planes(frame)
                yuv = ycbcr420_to_444_np(y, uv)
                x = torch.from_numpy(yuv).to(device=device, dtype=torch.float32) / 255.0
                x = x.unsqueeze(0)
                if device.type == "cuda":
                    x = x.to(dtype=torch.float16)

                x_padded = replicate_pad(x, padding_b, padding_r)

                is_i = (enc_idx == 0) or (intra_period > 0 and enc_idx % intra_period == 0)
                if is_i:
                    curr_qp = qp_i
                    sps = {
                        "sps_id": -1,
                        "height": info.height,
                        "width": info.width,
                        "ec_part": 1 if use_two_entropy_coders else 0,
                        "use_ada_i": 0,
                    }
                    encoded = i_frame_net.compress(x_padded, qp_i)
                    p_frame_net.clear_dpb()
                    p_frame_net.add_ref_frame(None, encoded["x_hat"])
                    is_i_frame = True
                else:
                    fa_idx = index_map[enc_idx % 8]
                    if reset_interval > 0 and enc_idx % reset_interval == 1:
                        use_ada_i = 1
                        p_frame_net.prepare_feature_adaptor_i(last_qp)
                    else:
                        use_ada_i = 0
                    curr_qp = p_frame_net.shift_qp(qp_p, fa_idx)
                    sps = {
                        "sps_id": -1,
                        "height": info.height,
                        "width": info.width,
                        "ec_part": 1 if use_two_entropy_coders else 0,
                        "use_ada_i": use_ada_i,
                    }
                    encoded = p_frame_net.compress(x_padded, curr_qp)
                    last_qp = curr_qp
                    is_i_frame = False

                sps_id, sps_new = sps_helper.get_sps_id(sps)
                sps["sps_id"] = sps_id
                if sps_new:
                    write_sps(output_buff, sps)
                write_ip(output_buff, is_i_frame, sps_id, curr_qp, encoded["bit_stream"])

                frame_index_map.append(int(src_idx))
                encoded_frames += 1
                enc_idx += 1

        bitstream = output_buff.getvalue()
    finally:
        output_buff.close()

    meta = {
        "video_path": str(video_path),
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "frames_total": info.frames,
        "frames_encoded": int(encoded_frames),
        "qp_i": qp_i,
        "qp_p": qp_p,
        "reset_interval": reset_interval,
        "intra_period": intra_period,
        "device": str(device),
        "use_kept_frames": bool(use_kept_frames),
        "frame_index_map": frame_index_map,
        "compressed_bytes": len(bitstream),
    }

    return {"bitstream_bytes": bitstream, "meta": meta}


def _init_models(model_i: str, model_p: str, force_zero_thres=None, device=None):
    from src.models.image_model import DMCI  # noqa: E402
    from src.models.video_model import DMC  # noqa: E402
    from src.utils.common import get_state_dict, set_torch_env  # noqa: E402

    set_torch_env()

    i_frame_net = DMCI()
    i_frame_net.load_state_dict(get_state_dict(str(model_i)))
    i_frame_net = i_frame_net.to(device).eval()
    i_frame_net.update(force_zero_thres)
    if device.type == "cuda":
        i_frame_net.half()

    p_frame_net = DMC()
    p_frame_net.load_state_dict(get_state_dict(str(model_p)))
    p_frame_net = p_frame_net.to(device).eval()
    p_frame_net.update(force_zero_thres)
    if device.type == "cuda":
        p_frame_net.half()

    return i_frame_net, p_frame_net


def encode_dcvc_to_bytes(
    video_path: str,
    cfg: Dict[str, Any],
    keep_frames: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Encode video into a DCVC bitstream (bytes).

    cfg expects:
      cfg["dcvc"]: {repo_dir, model_i, model_p, force_zero_thres, use_cuda, cuda_idx, reset_interval, intra_period, max_frames}
      cfg["quality"]: {qp_i, qp_p}

    keep_frames:
      - If provided, only these source frame indices are encoded (in increasing order).
      - Returned meta includes `frame_index_map` (encoded_index -> source_index).
    """
    info = probe_video(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    keep_set = set(int(x) for x in keep_frames) if keep_frames is not None else None

    try:
        def _iter_frames() -> Iterable[Tuple[int, np.ndarray]]:
            src_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if keep_set is not None and src_idx not in keep_set:
                    src_idx += 1
                    continue

                src_idx += 1
                yield int(src_idx - 1), frame
        return _encode_dcvc_core(
            frame_iter=_iter_frames(),
            info=info,
            cfg=cfg,
            video_path=str(video_path),
            use_kept_frames=keep_set is not None,
        )
    finally:
        cap.release()


def encode_dcvc_frames_to_bytes(
    frames: Iterable[Tuple[int, np.ndarray]],
    *,
    info: VideoInfo,
    cfg: Dict[str, Any],
    video_path: str = "<rendered_frame_stream>",
) -> Dict[str, Any]:
    return _encode_dcvc_core(
        frame_iter=frames,
        info=info,
        cfg=cfg,
        video_path=str(video_path),
        use_kept_frames=True,
    )
