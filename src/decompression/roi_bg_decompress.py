# src/decompression/roi_bg_decompress.py
"""
ROI/BG decompression helpers (pure module).

Given:
  - roi.bin (DCVC stream encoding ROI-only frames)
  - bg.bin  (DCVC stream encoding BG-only frames)
  - meta.json produced by compression step

This module decodes ROI/BG streams into frame arrays.
Final timeline interpolation and ROI/BG compositing are handled by the decompression runner.

No filesystem writes here.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]


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


def _remap_container_path(raw_path: str) -> Optional[Path]:
    """
    Map container-rooted paths like /app/... to local project paths.
    """
    p = Path(str(raw_path)).expanduser()
    if not p.is_absolute():
        return None

    parts = p.parts
    if len(parts) >= 3 and str(parts[1]).lower() == "app":
        # /app/<tail...> -> <repo_root>/<tail...>
        return (ROOT / Path(*parts[2:])).resolve()
    return None


def _resolve_runtime_path(raw_path: str, kind: str) -> Path:
    """
    Resolve paths from archive metadata with local fallbacks.
    """
    p = Path(str(raw_path)).expanduser()
    candidates: List[Path] = []

    if p.is_absolute():
        candidates.append(p)
        mapped = _remap_container_path(str(p))
        if mapped is not None:
            candidates.append(mapped)
    else:
        candidates.append((ROOT / p).resolve())

    if kind == "repo":
        candidates.append((ROOT / "DCVC").resolve())
    elif kind in {"model_i", "model_p"}:
        candidates.append((ROOT / "models" / p.name).resolve())

    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists():
            return c.resolve()

    # Return best-effort path for caller-side error messages.
    if candidates:
        return candidates[0].resolve()
    return p.resolve()


def _resolve_hint(meta_hint: Any, user_limit: Optional[int]) -> Optional[int]:
    hint: Optional[int] = None
    if meta_hint is not None:
        try:
            hint = int(meta_hint)
        except (TypeError, ValueError):
            hint = None
    if user_limit is None:
        return hint
    try:
        lim = int(user_limit)
    except (TypeError, ValueError):
        lim = -1
    if lim <= 0:
        return hint
    if hint is None:
        return lim
    return min(hint, lim)



def ycbcr444_to_bgr(ycbcr: np.ndarray) -> np.ndarray:
    """
    Convert YCbCr (Y, Cb, Cr) uint8 HxWx3 -> BGR uint8 HxWx3 using OpenCV.
    OpenCV expects YCrCb ordering, so we swap channels to (Y, Cr, Cb).
    """
    if ycbcr.dtype != np.uint8:
        ycbcr = np.clip(ycbcr, 0, 255).astype(np.uint8)
    y = ycbcr[..., 0]
    cb = ycbcr[..., 1]
    cr = ycbcr[..., 2]
    ycrcb = np.stack([y, cr, cb], axis=-1)
    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return bgr


def _setup_dcvc_imports(dcvc_repo_dir: str) -> None:
    repo = Path(dcvc_repo_dir).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"DCVC repo not found: {repo}")

    # Avoid namespace collision: DCVC imports `src.*`
    if "src" in sys.modules:
        del sys.modules["src"]
    if str(repo) in sys.path:
        sys.path.remove(str(repo))
    sys.path.insert(0, str(repo))


def _normalize_cuda_index(cuda_idx: Any) -> Optional[int]:
    if cuda_idx is None or isinstance(cuda_idx, bool):
        return None
    raw = cuda_idx[0] if isinstance(cuda_idx, (list, tuple)) and cuda_idx else cuda_idx
    try:
        idx = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid decompression dcvc.cuda_idx: {cuda_idx!r}") from exc
    if idx < 0:
        raise ValueError("decompression dcvc.cuda_idx must be >= 0")
    return idx


def _configure_cuda(use_cuda: bool, cuda_idx: Any) -> int:
    if not bool(use_cuda):
        raise ValueError("Strict GPU runtime forbids decompression dcvc.use_cuda=false.")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Strict GPU runtime requires CUDA, but torch.cuda.is_available() is false.")
    selected_idx = _normalize_cuda_index(cuda_idx)
    if selected_idx is None:
        selected_idx = 0
    device_count = int(torch.cuda.device_count() or 0)
    if device_count > 0 and selected_idx >= device_count:
        raise ValueError(
            f"decompression dcvc.cuda_idx={selected_idx} is out of range for {device_count} visible CUDA device(s)."
        )
    torch.cuda.set_device(int(selected_idx))
    return int(selected_idx)


def _decode_stream_bytes_dcvc(
    stream_bytes: bytes,
    dcvc_cfg: Dict[str, Any],
    video_info: Dict[str, Any],
    frame_count_hint: Optional[int] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    frame_consumer: Optional[Callable[[np.ndarray], None]] = None,
) -> List[np.ndarray]:
    """
    Decode a DCVC bitstream from bytes into a list of BGR frames (uint8).

    This uses DCVC repo utilities. Different DCVC forks name helpers differently, so we:
      - import stream_helper module
      - try to find read_sps/read_ip or BitStreamReader-like APIs
      - fall back to minimal parsing assumptions

    If your DCVC fork uses different names, this function is the only place you should adapt.
    """
    repo_dir = str(_resolve_runtime_path(str(dcvc_cfg.get("repo_dir", "DCVC")), kind="repo"))
    model_i = str(_resolve_runtime_path(str(dcvc_cfg.get("model_i", "")), kind="model_i"))
    model_p = str(_resolve_runtime_path(str(dcvc_cfg.get("model_p", "")), kind="model_p"))
    force_zero_thres = dcvc_cfg.get("force_zero_thres", None)
    use_cuda = _coerce_bool(dcvc_cfg.get("use_cuda", True), default=True)
    cuda_idx = dcvc_cfg.get("cuda_idx", None)

    if not model_i or not model_p:
        raise ValueError("meta.dcvc.model_i/model_p missing; cannot decode")

    selected_cuda_idx = _configure_cuda(use_cuda, cuda_idx)
    _setup_dcvc_imports(repo_dir)

    import torch
    from src.utils.common import get_state_dict, set_torch_env  # type: ignore
    from src.utils.transforms import yuv_444_to_420  # type: ignore

    set_torch_env()
    device = torch.device(f"cuda:{int(selected_cuda_idx)}")

    from src.models.image_model import DMCI  # type: ignore
    from src.models.video_model import DMC  # type: ignore

    i_net = DMCI()
    i_net.load_state_dict(get_state_dict(model_i))
    i_net = i_net.to(device).eval()
    i_net.update(force_zero_thres)

    p_net = DMC()
    p_net.load_state_dict(get_state_dict(model_p))
    p_net = p_net.to(device).eval()
    p_net.update(force_zero_thres)

    if device.type == "cuda":
        i_net.half()
        p_net.half()

    width = int(video_info["width"])
    height = int(video_info["height"])
    use_two_entropy_coders = (width * height) > (1280 * 720)
    i_net.set_use_two_entropy_coders(use_two_entropy_coders)
    p_net.set_use_two_entropy_coders(use_two_entropy_coders)

    # Stream parsing helpers (best-effort)
    sh = None
    try:
        import importlib
        sh = importlib.import_module("src.utils.stream_helper")  # type: ignore
    except Exception as e:
        raise RuntimeError("Could not import DCVC stream_helper. Check repo_dir.") from e

    buff = io.BytesIO(stream_bytes)

    # Heuristic: DCVC stream usually alternates:
    # - SPS packets (optional)
    # - IP packets each frame (I/P marker + sps_id + qp + bit_stream)
    #
    # Different DCVC forks expose different parsers:
    #   A) decode_one_frame(BytesIO, i_net, p_net, device=...)
    #   B) read_ip(BytesIO)  -> packets
    #   C) read_header/read_sps_remaining/read_ip_remaining(BytesIO) -> NAL parsing
    #
    # We support all three, preferring C (matches write_sps/write_ip style).
    read_sps = getattr(sh, "read_sps", None)
    read_ip = getattr(sh, "read_ip", None)

    read_header = getattr(sh, "read_header", None)
    read_sps_remaining = getattr(sh, "read_sps_remaining", None)
    read_ip_remaining = getattr(sh, "read_ip_remaining", None)
    NalType = getattr(sh, "NalType", None)

    SPSHelper = getattr(sh, "SPSHelper", None)

    sps_map: Dict[int, Any] = {}
    sps_helper = SPSHelper() if SPSHelper else None

    frames: List[np.ndarray] = []
    frames_decoded = 0
    p_net.set_curr_poc(0)
    p_net.clear_dpb()

    max_frames = int(frame_count_hint) if frame_count_hint is not None else None
    if max_frames is not None and max_frames < 0:
        max_frames = None

    def _limit_reached() -> bool:
        return max_frames is not None and frames_decoded >= max_frames

    def _emit_frame(frame: np.ndarray) -> None:
        nonlocal frames_decoded
        if frame_consumer is None:
            frames.append(frame)
        else:
            frame_consumer(frame)
        frames_decoded += 1
        if progress_cb is not None:
            progress_cb(1)

    # Some repos expose `decode_one_frame` helpers; try them first
    decode_one = getattr(sh, "decode_one_frame", None)
    if callable(decode_one):
        while True:
            if _limit_reached():
                break
            try:
                out = decode_one(buff, i_net, p_net, device=device)  # type: ignore
            except Exception:
                break
            if out is None:
                break
            if isinstance(out, np.ndarray):
                frame = out
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                # assume RGB -> BGR
                if frame.ndim == 3 and frame.shape[-1] == 3:
                    frame = frame[..., ::-1].copy()
                _emit_frame(frame)
        return frames

    def _decompress_i(bit_stream, sps, qp):
        try:
            return i_net.decompress(bit_stream, sps, qp)  # type: ignore
        except TypeError:
            return i_net.decompress(bit_stream, qp)  # type: ignore

    def _decompress_p(bit_stream, sps, qp):
        try:
            return p_net.decompress(bit_stream, sps, qp)  # type: ignore
        except TypeError:
            return p_net.decompress(bit_stream, qp)  # type: ignore

    def _to_bgr(x_hat) -> np.ndarray:
        # Match phase5 decode path as closely as possible:
        # x_hat (YCbCr444) -> YUV420 (rounded uint8) -> BGR
        if hasattr(x_hat, "detach"):
            x = x_hat.detach()
            if hasattr(x, "__getitem__"):
                try:
                    x = x[:, :, :height, :width]
                except Exception:
                    pass
            x = x.float()
            y_rec_t, uv_rec_t = yuv_444_to_420(x)
            y_rec = (
                torch.clamp(y_rec_t * 255.0, 0, 255)
                .round()
                .to(dtype=torch.uint8)
                .squeeze(0)
                .cpu()
                .numpy()
            )
            uv_rec = (
                torch.clamp(uv_rec_t * 255.0, 0, 255)
                .round()
                .to(dtype=torch.uint8)
                .squeeze(0)
                .cpu()
                .numpy()
            )
            if y_rec.ndim == 3:
                y_rec = y_rec[0]
            u = uv_rec[0]
            v = uv_rec[1]
            yuv_i420 = np.concatenate([y_rec.reshape(-1), u.reshape(-1), v.reshape(-1)]).reshape(
                (height * 3 // 2, width)
            )
            return cv2.cvtColor(yuv_i420, cv2.COLOR_YUV2BGR_I420)
        arr = np.array(x_hat)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # If already BGR, just return; otherwise assume YCbCr and convert
        if arr.ndim == 3 and arr.shape[-1] == 3:
            try:
                return ycbcr444_to_bgr(arr)
            except Exception:
                return arr
        return arr

    # Prefer NAL parsing API (read_header/read_sps_remaining/read_ip_remaining)
    if callable(read_header) and callable(read_sps_remaining) and callable(read_ip_remaining):
        import torch
        with torch.no_grad():
            if max_frames is not None:
                frame_iter = range(max_frames)
            else:
                frame_iter = iter(int, 1)  # infinite iterator

            for _ in frame_iter:
                try:
                    header = read_header(buff)  # type: ignore
                except Exception:
                    if max_frames is not None:
                        raise RuntimeError(
                            f"Failed to parse bitstream header before decoding all expected frames ({max_frames})."
                        )
                    break
                if not header:
                    if max_frames is not None:
                        raise RuntimeError(
                            f"Unexpected EOF before decoding all expected frames ({max_frames})."
                        )
                    break

                # Consume SPS NALs until current frame header is I/P.
                if NalType is not None:
                    nal_sps = getattr(NalType, "NAL_SPS", None)
                    nal_i = getattr(NalType, "NAL_I", None)
                    nal_p = getattr(NalType, "NAL_P", None)
                else:
                    nal_sps = None
                    nal_i = None
                    nal_p = None

                while nal_sps is not None and header.get("nal_type") == nal_sps:
                    sps = read_sps_remaining(buff, header["sps_id"])  # type: ignore
                    if sps_helper is not None:
                        sps_helper.add_sps_by_id(sps)
                    else:
                        sps_map[int(header["sps_id"])] = sps
                    try:
                        header = read_header(buff)  # type: ignore
                    except Exception:
                        if max_frames is not None:
                            raise RuntimeError(
                                f"Unexpected EOF after SPS before decoding all expected frames ({max_frames})."
                            )
                        header = None
                    if not header:
                        break

                if not header:
                    if max_frames is not None:
                        raise RuntimeError(
                            f"Unexpected EOF before decoding all expected frames ({max_frames})."
                        )
                    break

                if sps_helper is not None:
                    sps = sps_helper.get_sps_by_id(header["sps_id"])
                else:
                    sps = sps_map.get(int(header["sps_id"]))
                if sps is None:
                    raise RuntimeError(f"Missing SPS id={header['sps_id']} before frame decode.")

                qp, bit_stream = read_ip_remaining(buff)  # type: ignore
                nal_type = header.get("nal_type")

                if (nal_i is not None and nal_type == nal_i) or (isinstance(nal_type, int) and nal_type == 1):
                    out = _decompress_i(bit_stream, sps, qp)
                    x_hat = out["x_hat"] if isinstance(out, dict) and "x_hat" in out else out
                    p_net.clear_dpb()
                    p_net.add_ref_frame(None, x_hat)
                elif (nal_p is not None and nal_type == nal_p) or (isinstance(nal_type, int) and nal_type == 2):
                    if isinstance(sps, dict) and int(sps.get("use_ada_i", 0)) == 1:
                        try:
                            # Match DCVC decode path used in test_pipeline.
                            p_net.reset_ref_feature()
                        except Exception:
                            pass
                    out = _decompress_p(bit_stream, sps, qp)
                    x_hat = out["x_hat"] if isinstance(out, dict) and "x_hat" in out else out
                else:
                    if max_frames is not None:
                        raise RuntimeError(f"Unexpected NAL type while decoding frame {frames_decoded}: {nal_type}")
                    break

                # Crop decoded tensor to target spatial size before conversion.
                if hasattr(x_hat, "__getitem__"):
                    try:
                        x_hat = x_hat[:, :, :height, :width]
                    except Exception:
                        pass

                bgr = _to_bgr(x_hat)
                if bgr.shape[0] != height or bgr.shape[1] != width:
                    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
                _emit_frame(bgr)

        return frames

    # Fallback: manual read_ip API if present
    if not callable(read_ip):
        raise RuntimeError(
            "Your DCVC repo does not expose read_header/read_sps_remaining/read_ip_remaining or read_ip/decode_one_frame. "
            "Please adapt _decode_stream_bytes_dcvc() to your fork's decoder API."
        )

    import torch
    with torch.no_grad():

        while True:
            if _limit_reached():
                break
            # Some read_ip implementations may raise EOFError at end.
            try:
                # read_ip(...) typically returns (is_i, sps_id, qp, bit_stream)
                pkt = read_ip(buff)  # type: ignore
            except Exception:
                break
            if pkt is None:
                break

            # Unpack flexibly
            if isinstance(pkt, dict):
                is_i = bool(pkt.get("is_i", pkt.get("is_i_frame", False)))
                sps_id = int(pkt.get("sps_id", 0))
                qp = int(pkt.get("qp", 0))
                bit_stream = pkt.get("bit_stream", None)
            else:
                # tuple-style
                if len(pkt) < 4:
                    break
                is_i, sps_id, qp, bit_stream = pkt[0], pkt[1], pkt[2], pkt[3]

            # If SPS needed and read_sps exists, try loading SPS map
            if sps_id not in sps_map and callable(read_sps):
                try:
                    # Some repos store SPS earlier; attempt to rewind? can't.
                    pass
                except Exception:
                    pass

            # Decompress
            if is_i:
                out = i_net.decompress(bit_stream, qp)  # type: ignore
                x_hat = out["x_hat"] if isinstance(out, dict) and "x_hat" in out else out
                p_net.clear_dpb()
                p_net.add_ref_frame(None, x_hat)
            else:
                out = p_net.decompress(bit_stream, qp)  # type: ignore
                x_hat = out["x_hat"] if isinstance(out, dict) and "x_hat" in out else out

            # x_hat is usually torch tensor in YCbCr444 [1,3,H,W], float in [0,1]
            if hasattr(x_hat, "detach"):
                x = x_hat.detach().float()
                if x.device.type != "cpu":
                    x = x.cpu()
                x = x.squeeze(0).permute(1, 2, 0).numpy()  # HWC
                x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
                # If it's YCbCr, convert to BGR
                bgr = ycbcr444_to_bgr(x)
            else:
                arr = np.array(x_hat)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                bgr = arr

            # Ensure correct size
            if bgr.shape[0] != height or bgr.shape[1] != width:
                bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)

            _emit_frame(bgr)

    return frames


def decode_roi_bg_streams(
    roi_bin_bytes: bytes,
    bg_bin_bytes: bytes,
    meta: Dict[str, Any],
    progress_cb_roi: Optional[Callable[[int], None]] = None,
    progress_cb_bg: Optional[Callable[[int], None]] = None,
    max_frames_roi: Optional[int] = None,
    max_frames_bg: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Decode ROI and BG streams and return them separately.
    """
    dcvc_cfg = meta.get("dcvc", {}) or {}
    video_info = meta.get("video", {}) or {}
    streams = meta.get("streams", {}) or {}
    roi_stream_meta = streams.get("roi", {}) or {}
    bg_stream_meta = streams.get("bg", {}) or {}

    roi_count_hint = roi_stream_meta.get("frames_encoded", None)
    if roi_count_hint is None and isinstance(roi_stream_meta.get("frame_index_map", None), list):
        roi_count_hint = len(roi_stream_meta.get("frame_index_map", []))
    bg_count_hint = bg_stream_meta.get("frames_encoded", None)
    if bg_count_hint is None and isinstance(bg_stream_meta.get("frame_index_map", None), list):
        bg_count_hint = len(bg_stream_meta.get("frame_index_map", []))

    roi_hint = _resolve_hint(roi_count_hint, max_frames_roi)
    bg_hint = _resolve_hint(bg_count_hint, max_frames_bg)

    roi_frames = _decode_stream_bytes_dcvc(
        roi_bin_bytes,
        dcvc_cfg,
        video_info,
        frame_count_hint=roi_hint,
        progress_cb=progress_cb_roi,
    )
    bg_frames = _decode_stream_bytes_dcvc(
        bg_bin_bytes,
        dcvc_cfg,
        video_info,
        frame_count_hint=bg_hint,
        progress_cb=progress_cb_bg,
    )
    return roi_frames, bg_frames


def decode_roi_bg_streams_to_memmap(
    roi_bin_bytes: bytes,
    bg_bin_bytes: bytes,
    meta: Dict[str, Any],
    work_dir: Path,
    progress_cb_roi: Optional[Callable[[int], None]] = None,
    progress_cb_bg: Optional[Callable[[int], None]] = None,
    max_frames_roi: Optional[int] = None,
    max_frames_bg: Optional[int] = None,
) -> Tuple[np.memmap, int, np.memmap, int]:
    dcvc_cfg = meta.get("dcvc", {}) or {}
    video_info = meta.get("video", {}) or {}
    streams = meta.get("streams", {}) or {}
    roi_stream_meta = streams.get("roi", {}) or {}
    bg_stream_meta = streams.get("bg", {}) or {}

    width = int(video_info.get("width", 0) or 0)
    height = int(video_info.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid video width/height in archive metadata")

    roi_count_hint = roi_stream_meta.get("frames_encoded", None)
    if roi_count_hint is None and isinstance(roi_stream_meta.get("frame_index_map", None), list):
        roi_count_hint = len(roi_stream_meta.get("frame_index_map", []))
    bg_count_hint = bg_stream_meta.get("frames_encoded", None)
    if bg_count_hint is None and isinstance(bg_stream_meta.get("frame_index_map", None), list):
        bg_count_hint = len(bg_stream_meta.get("frame_index_map", []))

    roi_hint = _resolve_hint(roi_count_hint, max_frames_roi)
    bg_hint = _resolve_hint(bg_count_hint, max_frames_bg)
    if roi_hint is None or bg_hint is None:
        raise RuntimeError("Archive is missing ROI/BG frame count hints required for low-memory decode.")

    work_dir.mkdir(parents=True, exist_ok=True)
    roi_path = work_dir / "roi_frames.npy"
    bg_path = work_dir / "bg_frames.npy"
    roi_map = np.lib.format.open_memmap(
        roi_path,
        mode="w+",
        dtype=np.uint8,
        shape=(int(roi_hint), int(height), int(width), 3),
    )
    bg_map = np.lib.format.open_memmap(
        bg_path,
        mode="w+",
        dtype=np.uint8,
        shape=(int(bg_hint), int(height), int(width), 3),
    )

    roi_written = 0
    bg_written = 0

    def _consume_roi(frame: np.ndarray) -> None:
        nonlocal roi_written
        if roi_written >= int(roi_map.shape[0]):
            raise RuntimeError("Decoded more ROI frames than expected from archive metadata.")
        roi_map[roi_written] = frame
        roi_written += 1

    def _consume_bg(frame: np.ndarray) -> None:
        nonlocal bg_written
        if bg_written >= int(bg_map.shape[0]):
            raise RuntimeError("Decoded more BG frames than expected from archive metadata.")
        bg_map[bg_written] = frame
        bg_written += 1

    _decode_stream_bytes_dcvc(
        roi_bin_bytes,
        dcvc_cfg,
        video_info,
        frame_count_hint=roi_hint,
        progress_cb=progress_cb_roi,
        frame_consumer=_consume_roi,
    )
    _decode_stream_bytes_dcvc(
        bg_bin_bytes,
        dcvc_cfg,
        video_info,
        frame_count_hint=bg_hint,
        progress_cb=progress_cb_bg,
        frame_consumer=_consume_bg,
    )
    roi_map.flush()
    bg_map.flush()
    return roi_map, roi_written, bg_map, bg_written


def decode_roi_bg_streams_to_cache(
    roi_bin_bytes: bytes,
    bg_bin_bytes: bytes,
    meta: Dict[str, Any],
    work_dir: Path,
    progress_cb_roi: Optional[Callable[[int], None]] = None,
    progress_cb_bg: Optional[Callable[[int], None]] = None,
    max_frames_roi: Optional[int] = None,
    max_frames_bg: Optional[int] = None,
) -> Tuple[List[Path], int, List[Path], int]:
    dcvc_cfg = meta.get("dcvc", {}) or {}
    video_info = meta.get("video", {}) or {}
    streams = meta.get("streams", {}) or {}
    roi_stream_meta = streams.get("roi", {}) or {}
    bg_stream_meta = streams.get("bg", {}) or {}

    roi_count_hint = roi_stream_meta.get("frames_encoded", None)
    if roi_count_hint is None and isinstance(roi_stream_meta.get("frame_index_map", None), list):
        roi_count_hint = len(roi_stream_meta.get("frame_index_map", []))
    bg_count_hint = bg_stream_meta.get("frames_encoded", None)
    if bg_count_hint is None and isinstance(bg_stream_meta.get("frame_index_map", None), list):
        bg_count_hint = len(bg_stream_meta.get("frame_index_map", []))

    roi_hint = _resolve_hint(roi_count_hint, max_frames_roi)
    bg_hint = _resolve_hint(bg_count_hint, max_frames_bg)
    if roi_hint is None or bg_hint is None:
        raise RuntimeError("Archive is missing ROI/BG frame count hints required for low-memory decode.")

    roi_dir = work_dir / "roi"
    bg_dir = work_dir / "bg"
    roi_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)

    roi_paths: List[Path] = []
    bg_paths: List[Path] = []
    roi_written = 0
    bg_written = 0

    def _write_frame(dst_dir: Path, idx: int, frame: np.ndarray, suffix: str, params: List[int], out: List[Path]) -> None:
        path = dst_dir / f"{idx:06d}{suffix}"
        ok = cv2.imwrite(str(path), frame, params)
        if not ok:
            raise RuntimeError(f"Failed to write decoded temp frame: {path}")
        out.append(path)

    def _consume_roi(frame: np.ndarray) -> None:
        nonlocal roi_written
        _write_frame(roi_dir, roi_written, frame, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 97], roi_paths)
        roi_written += 1

    def _consume_bg(frame: np.ndarray) -> None:
        nonlocal bg_written
        _write_frame(bg_dir, bg_written, frame, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95], bg_paths)
        bg_written += 1

    _decode_stream_bytes_dcvc(
        roi_bin_bytes,
        dcvc_cfg,
        video_info,
        frame_count_hint=roi_hint,
        progress_cb=progress_cb_roi,
        frame_consumer=_consume_roi,
    )
    _decode_stream_bytes_dcvc(
        bg_bin_bytes,
        dcvc_cfg,
        video_info,
        frame_count_hint=bg_hint,
        progress_cb=progress_cb_bg,
        frame_consumer=_consume_bg,
    )
    return roi_paths, roi_written, bg_paths, bg_written
