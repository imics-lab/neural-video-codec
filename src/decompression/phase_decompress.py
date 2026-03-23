"""
Decompression phase orchestrator.

Pipeline:
  1. Unpack ZIP archive — extract ROI.bin, BG.bin, meta.json, detections.json
  2. Decode ROI stream with DCVC → list of BGR frames
  3. Decode BG  stream with DCVC → list of BGR frames
  4. Per-frame compositing: ROI alpha-blend over BG using stored bounding boxes
  5. Return composited frames (and optionally write to disk)

Public API:
    decompress_archive(archive_path_or_bytes, cfg) -> List[np.ndarray]
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np


def decompress_archive(
    archive: Union[str, Path, bytes],
    cfg: Dict[str, Any],
    progress_cb: Optional[Callable[[str, int], None]] = None,
) -> Dict[str, Any]:
    """
    Decompress a ZIP archive produced by compress_video().

    Args:
        archive:     Path to ZIP file, or raw ZIP bytes.
        cfg:         Decompression config dict (decompression.dcvc, compositing).
        progress_cb: Called as progress_cb(stage, n_frames_done).

    Returns:
        {
            "frames":     List[np.ndarray],  # composited BGR uint8 frames
            "fps":        float,
            "width":      int,
            "height":     int,
            "detections": dict,              # raw detection data from archive
            "meta":       dict,              # archive metadata
        }
    """
    # ── Load archive ──────────────────────────────────────────────────────────
    if isinstance(archive, (str, Path)):
        with open(str(archive), "rb") as f:
            archive_bytes = f.read()
    else:
        archive_bytes = bytes(archive)

    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
        meta       = json.loads(zf.read("meta.json"))
        detections = json.loads(zf.read("detections.json"))
        roi_bytes  = zf.read("roi.bin")
        bg_bytes   = zf.read("bg.bin")

    video_info  = meta.get("video",  {})
    archive_dcvc= meta.get("dcvc",   {})
    streams     = meta.get("streams",{})

    width  = int(video_info.get("width",  0))
    height = int(video_info.get("height", 0))
    fps    = float(video_info.get("fps",  30.0))

    # Merge archive DCVC config with user overrides (user wins)
    user_dcvc = (cfg.get("decompression", {}) or {}).get("dcvc", {}) or {}
    dcvc_cfg = {**archive_dcvc, **{k: v for k, v in user_dcvc.items() if v}}

    roi_meta = streams.get("roi", {})
    bg_meta  = streams.get("bg",  {})
    roi_hint = roi_meta.get("frames_encoded") or len(roi_meta.get("frame_index_map", []))
    bg_hint  = bg_meta.get("frames_encoded")  or len(bg_meta.get("frame_index_map", []))

    # ── Decode streams ────────────────────────────────────────────────────────
    from .dcvc_decoder import decode_stream_bytes

    _cb(progress_cb, "decode_roi", 0)
    roi_frames = decode_stream_bytes(
        roi_bytes, dcvc_cfg, video_info,
        frame_count_hint=roi_hint or None,
        progress_cb=lambda n: _cb(progress_cb, "decode_roi", n),
    )

    _cb(progress_cb, "decode_bg", 0)
    bg_frames = decode_stream_bytes(
        bg_bytes, dcvc_cfg, video_info,
        frame_count_hint=bg_hint or None,
        progress_cb=lambda n: _cb(progress_cb, "decode_bg", n),
    )

    n_frames = max(len(roi_frames), len(bg_frames))

    # ── Per-frame compositing ─────────────────────────────────────────────────
    feather_px = int(
        (cfg.get("decompression", {}) or {})
        .get("compositing", {})
        .get("feather_px", 8)
    )
    from ..roi_masking.roi_masking import build_frame_mask, compose_soft, mask_to_alpha

    composited: List[np.ndarray] = []
    _cb(progress_cb, "composite", 0)

    for fi in range(n_frames):
        roi_f = roi_frames[fi] if fi < len(roi_frames) else None
        bg_f  = bg_frames[fi]  if fi < len(bg_frames)  else None

        if roi_f is None and bg_f is None:
            continue
        if bg_f is None:
            composited.append(roi_f)
            continue
        if roi_f is None:
            composited.append(bg_f)
            continue

        mask = build_frame_mask(
            frame_idx=fi,
            width=width,
            height=height,
            roi_map=detections,
            min_conf=0.0,
            dilate_px=0,
        )
        alpha = mask_to_alpha(mask, feather_px=feather_px)
        composited.append(compose_soft(roi_f, bg_f, alpha))
        _cb(progress_cb, "composite", 1)

    return {
        "frames":     composited,
        "fps":        fps,
        "width":      width,
        "height":     height,
        "detections": detections,
        "meta":       meta,
    }


def _cb(cb: Optional[Callable], stage: str, n: int) -> None:
    if cb is not None:
        cb(stage, n)
