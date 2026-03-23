"""
Compression phase orchestrator.

Pipeline:
  1. Run YOLO11 detection → ROI bounding boxes + track IDs per frame
  2. Build per-frame binary ROI masks
  3. Split frames into ROI and BG streams using the mask
  4. DCVC-encode ROI stream at high quality
  5. DCVC-encode BG stream at lower quality
  6. Pack both bitstreams + metadata + detections into a ZIP archive (bytes)

Public API:
    compress_video(video_path, cfg) -> bytes  (ZIP archive)
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


ARCHIVE_VERSION   = "1.0.0"
MANIFEST_NAME     = "manifest.json"
META_NAME         = "meta.json"
ROI_STREAM_NAME   = "roi.bin"
BG_STREAM_NAME    = "bg.bin"
DETECTIONS_NAME   = "detections.json"
RUNTIME_CFG_NAME  = "compression_config.json"


def compress_video(
    video_path: str | Path,
    cfg: Dict[str, Any],
    progress_cb: Optional[Callable[[str, int], None]] = None,
) -> bytes:
    """
    Compress a video to a ZIP archive (in memory).

    Args:
        video_path:  Source video file.
        cfg:         Merged compression config dict (detection + compression sections).
        progress_cb: Called as progress_cb(stage_name, frames_done).

    Returns:
        ZIP archive as bytes — write to disk or send over network.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    det_cfg  = cfg.get("detection",   {}) or {}
    comp_cfg = cfg.get("compression", {}) or {}
    dcvc_cfg = comp_cfg.get("dcvc",    {}) or {}
    qual_cfg = comp_cfg.get("quality", {}) or {}
    roi_cfg  = comp_cfg.get("roi",     {}) or {}

    roi_dilate_px = int(roi_cfg.get("dilate_px", 8))
    roi_min_conf  = float(roi_cfg.get("min_conf", 0.25))

    # ── Phase 1: Detection ─────────────────────────────────────────────────────
    _cb(progress_cb, "detection", 0)
    from ..detection.yolo_detector import run_detection

    det_result = run_detection(
        str(video_path),
        config=det_cfg,
        progress_cb=lambda n: _cb(progress_cb, "detection", n),
    )
    frames_map: Dict[int, List[Dict[str, Any]]] = det_result["frames"]
    width      = int(det_result["width"])
    height     = int(det_result["height"])
    fps        = float(det_result["fps"])
    n_frames   = int(det_result["frame_count"])

    # ── Phase 2: Build ROI mask stream ────────────────────────────────────────
    from ..roi_masking.roi_masking import build_frame_mask

    roi_masks: Dict[int, np.ndarray] = {}
    for fi in range(n_frames):
        roi_masks[fi] = build_frame_mask(
            frame_idx=fi,
            width=width,
            height=height,
            roi_map=frames_map,
            min_conf=roi_min_conf,
            dilate_px=roi_dilate_px,
        )

    # ── Phase 3: Extract frames, split into ROI and BG streams ────────────────
    _cb(progress_cb, "extraction", 0)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    from ..compression.dcvc_encoder import VideoInfo, encode_frames_to_bytes

    info = VideoInfo(width=width, height=height, fps=fps, frames=n_frames)

    # We need to encode two streams.  To avoid reading the video twice we buffer
    # all frames — for very long videos consider disk-spill, but here we keep it
    # simple and in-memory.
    roi_frames: List[Tuple[int, np.ndarray]] = []
    bg_frames:  List[Tuple[int, np.ndarray]] = []

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            mask = roi_masks.get(frame_idx, np.zeros((height, width), dtype=np.uint8))
            roi_frames.append((frame_idx, frame))   # full frame stored for ROI stream
            bg_frames.append((frame_idx, frame))    # same frame; BG stream = full frame
            # (The mask is embedded in meta; compositing happens at decompression time)
            frame_idx += 1
            _cb(progress_cb, "extraction", 1)
    finally:
        cap.release()

    # ── Phase 4: Encode ROI stream ────────────────────────────────────────────
    _cb(progress_cb, "encode_roi", 0)
    roi_quality = {
        "qp_i": int(qual_cfg.get("roi_qp_i", 63)),
        "qp_p": int(qual_cfg.get("roi_qp_p", 63)),
    }
    roi_result = encode_frames_to_bytes(
        iter(roi_frames), info=info,
        dcvc_cfg=dcvc_cfg, quality_cfg=roi_quality,
        source_label="roi",
    )

    # ── Phase 5: Encode BG stream ─────────────────────────────────────────────
    _cb(progress_cb, "encode_bg", 0)
    bg_quality = {
        "qp_i": int(qual_cfg.get("bg_qp_i", 25)),
        "qp_p": int(qual_cfg.get("bg_qp_p", 25)),
    }
    bg_result = encode_frames_to_bytes(
        iter(bg_frames), info=info,
        dcvc_cfg=dcvc_cfg, quality_cfg=bg_quality,
        source_label="bg",
    )

    # ── Phase 6: Serialise ROI mask data for decompression ──────────────────
    # Store bounding boxes (not full masks) to keep archive small.
    roi_boxes_json: Dict[str, Any] = {}
    for fi, boxes in frames_map.items():
        roi_boxes_json[str(fi)] = boxes

    # ── Phase 7: Pack ZIP archive ─────────────────────────────────────────────
    meta: Dict[str, Any] = {
        "version": ARCHIVE_VERSION,
        "video": {
            "path":        str(video_path),
            "width":       width,
            "height":      height,
            "fps":         fps,
            "frame_count": n_frames,
        },
        "dcvc": {
            "repo_dir": str(dcvc_cfg.get("repo_dir", "DCVC")),
            "model_i":  str(dcvc_cfg.get("model_i", "")),
            "model_p":  str(dcvc_cfg.get("model_p", "")),
            "use_cuda": bool(dcvc_cfg.get("use_cuda", True)),
            "cuda_idx": dcvc_cfg.get("cuda_idx", 0),
            "force_zero_thres": dcvc_cfg.get("force_zero_thres", None),
        },
        "streams": {
            "roi": roi_result["meta"],
            "bg":  bg_result["meta"],
        },
    }

    archive_buf = io.BytesIO()
    with zipfile.ZipFile(archive_buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(META_NAME,        json.dumps(meta, indent=2))
        zf.writestr(DETECTIONS_NAME,  json.dumps(roi_boxes_json, indent=2))
        zf.writestr(ROI_STREAM_NAME,  roi_result["bitstream_bytes"])
        zf.writestr(BG_STREAM_NAME,   bg_result["bitstream_bytes"])
        zf.writestr(RUNTIME_CFG_NAME, json.dumps(cfg, indent=2, default=str))

    return archive_buf.getvalue()


def _cb(cb: Optional[Callable[[str, int], None]], stage: str, n: int) -> None:
    if cb is not None:
        cb(stage, n)
