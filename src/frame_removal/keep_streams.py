from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import cv2


def _boxes_for_frame(roi_bbox_map: Mapping[Any, Any], frame_idx: int) -> list[Any]:
    boxes = roi_bbox_map.get(frame_idx, None)
    if boxes is None:
        boxes = roi_bbox_map.get(str(frame_idx), None)
    if isinstance(boxes, list):
        return boxes
    return []


def _mask_from_boxes(frame, boxes: list[Any]) -> Any:
    h, w = frame.shape[:2]
    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask[:] = 0
    for roi in boxes:
        if not isinstance(roi, dict):
            continue
        x1 = max(0, min(w - 1, int(roi.get("x1", 0))))
        y1 = max(0, min(h - 1, int(roi.get("y1", 0))))
        x2 = max(0, min(w - 1, int(roi.get("x2", 0))))
        y2 = max(0, min(h - 1, int(roi.get("y2", 0))))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def write_kept_frames_video(
    *,
    video_path: str | Path,
    kept_frames: Iterable[int],
    roi_bbox_map: Mapping[Any, Any],
    render_mode: str,
    output_path: str | Path,
) -> Dict[str, Any]:
    """
    Render phase-2 keep stream video.

    render_mode:
      - "roi": keep only ROI pixels, black elsewhere
      - "bg": keep full frame (BG timeline differs only by selected indices)
    """
    mode = str(render_mode).strip().lower()
    if mode not in {"roi", "bg"}:
        raise ValueError("render_mode must be one of: roi, bg")

    src_path = Path(video_path).expanduser().resolve()
    out_path = Path(output_path).expanduser().resolve()
    keep_set = set(int(x) for x in kept_frames)

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for kept-frame rendering: {src_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions while creating keep-stream video")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1e-6, fps),
        (width, height),
        True,
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create keep-stream writer: {out_path}")

    src_idx = 0
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if src_idx not in keep_set:
                src_idx += 1
                continue

            if mode == "roi":
                boxes = _boxes_for_frame(roi_bbox_map, src_idx)
                if boxes:
                    mask = _mask_from_boxes(frame, boxes)
                    frame = cv2.bitwise_and(frame, frame, mask=mask)
                else:
                    frame[:] = 0

            writer.write(frame)
            written += 1
            src_idx += 1
    finally:
        cap.release()
        writer.release()

    return {
        "output_path": str(out_path),
        "render_mode": mode,
        "written_frames": int(written),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
    }
