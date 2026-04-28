#!/usr/bin/env python3
"""
debug_roi.py — Save ROI regions from detections.json to verify bboxes are correct.

Usage:
    python debug_roi.py --video outputs/compression/bird1_decompressed.mp4 \
                        --detections outputs/compression/bird1_detections.json
    # or re-run detection directly on a video:
    python debug_roi.py --video outputs/compression/bird1_decompressed.mp4 --detect \
                        --config configs/gpu/compression.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def draw_bboxes(frame: np.ndarray, bboxes: list, color=(0, 255, 0), thickness=2) -> np.ndarray:
    out = frame.copy()
    for b in bboxes:
        if isinstance(b, dict):
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            label = f"{b.get('label','?')} {b.get('conf', 0):.2f}"
        else:
            x1, y1, x2, y2 = b[:4]
            label = ""
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(out, label, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video",       required=True)
    p.add_argument("--detections",  default=None,
                   help="Path to detections.json saved from archive")
    p.add_argument("--detect",      action="store_true",
                   help="Re-run MegaDetector on --video instead of loading detections file")
    p.add_argument("--config",      default="configs/gpu/compression.yaml")
    p.add_argument("--out",         default="debug_roi.mp4")
    p.add_argument("--max-frames",  type=int, default=300)
    args = p.parse_args()

    # ── Load detections ───────────────────────────────────────────────────────
    if args.detect:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        from src.detection.yolo_detector import run_detection
        print("Running detector ...", flush=True)
        det_result = run_detection(args.video, config=cfg["detection"])
        detections = {str(fi): boxes for fi, boxes in det_result["frames"].items()}
        # Save for inspection
        det_out = Path(args.out).with_suffix(".detections.json")
        det_out.write_text(json.dumps(detections, indent=2))
        print(f"Saved detections → {det_out}")
    elif args.detections:
        detections = json.loads(Path(args.detections).read_text())
    else:
        # Try to extract from archive zip next to the video
        vid = Path(args.video)
        archive = vid.parent / (vid.stem.replace("_decompressed", "") + ".zip")
        if not archive.exists():
            print("ERROR: no --detections file and no archive found. "
                  "Use --detect to re-run detection.", file=sys.stderr)
            return 1
        import zipfile, io
        with zipfile.ZipFile(str(archive)) as zf:
            detections = json.loads(zf.read("detections.json"))
        print(f"Loaded detections from {archive}")

    # ── Read video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.out)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))

    fi = 0
    n_with_roi = 0
    while fi < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        bboxes = detections.get(fi) or detections.get(str(fi)) or []
        if bboxes:
            n_with_roi += 1
            frame = draw_bboxes(frame, bboxes)
            # Also print first few to terminal
            if n_with_roi <= 5:
                print(f"  frame {fi}: {len(bboxes)} detections → {bboxes[0]}")
        writer.write(frame)
        fi += 1

    cap.release()
    writer.release()

    print(f"\nProcessed {fi} frames, {n_with_roi} had ROI detections.")
    print(f"Output → {out_path}")
    if n_with_roi == 0:
        print("WARNING: NO detections found — bboxes are empty!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
