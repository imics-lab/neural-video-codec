#!/usr/bin/env python3
"""
debug_roi.py — Overlay detection bboxes from an existing archive onto the decompressed video.

Usage:
    python debug_roi.py --archive outputs/compression/bird1.zip \
                        --video   outputs/compression/bird1_decompressed.mp4
"""
import argparse, io, json, sys, zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True, help="ZIP archive from compression stage")
    p.add_argument("--video",   required=True, help="Decompressed video to overlay on")
    p.add_argument("--output",  default="debug_roi_overlay.mp4")
    args = p.parse_args()

    with zipfile.ZipFile(args.archive) as arc:
        detections = json.loads(arc.read("detections.json"))

    print(f"Frames with detections: {len(detections)}")
    print(f"Total boxes: {sum(len(v) for v in detections.values())}")
    if detections:
        k = list(detections.keys())[0]
        print(f"Sample [{k}]: {detections[k][0]}")
    else:
        print("ERROR: detections.json is empty — no ROI was detected")
        return 1

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = detections.get(fi) or detections.get(str(fi)) or []
        for b in boxes:
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{b.get('label','?')} {b.get('conf', 0):.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out.write(frame)
        fi += 1

    cap.release()
    out.release()
    print(f"Saved {fi} frames → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
