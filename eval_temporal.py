"""
Temporal consistency evaluation via motion-compensated warping error.

For each pair of consecutive frames in a video:
  1. Compute Farneback optical flow from frame t to frame t+1.
  2. Warp frame t using the flow.
  3. Record MSE between the warped frame and frame t+1.

A lower mean warping error means the restored sequence is more temporally
consistent (less flickering independent of scene motion).
"""

import argparse
import sys

import cv2
import numpy as np


def _warp(frame_f32, flow):
    h, w = flow.shape[:2]
    map_x = (np.arange(w, dtype=np.float32)[None, :] + flow[..., 0])
    map_y = (np.arange(h, dtype=np.float32)[:, None] + flow[..., 1])
    return cv2.remap(frame_f32, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def warping_errors(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"no frames in {video_path}")

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_f = frame.astype(np.float32) / 255.0
    errors = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_f = frame.astype(np.float32) / 255.0

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        warped = _warp(prev_f, flow)
        errors.append(float(np.mean((warped - curr_f) ** 2)))

        prev_gray = curr_gray
        prev_f = curr_f

    cap.release()
    return np.array(errors)


def main():
    ap = argparse.ArgumentParser(description="Temporal consistency (warping error)")
    ap.add_argument("--videos", nargs="+", required=True,
                    help="Video file paths to evaluate")
    ap.add_argument("--labels", nargs="+",
                    help="Display labels (same order as --videos)")
    args = ap.parse_args()

    labels = args.labels if args.labels else args.videos
    if len(labels) != len(args.videos):
        sys.exit("--labels count must match --videos count")

    print(f"\n{'Label':<22} {'Mean WE x1e4':>13} {'Std WE x1e4':>12} {'Frames':>7}")
    print("-" * 58)
    for label, path in zip(labels, args.videos):
        errs = warping_errors(path)
        mean_we = np.mean(errs) * 1e4
        std_we = np.std(errs) * 1e4
        print(f"{label:<22} {mean_we:>13.4f} {std_we:>12.4f} {len(errs):>7}")
    print()


if __name__ == "__main__":
    main()
