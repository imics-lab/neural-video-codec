from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roi_detection import roi_detector as rd  # noqa: E402


class TestRoiDetectionAblation(unittest.TestCase):
    def _write_video(self, path: Path, num_frames: int = 3) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 64), True)
        if not writer.isOpened():
            raise RuntimeError("Failed to open test video writer")
        try:
            for _ in range(num_frames):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

    def _base_cfg(
        self,
        *,
        keyframe_interval: int,
        min_hits: int = 1,
        enable_propagation: bool = True,
    ) -> dict:
        return {
            "paths": {"animal_model_path": "models/dummy.pt"},
            "runtime": {
                "processing_scale": 1.0,
                "imgsz": 64,
                "conf": 0.25,
                "iou_nms": 0.5,
                "keyframe_interval": keyframe_interval,
            },
            "tracking": {
                "min_hits": min_hits,
                "match_iou": 0.3,
                "max_det_gap": 90,
                "klt_max_points": 8,
                "klt_min_points": 1,
                "enable_propagation": enable_propagation,
            },
        }

    def test_enable_propagation_controls_intermediate_roi_emission(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video_path = tmp / "clip.avi"
            self._write_video(video_path)

            base_cfg = self._base_cfg(keyframe_interval=2, enable_propagation=True)

            def fake_detect_proc_boxes(**_kwargs: object) -> tuple[list[tuple[float, float, float, float]], list[float]]:
                return [(10.0, 10.0, 20.0, 20.0)], [0.9]

            def fake_seed_points(
                _gray: np.ndarray,
                _bbox: tuple[float, float, float, float],
                max_pts: int,
            ) -> np.ndarray:
                self.assertGreaterEqual(max_pts, 1)
                return np.array([[[10.0, 10.0]], [[15.0, 15.0]]], dtype=np.float32)

            def fake_klt_translation(
                _prev_gray: np.ndarray,
                _cur_gray: np.ndarray,
                _pts_prev: np.ndarray,
                min_valid: int,
            ) -> tuple[np.ndarray, tuple[float, float], int]:
                self.assertEqual(min_valid, 1)
                pts_cur = np.array([[[15.0, 10.0]], [[20.0, 15.0]]], dtype=np.float32)
                return pts_cur, (5.0, 0.0), 2

            with patch.object(rd, "YOLO", return_value=object()):
                with patch.object(rd, "_detect_proc_boxes", side_effect=fake_detect_proc_boxes):
                    with patch.object(rd, "_seed_klt_points", side_effect=fake_seed_points):
                        with patch.object(rd, "_klt_translation", side_effect=fake_klt_translation):
                            cfg_prop = {**base_cfg, "tracking": {**base_cfg["tracking"], "enable_propagation": True}}
                            result_prop = rd.run_roi_detection(str(video_path), cfg_prop)

                            cfg_no_prop = {**base_cfg, "tracking": {**base_cfg["tracking"], "enable_propagation": False}}
                            result_no_prop = rd.run_roi_detection(str(video_path), cfg_no_prop)

            frames_prop = {int(k): v for k, v in (result_prop.get("frames", {}) or {}).items()}
            frames_no_prop = {int(k): v for k, v in (result_no_prop.get("frames", {}) or {}).items()}

            self.assertIn(0, frames_prop)
            self.assertIn(1, frames_prop)
            self.assertIn(2, frames_prop)
            self.assertNotIn(1, frames_no_prop)
            self.assertIn(0, frames_no_prop)
            self.assertIn(2, frames_no_prop)

            propagated_box = frames_prop[1][0]
            self.assertEqual(int(propagated_box["x1"]), 15)
            self.assertEqual(int(propagated_box["x2"]), 25)

    def test_first_detection_is_emitted_immediately_with_sparse_keyframes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video_path = tmp / "clip.avi"
            self._write_video(video_path, num_frames=16)

            cfg = self._base_cfg(keyframe_interval=15, min_hits=2, enable_propagation=True)

            def fake_detect_proc_boxes(**_kwargs: object) -> tuple[list[tuple[float, float, float, float]], list[float]]:
                return [(10.0, 10.0, 20.0, 20.0)], [0.9]

            def fake_seed_points(
                _gray: np.ndarray,
                _bbox: tuple[float, float, float, float],
                max_pts: int,
            ) -> np.ndarray:
                self.assertGreaterEqual(max_pts, 1)
                return np.array([[[10.0, 10.0]], [[15.0, 15.0]]], dtype=np.float32)

            def fake_klt_translation(
                _prev_gray: np.ndarray,
                _cur_gray: np.ndarray,
                _pts_prev: np.ndarray,
                min_valid: int,
            ) -> tuple[np.ndarray, tuple[float, float], int]:
                self.assertEqual(min_valid, 1)
                pts_cur = np.array([[[10.0, 10.0]], [[15.0, 15.0]]], dtype=np.float32)
                return pts_cur, (0.0, 0.0), 2

            with patch.object(rd, "YOLO", return_value=object()):
                with patch.object(rd, "_detect_proc_boxes", side_effect=fake_detect_proc_boxes):
                    with patch.object(rd, "_seed_klt_points", side_effect=fake_seed_points):
                        with patch.object(rd, "_klt_translation", side_effect=fake_klt_translation):
                            result = rd.run_roi_detection(str(video_path), cfg)

            frames = {int(k): v for k, v in (result.get("frames", {}) or {}).items()}
            self.assertEqual(list(range(16)), sorted(frames.keys()))

    def test_track_is_retired_when_detector_misses_on_keyframe(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video_path = tmp / "clip.avi"
            self._write_video(video_path, num_frames=4)

            cfg = self._base_cfg(keyframe_interval=2, min_hits=1, enable_propagation=True)
            detector_calls = {"count": 0}

            def fake_detect_proc_boxes(**_kwargs: object) -> tuple[list[tuple[float, float, float, float]], list[float]]:
                detector_calls["count"] += 1
                if detector_calls["count"] == 1:
                    return [(10.0, 10.0, 20.0, 20.0)], [0.9]
                return [], []

            def fake_seed_points(
                _gray: np.ndarray,
                _bbox: tuple[float, float, float, float],
                max_pts: int,
            ) -> np.ndarray:
                self.assertGreaterEqual(max_pts, 1)
                return np.array([[[10.0, 10.0]], [[15.0, 15.0]]], dtype=np.float32)

            def fake_klt_translation(
                _prev_gray: np.ndarray,
                _cur_gray: np.ndarray,
                _pts_prev: np.ndarray,
                min_valid: int,
            ) -> tuple[np.ndarray, tuple[float, float], int]:
                self.assertEqual(min_valid, 1)
                pts_cur = np.array([[[10.0, 10.0]], [[15.0, 15.0]]], dtype=np.float32)
                return pts_cur, (0.0, 0.0), 2

            with patch.object(rd, "YOLO", return_value=object()):
                with patch.object(rd, "_detect_proc_boxes", side_effect=fake_detect_proc_boxes):
                    with patch.object(rd, "_seed_klt_points", side_effect=fake_seed_points):
                        with patch.object(rd, "_klt_translation", side_effect=fake_klt_translation):
                            result = rd.run_roi_detection(str(video_path), cfg)

            frames = {int(k): v for k, v in (result.get("frames", {}) or {}).items()}
            self.assertIn(0, frames)
            self.assertIn(1, frames)
            self.assertNotIn(2, frames)
            self.assertNotIn(3, frames)

    def test_invalid_keyframe_interval_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video_path = tmp / "clip.avi"
            self._write_video(video_path, num_frames=2)

            cfg = self._base_cfg(keyframe_interval=0, min_hits=1, enable_propagation=False)

            with patch.object(rd, "YOLO", return_value=object()):
                with self.assertRaises(ValueError):
                    rd.run_roi_detection(str(video_path), cfg)


if __name__ == "__main__":
    unittest.main()
