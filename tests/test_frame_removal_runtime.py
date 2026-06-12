from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frame_removal import remove_redundant_frames, validate_frame_removal_config  # noqa: E402
from frame_removal.remove_frames import parse_box_any  # noqa: E402


def _write_test_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (16, 16), True)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open test video writer: {path}")
    try:
        for idx in range(int(frame_count)):
            frame = np.full((16, 16, 3), idx * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class TestFrameRemovalRuntime(unittest.TestCase):
    def test_parse_box_any_returns_none_for_bad_dict_values(self) -> None:
        bad = {"x1": "bad", "y1": 0, "x2": 1, "y2": 1}
        self.assertIsNone(parse_box_any(bad))

    def test_validate_frame_removal_config_rejects_string_booleans(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame_removal.params.roi.gray"):
            validate_frame_removal_config({"params": {"roi": {"gray": "false"}}})

        with self.assertRaisesRegex(ValueError, "frame_removal.params.bbox_smoothing.enable"):
            validate_frame_removal_config({"params": {"bbox_smoothing": {"enable": "false"}}})

        with self.assertRaisesRegex(ValueError, "frame_removal.dual_timeline.enable"):
            validate_frame_removal_config({"dual_timeline": {"enable": "true"}})

    def test_remove_redundant_frames_skips_malformed_roi_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            video_path = Path(td) / "test.avi"
            _write_test_video(video_path, frame_count=2)

            result = remove_redundant_frames(
                str(video_path),
                {0: [{"x1": "bad", "y1": 0, "x2": 8, "y2": 8}]},
                {},
            )

        self.assertEqual(result["per_frame"]["0"]["roi_count"], 0)
        self.assertTrue(result["per_frame"]["0"]["bbox_missing"])

    def test_remove_redundant_frames_rejects_string_boolean_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            video_path = Path(td) / "test.avi"
            _write_test_video(video_path, frame_count=2)

            with self.assertRaisesRegex(ValueError, "frame_removal.params.roi.gray"):
                remove_redundant_frames(
                    str(video_path),
                    {},
                    {"params": {"roi": {"gray": "false"}}},
                )


if __name__ == "__main__":
    unittest.main()
