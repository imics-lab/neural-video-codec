"""Unit tests for YOLO detection + KLT tracking (run_detection API)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_video(path: str, n: int = 5, h: int = 64, w: int = 64) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wtr    = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
    for _ in range(n):
        wtr.write(np.random.randint(0, 256, (h, w, 3), dtype=np.uint8))
    wtr.release()


def _default_cfg(keyframe_interval: int = 1) -> dict:
    return {
        "model_path": "yolo11n.pt",
        "animal_classes": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        "conf": 0.3,
        "iou_nms": 0.45,
        "processing_scale": 1.0,
        "imgsz": 64,
        "half": False,
        "device": "cpu",
        "bbox_pad_frac": 0.0,
        "bbox_pad_px": 0.0,
        "keyframe_interval": keyframe_interval,
        "tracking": {
            "enabled": True,
            "min_hits": 1,
            "max_det_gap": 90,
            "match_iou": 0.30,
            "vel_smooth": 0.70,
            "klt_max_points": 80,
            "klt_min_points": 10,
        },
    }


class TestRunDetection:
    """Tests for run_detection() (mocked YOLO and _detect_boxes)."""

    @pytest.fixture
    def tmp_video(self, tmp_path):
        p = tmp_path / "test.mp4"
        _make_video(str(p), n=3)
        return str(p)

    @patch("src.detection.yolo_detector._detect_boxes")
    def test_returns_required_keys(self, mock_detect_boxes, tmp_video):
        """run_detection must return frame_count, fps, width, height, frames."""
        mock_detect_boxes.return_value = ([], [], [])
        _fake_ultralytics = MagicMock()
        _fake_ultralytics.YOLO.return_value = MagicMock()

        with patch.dict("sys.modules", {"ultralytics": _fake_ultralytics}):
            from src.detection.yolo_detector import run_detection
            result = run_detection(tmp_video, _default_cfg())

        for key in ("frame_count", "fps", "width", "height", "frames"):
            assert key in result, f"Missing key: {key}"
        assert isinstance(result["frames"], dict)

    @patch("src.detection.yolo_detector._detect_boxes")
    def test_no_detections_empty_frames(self, mock_detect_boxes, tmp_video):
        """With no detections, frames dict should be empty."""
        mock_detect_boxes.return_value = ([], [], [])
        _fake_ultralytics = MagicMock()
        _fake_ultralytics.YOLO.return_value = MagicMock()

        with patch.dict("sys.modules", {"ultralytics": _fake_ultralytics}):
            from src.detection.yolo_detector import run_detection
            result = run_detection(tmp_video, _default_cfg())
        assert result["frames"] == {}

    @patch("src.detection.yolo_detector._detect_boxes")
    def test_detection_with_animal(self, mock_detect_boxes, tmp_video):
        """Detected animal boxes should appear in the frames dict."""
        # Single box: class 16 (dog), conf 0.9
        mock_detect_boxes.return_value = (
            [(5.0, 5.0, 30.0, 30.0)],  # boxes
            [0.9],                       # confs
            [16],                        # cls
        )
        _fake_ultralytics = MagicMock()
        _fake_ultralytics.YOLO.return_value = MagicMock()

        cfg = _default_cfg(keyframe_interval=1)
        cfg["tracking"]["min_hits"] = 1  # confirm on first hit

        with patch.dict("sys.modules", {"ultralytics": _fake_ultralytics}):
            from src.detection.yolo_detector import run_detection
            result = run_detection(tmp_video, cfg)

        assert len(result["frames"]) > 0
        # At least one frame should have a detection
        first_frame = next(iter(result["frames"].values()))
        assert len(first_frame) > 0
        det = first_frame[0]
        assert "x1" in det and "y1" in det and "x2" in det and "y2" in det
        assert "conf" in det and "cls" in det and "track_id" in det

    def test_missing_video_raises(self, tmp_path):
        """run_detection should raise when video file does not exist."""
        from src.detection.yolo_detector import run_detection
        with pytest.raises(Exception):
            run_detection(str(tmp_path / "nonexistent.mp4"), _default_cfg())

    @pytest.mark.skip(reason="Requires real YOLO weights (yolo11n.pt)")
    def test_run_detection_smoke_real(self, tmp_video):
        """Full smoke test with real YOLO model."""
        from src.detection.yolo_detector import run_detection
        result = run_detection(tmp_video, _default_cfg())
        assert "frames" in result


class TestDetectorHelpers:
    """Unit tests for geometry helpers."""

    def test_iou_identical_boxes(self):
        from src.detection.yolo_detector import _iou
        box = (0.0, 0.0, 10.0, 10.0)
        assert abs(_iou(box, box) - 1.0) < 1e-6

    def test_iou_non_overlapping(self):
        from src.detection.yolo_detector import _iou
        a = (0.0, 0.0, 5.0, 5.0)
        b = (10.0, 10.0, 20.0, 20.0)
        assert _iou(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_iou_partial_overlap(self):
        from src.detection.yolo_detector import _iou
        a = (0.0, 0.0, 10.0, 10.0)  # area=100
        b = (5.0, 0.0, 15.0, 10.0)  # area=100, overlap=(5,0,10,10)=50
        # IoU = 50 / (100 + 100 - 50) = 50/150 ≈ 0.333
        assert 0.3 < _iou(a, b) < 0.4

    def test_clip_xyxy_clamps(self):
        from src.detection.yolo_detector import _clip_xyxy
        clipped = _clip_xyxy((-5.0, -5.0, 999.0, 999.0), w=100, h=80)
        assert clipped[0] >= 0
        assert clipped[1] >= 0
        assert clipped[2] < 100
        assert clipped[3] < 80
