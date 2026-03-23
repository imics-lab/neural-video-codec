"""Unit tests for compression pipeline (phase_compress + roi_masking)."""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _make_frames(n: int = 5, h: int = 180, w: int = 320) -> list:
    return [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _minimal_cfg() -> dict:
    return {
        "compression": {
            "quality": {
                "roi_qp_i": 63, "roi_qp_p": 63,
                "bg_qp_i": 25,  "bg_qp_p": 25,
            },
            "dcvc": {
                "repo_dir": "DCVC",
                "model_i": "checkpoints/IntraNoAR/model.pth.tar",
                "model_p": "checkpoints/video/model.pth.tar",
                "device": "cpu",
            },
        },
        "detection": {
            "model_path": "yolo11n.pt",
            "animal_classes": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            "conf": 0.3,
            "iou_nms": 0.45,
            "keyframe_interval": 5,
            "processing_scale": 1.0,
        },
    }


class TestBuildBoxesMask:
    """Tests for build_boxes_mask in roi_masking (used by phase_compress)."""

    def test_empty_boxes_gives_zero_mask(self):
        from src.roi_masking.roi_masking import build_boxes_mask
        mask = build_boxes_mask(width=320, height=180, boxes=[])
        assert mask.shape == (180, 320)
        assert mask.max() == 0

    def test_full_frame_box_gives_full_mask(self):
        from src.roi_masking.roi_masking import build_boxes_mask
        H, W = 180, 320
        boxes = [{"x1": 0, "y1": 0, "x2": W, "y2": H, "conf": 0.9}]
        mask = build_boxes_mask(width=W, height=H, boxes=boxes)
        assert mask.min() > 0

    def test_partial_box(self):
        from src.roi_masking.roi_masking import build_boxes_mask
        H, W = 100, 100
        boxes = [{"x1": 10, "y1": 10, "x2": 50, "y2": 50, "conf": 0.9}]
        mask = build_boxes_mask(width=W, height=H, boxes=boxes)
        assert mask[30, 30] > 0
        assert mask[0, 0] == 0

    def test_multiple_boxes_union(self):
        from src.roi_masking.roi_masking import build_boxes_mask
        H, W = 100, 100
        boxes = [
            {"x1": 0,  "y1": 0,  "x2": 40, "y2": 40, "conf": 0.9},
            {"x1": 60, "y1": 60, "x2": 99, "y2": 99, "conf": 0.9},
        ]
        mask = build_boxes_mask(width=W, height=H, boxes=boxes)
        assert mask[20, 20] > 0
        assert mask[80, 80] > 0
        assert mask[50, 50] == 0

    def test_low_conf_filtered(self):
        from src.roi_masking.roi_masking import build_boxes_mask
        H, W = 100, 100
        boxes = [{"x1": 10, "y1": 10, "x2": 90, "y2": 90, "conf": 0.1}]
        mask = build_boxes_mask(width=W, height=H, boxes=boxes, min_conf=0.5)
        assert mask.max() == 0


class TestCompressVideoIntegration:
    """Integration smoke-test for compress_video (mocked DCVC encoder + detection)."""

    @patch("src.detection.yolo_detector.run_detection")
    @patch("src.compression.dcvc_encoder.encode_frames_to_bytes")
    def test_compress_returns_zip_bytes(self, mock_encode, mock_detect):
        """compress_video should return valid ZIP bytes with expected entries."""
        from src.compression.phase_compress import compress_video

        # Mock detection: no animals (all background)
        mock_detect.return_value = {
            "frames": {},
            "width": 320, "height": 180, "fps": 30.0, "frame_count": 5,
        }

        # Mock encoder: return fake bitstream
        mock_encode.return_value = {
            "bitstream_bytes": b"\x00" * 100,
            "meta": {"frames_encoded": 5, "fps": 30.0},
        }

        import tempfile, cv2
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name
        try:
            h, w = 180, 320
            wtr = cv2.VideoWriter(
                tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h)
            )
            for _ in range(5):
                wtr.write(np.random.randint(0, 256, (h, w, 3), dtype=np.uint8))
            wtr.release()

            result = compress_video(tmp_path, _minimal_cfg())
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        assert isinstance(result, bytes)
        zf = zipfile.ZipFile(io.BytesIO(result))
        names = zf.namelist()
        assert "meta.json" in names
        assert "roi.bin" in names
        assert "bg.bin" in names
        assert "detections.json" in names
