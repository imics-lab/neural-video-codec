"""Unit tests for decompression pipeline (phase_decompress)."""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _make_zip(
    width: int = 64,
    height: int = 64,
    n_frames: int = 3,
    roi_bytes: bytes = b"\x00" * 50,
    bg_bytes:  bytes = b"\x00" * 50,
    detections: dict | None = None,
) -> bytes:
    """Build a minimal fake archive matching phase_decompress's expected format."""
    meta = {
        "version": "1.0.0",
        "video": {
            "width": width, "height": height,
            "fps": 30.0, "frame_count": n_frames,
        },
        "dcvc": {},
        "streams": {
            "roi": {"frames_encoded": n_frames},
            "bg":  {"frames_encoded": n_frames},
        },
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("meta.json",               json.dumps(meta))
        zf.writestr("detections.json",          json.dumps(detections or {}))
        zf.writestr("roi.bin",                  roi_bytes)
        zf.writestr("bg.bin",                   bg_bytes)
        zf.writestr("compression_config.json",  json.dumps({}))
    return buf.getvalue()


class TestDecompressArchive:
    """Tests for decompress_archive."""

    @patch("src.decompression.dcvc_decoder.decode_stream_bytes")
    def test_returns_frames_and_fps(self, mock_decode):
        """Should return dict with 'frames' list and 'fps' float."""
        from src.decompression.phase_decompress import decompress_archive

        n = 3
        h, w = 64, 64
        fake_frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]
        mock_decode.return_value = fake_frames

        archive = _make_zip(width=w, height=h, n_frames=n)
        cfg = {"decompression": {"dcvc": {}}}
        result = decompress_archive(archive, cfg)

        assert "frames" in result
        assert "fps" in result
        assert isinstance(result["frames"], list)
        assert len(result["frames"]) == n
        assert result["fps"] == 30.0

    @patch("src.decompression.dcvc_decoder.decode_stream_bytes")
    def test_composite_blending(self, mock_decode):
        """ROI region should overwrite BG region in output frames."""
        from src.decompression.phase_decompress import decompress_archive

        h, w = 64, 64
        # ROI frames: all red (BGR: 0, 0, 255)
        roi_frames = [np.full((h, w, 3), [0, 0, 255], dtype=np.uint8)]
        # BG frames: all green (BGR: 0, 255, 0)
        bg_frames  = [np.full((h, w, 3), [0, 255, 0], dtype=np.uint8)]

        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return roi_frames if call_count[0] == 1 else bg_frames

        mock_decode.side_effect = side_effect

        # Detections covering the full frame
        dets = {"0": [{"x1": 0, "y1": 0, "x2": w, "y2": h,
                        "conf": 0.9, "cls": 16, "label": "dog", "track_id": 0}]}
        archive = _make_zip(width=w, height=h, n_frames=1, detections=dets)

        cfg    = {"decompression": {"dcvc": {}}}
        result = decompress_archive(archive, cfg)

        assert len(result["frames"]) == 1
        frame = result["frames"][0]
        # ROI (full frame) should be red
        assert np.all(frame[:, :, 2] == 255), "Red channel should be 255 in ROI region"
        assert np.all(frame[:, :, 1] == 0),   "Green channel should be 0 in ROI region"

    def test_malformed_archive_raises(self):
        """Non-ZIP bytes should raise an exception."""
        from src.decompression.phase_decompress import decompress_archive
        with pytest.raises(Exception):
            decompress_archive(b"not a zip file at all", {})
