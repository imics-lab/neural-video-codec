from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import test_compression  # noqa: E402
import test_decompression  # noqa: E402


class TestSanityArchiveHelpers(unittest.TestCase):
    def test_test_compression_reads_manifest_mapped_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            archive_path = Path(td) / "custom.zip"
            manifest = {
                "version": 1,
                "entries": {
                    "meta.json": "custom_meta.json",
                    "roi_detections.json": "custom_roi.json",
                    "frame_drop.json": "custom_frame_drop.json",
                    "roi.bin": "custom_roi.bin",
                    "bg.bin": "custom_bg.bin",
                },
            }
            meta = {
                "streams": {
                    "roi": {"frames_encoded": 1, "frame_index_map": [0]},
                    "bg": {"frames_encoded": 1, "frame_index_map": [0]},
                },
                "roi_detection": {"model_selection": {"selected_format": "pt"}},
                "runtime": {"codec_backend_selected": "ffmpeg"},
                "sizes": {},
            }
            frame_drop = {
                "roi_kept_frames": [0],
                "roi_dropped_frames": [],
                "bg_kept_frames": [0],
                "bg_dropped_frames": [],
            }
            roi_json = {"frames": {"0": [{"x1": 0, "y1": 0, "x2": 1, "y2": 1}]}}

            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("archive_manifest.json", json.dumps(manifest))
                zf.writestr("custom_meta.json", json.dumps(meta))
                zf.writestr("custom_roi.json", json.dumps(roi_json))
                zf.writestr("custom_frame_drop.json", json.dumps(frame_drop))
                zf.writestr("custom_roi.bin", b"roi")
                zf.writestr("custom_bg.bin", b"bg")

            metrics = test_compression._read_archive_metrics(archive_path)

        self.assertEqual(metrics["roi_bytes"], 3)
        self.assertEqual(metrics["bg_bytes"], 2)
        self.assertEqual(metrics["roi_frames_with_boxes"], 1)
        self.assertEqual(metrics["roi_stream_frames_encoded"], 1)

    def test_test_decompression_reads_manifest_mapped_meta(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            archive_path = Path(td) / "custom.zip"
            manifest = {
                "version": 1,
                "entries": {
                    "meta.json": "custom_meta.json",
                    "roi_detections.json": "custom_roi.json",
                    "frame_drop.json": "custom_frame_drop.json",
                    "roi.bin": "custom_roi.bin",
                    "bg.bin": "custom_bg.bin",
                },
            }
            meta = {"video": {"frames_total": 12}}

            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("archive_manifest.json", json.dumps(manifest))
                zf.writestr("custom_meta.json", json.dumps(meta))
                zf.writestr("custom_roi.json", json.dumps({"frames": {}}))
                zf.writestr("custom_frame_drop.json", json.dumps({"stats": {}}))
                zf.writestr("custom_roi.bin", b"roi")
                zf.writestr("custom_bg.bin", b"bg")

            loaded = test_decompression._read_archive_meta(archive_path)

        self.assertEqual(loaded["video"]["frames_total"], 12)


if __name__ == "__main__":
    unittest.main()
