from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest.mock import patch

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_compression  # noqa: E402
from src.decompression import common as rd  # noqa: E402


def _codec_cfg() -> dict:
    return {
        "ffmpeg_bin": "ffmpeg",
        "ffprobe_bin": "ffprobe",
        "roi": {"codec": "av1", "encoder": "libsvtav1", "container": "mkv", "preset": 0, "qp": 10},
        "bg": {"codec": "hevc", "encoder": "libx265", "container": "mkv", "preset": "medium", "qp": 35},
    }


class TestRunOutputContract(unittest.TestCase):
    def test_main_writes_zip_and_removes_raw_outputs(self) -> None:
        run_suffix = uuid.uuid4().hex
        out_dir_rel = f"outputs/test_run_contract_{run_suffix}"
        out_dir_abs = ROOT / out_dir_rel

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video_path = tmp / "video.mp4"
            yolo_model = tmp / "yolo.pt"

            video_path.write_bytes(b"v")
            yolo_model.write_bytes(b"m")

            cfg = {
                "input": {"video_path": str(video_path)},
                "roi_detection": {"enable": False, "paths": {"animal_model_path": str(yolo_model)}},
                "frame_removal": {"params": {"motion": {"t_low": 1.0, "t_high": 2.0}}},
                "compression": {
                    "codec": _codec_cfg(),
                    "quality": {"roi_qp_i": 10, "roi_qp_p": 10, "bg_qp_i": 35, "bg_qp_p": 35},
                    "roi": {"min_conf": 0.25},
                },
                "output": {
                    "write_outputs": True,
                    "out_dir": out_dir_rel,
                    "roi_json": "roi_detections.json",
                    "frame_drop_json": "frame_drop.json",
                    "roi_bin": "roi.bin",
                    "bg_bin": "bg.bin",
                    "meta_json": "meta.json",
                },
            }
            cfg_path = tmp / "cfg.yaml"
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

            fake_frame_result = {
                "kept_frames": [0],
                "dropped_frames": [],
                "per_frame": {},
                "stats": {"num_frames_read": 1},
            }
            fake_compression_result = {"roi_bin_bytes": b"roi", "bg_bin_bytes": b"bg", "meta": {"ok": True}}

            with patch.object(run_compression, "_cuda_is_available", return_value=True):
                with patch.object(run_compression, "remove_redundant_frames", return_value=fake_frame_result):
                    with patch.object(run_compression, "compress_keep_streams", return_value=fake_compression_result):
                        with patch.object(sys, "argv", ["run_compression.py", str(video_path), "--config", str(cfg_path)]):
                            run_compression.main()

            zip_path = out_dir_abs / "video.zip"
            self.assertTrue(zip_path.exists(), "Expected output zip file")
            self.assertFalse((out_dir_abs / "roi_detections.json").exists())
            self.assertFalse((out_dir_abs / "frame_drop.json").exists())
            self.assertFalse((out_dir_abs / "roi.bin").exists())
            self.assertFalse((out_dir_abs / "bg.bin").exists())
            self.assertFalse((out_dir_abs / "meta.json").exists())

            with zipfile.ZipFile(zip_path, "r") as zf:
                self.assertEqual(
                    set(zf.namelist()),
                    {
                        "roi_detections.json",
                        "frame_drop.json",
                        "roi.bin",
                        "bg.bin",
                        "meta.json",
                        "compression.runtime_config.json",
                        "archive_manifest.json",
                    },
                )
                meta = json.loads(zf.read("meta.json").decode("utf-8"))
                self.assertIn("roi_detection", meta)
                self.assertIn("runtime", meta)
                self.assertIn("provenance", meta)

        if out_dir_abs.exists():
            shutil.rmtree(out_dir_abs)

    def test_runtime_fallbacks_record_ffmpeg_backend(self) -> None:
        cfg = {
            "roi_detection": {"enable": False},
            "compression": {"codec": _codec_cfg()},
        }
        report = run_compression._apply_runtime_device_fallbacks(cfg)
        self.assertEqual(report["codec_backend_selected"], "ffmpeg")
        self.assertEqual(report["codec_ffmpeg_bin"], "ffmpeg")

    def test_model_select_prefers_onnx_with_gpu_provider(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pt_model = tmp / "model.pt"
            onnx_model = tmp / "model.onnx"
            pt_model.write_bytes(b"pt")
            onnx_model.write_bytes(b"onnx")
            cfg = {
                "roi_detection": {
                    "paths": {
                        "animal_model_path": str(pt_model),
                        "animal_model_path_onnx": str(onnx_model),
                    },
                    "runtime": {},
                }
            }
            with patch.object(
                run_compression,
                "_detect_onnx_gpu_provider",
                return_value=(True, "CUDAExecutionProvider"),
            ):
                selected = run_compression._select_roi_model_for_runtime(cfg)
        self.assertEqual(selected["selected_format"], "onnx")
        self.assertEqual(selected["selected_model_path"], str(onnx_model))
        self.assertEqual(selected["selected_model_resolved_path"], str(onnx_model.resolve()))
        self.assertIsNone(selected["fallback_reason"])

    def test_model_select_falls_back_to_pt_when_onnx_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pt_model = tmp / "model.pt"
            pt_model.write_bytes(b"pt")
            cfg = {
                "roi_detection": {
                    "paths": {
                        "animal_model_path": str(pt_model),
                        "animal_model_path_onnx": str(tmp / "missing.onnx"),
                    },
                    "runtime": {},
                }
            }
            with patch.object(
                run_compression,
                "_detect_onnx_gpu_provider",
                return_value=(True, "CUDAExecutionProvider"),
            ):
                selected = run_compression._select_roi_model_for_runtime(cfg)
        self.assertEqual(selected["selected_format"], "pt")
        self.assertEqual(selected["selected_model_path"], str(pt_model))
        self.assertEqual(selected["fallback_reason"], "onnx_missing_fallback_pt")
        self.assertFalse(selected["prefer_onnx"])

    def test_model_select_falls_back_to_pt_when_onnx_runtime_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pt_model = tmp / "model.pt"
            onnx_model = tmp / "model.onnx"
            pt_model.write_bytes(b"pt")
            onnx_model.write_bytes(b"onnx")
            cfg = {
                "roi_detection": {
                    "paths": {
                        "animal_model_path": str(pt_model),
                        "animal_model_path_onnx": str(onnx_model),
                    },
                    "runtime": {},
                }
            }
            with patch.object(
                run_compression,
                "_detect_onnx_gpu_provider",
                return_value=(False, "onnxruntime_import_error:ModuleNotFoundError"),
            ):
                selected = run_compression._select_roi_model_for_runtime(cfg)
        self.assertEqual(selected["selected_format"], "pt")
        self.assertEqual(selected["selected_model_path"], str(pt_model))
        self.assertEqual(selected["fallback_reason"], "onnx_runtime_unavailable_fallback_pt")

    def test_model_select_strict_onnx_raises_when_runtime_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pt_model = tmp / "model.pt"
            onnx_model = tmp / "model.onnx"
            pt_model.write_bytes(b"pt")
            onnx_model.write_bytes(b"onnx")
            cfg = {
                "roi_detection": {
                    "paths": {
                        "animal_model_path": str(pt_model),
                        "animal_model_path_onnx": str(onnx_model),
                    },
                    "runtime": {
                        "prefer_onnx_strict": True,
                    },
                }
            }
            with patch.object(
                run_compression,
                "_detect_onnx_gpu_provider",
                return_value=(False, "onnxruntime_import_error:ModuleNotFoundError"),
            ):
                with self.assertRaises(RuntimeError):
                    run_compression._select_roi_model_for_runtime(cfg)

    def test_model_select_raises_when_both_detector_models_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = {
                "roi_detection": {
                    "paths": {
                        "animal_model_path": str(tmp / "missing.pt"),
                        "animal_model_path_onnx": str(tmp / "missing.onnx"),
                    },
                    "runtime": {},
                }
            }

            with self.assertRaises(FileNotFoundError):
                run_compression._select_roi_model_for_runtime(cfg)

    def test_archive_loader_supports_manifest_for_custom_entry_names(self) -> None:
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
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr("archive_manifest.json", json.dumps(manifest))
                zf.writestr("custom_meta.json", "{}")
                zf.writestr("custom_roi.json", '{"frames": {}}')
                zf.writestr("custom_frame_drop.json", '{"stats": {}}')
                zf.writestr("custom_roi.bin", b"roi")
                zf.writestr("custom_bg.bin", b"bg")

            payloads = rd._load_archive_payloads(archive_path)

        self.assertEqual(json.loads(payloads["meta.json"].decode("utf-8")), {})
        self.assertEqual(payloads["roi.bin"], b"roi")
        self.assertEqual(payloads["bg.bin"], b"bg")

    def test_pick_stream_indices_prefers_metadata_frame_index_map(self) -> None:
        frame_drop_json = {"roi_kept_frames": [0, 10, 20]}
        meta = {"streams": {"roi": {"frame_index_map": [1, 11, 21]}}}

        picked = rd._pick_stream_indices(frame_drop_json, meta, "roi")

        self.assertEqual(picked, [1, 11, 21])


if __name__ == "__main__":
    unittest.main()
