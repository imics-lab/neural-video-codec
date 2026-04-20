from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.config_schema import validate_pipeline_config  # noqa: E402


def _codec_cfg() -> dict:
    return {
        "ffmpeg_bin": "ffmpeg",
        "ffprobe_bin": "ffprobe",
        "roi": {"codec": "av1", "encoder": "libsvtav1", "container": "mkv", "preset": 0, "qp": 10},
        "bg": {"codec": "hevc", "encoder": "libx265", "container": "mkv", "preset": "medium", "qp": 35},
    }


class TestConfigSchema(unittest.TestCase):
    def _base_cfg(self, root: Path) -> dict:
        return {
            "input": {"video_path": "video.mp4"},
            "roi_detection": {
                "paths": {"animal_model_path": "models/yolo.pt"},
                "runtime": {"processing_scale": 0.5, "imgsz": 640, "conf": 0.25, "iou_nms": 0.5, "keyframe_interval": 15},
                "tracking": {"match_iou": 0.3, "min_hits": 2},
            },
            "frame_removal": {
                "params": {
                    "motion": {"t_low": 4.0, "t_high": 7.0},
                    "bbox_smoothing": {"enable": True, "ema_alpha": 0.6},
                }
            },
            "compression": {
                "codec": _codec_cfg(),
                "quality": {"roi_qp_i": 10, "roi_qp_p": 10, "bg_qp_i": 35, "bg_qp_p": 35},
                "roi": {"min_conf": 0.25},
            },
            "output": {"write_outputs": True, "out_dir": "outputs/test"},
        }

    def test_valid_config_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "video.mp4").write_bytes(b"x")
            (root / "models").mkdir(parents=True, exist_ok=True)
            (root / "models" / "yolo.pt").write_bytes(b"x")
            cfg = self._base_cfg(root)
            validate_pipeline_config(cfg, root_dir=root)

    def test_invalid_thresholds_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "video.mp4").write_bytes(b"x")
            (root / "models").mkdir(parents=True, exist_ok=True)
            (root / "models" / "yolo.pt").write_bytes(b"x")

            cfg = self._base_cfg(root)
            cfg["frame_removal"]["params"]["motion"]["t_high"] = 1.0

            with self.assertRaises(ValueError):
                validate_pipeline_config(cfg, root_dir=root)

    def test_roi_model_not_required_when_roi_detection_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "video.mp4").write_bytes(b"x")

            cfg = self._base_cfg(root)
            cfg["roi_detection"]["enable"] = False
            cfg["roi_detection"]["paths"].pop("animal_model_path", None)

            validate_pipeline_config(cfg, root_dir=root)

    def test_tracking_enable_propagation_must_be_bool(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "video.mp4").write_bytes(b"x")
            (root / "models").mkdir(parents=True, exist_ok=True)
            (root / "models" / "yolo.pt").write_bytes(b"x")

            cfg = self._base_cfg(root)
            cfg["roi_detection"]["tracking"]["enable_propagation"] = "false"

            with self.assertRaises(ValueError):
                validate_pipeline_config(cfg, root_dir=root)

    def test_missing_onnx_path_can_fall_back_to_pt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "video.mp4").write_bytes(b"x")
            (root / "models").mkdir(parents=True, exist_ok=True)
            (root / "models" / "yolo.pt").write_bytes(b"x")

            cfg = self._base_cfg(root)
            cfg["roi_detection"]["paths"]["animal_model_path_onnx"] = "models/yolo.onnx"

            validate_pipeline_config(cfg, root_dir=root)

    def test_onnx_model_can_be_used_without_pt_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "video.mp4").write_bytes(b"x")
            (root / "models").mkdir(parents=True, exist_ok=True)
            (root / "models" / "yolo.onnx").write_bytes(b"x")

            cfg = self._base_cfg(root)
            cfg["roi_detection"]["paths"].pop("animal_model_path", None)
            cfg["roi_detection"]["paths"]["animal_model_path_onnx"] = "models/yolo.onnx"

            validate_pipeline_config(cfg, root_dir=root)

    def test_roi_detection_requires_one_existing_model(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "video.mp4").write_bytes(b"x")
            (root / "models").mkdir(parents=True, exist_ok=True)

            cfg = self._base_cfg(root)
            cfg["roi_detection"]["paths"]["animal_model_path"] = "models/missing.pt"
            cfg["roi_detection"]["paths"]["animal_model_path_onnx"] = "models/missing.onnx"

            with self.assertRaises(ValueError):
                validate_pipeline_config(cfg, root_dir=root)

    def test_prefer_onnx_strict_no_longer_requires_prefer_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "video.mp4").write_bytes(b"x")
            (root / "models").mkdir(parents=True, exist_ok=True)
            (root / "models" / "yolo.pt").write_bytes(b"x")

            cfg = self._base_cfg(root)
            cfg["roi_detection"]["runtime"]["prefer_onnx_strict"] = True

            validate_pipeline_config(cfg, root_dir=root)

    def test_ffmpeg_qp_must_be_in_range(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "video.mp4").write_bytes(b"x")
            (root / "models").mkdir(parents=True, exist_ok=True)
            (root / "models" / "yolo.pt").write_bytes(b"x")

            cfg = self._base_cfg(root)
            cfg["compression"]["codec"]["roi"]["qp"] = 99

            with self.assertRaises(ValueError):
                validate_pipeline_config(cfg, root_dir=root)


if __name__ == "__main__":
    unittest.main()
