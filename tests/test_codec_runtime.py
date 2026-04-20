from __future__ import annotations

from pathlib import Path
from unittest import mock
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_decompression  # noqa: E402
from src.decompression import interpolation_amt  # noqa: E402


class TestCodecRuntime(unittest.TestCase):
    def test_amt_resolve_device_honors_explicit_index(self) -> None:
        with mock.patch.object(interpolation_amt.torch.cuda, "is_available", return_value=True):
            with mock.patch.object(interpolation_amt.torch.cuda, "device_count", return_value=4):
                with mock.patch.object(interpolation_amt.torch.cuda, "set_device") as mock_set_device:
                    device = interpolation_amt._resolve_device("cuda:3")

        self.assertEqual(str(device), "cuda:3")
        mock_set_device.assert_called_once_with(3)

    def test_enforce_runtime_preserves_interpolation_index(self) -> None:
        dec_cfg = {"interpolate": {"enable": True, "device": "cuda:2"}}
        meta = {"codec": {"implementation": "ffmpeg", "ffmpeg_bin": "ffmpeg"}}
        with mock.patch.object(run_decompression.torch.cuda, "is_available", return_value=True):
            report = run_decompression._enforce_strict_gpu_runtime(dec_cfg, meta)

        self.assertEqual(dec_cfg["interpolate"]["device"], "cuda:2")
        self.assertEqual(report["interpolate_device_selected"], "cuda:2")
        self.assertEqual(report["interpolate_cuda_idx_selected"], 2)
        self.assertEqual(report["codec_backend_selected"], "ffmpeg")

    def test_codec_override_is_applied_before_runtime_enforcement(self) -> None:
        dec_cfg = {
            "archive_codec": {"ffmpeg_bin": "custom-ffmpeg"},
            "interpolate": {"enable": False},
        }
        meta = {"codec": {"implementation": "ffmpeg", "ffmpeg_bin": "ffmpeg"}}

        run_decompression._apply_codec_overrides(meta, dec_cfg)
        with mock.patch.object(run_decompression.torch.cuda, "is_available", return_value=False):
            report = run_decompression._enforce_strict_gpu_runtime(dec_cfg, meta)

        self.assertEqual(meta["codec"]["ffmpeg_bin"], "custom-ffmpeg")
        self.assertEqual(report["runtime_mode"], "cpu_codec_only")

    def test_interpolate_many_batched_pads_small_roi_crops_like_reference(self) -> None:
        class _FakeModel:
            def __init__(self) -> None:
                self.last_shape = None

            def __call__(self, in0_b, in1_b, embt, scale_factor=1.0, eval=True):
                self.last_shape = tuple(in0_b.shape)
                return {"imgt_pred": in0_b}

        class _FakeInterpolator:
            def __init__(self) -> None:
                self.device = torch.device("cpu")
                self.pad_to = 16
                self.fp16 = False
                self.model = _FakeModel()

            @staticmethod
            def _bgr_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
                return torch.from_numpy(frame_bgr).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            @staticmethod
            def _tensor_to_bgr(frame_t: torch.Tensor) -> np.ndarray:
                return (
                    frame_t.clamp(0.0, 1.0)
                    .mul(255.0)
                    .byte()
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )

        interpolator = _FakeInterpolator()
        frame0 = np.zeros((32, 48, 3), dtype=np.uint8)
        frame1 = np.full((32, 48, 3), 255, dtype=np.uint8)

        mids = run_decompression._interpolate_many_batched(
            interpolator=interpolator,
            frame0_bgr=frame0,
            frame1_bgr=frame1,
            count=2,
            batch_size=2,
            max_crop_side=0,
        )

        self.assertEqual(len(mids), 2)
        self.assertEqual(mids[0].shape, frame0.shape)
        self.assertEqual(mids[1].shape, frame0.shape)
        self.assertIsNotNone(interpolator.model.last_shape)
        self.assertGreaterEqual(interpolator.model.last_shape[-2], 128)
        self.assertGreaterEqual(interpolator.model.last_shape[-1], 128)


if __name__ == "__main__":
    unittest.main()
