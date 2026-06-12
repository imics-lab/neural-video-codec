from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_decompression  # noqa: E402
from src.compression import dcvc_encoder  # noqa: E402
from src.decompression import interpolation_amt  # noqa: E402
from src.decompression import roi_bg_decompress  # noqa: E402


class TestDcvcDeviceRuntime(unittest.TestCase):
    def test_encoder_configure_cuda_sets_selected_device(self) -> None:
        with mock.patch.object(dcvc_encoder.torch.cuda, "is_available", return_value=True):
            with mock.patch.object(dcvc_encoder.torch.cuda, "device_count", return_value=4):
                with mock.patch.object(dcvc_encoder.torch.cuda, "set_device") as mock_set_device:
                    device = dcvc_encoder._configure_cuda(True, 2)

        self.assertEqual(str(device), "cuda:2")
        mock_set_device.assert_called_once_with(2)

    def test_decoder_configure_cuda_sets_selected_device(self) -> None:
        fake_cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 3,
            set_device=mock.Mock(),
        )
        fake_torch = SimpleNamespace(cuda=fake_cuda)
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            selected = roi_bg_decompress._configure_cuda(True, 1)

        self.assertEqual(selected, 1)
        fake_cuda.set_device.assert_called_once_with(1)

    def test_amt_resolve_device_honors_explicit_index(self) -> None:
        with mock.patch.object(interpolation_amt.torch.cuda, "is_available", return_value=True):
            with mock.patch.object(interpolation_amt.torch.cuda, "device_count", return_value=4):
                with mock.patch.object(interpolation_amt.torch.cuda, "set_device") as mock_set_device:
                    device = interpolation_amt._resolve_device("cuda:3")

        self.assertEqual(str(device), "cuda:3")
        mock_set_device.assert_called_once_with(3)

    def test_enforce_strict_gpu_runtime_preserves_interpolation_index(self) -> None:
        dec_cfg = {"interpolate": {"enable": True, "device": "cuda:2"}}
        meta = {"dcvc": {"use_cuda": True, "device": "cuda", "cuda_idx": 1}}
        with mock.patch.object(run_decompression.torch.cuda, "is_available", return_value=True):
            report = run_decompression._enforce_strict_gpu_runtime(dec_cfg, meta)

        self.assertEqual(dec_cfg["interpolate"]["device"], "cuda:2")
        self.assertEqual(report["interpolate_device_selected"], "cuda:2")
        self.assertEqual(report["interpolate_cuda_idx_selected"], 2)
        self.assertEqual(meta["dcvc"]["device"], "cuda:1")
        self.assertEqual(report["dcvc_device_selected"], "cuda:1")
        self.assertEqual(report["dcvc_cuda_idx_selected"], 1)

    def test_dcvc_device_override_is_applied_before_runtime_enforcement(self) -> None:
        dec_cfg = {
            "dcvc": {"device": "cuda:2", "use_cuda": True},
            "interpolate": {"enable": False},
        }
        meta = {"dcvc": {"use_cuda": True, "device": "cuda:0", "cuda_idx": 0}}

        run_decompression._apply_dcvc_overrides(meta, dec_cfg)
        with mock.patch.object(run_decompression.torch.cuda, "is_available", return_value=True):
            report = run_decompression._enforce_strict_gpu_runtime(dec_cfg, meta)

        self.assertEqual(meta["dcvc"]["device"], "cuda:2")
        self.assertEqual(report["dcvc_device_selected"], "cuda:2")
        self.assertEqual(report["dcvc_cuda_idx_selected"], 2)


if __name__ == "__main__":
    unittest.main()
