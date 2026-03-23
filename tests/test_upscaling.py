"""Unit tests for Model S super-resolution network and phase_upscale."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")


class TestSRUNet:
    """Shape and forward-pass tests for SRUNet."""

    @pytest.fixture
    def model(self):
        from src.upscaling._network import SRUNet
        return SRUNet(
            base_channels=16,
            encoder_channels=(32, 64, 128),
            num_res_blocks=1,
        ).eval()

    def test_output_shape(self, model):
        """Output should be (B, 3, H, W) matching input spatial dims."""
        import torch
        B, H, W = 2, 64, 64
        x = torch.randn(B, 6, H, W)
        t = torch.randint(0, 1000, (B,))
        with torch.no_grad():
            out = model(x, t)
        assert out.shape == (B, 3, H, W)

    def test_no_nan(self, model):
        import torch
        x = torch.randn(1, 6, 64, 64)
        t = torch.zeros(1, dtype=torch.long)
        with torch.no_grad():
            out = model(x, t)
        assert not torch.isnan(out).any()

    def test_in_channels_constant(self):
        from src.upscaling._network import SRUNet
        assert SRUNet.IN_CHANNELS == 6

    def test_no_temporal_attention(self):
        """SRUNet must NOT contain TemporalAttention blocks."""
        from src.upscaling._network import SRUNet
        try:
            from src.restoration._blocks import TemporalAttention
        except ImportError:
            pytest.skip("TemporalAttention not importable from restoration._blocks")

        model = SRUNet(base_channels=16)
        for module in model.modules():
            assert not isinstance(module, TemporalAttention), \
                "SRUNet should not contain TemporalAttention layers"


# ── Checkpoint fixture ────────────────────────────────────────────────────────

def _make_sr_checkpoint(tmp_path: Path) -> tuple:
    """Create a tiny SRUNet checkpoint and return (ckpt_path, model_cfg)."""
    import torch
    from src.upscaling._network import SRUNet

    model_cfg = dict(
        base_channels=8,
        encoder_channels=(16, 32, 64),
        num_res_blocks=1,
    )
    model = SRUNet(**model_cfg)
    ckpt_path = tmp_path / "sr.pt"
    torch.save({"model": model.state_dict(), "model_cfg": model_cfg}, str(ckpt_path))
    return ckpt_path, model_cfg


class TestUpscaler:
    """Tests for the high-level Upscaler class."""

    @pytest.fixture
    def upscaler(self, tmp_path):
        from omegaconf import OmegaConf
        from src.upscaling.upscaler import Upscaler

        ckpt_path, model_cfg = _make_sr_checkpoint(tmp_path)
        cfg_dict = {
            "device": "cpu",
            "scale": 2,
            "model": {
                "checkpoint": str(ckpt_path),
                "base_channels": model_cfg["base_channels"],
                "encoder_channels": list(model_cfg["encoder_channels"]),
                "num_res_blocks": model_cfg["num_res_blocks"],
                "timesteps": 100,
            },
            "inference": {
                "ddim_steps": 2,
                "t_start": 50,
                "tile_size": 0,
                "tile_overlap": 8,
                "color_fix": False,
            },
        }
        return Upscaler(OmegaConf.create(cfg_dict))

    def test_output_resolution_doubled(self, upscaler):
        """Upscaled frame should be 2× the input resolution."""
        H, W = 32, 32
        frame = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        out = upscaler.upscale_frame(frame)
        assert out.shape == (H * 2, W * 2, 3)

    def test_output_dtype_and_range(self, upscaler):
        frame = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        out = upscaler.upscale_frame(frame)
        assert out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255

    def test_upscale_frames_wrapper(self, tmp_path):
        """phase_upscale.upscale_frames should return list of same length."""
        from src.upscaling.phase_upscale import upscale_frames
        from omegaconf import OmegaConf

        ckpt_path, model_cfg = _make_sr_checkpoint(tmp_path)
        cfg = {
            "device": "cpu",
            "scale": 2,
            "model": {
                "checkpoint": str(ckpt_path),
                "base_channels": model_cfg["base_channels"],
                "encoder_channels": list(model_cfg["encoder_channels"]),
                "num_res_blocks": model_cfg["num_res_blocks"],
                "timesteps": 100,
            },
            "inference": {
                "ddim_steps": 2,
                "t_start": 50,
                "tile_size": 0,
                "tile_overlap": 8,
                "color_fix": False,
            },
        }
        frames = [np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(4)]
        out = upscale_frames(frames, cfg)
        assert len(out) == 4

    def test_upscaled_larger_than_input(self, upscaler):
        frame = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        out = upscaler.upscale_frame(frame)
        assert out.shape[0] > frame.shape[0]
        assert out.shape[1] > frame.shape[1]
