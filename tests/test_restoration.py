"""Unit tests for Model R restoration network and phase_restore."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")


class TestRestoreUNet:
    """Shape and forward-pass tests for RestoreUNet."""

    @pytest.fixture
    def model(self):
        from src.restoration._network import RestoreUNet
        return RestoreUNet(
            base_channels=16,
            encoder_channels=(32, 64, 128),
            num_res_blocks=1,
            T=3,
            n_heads=2,
        ).eval()

    def test_output_shape(self, model):
        """Output should be (B*T, 3, H, W)."""
        import torch
        B, T, H, W = 2, 3, 64, 64
        x = torch.randn(B * T, 6, H, W)
        t = torch.randint(0, 1000, (B * T,))
        with torch.no_grad():
            out = model(x, t)
        assert out.shape == (B * T, 3, H, W)

    def test_output_no_nan(self, model):
        """No NaN values in output."""
        import torch
        x = torch.randn(3, 6, 64, 64)
        t = torch.zeros(3, dtype=torch.long)
        with torch.no_grad():
            out = model(x, t)
        assert not torch.isnan(out).any()

    def test_different_timesteps(self, model):
        """Model should handle different t values per sample."""
        import torch
        B_T = 6
        x   = torch.randn(B_T, 6, 64, 64)
        t   = torch.tensor([0, 100, 250, 500, 750, 999])
        with torch.no_grad():
            out = model(x, t)
        assert out.shape[0] == B_T

    def test_in_channels_constant(self):
        from src.restoration._network import RestoreUNet
        assert RestoreUNet.IN_CHANNELS == 6


class TestTemporalAttention:
    """Tests for the TemporalAttention block."""

    def test_shape_preserved(self):
        import torch
        from src.restoration._blocks import TemporalAttention
        T, B, C, H, W = 3, 2, 32, 16, 16
        block = TemporalAttention(channels=C, T=T, n_heads=2)
        x = torch.randn(B * T, C, H, W)
        with torch.no_grad():
            out = block(x)
        assert out.shape == x.shape

    def test_residual_path(self):
        """With zero-init projection weight, output ≈ input (residual identity)."""
        import torch
        from src.restoration._blocks import TemporalAttention
        T, B, C, H, W = 3, 1, 16, 8, 8
        block = TemporalAttention(channels=C, T=T, n_heads=2)
        # Zero out all parameters so the block contribution is zero
        for p in block.parameters():
            p.data.zero_()
        x = torch.randn(B * T, C, H, W)
        with torch.no_grad():
            out = block(x)
        # Residual: output = input + zero = input
        assert torch.allclose(out, x, atol=1e-5)


# ── Checkpoint fixture ────────────────────────────────────────────────────────

def _make_restore_checkpoint(tmp_path: Path) -> tuple:
    """Create a tiny RestoreUNet checkpoint and return (ckpt_path, model_cfg)."""
    import torch
    from src.restoration._network import RestoreUNet

    model_cfg = dict(
        base_channels=8,
        encoder_channels=(16, 32, 64),
        num_res_blocks=1,
        T=3,
        n_heads=1,
    )
    model = RestoreUNet(**model_cfg)
    ckpt_path = tmp_path / "restoration.pt"
    torch.save({"model": model.state_dict(), "model_cfg": model_cfg}, str(ckpt_path))
    return ckpt_path, model_cfg


class TestRestorer:
    """Tests for the high-level Restorer class."""

    @pytest.fixture
    def restorer_and_cfg(self, tmp_path):
        from omegaconf import OmegaConf
        from src.restoration.restorer import Restorer

        ckpt_path, model_cfg = _make_restore_checkpoint(tmp_path)
        cfg_dict = {
            "device": "cpu",
            "model": {
                "checkpoint": str(ckpt_path),
                "base_channels": model_cfg["base_channels"],
                "encoder_channels": list(model_cfg["encoder_channels"]),
                "num_res_blocks": model_cfg["num_res_blocks"],
                "timesteps": 100,
                "temporal_window": 3,
                "n_heads": model_cfg["n_heads"],
            },
            "inference": {
                "ddim_steps": 2,
                "t_start": 50,
                "tile_size": 0,
                "tile_overlap": 8,
                "color_fix": False,
                "batch_size": 1,
            },
        }
        cfg_node = OmegaConf.create(cfg_dict)
        restorer = Restorer(cfg_node)
        return restorer, cfg_dict

    def test_restore_sequence_length(self, restorer_and_cfg):
        """Output length should equal input length."""
        restorer, _ = restorer_and_cfg
        frames = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
        out = restorer.restore_sequence(frames)
        assert len(out) == len(frames)

    def test_output_dtype_and_range(self, restorer_and_cfg):
        """Each output frame should be uint8 in [0, 255]."""
        restorer, _ = restorer_and_cfg
        frames = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)]
        out = restorer.restore_sequence(frames)
        assert out[0].dtype == np.uint8
        assert out[0].min() >= 0
        assert out[0].max() <= 255

    def test_restore_frames_wrapper(self, tmp_path):
        """phase_restore.restore_frames should call Restorer and return same-length list."""
        from src.restoration.phase_restore import restore_frames

        ckpt_path, model_cfg = _make_restore_checkpoint(tmp_path)
        cfg = {
            "device": "cpu",
            "model": {
                "checkpoint": str(ckpt_path),
                "base_channels": model_cfg["base_channels"],
                "encoder_channels": list(model_cfg["encoder_channels"]),
                "num_res_blocks": model_cfg["num_res_blocks"],
                "timesteps": 100,
                "temporal_window": 3,
                "n_heads": model_cfg["n_heads"],
            },
            "inference": {
                "ddim_steps": 2,
                "t_start": 50,
                "tile_size": 0,
                "tile_overlap": 8,
                "color_fix": False,
                "batch_size": 1,
            },
        }
        frames = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        out = restore_frames(frames, cfg)
        assert len(out) == 3
