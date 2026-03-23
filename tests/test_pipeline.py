"""End-to-end pipeline smoke tests (all heavy calls mocked)."""
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


def _make_video(path: str, n: int = 5, h: int = 64, w: int = 64, fps: float = 30.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wtr    = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for _ in range(n):
        wtr.write(np.random.randint(0, 256, (h, w, 3), dtype=np.uint8))
    wtr.release()


def _fake_frames(n: int = 5, h: int = 64, w: int = 64) -> list:
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]


class TestPipelineIntegration:
    """Smoke tests for run_pipeline.main (mocked sub-stages)."""

    @pytest.fixture
    def tmp_video(self, tmp_path):
        p = tmp_path / "test.mp4"
        _make_video(str(p))
        return str(p)

    @pytest.fixture
    def pipeline_cfg(self, tmp_path):
        cfg_path = tmp_path / "pipeline.yaml"
        cfg_path.write_text(
            "input:\n  downscale_input: 1.0\n"
            "output:\n  out_dir: " + str(tmp_path / "out") + "\n"
            "compression: {}\ndecompression: {}\n"
            "restoration:\n  enable: false\n"
            "upscaling:\n  enable: false\n"
        )
        return str(cfg_path)

    @patch("src.compression.phase_compress.compress_video")
    @patch("src.decompression.phase_decompress.decompress_archive")
    @patch("src.postprocessing.video_assembler.assemble_video")
    def test_pipeline_runs_without_models(
        self, mock_assemble, mock_decomp, mock_comp, tmp_video, pipeline_cfg
    ):
        """Pipeline should complete when restore+upscale are disabled."""
        mock_comp.return_value   = b"\x00" * 50
        mock_decomp.return_value = {"frames": _fake_frames(5), "fps": 30.0}
        mock_assemble.return_value = None

        import run_pipeline
        import argparse
        args = argparse.Namespace(
            video=tmp_video,
            config=pipeline_cfg,
            output=None,
            downscale=None,
            skip_restore=True,
            skip_upscale=True,
            save_intermediate=False,
            verbose=False,
        )
        # Patch _parse_args to return our namespace
        with patch("run_pipeline._parse_args", return_value=args):
            ret = run_pipeline.main()
        assert ret == 0

    @patch("src.compression.phase_compress.compress_video")
    @patch("src.decompression.phase_decompress.decompress_archive")
    @patch("src.restoration.phase_restore.restore_frames")
    @patch("src.upscaling.phase_upscale.upscale_frames")
    @patch("src.postprocessing.video_assembler.assemble_video")
    def test_pipeline_with_all_stages(
        self, mock_assemble, mock_upscale, mock_restore, mock_decomp, mock_comp,
        tmp_video, pipeline_cfg
    ):
        """All four pipeline stages should be called."""
        frames = _fake_frames(5)
        mock_comp.return_value    = b"\x00" * 50
        mock_decomp.return_value  = {"frames": frames, "fps": 30.0}
        mock_restore.return_value = frames
        mock_upscale.return_value = [
            np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(5)
        ]
        mock_assemble.return_value = None

        # Patch the pipeline config to enable restore + upscale
        import run_pipeline
        import argparse

        cfg_with_models = {
            "input": {"downscale_input": 1.0},
            "output": {"out_dir": str(Path(tmp_video).parent / "out")},
            "compression": {},
            "decompression": {},
            "restoration": {"enable": True, "config": ""},
            "upscaling":   {"enable": True, "config": ""},
        }

        args = argparse.Namespace(
            video=tmp_video,
            config="irrelevant",
            output=None,
            downscale=None,
            skip_restore=False,
            skip_upscale=False,
            save_intermediate=False,
            verbose=False,
        )
        with patch("run_pipeline._parse_args", return_value=args), \
             patch("run_pipeline._load_config", return_value=cfg_with_models):
            ret = run_pipeline.main()

        assert ret == 0
        mock_restore.assert_called_once()
        mock_upscale.assert_called_once()

    def test_pipeline_missing_video_returns_1(self, tmp_path, pipeline_cfg):
        """Should return 1 when video does not exist."""
        import run_pipeline
        import argparse

        args = argparse.Namespace(
            video=str(tmp_path / "nonexistent.mp4"),
            config=pipeline_cfg,
            output=None,
            downscale=None,
            skip_restore=True,
            skip_upscale=True,
            save_intermediate=False,
            verbose=False,
        )
        with patch("run_pipeline._parse_args", return_value=args):
            ret = run_pipeline.main()
        assert ret == 1
