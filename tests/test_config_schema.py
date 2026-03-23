"""Tests for YAML config schema validation."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        import json
        with open(path) as f:
            return json.load(f)


class TestCompressionConfig:
    """Validate configs/gpu/compression.yaml."""

    @pytest.fixture
    def cfg(self):
        p = ROOT / "configs" / "gpu" / "compression.yaml"
        if not p.exists():
            pytest.skip("compression.yaml not found")
        return _load_yaml(p)

    def test_compression_section_present(self, cfg):
        assert "compression" in cfg

    def test_quality_qps(self, cfg):
        q = cfg["compression"]["quality"]
        for key in ("roi_qp_i", "roi_qp_p", "bg_qp_i", "bg_qp_p"):
            assert key in q, f"Missing QP key: {key}"
        assert 0 < q["roi_qp_i"] <= 63
        assert 0 < q["bg_qp_i"]  <= 63
        assert q["roi_qp_i"] >= q["bg_qp_i"], "ROI QP should be ≥ BG QP"

    def test_dcvc_keys(self, cfg):
        dcvc = cfg["compression"]["dcvc"]
        for key in ("model_i", "model_p", "repo_dir"):
            assert key in dcvc, f"Missing key: {key}"

    def test_detection_section(self, cfg):
        assert "detection" in cfg
        d = cfg["detection"]
        assert "model_path" in d
        assert "animal_classes" in d
        assert isinstance(d["animal_classes"], list)
        assert len(d["animal_classes"]) > 0


class TestRestorationConfig:
    """Validate configs/gpu/restoration.yaml."""

    @pytest.fixture
    def cfg(self):
        p = ROOT / "configs" / "gpu" / "restoration.yaml"
        if not p.exists():
            pytest.skip("restoration.yaml not found")
        return _load_yaml(p)

    def test_temporal_window(self, cfg):
        # temporal_window lives under model: in restoration.yaml
        T = cfg.get("model", {}).get("temporal_window") or cfg.get("temporal_window")
        assert T is not None, "temporal_window not found in cfg or cfg.model"
        assert T % 2 == 1, "Temporal window should be odd (symmetric)"
        assert T >= 3

    def test_inference_section(self, cfg):
        assert "inference" in cfg
        inf = cfg["inference"]
        assert "ddim_steps" in inf
        assert inf["ddim_steps"] > 0

    def test_t_start_in_range(self, cfg):
        t_start = cfg.get("inference", {}).get("t_start", None)
        if t_start is not None:
            assert 0 < t_start < 1000


class TestUpscalingConfig:
    """Validate configs/gpu/upscaling.yaml."""

    @pytest.fixture
    def cfg(self):
        p = ROOT / "configs" / "gpu" / "upscaling.yaml"
        if not p.exists():
            pytest.skip("upscaling.yaml not found")
        return _load_yaml(p)

    def test_scale_is_2(self, cfg):
        assert cfg.get("scale") == 2

    def test_inference_section(self, cfg):
        assert "inference" in cfg
        assert "ddim_steps" in cfg["inference"]

    def test_tile_size_reasonable(self, cfg):
        tile_size = cfg.get("inference", {}).get("tile_size", None)
        if tile_size is not None:
            assert tile_size >= 64
            assert tile_size % 8 == 0, "Tile size should be multiple of 8"


class TestPipelineConfig:
    """Validate configs/gpu/pipeline.yaml."""

    @pytest.fixture
    def cfg(self):
        p = ROOT / "configs" / "gpu" / "pipeline.yaml"
        if not p.exists():
            pytest.skip("pipeline.yaml not found")
        return _load_yaml(p)

    def test_all_stages_present(self, cfg):
        for stage in ("compression", "decompression", "restoration", "upscaling"):
            assert stage in cfg, f"Missing pipeline stage: {stage}"

    def test_output_section(self, cfg):
        assert "output" in cfg

    def test_restoration_enable_flag(self, cfg):
        assert "enable" in cfg.get("restoration", {})

    def test_upscaling_enable_flag(self, cfg):
        assert "enable" in cfg.get("upscaling", {})
