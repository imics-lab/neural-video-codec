from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import test_decompression  # noqa: E402


class TestDecompressionScriptContract(unittest.TestCase):
    def test_run_decompression_once_omits_amt_overrides_when_not_requested(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            archive = tmp / "archive.zip"
            cfg = tmp / "cfg.yaml"
            out_path = tmp / "reconstructed.mp4"
            lossless = tmp / "reconstructed_lossless.avi"

            archive.write_bytes(b"zip")
            cfg.write_text("decompression: {}", encoding="utf-8")
            out_path.write_bytes(b"mp4")
            lossless.write_bytes(b"avi")
            expected_lossless_sha = test_decompression._sha256_file(lossless)

            captured = {}

            def _fake_run(cmd, **kwargs):
                captured["cmd"] = list(cmd)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(test_decompression.subprocess, "run", side_effect=_fake_run):
                with mock.patch.object(
                    test_decompression,
                    "_probe_video",
                    return_value={
                        "width": 32,
                        "height": 24,
                        "fps": 30.0,
                        "frames": 4,
                        "pixels_sha256": "pixels",
                        "file_sha256": "file",
                        "size_bytes": 123,
                    },
                ):
                    result = test_decompression._run_decompression_once(
                        archive_path=archive,
                        config_path=cfg,
                        out_path=out_path,
                        lossless_out_path=lossless,
                        lossless_yuv420_out_path=None,
                        max_frames=0,
                        no_interpolate=False,
                        amt_batch_size=None,
                        amt_crop_margin=None,
                    )

        self.assertIn("--lossless-output", captured["cmd"])
        self.assertNotIn("--amt-batch-size", captured["cmd"])
        self.assertNotIn("--amt-crop-margin", captured["cmd"])
        self.assertEqual(result["lossless_file_sha256"], expected_lossless_sha)


if __name__ == "__main__":
    unittest.main()
