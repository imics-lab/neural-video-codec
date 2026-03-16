from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _download_models_common as dl  # noqa: E402


class TestModelDownloadCommon(unittest.TestCase):
    def test_parse_github_repo_slug_supports_ssh_remote(self) -> None:
        self.assertEqual(
            dl.parse_github_repo_slug("git@github.com:mathew-felix/edge-roi-video-compression.git"),
            "mathew-felix/edge-roi-video-compression",
        )

    def test_parse_github_repo_slug_supports_https_remote(self) -> None:
        self.assertEqual(
            dl.parse_github_repo_slug("https://github.com/mathew-felix/edge-roi-video-compression.git"),
            "mathew-felix/edge-roi-video-compression",
        )

    def test_build_release_api_url_supports_latest_and_tagged(self) -> None:
        self.assertEqual(
            dl.build_release_api_url("owner/repo", "latest"),
            "https://api.github.com/repos/owner/repo/releases/latest",
        )
        self.assertEqual(
            dl.build_release_api_url("owner/repo", "models-v1"),
            "https://api.github.com/repos/owner/repo/releases/tags/models-v1",
        )

    def test_select_group_specs_uses_manifest_entries_for_both_groups(self) -> None:
        manifest = dl.load_manifest(ROOT / "models" / "models.manifest.json")
        compression = dl.select_group_specs("compression", manifest)
        decompression = dl.select_group_specs("decompression", manifest)

        self.assertEqual(
            [spec.file for spec in compression],
            [
                "MDV6-yolov9-c.pt",
                "MDV6-yolov9-c.onnx",
                "cvpr2025_image.pth.tar",
                "cvpr2025_video.pth.tar",
            ],
        )
        self.assertEqual(
            [spec.file for spec in decompression],
            [
                "cvpr2025_image.pth.tar",
                "cvpr2025_video.pth.tar",
                "amt-s.pth",
                "amt-l.pth",
            ],
        )

    def test_manifest_defaults_return_none_when_not_set(self) -> None:
        payload = dl.load_manifest_document(ROOT / "models" / "models.manifest.json")
        self.assertIsNone(dl.manifest_default_repo_slug(payload))
        self.assertIsNone(dl.manifest_default_release_tag(payload))

    def test_infer_repo_slug_uses_origin_remote(self) -> None:
        with patch("subprocess.run") as run:
            run.return_value.stdout = "git@github.com:mathew-felix/edge-roi-video-compression.git\n"
            run.return_value.returncode = 0

            repo_slug = dl.infer_repo_slug(ROOT)

        self.assertEqual(repo_slug, "mathew-felix/edge-roi-video-compression")


if __name__ == "__main__":
    unittest.main()
