from __future__ import annotations

from _download_models_common import run_cli


if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            "all",
            "Download all release-backed model files from GitHub Releases into the local models folder.",
        )
    )
