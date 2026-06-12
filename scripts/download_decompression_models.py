from __future__ import annotations

from _download_models_common import run_cli


if __name__ == "__main__":
    raise SystemExit(
        run_cli(
            "decompression",
            "Download all decompression-stage models from GitHub Releases into the local models folder.",
        )
    )
