from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _download_models_common import (  # noqa: E402
    MODELS_DIR,
    download_model_group,
    infer_repo_slug,
    load_manifest_document,
    manifest_default_release_tag,
    manifest_default_repo_slug,
)


def _build_parser(default_repo: str | None, default_tag: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download all compression/decompression models from GitHub Releases.")
    parser.add_argument(
        "--repo",
        default=default_repo,
        help=f"GitHub repo slug like owner/repo. Default: {default_repo or 'git remote origin'}",
    )
    parser.add_argument(
        "--tag",
        default=default_tag,
        help=f"GitHub release tag to use. Default: {default_tag or 'latest'}",
    )
    parser.add_argument(
        "--models-dir",
        default=str(MODELS_DIR),
        help=f"Destination folder for downloaded models. Default: {MODELS_DIR}",
    )
    parser.add_argument(
        "--github-token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub token for private repos or higher API rate limits. Defaults to GITHUB_TOKEN.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even if matching files already exist.",
    )
    return parser


def main() -> int:
    manifest_payload = load_manifest_document()
    default_repo = manifest_default_repo_slug(manifest_payload)
    default_tag = manifest_default_release_tag(manifest_payload) or "latest"
    parser = _build_parser(default_repo, default_tag)
    args = parser.parse_args()

    repo_slug = str(args.repo).strip() if args.repo else infer_repo_slug(ROOT)
    tag = str(args.tag).strip() if args.tag else "latest"
    models_dir = Path(args.models_dir).expanduser().resolve()
    token = str(args.github_token).strip() if args.github_token else None

    downloaded = 0
    for group, description in (
        ("compression", "compression-stage models"),
        ("decompression", "decompression-stage models"),
    ):
        print(f"Downloading {description} from {repo_slug}@{tag} into {models_dir}")
        downloaded += len(
            download_model_group(
                group=group,
                repo_slug=repo_slug,
                tag=tag,
                models_dir=models_dir,
                token=token,
                overwrite=bool(args.overwrite),
            )
        )

    print(f"Downloaded/validated {downloaded} model file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
