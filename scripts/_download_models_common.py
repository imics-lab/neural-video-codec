from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "models.manifest.json"
USER_AGENT = "edge-roi-video-compression-model-downloader/1.0"

MODEL_GROUPS = {
    "compression": [
        "MDV6-yolov9-c.pt",
        "MDV6-yolov9-c.onnx",
    ],
    "decompression": [
        "amt-s.pth",
        "amt-l.pth",
    ],
}


@dataclass(frozen=True)
class ModelSpec:
    file: str
    sha256: str | None
    size_bytes: int | None


def load_manifest_document(path: Path = MANIFEST_PATH) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid manifest format in {path}")
    return payload


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, ModelSpec]:
    payload = load_manifest_document(path)
    items = payload.get("models", [])
    if not isinstance(items, list):
        raise ValueError(f"Invalid manifest format in {path}")
    result: dict[str, ModelSpec] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid model entry in {path}")
        file_name = str(item.get("file", "")).strip()
        if not file_name:
            raise ValueError(f"Manifest entry missing file name in {path}")
        sha256 = item.get("sha256")
        size_bytes = item.get("size_bytes")
        result[file_name] = ModelSpec(
            file=file_name,
            sha256=str(sha256).strip() if sha256 else None,
            size_bytes=int(size_bytes) if size_bytes is not None else None,
        )
    return result


def manifest_default_repo_slug(payload: dict) -> str | None:
    raw = payload.get("github_repo", None)
    if raw is None:
        raw = payload.get("repo_slug", None)
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def manifest_default_release_tag(payload: dict) -> str | None:
    raw = payload.get("release_tag", None)
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def select_group_specs(group: str, manifest: dict[str, ModelSpec]) -> list[ModelSpec]:
    try:
        files = MODEL_GROUPS[group]
    except KeyError as exc:
        raise ValueError(f"Unknown model group: {group}") from exc
    missing = [file_name for file_name in files if file_name not in manifest]
    if missing:
        raise KeyError(f"Manifest is missing model entries: {missing}")
    return [manifest[file_name] for file_name in files]


def parse_github_repo_slug(remote_url: str) -> str:
    value = remote_url.strip()
    if value.startswith("git@github.com:"):
        slug = value.split(":", 1)[1]
    elif value.startswith("ssh://git@github.com/"):
        slug = value.split("ssh://git@github.com/", 1)[1]
    elif value.startswith("https://github.com/"):
        slug = value.split("https://github.com/", 1)[1]
    elif value.startswith("http://github.com/"):
        slug = value.split("http://github.com/", 1)[1]
    else:
        raise ValueError(f"Unsupported GitHub remote URL: {remote_url}")
    if slug.endswith(".git"):
        slug = slug[:-4]
    slug = slug.strip("/")
    if slug.count("/") != 1:
        raise ValueError(f"Could not parse GitHub repo slug from: {remote_url}")
    return slug


def infer_repo_slug(repo_root: Path = REPO_ROOT) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
    )
    remote_url = completed.stdout.strip()
    if not remote_url:
        raise RuntimeError("git remote 'origin' is not configured")
    return parse_github_repo_slug(remote_url)


def build_release_api_url(repo_slug: str, tag: str) -> str:
    safe_repo = repo_slug.strip("/")
    if tag == "latest":
        return f"https://api.github.com/repos/{safe_repo}/releases/latest"
    return f"https://api.github.com/repos/{safe_repo}/releases/tags/{quote(tag, safe='')}"


def _github_headers(token: str | None, accept: str) -> dict[str, str]:
    headers = {
        "Accept": accept,
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _read_json(url: str, token: str | None) -> dict:
    req = Request(url, headers=_github_headers(token, "application/vnd.github+json"))
    try:
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        hint = " If this repo is private, set GITHUB_TOKEN." if exc.code in (401, 403, 404) and not token else ""
        raise RuntimeError(f"GitHub API request failed ({exc.code}) for {url}: {detail}{hint}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach GitHub API at {url}: {exc}") from exc


def fetch_release(repo_slug: str, tag: str, token: str | None) -> dict:
    payload = _read_json(build_release_api_url(repo_slug, tag), token)
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected GitHub release payload")
    return payload


def release_assets_by_name(release_payload: dict) -> dict[str, dict]:
    assets = release_payload.get("assets", [])
    if not isinstance(assets, list):
        raise RuntimeError("GitHub release payload is missing asset metadata")
    result: dict[str, dict] = {}
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name", "")).strip()
        if not name:
            continue
        result[name] = asset
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_release_asset(asset: dict, dest_path: Path, token: str | None) -> None:
    asset_url = str(asset.get("url", "")).strip()
    if not asset_url:
        raise RuntimeError(f"Release asset metadata is missing an API URL for {asset.get('name')}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest_path.name + ".", suffix=".part", dir=str(dest_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    req = Request(asset_url, headers=_github_headers(token, "application/octet-stream"))
    try:
        with urlopen(req, timeout=300) as resp, tmp_path.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        tmp_path.replace(dest_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _verify_file(path: Path, spec: ModelSpec) -> None:
    if spec.size_bytes is not None and path.stat().st_size != spec.size_bytes:
        raise RuntimeError(
            f"{path.name} size mismatch: expected {spec.size_bytes} bytes, got {path.stat().st_size} bytes"
        )
    if spec.sha256:
        actual_sha = sha256_file(path)
        if actual_sha != spec.sha256:
            raise RuntimeError(f"{path.name} sha256 mismatch: expected {spec.sha256}, got {actual_sha}")


def _needs_download(path: Path, spec: ModelSpec, overwrite: bool) -> bool:
    if overwrite or not path.exists():
        return True
    try:
        _verify_file(path, spec)
        return False
    except RuntimeError:
        return True


def download_model_group(
    *,
    group: str,
    repo_slug: str,
    tag: str,
    models_dir: Path,
    token: str | None,
    overwrite: bool,
) -> list[Path]:
    manifest = load_manifest()
    specs = select_group_specs(group, manifest)
    release_payload = fetch_release(repo_slug, tag, token)
    assets = release_assets_by_name(release_payload)
    available_assets = sorted(assets)
    downloaded: list[Path] = []
    for spec in specs:
        dest = models_dir / spec.file
        if not _needs_download(dest, spec, overwrite):
            print(f"[skip] {dest.name} already exists and matches manifest")
            downloaded.append(dest)
            continue
        asset = assets.get(spec.file)
        if asset is None:
            raise FileNotFoundError(
                f"Release does not contain asset {spec.file}. Available assets: {available_assets}"
            )
        print(f"[download] {spec.file} -> {dest}")
        _download_release_asset(asset, dest, token)
        try:
            _verify_file(dest, spec)
        except RuntimeError:
            if dest.exists():
                dest.unlink()
            raise
        downloaded.append(dest)
        print(f"[ok] {spec.file}")
    return downloaded


def build_parser(group: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repo slug like owner/repo. Defaults to git remote origin.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="GitHub release tag to use. Defaults to manifest release_tag, otherwise latest.",
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
        help="Redownload files even if a matching file already exists.",
    )
    parser.epilog = (
        f"This downloads the {group} model set defined in models/models.manifest.json "
        "using GitHub Release asset names that match the manifest filenames."
    )
    return parser


def run_cli(group: str, description: str) -> int:
    parser = build_parser(group, description)
    args = parser.parse_args()
    manifest_payload = load_manifest_document()
    repo_slug = args.repo or manifest_default_repo_slug(manifest_payload) or infer_repo_slug(REPO_ROOT)
    raw_tag = "" if args.tag is None else str(args.tag).strip()
    tag = raw_tag or (manifest_default_release_tag(manifest_payload) or "latest")
    models_dir = Path(args.models_dir).expanduser().resolve()
    downloaded = download_model_group(
        group=group,
        repo_slug=repo_slug,
        tag=tag,
        models_dir=models_dir,
        token=args.github_token,
        overwrite=bool(args.overwrite),
    )
    print(f"Downloaded/validated {len(downloaded)} {group} model file(s) into {models_dir}")
    return 0


__all__ = [
    "MODEL_GROUPS",
    "ModelSpec",
    "build_release_api_url",
    "download_model_group",
    "infer_repo_slug",
    "load_manifest",
    "load_manifest_document",
    "manifest_default_release_tag",
    "manifest_default_repo_slug",
    "parse_github_repo_slug",
    "release_assets_by_name",
    "run_cli",
    "select_group_specs",
    "sha256_file",
]
