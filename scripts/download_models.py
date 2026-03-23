#!/usr/bin/env python3
"""
download_models.py — Download / verify model weights.

Reads models/models.manifest.json and downloads any missing or corrupted
checkpoints to their target paths.

Usage:
    python scripts/download_models.py [--force] [--manifest models/models.manifest.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DEFAULT = ROOT / "models" / "models.manifest.json"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}")
    print(f"  → {dest}")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            print(f"\r  {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    print()  # newline after progress


def _verify(path: Path, expected_sha256: str | None) -> bool:
    if not path.exists():
        return False
    if expected_sha256 is None:
        return True
    actual = _sha256(path)
    if actual != expected_sha256:
        print(f"  WARN: checksum mismatch for {path.name}")
        print(f"        expected: {expected_sha256}")
        print(f"        actual:   {actual}")
        return False
    return True


def download_all(manifest_path: Path, force: bool) -> int:
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    models = manifest.get("models", [])
    print(f"Manifest contains {len(models)} model(s).\n")

    errors = 0
    for entry in models:
        name     = entry.get("name", "?")
        dest     = ROOT / entry["path"]
        url      = entry.get("url", None)
        sha256   = entry.get("sha256", None)
        required = entry.get("required", True)

        print(f"[{name}]")
        print(f"  Target: {dest}")

        already_ok = _verify(dest, sha256)

        if already_ok and not force:
            print("  Status: OK (already present)")
            continue

        if already_ok and force:
            print("  Forcing re-download ...")

        if url is None:
            msg = "  SKIP: no download URL in manifest"
            if required:
                print(f"  ERROR: {msg} (required model!)")
                errors += 1
            else:
                print(msg)
            continue

        try:
            _download(url, dest)
        except Exception as e:
            print(f"  ERROR downloading: {e}")
            errors += 1
            continue

        if not _verify(dest, sha256):
            print("  ERROR: checksum failed after download")
            errors += 1
        else:
            print("  Downloaded and verified OK")

    print(f"\nDone. {errors} error(s).")
    return 0 if errors == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Download model checkpoints.")
    p.add_argument("--manifest", default=str(MANIFEST_DEFAULT))
    p.add_argument("--force",    action="store_true",
                   help="Re-download even if file already exists")
    args = p.parse_args()
    return download_all(Path(args.manifest), args.force)


if __name__ == "__main__":
    sys.exit(main())
