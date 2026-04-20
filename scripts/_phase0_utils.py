from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPRESSION_CONFIG = ROOT / "configs" / "gpu" / "compression.yaml"
DEFAULT_DECOMPRESSION_CONFIG = ROOT / "configs" / "gpu" / "decompression.yaml"
DEFAULT_SAMPLE_VIDEO = ROOT / "data" / "test.mp4"
ARCHIVE_MANIFEST_NAME = "archive_manifest.json"
ARCHIVE_REQUIRED_ENTRIES = (
    "meta.json",
    "roi_detections.json",
    "frame_drop.json",
    "roi.bin",
    "bg.bin",
)


def resolve_from_root(raw_path: str | Path) -> Path:
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        return p.resolve()

    candidates = []
    cwd_candidate = (Path.cwd() / p).resolve()
    candidates.append(cwd_candidate)

    root_candidate = (ROOT / p).resolve()
    candidates.append(root_candidate)

    if p.parts and str(p.parts[0]).lower() == ROOT.name.lower():
        candidates.append((ROOT.parent / p).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return root_candidate


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_archive_payloads(path: Path) -> Dict[str, bytes]:
    import zipfile

    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        entry_map = {name: name for name in ARCHIVE_REQUIRED_ENTRIES if name in names}
        missing = [name for name in ARCHIVE_REQUIRED_ENTRIES if name not in entry_map]
        if missing:
            if ARCHIVE_MANIFEST_NAME not in names:
                raise FileNotFoundError(f"Missing archive entries {missing} in {path}")
            manifest = json.loads(zf.read(ARCHIVE_MANIFEST_NAME).decode("utf-8"))
            entries = manifest.get("entries", {}) if isinstance(manifest, dict) else {}
            if not isinstance(entries, dict):
                raise RuntimeError(f"Invalid archive manifest in {path}")
            entry_map = {}
            manifest_missing = []
            for canonical_name in ARCHIVE_REQUIRED_ENTRIES:
                actual_name = entries.get(canonical_name, canonical_name)
                if not isinstance(actual_name, str) or not actual_name.strip():
                    manifest_missing.append(canonical_name)
                    continue
                if actual_name not in names:
                    manifest_missing.append(canonical_name)
                    continue
                entry_map[canonical_name] = actual_name
            if manifest_missing:
                raise FileNotFoundError(f"Missing archive entries {manifest_missing} in {path}")
        return {canonical_name: zf.read(entry_map[canonical_name]) for canonical_name in ARCHIVE_REQUIRED_ENTRIES}


def resolve_video_path(cfg: Dict[str, Any], cli_video: Optional[str]) -> Path:
    if cli_video:
        p = resolve_from_root(cli_video)
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {p}")
        return p

    input_cfg = cfg.get("input", {}) or {}
    raw = input_cfg.get("video_path") or input_cfg.get("video") or cfg.get("input_video") or cfg.get("video")
    if raw:
        p = resolve_from_root(str(raw))
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {p}")
        return p
    if DEFAULT_SAMPLE_VIDEO.exists():
        return DEFAULT_SAMPLE_VIDEO.resolve()
    raise KeyError(
        "Video path not found. Pass --video, set input.video_path in a custom config, "
        "or add data/test.mp4 as the default sample clip."
    )


def resolve_out_dir(cli_out_dir: Optional[str], default_rel: str) -> Path:
    if cli_out_dir:
        out = resolve_from_root(cli_out_dir)
    else:
        out = resolve_from_root(default_rel)
    out.mkdir(parents=True, exist_ok=True)
    return out


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_json(data: Any) -> str:
    return sha256_bytes(_canonical_json_bytes(data))


def human_bytes(size_bytes: int) -> str:
    size = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f} {units[idx]}"

