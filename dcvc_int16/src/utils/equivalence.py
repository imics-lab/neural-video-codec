"""Utilities for bitstream equivalence and reproducibility metadata."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_RELEVANT_ENV_KEYS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "DCVC_ENABLE_INT16_PFRAME_GRAPHS",
    "DCVC_INT16_ASYNC_ENTROPY_PREP",
    "DCVC_INT16_ENCODE_ONLY",
    "DCVC_PROFILE_INT16_PIPELINE",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCHINDUCTOR_CUDAGRAPHS",
)


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA-256 digest for a file without loading it all at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            chunk = fp.read(chunk_size)
            if len(chunk) == 0:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload):
    """Return the SHA-256 digest for an in-memory bytes payload."""
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def compare_files_bytewise(left_path, right_path, chunk_size=1024 * 1024):
    """Compare two files and report the first differing byte offset."""
    left_path = Path(left_path)
    right_path = Path(right_path)
    offset = 0
    with open(left_path, "rb") as left_fp, open(right_path, "rb") as right_fp:
        while True:
            left_chunk = left_fp.read(chunk_size)
            right_chunk = right_fp.read(chunk_size)
            if left_chunk != right_chunk:
                common = min(len(left_chunk), len(right_chunk))
                mismatch_offset = offset + common
                for idx in range(common):
                    if left_chunk[idx] != right_chunk[idx]:
                        mismatch_offset = offset + idx
                        break
                return {
                    "equal": False,
                    "first_mismatch_offset": mismatch_offset,
                    "left_size_bytes": int(left_path.stat().st_size),
                    "right_size_bytes": int(right_path.stat().st_size),
                }
            if len(left_chunk) == 0:
                break
            offset += len(left_chunk)
    return {
        "equal": True,
        "first_mismatch_offset": None,
        "left_size_bytes": int(left_path.stat().st_size),
        "right_size_bytes": int(right_path.stat().st_size),
    }


def find_git_root(start_path):
    """Find the nearest parent containing a `.git` directory."""
    start_path = Path(start_path).resolve()
    for candidate in (start_path,) + tuple(start_path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _run_capture(command, cwd=None):
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "command failed")
    return completed.stdout.strip()


def collect_git_metadata(start_path):
    """Collect source revision metadata for an equivalence sidecar."""
    git_root = find_git_root(start_path)
    payload = {
        "available": False,
        "root": str(git_root) if git_root is not None else None,
        "commit": None,
        "short_commit": None,
        "tracked_dirty": None,
    }
    if git_root is None:
        return payload
    try:
        payload["commit"] = _run_capture(["git", "rev-parse", "HEAD"], cwd=git_root)
        payload["short_commit"] = _run_capture(["git", "rev-parse", "--short", "HEAD"], cwd=git_root)
        tracked_status = _run_capture(
            ["git", "status", "--short", "--untracked-files=no"],
            cwd=git_root,
        )
        payload["tracked_dirty"] = len(tracked_status.strip()) > 0
        payload["available"] = True
    except Exception as exc:  # pylint: disable=broad-except
        payload["error"] = str(exc)
    return payload


def _query_nvidia_driver_version(device_index=None):
    try:
        output = _run_capture(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )
    except Exception:  # pylint: disable=broad-except
        return None
    versions = [line.strip() for line in output.splitlines() if len(line.strip()) > 0]
    if len(versions) == 0:
        return None
    if device_index is not None and 0 <= device_index < len(versions):
        return versions[device_index]
    return versions[0]


def collect_torch_device_metadata(device):
    """Collect PyTorch/CUDA device metadata when PyTorch is installed."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for device metadata collection.") from exc

    device = torch.device(device)
    payload = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "device_type": device.type,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": None,
        "device_capability": None,
        "driver_version": None,
        "total_memory_bytes": None,
    }
    if device.type != "cuda" or not torch.cuda.is_available():
        return payload
    props = torch.cuda.get_device_properties(device)
    payload["device_name"] = props.name
    payload["device_capability"] = [int(props.major), int(props.minor)]
    payload["total_memory_bytes"] = int(props.total_memory)
    payload["driver_version"] = _query_nvidia_driver_version(device.index)
    return payload


def collect_model_bundle_metadata(bundle_path):
    """Collect checksum metadata for a model bundle without reading weights into chat."""
    bundle_path = Path(bundle_path)
    payload = {
        "path": str(bundle_path),
        "exists": bundle_path.exists(),
        "sha256": None,
        "size_bytes": None,
    }
    if bundle_path.exists():
        payload["sha256"] = sha256_file(bundle_path)
        payload["size_bytes"] = int(bundle_path.stat().st_size)
    return payload


def collect_relevant_env(extra_keys=None):
    """Collect environment variables that affect deterministic equivalence."""
    keys = set(DEFAULT_RELEVANT_ENV_KEYS)
    if extra_keys is not None:
        keys.update(extra_keys)
    for key in os.environ:
        if key.startswith("DCVC_"):
            keys.add(key)
    return {key: os.environ.get(key) for key in sorted(keys)}


def collect_entrypoint_metadata(script_path, argv):
    """Capture the command used to produce an artifact."""
    command_parts = [sys.executable, str(Path(script_path).resolve()), *list(argv)]
    return {
        "script_path": str(Path(script_path).resolve()),
        "python_executable": sys.executable,
        "cwd": str(Path.cwd()),
        "argv": list(argv),
        "command_line": subprocess.list2cmdline(command_parts),
    }


def load_json_if_exists(path):
    """Load a JSON file if present; return None when absent."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def infer_metrics_json_path(bitstream_path):
    """Infer the sibling metrics path for a bitstream."""
    return Path(bitstream_path).with_suffix(".json")
