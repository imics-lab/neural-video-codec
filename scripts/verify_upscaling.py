"""
verify_upscaling.py — Standalone sanity-check script for the upscaling integration.

Checks:
  1. Python version >= 3.10
  2. Required imports: torch, cv2, numpy, loguru, omegaconf
  3. torch.cuda.is_available() — prints GPU name and VRAM
  4. Model checkpoint exists and SHA256 matches models/models.manifest.json
  5. Dummy forward pass: DiffusionUNet with base_channels=16, 64x64 input, 2 DDIM steps

Exits with code 0 if all checks pass, 1 if any fail.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    icon = "+" if ok else "x"
    line = f"  [{icon}] {label:<45}  {status}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return ok


def run_checks() -> bool:
    print("=" * 70)
    print("  neural-video-codec: Upscaling Integration Verification")
    print("=" * 70)
    print()

    all_ok = True

    # ------------------------------------------------------------------
    # 1. Python version
    # ------------------------------------------------------------------
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 10)
    all_ok &= _check(
        "Python >= 3.10",
        ok,
        f"{major}.{minor}.{sys.version_info[2]}",
    )

    # ------------------------------------------------------------------
    # 2. Required imports
    # ------------------------------------------------------------------
    print()
    print("  Package imports:")
    required_mods = [
        ("torch",     "PyTorch",   True),
        ("cv2",       "OpenCV",    True),
        ("numpy",     "NumPy",     True),
        ("loguru",    "loguru",    True),
        ("omegaconf", "omegaconf", True),
        ("yaml",      "PyYAML",    True),
        ("tqdm",      "tqdm",      True),
    ]
    for mod_name, display, critical in required_mods:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "ok")
            ok = True
        except Exception as exc:
            ver = str(exc)[:60]
            ok = False
        all_ok &= (ok or not critical)
        _check(f"  import {display}", ok, ver)

    # ------------------------------------------------------------------
    # 3. CUDA check
    # ------------------------------------------------------------------
    print()
    print("  CUDA:")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            detail = f"{gpu_name}, {vram_gb:.1f} GB VRAM"
        else:
            detail = "CUDA not available"
        all_ok &= _check("  torch.cuda.is_available()", cuda_ok, detail)
    except Exception as exc:
        all_ok &= _check("  torch.cuda.is_available()", False, str(exc)[:60])

    # ------------------------------------------------------------------
    # 4. Model checkpoint + SHA256
    # ------------------------------------------------------------------
    print()
    print("  Model checkpoint:")
    manifest_path = ROOT / "models" / "models.manifest.json"
    if not manifest_path.exists():
        _check("  models/models.manifest.json exists", False, "file not found")
        all_ok = False
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in manifest.get("models", []):
                name = entry.get("file") or entry.get("name")
                if not name:
                    all_ok &= _check("  manifest entry", False, "missing file/name")
                    continue
                model_path = ROOT / "models" / name
                expected_sha = entry.get("sha256", "")

                if not model_path.exists():
                    ok = _check(f"  {name} exists", False, "file not found")
                    all_ok &= ok
                    continue

                _check(f"  {name} exists", True, f"{model_path.stat().st_size // (1024*1024)} MB")

                if expected_sha and not expected_sha.startswith("PLACEHOLDER"):
                    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
                    sha_ok = sha == expected_sha
                    all_ok &= _check(
                        f"  {name} SHA256",
                        sha_ok,
                        sha[:16] + "..." if sha_ok else f"expected {expected_sha[:16]}...",
                    )
                else:
                    _check(f"  {name} SHA256", True, "placeholder — skipped")
        except Exception as exc:
            all_ok &= _check("  manifest parse", False, str(exc)[:60])

    # ------------------------------------------------------------------
    # 5. Dummy forward pass
    # ------------------------------------------------------------------
    print()
    print("  Dummy forward pass (DiffusionUNet base_channels=16, 64x64, 2 DDIM steps):")
    try:
        import torch
        from upscaling._network import DiffusionUNet
        from upscaling._diffusion import GaussianDiffusion

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = DiffusionUNet(
            base_channels=16,
            encoder_channels=(32, 64, 128),
            num_res_blocks=1,
        ).to(device).eval()

        diffusion = GaussianDiffusion(T=100)

        lr_up  = torch.rand(1, 3, 64, 64, device=device)
        roi_mask = torch.ones(1, 1, 64, 64, device=device)

        with torch.no_grad():
            out = diffusion.ddim_sample(
                model,
                lr_up,
                roi_mask,
                steps=2,
                device=device,
                t_start=50,
            )

        shape_ok = out.shape == (1, 3, 64, 64)
        range_ok = float(out.min()) >= 0.0 and float(out.max()) <= 1.0
        ok = shape_ok and range_ok
        detail = f"output shape {tuple(out.shape)}, range [{float(out.min()):.3f}, {float(out.max()):.3f}]"
        all_ok &= _check("  forward pass", ok, detail)

    except Exception as exc:
        all_ok &= _check("  forward pass", False, str(exc)[:80])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    if all_ok:
        print("  [+] ALL CHECKS PASSED — installation is valid")
    else:
        print("  [x] SOME CHECKS FAILED — review errors above")
    print("=" * 70)

    return all_ok


if __name__ == "__main__":
    ok = run_checks()
    sys.exit(0 if ok else 1)
