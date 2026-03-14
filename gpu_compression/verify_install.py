"""
Verify that all dependencies and DCVC extensions are installed correctly.
Run inside the Docker container to validate the build.
"""

import sys
import importlib

CHECKS = [
    # (module_name, display_name, critical)
    ("torch", "PyTorch", True),
    ("numpy", "NumPy", True),
    ("cv2", "OpenCV", True),
    ("yaml", "PyYAML", True),
    ("ultralytics", "Ultralytics (YOLO)", True),
    ("pybind11", "pybind11", True),
    ("MLCodec_extensions_cpp", "DCVC rANS codec (C++)", True),
    ("inference_extensions_cuda", "DCVC CUDA inference kernels", True),
]

def main():
    print("=" * 60)
    print("  neural-video-codec: Install Verification")
    print("=" * 60)
    print()

    all_ok = True
    results = []

    for mod_name, display, critical in CHECKS:
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "OK")
            results.append((display, "PASS", version))
        except Exception as exc:
            results.append((display, "FAIL", str(exc)[:80]))
            if critical:
                all_ok = False

    # Print results table
    max_name = max(len(r[0]) for r in results)
    for name, status, detail in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {name:<{max_name}}  {status}  ({detail})")

    print()

    # PyTorch CUDA check
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu = torch.cuda.get_device_name(0)
            print(f"  ✓ CUDA available: {gpu}")
        else:
            print(f"  ⚠ CUDA not available (expected if testing without GPU)")
    except Exception as exc:
        print(f"  ⚠ CUDA check skipped: {exc}")

    print()

    # Pipeline module imports
    print("  Pipeline module imports:")
    sys.path.insert(0, "/app/src")
    pipeline_modules = [
        ("roi_detection.roi_detector", "ROI Detection"),
        ("frame_removal.remove_frames", "Frame Removal"),
        ("frame_removal.dual_timeline", "Dual Timeline"),
        ("compression.phase4_dcvc", "DCVC Compression"),
        ("compression.dcvc_encoder", "DCVC Encoder"),
        ("pipeline.config_schema", "Config Validation"),
    ]
    for mod_name, display in pipeline_modules:
        try:
            importlib.import_module(mod_name)
            print(f"    ✓ {display}")
        except Exception as exc:
            print(f"    ✗ {display}: {exc}")
            all_ok = False

    print()
    print("=" * 60)
    if all_ok:
        print("  ✓ ALL CHECKS PASSED — Docker build is valid")
    else:
        print("  ✗ SOME CHECKS FAILED — review errors above")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
