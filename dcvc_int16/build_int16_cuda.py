#!/usr/bin/env python3
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    if importlib.util.find_spec("torch") is None:
        print(
            "PyTorch is not installed, so the optional INT16 CUDA extension "
            "cannot be built in this environment."
        )
        print("Install a CUDA-enabled PyTorch build, then rerun this helper.")
        return

    from src.layers.int16_cuda_ext import is_available, load_int16_ext

    print("Checking int16 CUDA extension availability...")
    available = is_available()
    print(f"is_available: {available}")
    if available:
        ext = load_int16_ext()
        print(f"loaded extension: {ext}")
    else:
        print(
            "int16 CUDA extension did not build/load. "
            "The codec will fall back to the slower Python path."
        )


if __name__ == "__main__":
    main()
