#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command, cwd=None, check=True):
    print("+", " ".join(str(part) for part in command))
    return subprocess.run(command, cwd=cwd or ROOT, check=check)


def main():
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            str(ROOT / "src" / "cpp"),
        ]
    )

    try:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                str(ROOT / "src" / "layers" / "extensions" / "inference"),
            ]
        )
    except subprocess.CalledProcessError:
        print("warning: inference_extensions_cuda build failed; continuing with PyTorch fallback.")

    try:
        run([sys.executable, str(ROOT / "build_int16_cuda.py")])
    except subprocess.CalledProcessError:
        print("warning: int16 CUDA extension preload failed; continuing with Python fallback.")

    print("bootstrap complete")


if __name__ == "__main__":
    main()
