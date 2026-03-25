#!/usr/bin/env bash
# =============================================================================
# setup.sh — Install neural_video_codec on a Linux GPU server (conda-based)
#
# Usage:
#   bash setup.sh                    # auto-detect CUDA version
#   bash setup.sh --cuda 12.1        # specify CUDA version
#   bash setup.sh --env myenv        # custom conda env name (default: nvc)
#   bash setup.sh --no-conda         # skip conda env, install into active env
# =============================================================================
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
ENV_NAME="nvc"
CUDA_VERSION=""
USE_CONDA=true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)    CUDA_VERSION="$2"; shift 2 ;;
        --env)     ENV_NAME="$2"; shift 2 ;;
        --no-conda) USE_CONDA=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Detect CUDA version if not provided ─────────────────────────────────────
if [[ -z "$CUDA_VERSION" ]]; then
    if command -v nvcc &>/dev/null; then
        CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    elif command -v nvidia-smi &>/dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
    else
        echo "WARNING: Could not detect CUDA version. Defaulting to 12.1."
        echo "         Re-run with --cuda <version> to override."
        CUDA_VERSION="12.1"
    fi
fi

CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

# Map CUDA version to PyTorch extra-index-url
if [[ "$CUDA_MAJOR" -ge 12 && "$CUDA_MINOR" -ge 6 ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu126"
elif [[ "$CUDA_MAJOR" -ge 12 && "$CUDA_MINOR" -ge 4 ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
fi

echo "============================================================"
echo " Neural Video Codec — setup"
echo "  CUDA:         $CUDA_VERSION  (index: $TORCH_INDEX)"
echo "  Conda env:    $ENV_NAME  (use_conda=$USE_CONDA)"
echo "  Project root: $SCRIPT_DIR"
echo "============================================================"

# ── 1. Create / activate conda environment ───────────────────────────────────
if $USE_CONDA; then
    if ! command -v conda &>/dev/null; then
        echo "ERROR: conda not found. Install Miniconda or use --no-conda."
        exit 1
    fi

    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "[1/5] Conda env '${ENV_NAME}' already exists — skipping create."
    else
        echo "[1/5] Creating conda env '${ENV_NAME}' with Python 3.11 ..."
        conda create -y -n "${ENV_NAME}" python=3.11
    fi

    # Re-run inside the env
    CONDA_BASE=$(conda info --base)
    PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
    PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"
else
    echo "[1/5] Skipping conda env creation (--no-conda)."
    PYTHON=$(command -v python3 || command -v python)
    PIP="${PYTHON} -m pip"
fi

# ── 2. Install Python dependencies ───────────────────────────────────────────
echo "[2/5] Installing Python dependencies ..."
$PIP install --upgrade pip

# Install PyTorch first with the right CUDA wheel
$PIP install --extra-index-url "$TORCH_INDEX" \
    "torch>=2.2.0" "torchvision>=0.17.0"

# Install the rest
$PIP install -r "${SCRIPT_DIR}/requirements.txt"

# ── 3. Build DCVC C++ entropy-coder extension ────────────────────────────────
echo "[3/5] Building DCVC MLCodec_extensions_cpp ..."
DCVC_CPP="${SCRIPT_DIR}/DCVC/src/cpp"

if [[ ! -d "$DCVC_CPP" ]]; then
    echo "ERROR: DCVC repo not found at ${SCRIPT_DIR}/DCVC"
    echo "       Clone it with: git clone https://github.com/microsoft/DCVC DCVC"
    exit 1
fi

$PIP install pybind11 setuptools
cd "$DCVC_CPP"
# Build in-place and install; --no-build-isolation ensures pybind11 is visible
$PYTHON setup.py build_ext --inplace
$PIP install --no-build-isolation .
cd "$SCRIPT_DIR"
echo "   MLCodec_extensions_cpp built and installed."

# ── 4. Verify imports ─────────────────────────────────────────────────────────
echo "[4/5] Verifying imports ..."
$PYTHON - <<'EOF'
import sys
errors = []

try:
    import torch
    print(f"  torch {torch.__version__}  CUDA={torch.cuda.is_available()}")
except ImportError as e:
    errors.append(f"torch: {e}")

try:
    import cv2
    print(f"  opencv {cv2.__version__}")
except ImportError as e:
    errors.append(f"opencv: {e}")

try:
    from MLCodec_extensions_cpp import RansEncoder, RansDecoder
    print("  MLCodec_extensions_cpp: OK")
except ImportError as e:
    errors.append(f"MLCodec_extensions_cpp: {e}")

try:
    import ultralytics
    print(f"  ultralytics {ultralytics.__version__}")
except ImportError as e:
    errors.append(f"ultralytics: {e}")

if errors:
    print("\nFAILED imports:")
    for err in errors:
        print(f"  {err}")
    sys.exit(1)
else:
    print("\nAll imports OK.")
EOF

# ── 5. Print activation instructions ─────────────────────────────────────────
echo ""
echo "[5/5] Setup complete."
echo ""
if $USE_CONDA; then
    echo "Activate the environment with:"
    echo "  conda activate ${ENV_NAME}"
    echo ""
fi
echo "Quick test:"
echo "  python scripts/sanity_check.py"
echo ""
echo "Compress a video:"
echo "  python run_compress.py --video dataset/bird1.mp4"
