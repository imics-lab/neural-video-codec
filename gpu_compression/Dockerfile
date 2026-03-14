# =============================================================================
# Dockerfile for gpu_based_compression
# =============================================================================
# Runs as a SEPARATE container alongside BioCoder-Edge.
#
# DCVC CUDA extension requires CUDA toolkit 12.4+ and PyTorch cu126.
# Both Jetson and PC use nvidia/cuda:12.4.1-devel + pip-installed PyTorch.
#
# Build: Compiles DCVC C++ extensions (requires CUDA toolkit + PyTorch)
# Run:   python run_compression.py <video_path>
# =============================================================================

ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    libgl1 \
    libglib2.0-0 \
    python3 \
    python3-pip \
    python3-dev \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ---- Install PyTorch (only on bare CUDA images; skipped if already present) ----
ARG INSTALL_PYTORCH=false
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
RUN if [ "$INSTALL_PYTORCH" = "true" ]; then \
    pip install --no-cache-dir \
    torch torchvision \
    --index-url ${PYTORCH_INDEX_URL} ; \
    fi

# Ensure PyTorch shared libraries (libc10.so etc.) are discoverable.
# On pip-installed PyTorch this is required for CUDA extensions to link/import.
RUN TORCH_LIB=$(python -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null | sed 's|/share/cmake/Torch|/lib|' || true) \
    && echo "PyTorch lib dir: ${TORCH_LIB:-not found}"
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}"

# ---- Set working directory ----
WORKDIR /app

# ---- Python dependencies ----
# PyTorch is NOT in requirements.txt — it comes from base image or INSTALL_PYTORCH step
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- Copy project source code ----
# Models and videos are bind-mounted at runtime (not baked in)
COPY DCVC/              /app/DCVC/
COPY src/               /app/src/
COPY configs/           /app/configs/
COPY run_compression.py /app/run_compression.py
COPY verify_install.py  /app/verify_install.py

# ---- Build DCVC C++ extensions ----
# NOTE: Docker build has no GPU access, so we compile only — import
#       verification happens at runtime via verify_install.py.

# 1) pybind11 rANS codec (CPU-only, always works)
WORKDIR /app/DCVC/src/cpp
RUN pip install --no-cache-dir . \
    && python -c "import MLCodec_extensions_cpp; print('MLCodec_extensions_cpp OK')"

# 2) CUDA inference kernels (REQUIRED — no fallback)
#    Needs: CUDA toolkit 12.4 + PyTorch cu126
#    Jetson Orin = sm_87, Jetson Xavier = sm_72
#    PC test:   use a broad arch list like "7.0;7.5;8.0;8.6;8.9;9.0"
ARG TORCH_CUDA_ARCH_LIST="8.7"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

WORKDIR /app/DCVC/src/layers/extensions/inference
# Compile only — import requires GPU runtime which isn't available during build
RUN pip install --no-cache-dir .

# ---- Back to project root ----
WORKDIR /app

# ---- Default environment ----
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONUNBUFFERED=1

# ---- Entrypoint ----
ENTRYPOINT ["python", "run_compression.py"]
CMD ["video.mp4", "--config", "configs/compression.yaml"]
