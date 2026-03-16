# Neural ROI-Aware Video Compression for Wildlife Monitoring on Edge Devices

## Pre-requisites & Models

The pre-trained YOLOv9 and DCVC models are hosted in the [GitHub Releases](../../releases) section of this repository.

1. Go to the [Releases page](../../releases) and download the model files attached to the latest release:
   - `MDV6-yolov9-c.onnx`
   - `MDV6-yolov9-c.pt`
   - `cvpr2025_image.pth.tar`
   - `cvpr2025_video.pth.tar`
   - `amt-s.pth` (only needed if AMT interpolation is enabled)
2. Place these downloaded files into the `models/` directory before running the pipeline.

---

The entry points are:

- `run_compression.py`
- `run_decompression.py`

## What This Runs

- `run_compression.py` creates a compressed archive (`.zip`)
- `run_decompression.py` reconstructs video from that archive
- Default compression output location from the shipped GPU config: `outputs/compression/`
- Decompression output should usually be passed with `--output`

## GPU Profile Layout

- `configs/gpu/compression.yaml`
- `configs/gpu/decompression.yaml`
- `docker/Dockerfile.gpu`
- `docker/compose.gpu.yaml`

## Required Model Files

Place these files in `gpu/models/`:

- `MDV6-yolov9-c.pt`
- `cvpr2025_image.pth.tar`
- `cvpr2025_video.pth.tar`
- `amt-s.pth` (only needed if AMT interpolation is enabled)

Optional:

- `MDV6-yolov9-c.onnx`

## Manual Setup (No Docker)

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools==82.0.0 wheel==0.46.3
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.10.0+cu126 torchvision==0.25.0+cu126
pip install -r docker/requirements.gpu.txt
cd DCVC/src/cpp
pip install --no-build-isolation .
cd ../layers/extensions/inference
pip install --no-build-isolation .
cd ../../../..
```

### Windows (`cmd.exe`)

```cmd
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools==82.0.0 wheel==0.46.3
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.10.0+cu126 torchvision==0.25.0+cu126
pip install -r docker\requirements.gpu.txt
cd DCVC\src\cpp
pip install --no-build-isolation .
cd ..\layers\extensions\inference
pip install --no-build-isolation .
cd ..\..\..\..
```

## Manual Run

### Compression

```bash
python run_compression.py video.mp4 --config configs/gpu/compression.yaml --output outputs/video.zip
```

Output:

- `outputs/video.zip`
- Console output is production-style by default; use `--verbose` for detailed diagnostic logs.

### Decompression

```bash
python run_decompression.py outputs/video.zip --config configs/gpu/decompression.yaml --output outputs/video_reconstructed.mp4
```

Output:

- `outputs/video_reconstructed.mp4`
- Console output is production-style by default; use `--verbose` for detailed diagnostic logs.

## Stage-by-Stage Test Scripts

Run these from `gpu/` after environment activation.

### 1) ROI Detection

```bash
python scripts/test_roi_detection.py --config configs/gpu/compression.yaml --video video.mp4
```

Outputs:

- `outputs/sanity_checks/roi_detection/roi_detections.json`
- `outputs/sanity_checks/roi_detection/roi_overlay.mp4`

### 2) Frame Removal

```bash
python scripts/test_frame_removal.py --config configs/gpu/compression.yaml --video video.mp4
```

Outputs:

- `outputs/sanity_checks/frame_removal/frame_drop.json`
- `outputs/sanity_checks/frame_removal/frame_drop_overlay.mp4`
- `outputs/sanity_checks/frame_removal/roi_kept_preview.mp4`
- `outputs/sanity_checks/frame_removal/bg_kept_preview.mp4`

### 3) Compression Sanity + Reproducibility

```bash
python scripts/test_compression.py --config configs/gpu/compression.yaml --video video.mp4 --repeat 2
```

Outputs:

- `outputs/sanity_checks/compression/session_*/summary.json`
- Per-run archives in the same session folder

### 4) Decompression Sanity + Reproducibility

```bash
python scripts/test_decompression.py outputs/video.zip --config configs/gpu/decompression.yaml --repeat 1
```

Outputs:

- `outputs/sanity_checks/decompression/session_*/summary.json`
- Per-run reconstructed videos in the same session folder

## Docker Setup

Run these Docker commands from `gpu/`.

### Build Image

Linux/macOS:

```bash
docker build -f docker/Dockerfile.gpu -t edge-roi-gpu .
```

Windows (`cmd.exe`):

```cmd
docker build -f docker\Dockerfile.gpu -t edge-roi-gpu .
```

For a clean rebuild with full logs:

```bash
docker build --no-cache --progress=plain -f docker/Dockerfile.gpu -t edge-roi-gpu .
```

If you intentionally want the slower PyTorch fallback instead of the CUDA inference extension:

```bash
docker build -t edge-roi-gpu --build-arg BUILD_INFERENCE_EXT=0 --build-arg REQUIRE_INFERENCE_EXT=0 -f docker/Dockerfile.gpu .
```

### Run Full Pipeline In Docker

Recommended compose workflow:

```bash
docker compose -f docker/compose.gpu.yaml run --rm pipeline-gpu
```

This runs compression and decompression sequentially with the sample paths already wired in:

- input: `data/tune/video.mp4`
- archive: `outputs/video.zip`
- reconstruction: `outputs/video_reconstructed.mp4`

### Run Compression In Docker

Linux/macOS:

```bash
mkdir -p outputs
docker run --rm -it --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/outputs:/app/outputs" \
  edge-roi-gpu \
  python run_compression.py video.mp4 --config configs/gpu/compression.yaml --output /app/outputs/video.zip
```

Windows (`cmd.exe`):

```cmd
if not exist outputs mkdir outputs
docker run --rm -it --gpus all -v "%cd%/data:/app/data" -v "%cd%/models:/app/models" -v "%cd%/outputs:/app/outputs" edge-roi-gpu python run_compression.py video.mp4 --config configs/gpu/compression.yaml --output /app/outputs/video.zip
```

Host output:

- `outputs/video.zip`

### Run Decompression In Docker

Linux/macOS:

```bash
docker run --rm -it --gpus all \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/outputs:/app/outputs" \
  edge-roi-gpu \
  python run_decompression.py /app/outputs/video.zip --config configs/gpu/decompression.yaml --output /app/outputs/video_reconstructed.mp4
```

Windows (`cmd.exe`):

```cmd
docker run --rm -it --gpus all -v "%cd%/models:/app/models" -v "%cd%/outputs:/app/outputs" edge-roi-gpu python run_decompression.py /app/outputs/video.zip --config configs/gpu/decompression.yaml --output /app/outputs/video_reconstructed.mp4
```

Host output:

- `outputs/video_reconstructed.mp4`

## Notes

- Runtime is strict GPU-only (no CPU/MPS fallback path).
- `run_compression.py` and `run_decompression.py` default to the GPU profile configs under `configs/gpu/`; `--config` can still override them explicitly.
- ROI detection keeps the `.pt` model by default. ONNX is used only when `roi_detection.runtime.prefer_onnx=true`; `prefer_onnx_strict=true` makes missing ONNX support fail fast instead of silently falling back.
- `run_compression.py` and `run_decompression.py` show short phase progress by default; pass `--verbose` to restore detailed logs.
- `run_decompression.py` takes the output video path from `--output`; if omitted, it writes next to the archive using a default filename.
- Docker image installs CUDA PyTorch wheels, but does not bake local models into the image. Mount `models/` at runtime.
- DCVC extension is installed with:
  - `pip install --no-build-isolation /app/DCVC/src/cpp`
- CUDA inference extension build is attempted with:
  - `pip install --no-build-isolation /app/DCVC/src/layers/extensions/inference`
  - By default, image build fails if this extension cannot be built or imported.
  - To allow PyTorch-kernel fallback intentionally, set `BUILD_INFERENCE_EXT=0` and `REQUIRE_INFERENCE_EXT=0`.
- If you still see `cannot import cuda implementation for inference, fallback to pytorch.`, rebuild with no cache:
  - `docker build --no-cache --progress=plain -t edge-roi-gpu .`
- The Dockerfile pins CUDA host compiler to `gcc-12/g++-12` for better nvcc compatibility.
- To verify the CUDA inference extension is importable in the built image:
  - `docker run --rm --gpus all edge-roi-gpu python -c "import inference_extensions_cuda; print('inference_extensions_cuda: OK')"`
- `docker/compose.gpu.yaml` exposes a default `pipeline-gpu` service for the end-to-end path. The per-stage `compression-gpu` and `decompression-gpu` services are left under the `stage-tools` profile for manual debugging only.
