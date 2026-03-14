# Neural ROI-Aware Video Compression for Wildlife Monitoring on Edge Devices

## Pre-requisites & Models

The pre-trained YOLOv9 and DCVC models are hosted in the [GitHub Releases](../../releases) section of this repository.

1. Go to the [Releases page](../../releases) and download the model files attached to the latest release:
   - `MDV6-yolov9-c.onnx`
   - `MDV6-yolov9-c.pt`
   - `cvpr2025_image.pth.tar`
   - `cvpr2025_video.pth.tar`
2. Place these downloaded files into the `gpu_compression/models/` directory before running the pipeline.

---

## Local Setup & Usage

### Installation (Local)

```bash
cd gpu_compression
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install opencv-python pyyaml ultralytics pybind11 onnxruntime-gpu wheel setuptools

cd DCVC/src/cpp
pip install .

cd ../layers/extensions/inference
pip install .

cd ../../../../../
```

### Usage (Local)

```bash
python run_compression.py video.mp4 --output output.zip
```

---

## Docker Deployment

### Docker — PC Test Build

Verify the Docker image builds correctly on a local PC (Windows/Linux x86_64):

```bash
cd gpu_compression
docker compose -f docker-compose-test.yml up --build
```

This uses `nvidia/cuda:12.4.1-devel-ubuntu22.04` with PyTorch cu126, compiles both DCVC C++ extensions, and runs `verify_install.py` to check all dependencies.

### Docker — Jetson Deployment

**Prerequisites:** NVIDIA Jetson with JetPack 6 (L4T R36), Docker, NVIDIA Container Runtime.

1. Clone the repo on the Jetson and ensure model files are placed in `gpu_compression/models/` (see Pre-requisites).
2. Place your input video:
   ```bash
   cp /path/to/video.mp4 gpu_compression/video.mp4
   ```

3. Build and run:
   ```bash
   cd gpu_compression
   docker compose -f docker-compose-jetson.yml up --build -d
   ```

4. View logs:
   ```bash
   docker compose -f docker-compose-jetson.yml logs -f
   ```

5. Output will be at `outputs/compressed.zip`.

> **Note:** For Jetson Xavier, change `TORCH_CUDA_ARCH_LIST` from `"8.7"` to `"7.2"` in `docker-compose-jetson.yml`.