# Neural ROI-Aware Video Compression

This repository now ships a single compression/decompression backend: FFmpeg-based ROI/BG coding. ROI frames are encoded as an AV1-style stream and background frames as an HEVC-style stream, then packaged into a `.zip` archive with `meta.codec.implementation == "ffmpeg"` and per-stream metadata in `meta.streams`.

Archives are FFmpeg-only. Compression writes FFmpeg-format archives, and decompression expects the same archive format.

For ROI detection, the runtime uses `models/MDV6-yolov9-c.onnx` automatically when that file exists and GPU ONNX Runtime is available; otherwise it falls back to `models/MDV6-yolov9-c.pt`.

## Entry Points

- `run_compression.py`
- `run_decompression.py`

Default configs:

- `configs/gpu/compression.yaml`
- `configs/gpu/decompression.yaml`

## Required Runtime Assets

Compression/decompression model files in `models/`:

- `MDV6-yolov9-c.pt`
- `amt-s.pth` if AMT interpolation is enabled during decompression

Optional:

- `MDV6-yolov9-c.onnx`
- `amt-l.pth`

FFmpeg and FFprobe must be available on `PATH`, unless you override their paths in the YAML config.
For local GPU runs, install a CUDA-capable PyTorch build that matches `docker/requirements.gpu.txt` and the pinned Docker image versions (`torch==2.10.0+cu126`, `torchvision==0.25.0+cu126`).

Upscaling remains separate from this pipeline. Any optional diffusion checkpoint used by upscaling is unchanged and is not part of the FFmpeg compression/decompression path.

## Quick Start

Create local working directories:

```bash
mkdir -p data models outputs
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data, models, outputs
```

Set up the environment:

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
sudo apt-get install -y ffmpeg
python -m pip install --upgrade pip==26.0.1 setuptools==82.0.0 wheel==0.46.3
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.10.0+cu126 torchvision==0.25.0+cu126
pip install -r docker/requirements.gpu.txt
```

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip==26.0.1 setuptools==82.0.0 wheel==0.46.3
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.10.0+cu126 torchvision==0.25.0+cu126
pip install -r docker\requirements.gpu.txt
# Ensure ffmpeg and ffprobe are available on PATH.
```

Download pipeline models:

```bash
python scripts/download_models.py
```

The public model release is [nevc-models](https://github.com/imics-lab/neural-video-codec/releases/tag/nevc-models). You can either:

1. Open the release page and download the published assets into `models/`
2. Run `python scripts/download_models.py`, which downloads from `imics-lab/neural-video-codec@nevc-models` and verifies each file against `models/models.manifest.json`

Published model files:
- `MDV6-yolov9-c.pt`
- `MDV6-yolov9-c.onnx`
- `amt-s.pth`
- `amt-l.pth`

Minimum runtime set for this FFmpeg compression/decompression pipeline:
- Keep `MDV6-yolov9-c.pt`
- Keep `amt-s.pth` when AMT interpolation is enabled
- Keep `MDV6-yolov9-c.onnx` if you want ONNX ROI detection
- `amt-l.pth` is optional unless you switch the decompression config to the larger AMT model

Compress a sample clip:

```bash
python run_compression.py data/test.mp4 --config configs/gpu/compression.yaml --output outputs/video.zip
```

Decompress it:

```bash
python run_decompression.py outputs/video.zip --config configs/gpu/decompression.yaml --output outputs/video_reconstructed.mp4
```

## Sanity Scripts

- Compression smoke/reproducibility:

```bash
python scripts/test_compression.py --config configs/gpu/compression.yaml --video data/test.mp4 --repeat 2
```

- Decompression smoke:

```bash
python scripts/test_decompression.py outputs/video.zip --config configs/gpu/decompression.yaml --repeat 1
```

More script notes are in [scripts/README.md](scripts/README.md).

## Docker

Build:

```bash
docker build -f docker/Dockerfile.gpu -t edge-roi-gpu .
```

Run:

```bash
docker compose -f docker/compose.gpu.yaml run --build --rm pipeline-gpu
```

The compose entrypoint uses `docker/run_pipeline.sh`. By default it reads `data/test.mp4` inside the container, and writes `outputs/video.zip` plus `outputs/video_reconstructed.mp4`.

Individual stages:

```bash
docker compose -f docker/compose.gpu.yaml --profile stage-tools run --build --rm compression-gpu
docker compose -f docker/compose.gpu.yaml --profile stage-tools run --build --rm decompression-gpu
```

`compression-gpu` writes `outputs/video.zip` by default. `decompression-gpu` expects that archive to already exist unless you override `WILDROI_ARCHIVE_PATH`.

Docker wrappers write intermediates to container-local `/tmp` and then copy the final archive or video into the mounted `outputs/` directory. This avoids bind-mount write failures on some Windows or OneDrive-backed paths.

Useful env vars:
- `WILDROI_INPUT_VIDEO` to override the container input path
- `WILDROI_ARCHIVE_PATH` to override the archive output path
- `WILDROI_RECON_PATH` to override the reconstructed video path
- `WILDROI_COMPRESSION_CONFIG` and `WILDROI_DECOMPRESSION_CONFIG` to point at alternate YAML files

The GPU image installs FFmpeg and the Python dependencies needed for the FFmpeg-only compression/decompression path.
