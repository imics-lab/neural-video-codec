# Neural-Video-Codec

## Quick Start for Jetson Compression/Decompression

### Installtion (Jetson) 

```bash
python3 -m venv venv
source venv/bin/activate
pip install "numpy<2"
pip install --no-cache-dir --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  torch==2.8.0 torchvision==0.23.0
pip install --no-cache-dir --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  onnxruntime-gpu==1.23.0
pip install -r docker/jetson/requirements.jetson.txt
```


Download pipeline models:

```bash
python scripts/download_models.py
```

Compress a video clip:

```bash
python run_compression.py {video_path/video.mp4} --config configs/jetson/compression.yaml --output {output_path/video.zip}
```
Note: video file name and zip file name must be mentioned

Decompress it:

```bash
python run_decompression.py {zip_path/video.zip} --config configs/gpu/decompression.yaml --output {recon_video_path/recon_video.mp4}
```
Note: video file name and zip file name must be mentioned

## Quick Start for Local Compression/Decompression

### Installtion 

```bash
python3 -m venv venv
source venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.10.0+cu126 torchvision==0.25.0+cu126
pip install -r docker/requirements.gpu.txt
```


Download pipeline models:

```bash
python scripts/download_models.py
```

Compress a video clip:

```bash
python run_compression.py {video_path/video.mp4} --config configs/jetson/compression.yaml --output {output_path/video.zip}
```
Note: video file name and zip file name must be mentioned

Decompress it:

```bash
python run_decompression.py {zip_path/video.zip} --config configs/gpu/decompression.yaml --output {recon_video_path/recon_video.mp4}
```
Note: video file name and zip file name must be mentioned

## Docker (Local)

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
