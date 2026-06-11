# FaunaScope - Architecture and Deployment

FaunaScope is an AI-powered wildlife camera trap system. A Jetson Orin Nano deployed
in the field detects animals, records video clips, compresses them, and uploads to a
server. The server decompresses, enhances, and posts to a website.

## Hardware (per unit)

- Jetson Orin Nano (8 GB) - edge AI inference and compression
- Solar panel + battery pack
- IR-equipped camera (records 1080p H.264 locally)
- LTE/cellular USB dongle (uplink)

## Pipeline split

```
[Edge - Jetson Orin Nano]              [Server - Cloud GPU]
---------------------------------      ---------------------------------
PIR/motion trigger                     receive ZIP over HTTPS
  -> start recording                   decompress (DCVC decode + composite)
  -> stop on inactivity                restore artifacts (RestoreUNet)
  -> run MegaDetector on clip          upscale 2x (S3Diff) - optional
  -> dual-stream DCVC compress         post video to website
  -> upload ZIP to server API          store metadata in database
```

## Code boundary

The ZIP archive is the interface between edge and server. No shared state beyond that.

### Edge runs:
- `src/preprocessing/` - frame extraction
- `src/detection/` - MegaDetector + KLT tracking
- `src/roi_masking/` - mask generation
- `src/compression/` - dual-stream DCVC encode

Entry point: `run_compress.py` (accepts a video path, produces a ZIP)

### Server runs:
- `src/decompression/` - DCVC decode + composite
- `src/restoration/` - RestoreUNet artifact removal
- `src/upscaling/` - S3Diff 2x upscale
- `src/postprocessing/` - frame -> video assembly

Entry point: `run_decompress.py`, `run_restore.py`, `run_upscale.py`
or all-in-one: `run_pipeline.py --stages decompress restore upscale`

## Dependencies per side

### Edge (Jetson, ARM64)
```
torch (Jetson wheel)
torchvision
ultralytics       # MegaDetector
opencv-python
numpy
pyyaml
```
DCVC compression requires DCVC's compiled CUDA extension (compiles on-device).

### Server (x86-64, CUDA)
```
torch>=2.0
torchvision
opencv-python
numpy
pyyaml
lpips
diffusers         # for S3Diff
```
DCVC decompression requires the same DCVC CUDA extension.

## Upload protocol (current: manual / future: HTTPS API)

Currently tested on a single machine (no network transfer needed).
For field deployment the plan is:

1. Edge: `curl -F file=@clip.zip https://server/api/upload`
2. Server: Flask/FastAPI endpoint that receives the ZIP, enqueues a processing job,
   runs the pipeline, and posts the result to the website.

This is not yet implemented; the focus is correctness on a single machine first.

## Scaling

Each camera sends one ZIP per animal event (typically a few seconds of video).
At moderate wildlife activity (10 events/day, 5 MB each) bandwidth is ~50 MB/day
per camera - well within LTE data budgets.

The server can queue and process ZIPs asynchronously. RestoreUNet at 3 DDIM steps
processes ~8 fps on a single A5000, so a 10-second clip (300 frames) takes ~40
seconds. A single GPU handles dozens of cameras in real time.

## Future work

- Dockerize edge (Dockerfile.edge, ARM64, Jetson L4T base)
- Implement server API endpoint (FastAPI + Celery queue)
- Website integration (post enhanced video + detection metadata)
- Species classification on the server after restoration
- Remote monitoring: push notification to researcher when a rare species is detected
