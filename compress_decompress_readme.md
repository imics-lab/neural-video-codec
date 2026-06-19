# Compression and Decompression Guide

This guide covers only the compression and decompression stages of the project. It does not cover restoration or upscaling.

## 1. Install System Requirements

Install these before running the Python commands:

| Requirement | Purpose |
|---|---|
| Python 3.10+ | Runtime |
| NVIDIA GPU driver + CUDA-compatible PyTorch | Required for DCVC/DCVC INT16 and AMT interpolation |
| FFmpeg | Required for `h264`, `h265`, `av1`, and video assembly |
| Git | Required for normal repo usage |

Check FFmpeg:

```bash
ffmpeg -version
```

Check CUDA from Python:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## 2. Install Python Packages

From the repo root:

```bash
pip install -r docker/requirements.gpu.txt
```

If you use the original Microsoft DCVC backend, build its C++ entropy extension:

```bash
pip install pybind11
cd DCVC/src/cpp
python setup.py build_ext --inplace
pip install --no-build-isolation .
cd ../../..
```

The deterministic INT16 backend uses the `dcvc_int16/` runtime. If that runtime needs bootstrapping on a fresh machine, run:

```bash
python dcvc_int16/bootstrap_runtime.py
```

## 3. Download Models

Download all release-backed models:

```bash
python scripts/download_models.py
```

Stage-specific downloads:

```bash
python scripts/download_compression_models.py
python scripts/download_decompression_models.py
```

Expected release-backed model files in `models/`:

| File | Used by |
|---|---|
| `MDV6-yolov9-c.pt` | ROI detection during compression |
| `MDV6-yolov9-c.onnx` | Optional ONNX ROI detection during compression |
| `int16_bundle_v1.0.0.pt` | `dcvc_int16` compression/decompression |
| `amt-s.pth` | Optional AMT interpolation model |
| `amt-l.pth` | Default AMT interpolation model |

## 4. Compression Command

Basic command:

```bash
python run_compress.py --video data/input.mp4 --config configs/gpu/compression.yaml
```

With explicit output archive:

```bash
python run_compress.py ^
  --video data/input.mp4 ^
  --config configs/gpu/compression.yaml ^
  --output outputs/compression/input.zip
```

Verbose mode:

```bash
python run_compress.py --video data/input.mp4 --config configs/gpu/compression.yaml --verbose
```

Default output location is controlled by:

```yaml
output:
  out_dir: "outputs/compression"
```

## 5. Decompression Command

Basic command:

```bash
python run_decompress.py --archive outputs/compression/input.zip --config configs/gpu/decompression.yaml
```

With explicit output video:

```bash
python run_decompress.py ^
  --archive outputs/compression/input.zip ^
  --config configs/gpu/decompression.yaml ^
  --output outputs/decompression/input_decompressed.mp4
```

Verbose mode:

```bash
python run_decompress.py --archive outputs/compression/input.zip --config configs/gpu/decompression.yaml --verbose
```

Default output location is controlled by:

```yaml
output:
  out_dir: "outputs/decompression"
```

If `output.out_dir` is not present in the decompression config, the script still defaults to `outputs/decompression`.

## 6. Codec Switching

Codec selection is controlled in `configs/gpu/compression.yaml`.

Use this block:

```yaml
compression:
  codec: "dcvc_int16"
  stream_codecs:
    roi: "dcvc_int16"
    bg: "dcvc_int16"
```

Supported codec values:

| Value | Backend |
|---|---|
| `dcvc` | Original Microsoft DCVC backend |
| `dcvc_int16` | Deterministic INT16 DCVC-RT backend |
| `dcvc_rt` | Alias for `dcvc_int16` |
| `h264` | FFmpeg H.264 backend |
| `h265` | FFmpeg H.265 backend |
| `hevc` | Alias for `h265` |
| `av1` | FFmpeg AV1 backend |

To use the same codec for ROI and background, set both fields to the same value.

### Example: DCVC INT16

```yaml
compression:
  codec: "dcvc_int16"
  stream_codecs:
    roi: "dcvc_int16"
    bg: "dcvc_int16"
  dcvc:
    repo_dir: "dcvc_int16"
    bundle_path: "../models/int16_bundle_v1.0.0.pt"
    device: "cuda"
    use_cuda: true
    cuda_idx: 0
```

### Example: Original DCVC

```yaml
compression:
  codec: "dcvc"
  stream_codecs:
    roi: "dcvc"
    bg: "dcvc"
  dcvc:
    repo_dir: "DCVC"
    model_i: "models/cvpr2025_image.pth.tar"
    model_p: "models/cvpr2025_video.pth.tar"
    device: "cuda"
    use_cuda: true
    cuda_idx: 0
```

Original DCVC requires its model files and the DCVC C++ extension.

### Example: H.264

```yaml
compression:
  codec: "h264"
  stream_codecs:
    roi: "h264"
    bg: "h264"
  quality:
    roi_crf: 20
    bg_crf: 38
  ffmpeg:
    h264_encoder: "libx264"
    preset:
      h264: "medium"
```

### Example: H.265 / HEVC

```yaml
compression:
  codec: "h265"
  stream_codecs:
    roi: "h265"
    bg: "h265"
  quality:
    roi_crf: 20
    bg_crf: 38
  ffmpeg:
    hevc_encoder: "libx265"
    preset:
      hevc: "ultrafast"
```

### Example: AV1

```yaml
compression:
  codec: "av1"
  stream_codecs:
    roi: "av1"
    bg: "av1"
  quality:
    roi_crf: 20
    bg_crf: 38
  ffmpeg:
    av1_encoder: "libaom-av1"
    preset:
      av1: "8"
```

## 7. Quality Settings

There are two quality-control systems:

| Codec family | Config fields | Meaning |
|---|---|---|
| `dcvc`, `dcvc_int16` | `roi_qp_i`, `roi_qp_p`, `bg_qp_i`, `bg_qp_p` | DCVC quantization parameters |
| `h264`, `h265`, `av1` | `roi_crf`, `bg_crf` | FFmpeg CRF values |

Current quality block:

```yaml
quality:
  roi_qp_i: 63
  roi_qp_p: 63
  bg_qp_i: 25
  bg_qp_p: 25
  roi_crf: 20
  bg_crf: 38
```

For FFmpeg codecs, lower CRF usually means higher quality and larger files.

For DCVC/DCVC INT16, use the QP fields. Do not compare DCVC QP values directly with FFmpeg CRF values; they are not the same scale.

## 8. Decompression Settings

Decompression reads codec metadata from the archive, so normally you do not manually select the codec again during decompression.

Important decompression config fields:

```yaml
decompression:
  dcvc:
    repo_dir: "dcvc_int16"
    bundle_path: "../models/int16_bundle_v1.0.0.pt"
    use_cuda: true
    cuda_idx: 0

  interpolate:
    enable: true
    model: "amt-l"
    weights_path: "models/amt-l.pth"
    repo_dir: "_third_party_amt"
    device: "cuda"
    fp16: true
```

Set `interpolate.enable: true` to reconstruct dropped frames with AMT interpolation.

Set `interpolate.enable: false` if you want decompression without interpolation.

## 9. Common Runs

Compress with DCVC INT16:

```bash
python run_compress.py --video data/input.mp4 --config configs/gpu/compression.yaml --output outputs/compression/input_dcvc_int16.zip
```

Decompress the archive:

```bash
python run_decompress.py --archive outputs/compression/input_dcvc_int16.zip --config configs/gpu/decompression.yaml --output outputs/decompression/input_dcvc_int16.mp4
```

Run on CPU-only FFmpeg codecs:

```yaml
compression:
  stream_codecs:
    roi: "h264"
    bg: "h264"
```

Then:

```bash
python run_compress.py --video data/input.mp4 --config configs/gpu/compression.yaml
python run_decompress.py --archive outputs/compression/input.zip --config configs/gpu/decompression.yaml
```

## 10. Troubleshooting

| Problem | Check |
|---|---|
| Model file missing | Run `python scripts/download_models.py` |
| `dcvc_int16 bundle not found` | Check `compression.dcvc.bundle_path` and `decompression.dcvc.bundle_path` |
| FFmpeg codec fails | Run `ffmpeg -encoders` and check for `libx264`, `libx265`, or `libaom-av1` |
| CUDA unavailable | Check NVIDIA driver, PyTorch CUDA install, and `torch.cuda.is_available()` |
| Original DCVC fails importing C++ extension | Rebuild `DCVC/src/cpp` extension |
| Decompressed video has fewer frames | Enable AMT interpolation in `configs/gpu/decompression.yaml` |

