# Project Log — Neural Video Codec

Dated record of design decisions, architecture changes, and model updates.
New entries go at the top. Format: `## YYYY-MM-DD — <short title>`.

---

## 2026-03-22 — Full codebase written

### Files added (second session)

- `training/train_restoration.py` — Model R training loop (L1 + VGG, temporal window batching, cosine schedule, resume support)
- `training/train_sr.py` — Model S training loop (L1 + VGG + LPIPS, cosine schedule, resume support)
- `eval_metrics.py` — PSNR / SSIM / LPIPS evaluator (video or frame-dir input, optional CSV export)
- `benchmark_qp.py` — QP sweep tool: compress → decompress at each (roi_qp, bg_qp) pair, report quality/bitrate table
- `tests/test_detection.py` — YoloDetector unit tests (mocked YOLO)
- `tests/test_compression.py` — ROI mask builder + compress_video smoke test
- `tests/test_decompression.py` — decompress_archive shape / blending / error tests
- `tests/test_restoration.py` — RestoreUNet shape, TemporalAttention residual, Restorer API tests
- `tests/test_upscaling.py` — SRUNet shape, no-TemporalAttention assertion, Upscaler API tests
- `tests/test_pipeline.py` — End-to-end pipeline smoke tests (all heavy calls mocked)
- `tests/test_config_schema.py` — YAML config validation tests
- `scripts/sanity_check.py` — CPU-only sanity check (synthetic video + tiny model forward pass)
- `scripts/download_models.py` — Checkpoint downloader using models.manifest.json
- `models/models.manifest.json` — Model metadata / SHA-256 / download URL registry

---

## 2026-03-22 — Initial design locked

### Project summary

Wildlife monitoring pipeline running on a Jetson Orin edge device (compression side)
and a cloud GPU cluster (decompression + restoration + upscaling side).
During development, the full pipeline is tested on prerecorded videos on one machine.

---

### End-to-end pipeline

```
[Edge / Compression]
  Video file
    → frame extraction
    → YOLO11 detection + ByteTrack multi-object tracking
        → ROI masks (animal bounding boxes) + detections.json
    → DCVC dual-stream compression
        ROI stream  : high quality (high QP)
        BG stream   : low quality (low QP)
    → ZIP archive  (ROI bitstream + BG bitstream + masks + detections + metadata)

[Server / Decompression + Enhancement]
  ZIP archive
    → DCVC dual-stream decode + composite
    → Model R  (restoration, same resolution)
    → Model S  (super-resolution, 2× upscale)
    → output MP4
```

Evaluation mode: pass `--downscale-input 0.5` to `run_compress.py` to simulate 540p
input, giving a 1080p Model S output that can be compared against the original.

---

### Models

#### Model R — Restoration (artifact removal)

| Property | Value |
|---|---|
| Task | Remove DCVC compression artifacts; same-resolution output |
| Architecture | Diffusion UNet with temporal self-attention (AnimateDiff-style) |
| Temporal window | 3 frames (current ± 1), configurable |
| Input / output | 1080p → 1080p |
| Loss | L1 pixel + VGG perceptual |
| DDIM steps | 50 (quality-first default); 8 available as fast preset |
| Training data | Auto-generated pairs: raw video → DCVC compress/decompress → PNG pairs |
| Training mode | Single-GPU; DDP to be added later |

#### Model S — Super-Resolution

| Property | Value |
|---|---|
| Task | 2× spatial upscaling |
| Architecture | Standard diffusion UNet (no temporal attention) |
| Temporal window | N/A — frame-by-frame |
| Input / output | 1080p → 2160p |
| Loss | L1 pixel + VGG perceptual + LPIPS |
| DDIM steps | 50 (quality-first default); 8 available as fast preset |
| Training data | Auto-generated pairs: raw 1080p → bicubic downsample to 540p → PNG pairs |
| Training mode | Single-GPU; DDP to be added later |

---

### Detection

| Property | Value |
|---|---|
| Model | YOLO11 (Ultralytics), configurable variant, default `yolo11n.pt` |
| Weights | COCO pretrained |
| Tracker | ByteTrack multi-object tracking |
| Target classes | Animals (all COCO animal classes) |
| Output | Per-frame bounding boxes + track IDs embedded in ZIP archive as `detections.json` |
| Timing | Before compression (detections drive ROI mask selection) |

---

### Compression

| Property | Value |
|---|---|
| Codec | DCVC (Deep Contextual Video Compression) |
| Checkpoints | CVPR 2025 I-frame + P-frame models |
| Stream layout | Dual-stream: ROI stream (high quality) + BG stream (low quality) |
| Frame selection | Every frame compressed (no frame removal) |
| Archive format | ZIP (ROI bitstream, BG bitstream, masks, detections.json, config.yaml) |

---

### Training & Data Preparation

| Script | Purpose |
|---|---|
| `data_prep/prepare_restoration.py` | Videos folder → DCVC compress/decompress → PNG pairs for Model R |
| `data_prep/prepare_sr.py` | Videos folder → bicubic downsample → PNG pairs (LR 540p / HR 1080p) for Model S |
| `train_restoration.py` | Train Model R |
| `train_sr.py` | Train Model S |

Dataset: ~1 hour of 1080p wildlife camera footage (one full day recording).
Pairs stored as PNG folders (degraded/ and original/ subdirectories).

---

### Evaluation & Benchmarking

| Script | Purpose |
|---|---|
| `eval_metrics.py` | Compute PSNR, SSIM, LPIPS between pipeline output and original |
| `benchmark_qp.py` | Sweep DCVC QP levels, report quality metrics table |

---

### Infrastructure

| Property | Value |
|---|---|
| Docker | Single `Dockerfile.gpu` (x86-64, cloud GPU, full pipeline) |
| CUDA | Any version (12.6 default in Docker, matches reference project) |
| CPU fallback | Best-effort via PyTorch; DCVC C++ kernels are GPU-only |
| OS | Windows + Linux (all paths via `pathlib.Path`) |
| Config format | YAML + OmegaConf + schema validation |
| Testing | Unit tests per module (`tests/` suite) |

---

### Repo structure

```
neural_video_codec/
├── src/
│   ├── compression/          dcvc_encoder.py, phase_compress.py
│   ├── decompression/        dcvc_decoder.py, phase_decompress.py
│   ├── detection/            yolo_detector.py  (YOLO11 + ByteTrack)
│   ├── restoration/          _network.py, _diffusion.py, _blocks.py,
│   │                         restorer.py, phase_restore.py
│   ├── upscaling/            _network.py, _diffusion.py, _blocks.py,
│   │                         upscaler.py, phase_upscale.py
│   ├── roi_masking/          roi_masking.py
│   ├── preprocessing/        frame_extractor.py
│   ├── postprocessing/       video_assembler.py
│   └── pipeline/             config_schema.py
├── data_prep/
│   ├── prepare_restoration.py
│   └── prepare_sr.py
├── training/
│   ├── train_restoration.py
│   └── train_sr.py
├── configs/gpu/
│   ├── compression.yaml
│   ├── decompression.yaml
│   ├── restoration.yaml
│   ├── upscaling.yaml
│   └── pipeline.yaml
├── tests/
├── docker/
│   ├── Dockerfile.gpu
│   ├── docker-compose.gpu.yaml
│   └── requirements.gpu.txt
├── models/                   weights downloaded separately
├── outputs/
├── scripts/                  sanity checks, model download helpers
├── DCVC/                     third-party DCVC codec
├── run_compress.py
├── run_decompress.py
├── run_restore.py
├── run_upscale.py
├── run_pipeline.py
├── eval_metrics.py
├── benchmark_qp.py
├── IDEA.md
└── PROJECT_LOG.md
```

---

### Explicitly out of scope (initial version)

- Frame interpolation (every frame is compressed)
- Network upload / transmission scripts
- Second YOLO pass after upscaling (server saves upscaled video only)
- Separate edge Docker image (Jetson deployment deferred)
- Multi-GPU / DDP training (single-GPU first)
- GAN / adversarial losses
