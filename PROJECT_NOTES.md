# Neural Video Codec — Project Notes

## Overview

End-to-end neural video codec combining DCVC compression with diffusion-based restoration and super-resolution. The pipeline allocates bits intelligently between regions of interest (animals) and background using object detection, then uses learned models to recover quality.

```
Video Input
    ↓
[1] COMPRESSION    — YOLO detection + KLT tracking → dual-stream DCVC encoding
    ↓
[2] DECOMPRESSION  — DCVC decoding + feathered blending
    ↓
[3] RESTORATION    — Temporal-attention diffusion UNet (Model R)
    ↓
[4] UPSCALING      — S3Diff one-step diffusion SR (Model S)
    ↓
Video Output
```

---

## Model R — RestoreUNet

**Architecture:** Temporal-attention diffusion U-Net (42M params)

- Input: `(B*T, 6, H, W)` — noisy frame (3ch) + degraded conditioning (3ch)
- Output: `(B*T, 3, H, W)` — predicted noise ε̂
- Temporal window: T=3 frames, center frame is target
- Temporal attention: AnimateDiff-style self-attention over T frames
  - Reshape `(B*T, C, H, W) → (B*H*W, T, C)` → attend → reshape back
  - Learnable positional bias (T×T) per head
- base_channels=64, encoder_channels=(128, 256, 512), num_res_blocks=4, n_heads=4

**Diffusion:** Cosine schedule, T=1000, DDIM inference
- Training: MSE(ε̂, ε) on random-timestep noised clean frames
- Inference: img2img starting from t_start (noised degraded frame)

**Training:**
- Dataset: original/degraded frame pairs from DCVC-compressed video
- Patch size: 256, augmentation: random crops/flips/rotations
- Hardware: 2× RTX PRO 6000 Blackwell (96GB each) on moon server
- 10 epochs trained: loss 0.98 → 0.17
- Checkpoint: `checkpoints/restore/best.pt` (484MB, DataParallel format with `module.` prefix)

**Inference config (`configs/gpu/restoration.yaml`):**
| Param | Value | Notes |
|-------|-------|-------|
| ddim_steps | 1 | 1=fast (~3min/20s video), 50=quality |
| t_start | 200 | lower = more faithful to input |
| tile_size | 0 | 0=full frame (96GB GPU fits 1080p) |
| batch_size | 8 | windows per DDIM pass |
| input_scale | 0.25 | downscale to 480×270 before restoring |

---

## Model S — S3Diff (Pretrained)

**Source:** `zhangap/S3Diff` on HuggingFace (paper: arxiv 2405.10044)

- One-step diffusion SR based on SD-Turbo
- LoRA-modulated UNet + degradation estimator (DEResNet)
- Weights: `weights/de_net.pth`, `weights/s3diff.pkl`
- Base model: `stabilityai/sd-turbo` (auto-downloaded)
- Scale: 4× upscaling
- Input: LR frame bilinearly upsampled to 4×, normalized to [-1,1], padded to 64× multiple
- DEResNet estimates degradation severity → guides LoRA modulation

**Why S3Diff over GAN-based (e.g., Real-ESRGAN):** Research focus is diffusion models.

---

## Compression

**Model:** DCVC (CVPR 2025)
- I-frame: `models/cvpr2025_image.pth.tar`
- P-frame: `models/cvpr2025_video.pth.tar`
- ROI stream: QP=63 (high quality)
- Background stream: QP=25 (lower quality)
- Note: DCVC CUDA kernels not compiled for sm_120 (Blackwell) → PyTorch fallback (~50 pairs/sec)

**Detection:** MegaDetector v6 (YOLOv9c)
- Checkpoint: `models/MDV6-yolov9-c.pt`
- Runs every 15 frames, KLT tracking between keyframes
- Animal classes (COCO IDs 14-23)
- Conf=0.25, IoU=0.50, scale=0.50

**Output:** ZIP archive with `roi.bin`, `bg.bin`, `meta.json`, `detections.json`

---

## Diffusion Framework

**Shared:** `src/restoration/_diffusion.py`

Cosine schedule:
```
ᾱ_t = cos((t/T + s)/(1+s) × π/2)² / cos(s/(1+s) × π/2)²,  s=0.008
```

DDIM sampling (η=0, deterministic):
```
x0_pred = (x_t - √(1-ᾱ_t) × ε̂) / √(ᾱ_t),  clamped to [-1, 2]
x_{t-1} = √(ᾱ_{t-1}) × x0_pred + √(1-ᾱ_{t-1}) × ε̂
```

---

## Servers

**moon** (primary training/inference):
- 2× NVIDIA RTX PRO 6000 Blackwell, 96GB each
- CUDA 13.0, PyTorch 2.11.0+cu128
- Conda env: `nvc`
- Repo: `~/neural_compression_codec`
- Shared server — check `nvidia-smi` before launching
- Use `CUDA_VISIBLE_DEVICES=1` to target GPU 1 (often less contested)

**sun** (backup, 8× A5000):
- DataParallel deadlocks without explicit `--gpus` flag
- Use `--gpu 2` for single-GPU training

---

## Key Implementation Notes

- **DataParallel checkpoints** save with `module.` prefix → stripped in `restorer.py` on load
- **xformers** must match PyTorch version exactly; uninstall if mismatch causes crashes
- **numpy<2.0** required (ABI incompatibility with packages compiled against numpy 1.x)
- **S3Diff** hardcodes `.cuda()` → use `CUDA_VISIBLE_DEVICES=N` to target specific GPU
- **S3Diff `__init__`** has no `device` argument — model auto-places to CUDA
- **`functional_tensor`** removed in new torchvision → patch: `sed -i 's/functional_tensor/functional/' S3Diff/basicsr/data/degradations.py`
- Restoration batching: windows reshaped `(B*T, 6, H, W)` so `_denoise_full` works unchanged

---

## Run Commands

**Full pipeline:**
```bash
CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --video dataset/bird1.mp4 \
    --output outputs/bird1_full.mp4 \
    --verbose
```

**Skip upscaling (test restoration only):**
```bash
python run_pipeline.py --video dataset/bird1.mp4 --output outputs/bird1_restored.mp4 --skip-upscale --verbose
```

**Standalone S3Diff upscaling:**
```bash
python run_upscale.py --input outputs/bird1_restored.mp4 --scale 4 --seed 42
```

**Train restoration:**
```bash
python training/train_restoration.py \
    --data dataset/restoration \
    --config configs/gpu/restoration.yaml \
    --epochs 20 --batch-size 128 --gpu 0 --no-val
```

---

## File Structure

```
neural_compression_codec/
├── run_pipeline.py              # main entry point
├── run_upscale.py               # standalone S3Diff upscaling
├── configs/gpu/
│   ├── pipeline.yaml
│   ├── compression.yaml
│   ├── decompression.yaml
│   ├── restoration.yaml         # Model R inference params
│   └── upscaling.yaml
├── src/
│   ├── compression/             # DCVC + YOLO + KLT
│   ├── decompression/           # DCVC decode + blend
│   ├── restoration/
│   │   ├── _network.py          # RestoreUNet architecture
│   │   ├── _diffusion.py        # GaussianDiffusion, DDIM
│   │   ├── restorer.py          # inference wrapper
│   │   └── phase_restore.py     # pipeline orchestrator
│   └── postprocessing/
├── training/
│   ├── train_restoration.py     # Model R training
│   └── train_sr.py              # SRUNet training (unused — replaced by S3Diff)
├── data_prep/
│   ├── prepare_restoration.py
│   └── prepare_sr_enhanced.py
├── models/                      # DCVC + YOLO weights (gitignored)
├── weights/                     # S3Diff weights (gitignored)
├── checkpoints/                 # training checkpoints (gitignored)
└── dataset/                     # video data (gitignored)
```
