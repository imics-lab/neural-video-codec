# Weekly Report — Neural Video Codec Pipeline
**Date:** 2026-03-27
**Repository:** https://github.com/smiksha1701/neural_compression_codec
**Server:** sun.cs.txstate.edu (Texas State University HPC)

---

## Project Overview

A two-stage neural video codec pipeline built on top of Microsoft's DCVC (Deep Contextual Video Compression, CVPR 2025):

1. **Compression** — DCVC dual-stream ROI/BG compression (DMCI I-frame + DMC P-frame models)
2. **Model R (RestoreUNet)** — Temporal-attention diffusion model that restores DCVC-compressed video artifacts
3. **Model S (SRUNet)** — Enhanced diffusion super-resolution: 2× upscale + denoising + sharpening in one pass

**Dataset:** 122 wildlife videos (~2.5 GB raw), generating ~442,000 training patch pairs for Model R.

---

## Architecture Details

### Model R — RestoreUNet (42.2M parameters)
- Temporal sliding window of T=3 frames (configurable)
- UNet backbone with ResBlock encoder/decoder
- Timestep embedding for diffusion schedule (T=1000 steps)
- VGG perceptual loss + L1 loss
- Input: `(B×T, 6, H, W)` — noised frame + degraded conditioning frame per temporal slot
- Output: predicted noise `(B×T, 3, H, W)`

### Model S — SRUNet (Enhanced SR)
- 2× upscaling diffusion model
- **Enhanced losses (new this session):**
  - L1 pixel loss
  - VGG perceptual loss (weight 0.1)
  - LPIPS loss (weight 0.1)
  - Gradient/Sobel loss (weight 0.05) — penalises blurry edges
  - Laplacian loss (weight 0.05) — fine high-frequency detail
  - Frequency/FFT loss (weight 0.05) — 4× weight on high-frequency amplitudes
- **Enhanced training data:** DCVC-degraded LR → clean HR pairs (not clean→clean)
- `t_start=700` for stronger denoising during inference

### DCVC C++ Extension (MLCodec_extensions_cpp)
- RANS entropy coder built with pybind11
- C++17, built via `setup.py build_ext --inplace`

---

## Infrastructure

### Hardware
- **Development:** Windows 11, NVIDIA RTX 4070 SUPER (12GB VRAM)
- **Training server:** sun.cs.txstate.edu — 8× NVIDIA RTX A5000 (24GB VRAM each)

### Software Stack
- Python 3.11.15, PyTorch 2.5.1+cu121, CUDA 12.2
- Conda environment: `nvc`
- Key packages: ultralytics (YOLO ROI detection), torchvision, lpips, opencv-python

---

## Work Completed This Session

### 1. Server Setup (`setup.sh`)
- Created automated setup script that:
  - Auto-detects CUDA version
  - Creates conda env `nvc`
  - Installs PyTorch with correct CUDA wheel
  - Builds DCVC MLCodec C++ extension

**Fixes applied to `setup.sh`:**
- `--no-build-isolation` flag for pip install of MLCodec (pip build isolation blocks pybind11 visibility)
- `sed` patch to change DCVC's `python_requires=">=3.12"` → `">=3.9"` (server runs Python 3.11)

### 2. PyTorch CUDA Version Fix
- Server had PyTorch 2.11.0+cu130 installed (CUDA 13.0) but driver only supports CUDA 12.2
- `torch.cuda.is_available()` returned `False`
- Fixed by reinstalling: `python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- Used `python -m pip` (not `pip`) because `/home/ukb12/.local/bin/pip` pointed to base env

### 3. sys.path Bug in Training Scripts
- `train_restoration.py` and `train_sr.py` had `sys.path.insert(0, str(ROOT / "src"))`
- Should be `sys.path.insert(0, str(ROOT))` so `import src.restoration` resolves correctly
- Same bug was previously fixed in `data_prep/` scripts

### 4. Multi-GPU DataParallel Support
- Added `nn.DataParallel` to both training scripts
- Automatically wraps model when `torch.cuda.device_count() > 1`
- Logs: `"Using N GPUs via DataParallel"`

### 5. Data Preparation (Server)
- Ran `prepare_restoration.py` on all 122 videos → 442,384 training samples
  - `data/restoration_pairs/original/<stem>/<frame>.png` — clean frames
  - `data/restoration_pairs/degraded/<stem>/<frame>.png` — DCVC-compressed frames
- Ran `prepare_sr_enhanced.py` → `data/sr_enhanced_pairs/` (degraded LR → clean HR pairs)

### 6. Training — First Epoch Attempt
- Launched `train_restoration.py` on GPUs 1-6 (A5000), batch_size=32
- GPUs confirmed at 100% utilization via `nvidia-smi`
- Identified logging issue: no epoch-start log, first print only at step `log_every` (default 50)
- Added epoch-start log: `"Epoch N/M starting (K steps) ..."`

### 7. Dataset Debug Script
- Created `debug_dataset.py` to isolate dataset loading from training
- Confirmed: 442,384 samples load correctly, `__getitem__` returns `([9,64,64], [3,64,64])` tensors

### 8. Model R Local Test (RTX 4070 SUPER)
- OOM at batch_size=8, patch_size=256 (default) on 12GB VRAM
- Solution: reduce to batch_size=2, patch_size=128

---

## Problems Encountered & Fixes

| Problem | Root Cause | Fix |
|---|---|---|
| `python_requires=">=3.12"` blocks install on Python 3.11 | Upstream DCVC constraint | `sed` patch in `setup.sh` |
| `ModuleNotFoundError: pybind11` during `pip install .` | pip build isolation creates fresh subprocess | `--no-build-isolation` flag |
| `torch.cuda.is_available() = False` despite GPU present | PyTorch cu130 on CUDA 12.2 driver | Reinstall PyTorch cu121 |
| Wrong `pip` binary used (`~/.local/bin/pip`) | Base env pip on PATH before nvc env | Use `python -m pip` |
| `ModuleNotFoundError: No module named 'src'` in training | `sys.path` pointed to `ROOT/src` not `ROOT` | Change to `sys.path.insert(0, str(ROOT))` |
| `scp` fails with `realpath ... No such file` | Windows OpenSSH scp bug with missing dest dirs | Create dir first; use destination without trailing slash |
| Training hangs silently after "Epoch 1 starting..." | `workers=8` + DataParallel deadlock on Linux | Restart with `workers=0`, `patch_size=128` |
| CUDA OOM on RTX 4070 | Default patch_size=256, batch_size=8 too large for 12GB | Reduce to batch_size=2, patch_size=128 |

---

## Current State (End of Session)

- **Server:** Training `train_restoration.py` running on GPUs 1-6 (A5000)
  - Restarted with `workers=0`, `patch_size=128`, `batch_size=16`, `log_every=1`
  - GPUs confirmed at 100% utilization
- **Local (Windows):** Training test running on RTX 4070 SUPER with batch_size=2, patch_size=128

---

## Further Steps

### Immediate
1. Confirm training produces loss output on server (first log line within ~3 min of restart)
2. Monitor training for ~20 epochs, save best checkpoint to `checkpoints/restore/best.pt`
3. Once Model R training finishes, start Model S (SR) training:
   ```bash
   CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python -u training/train_sr.py \
       --data data/sr_enhanced_pairs \
       --ckpt-dir checkpoints/sr \
       --epochs 20 --batch-size 16 --patch-size 128 --workers 0 --log-every 1
   ```

### Short-term
4. Run full compression pipeline on server with trained models:
   ```bash
   python run_pipeline.py --input dataset/bird1.mp4 --output outputs/bird1_neural.mp4
   ```
5. Evaluate metrics (PSNR, SSIM, LPIPS) against baseline HEVC compression
6. Fix `workers=0` bottleneck — add `multiprocessing_context='spawn'` to DataLoader for safe multi-worker loading with DataParallel

### Medium-term
7. Tune hyperparameters based on validation loss curves
8. Consider switching to `DistributedDataParallel` (DDP) for better multi-GPU scaling vs DataParallel
9. Add AMP (automatic mixed precision) training with `torch.cuda.amp.GradScaler` for 2× speedup
10. Run benchmark comparison: DCVC alone vs DCVC + Model R vs DCVC + Model R + Model S
