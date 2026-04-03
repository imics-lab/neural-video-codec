# Weekly Report — Neural Video Codec Pipeline
**Date:** 2026-04-03
**Repository:** https://github.com/smiksha1701/neural_compression_codec
**Server:** moon (Texas State University HPC) — 2× NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM each

---

## Summary

This week focused on three major areas: completing Model R (RestoreUNet) training on the moon server, replacing the custom SRUNet (Model S) with a pretrained one-step diffusion super-resolution model (S3Diff), and integrating the full end-to-end pipeline on real video. Significant effort went into resolving server, environment, and multi-GPU issues that blocked inference.

---

## 1. Completed Model R Training

RestoreUNet was trained to completion on moon (2× RTX PRO 6000 Blackwell, 96 GB each):

- **10 epochs, batch_size=128, no-val mode**
- Loss trajectory: 0.98 → ~0.17 (diffusion noise prediction MSE)
- Checkpoints saved: `checkpoints/restore/best.pt`, `epoch_0005.pt`, `epoch_0010.pt`, `latest.pt` (484 MB each, DataParallel format with `module.` prefix)
- Training speed: ~0.89 steps/sec, ~10.2 hours total

**Key fix during training:** CUDA misaligned address during validation on Blackwell + DataParallel. Root cause was not fully diagnosed; workaround was to skip validation entirely (`--no-val` flag) and validate by running inference on held-out frames.

---

## 2. Replaced Model S with S3Diff (Pretrained Diffusion SR)

The original plan used a custom-trained SRUNet for super-resolution. This was replaced with **S3Diff** — a one-step diffusion SR model based on SD-Turbo — for the following reasons:

- The project's research focus is diffusion models throughout the pipeline
- GAN-based upscalers (Real-ESRGAN) are inconsistent with the diffusion framing
- S3Diff provides state-of-the-art 4× SR quality without any training required

**S3Diff integration:**
- Source: `zhangap/S3Diff` (HuggingFace), paper: arxiv 2405.10044
- Base model: `stabilityai/sd-turbo` (auto-downloaded via HuggingFace hub)
- Weights: `de_net.pth` (degradation estimator), `s3diff.pkl` (LoRA checkpoint)
- Rewrote `run_upscale.py` from scratch for S3Diff
- Replaced Real-ESRGAN block in `run_pipeline.py` step 4 with inline S3Diff inference
- Fixed seed per frame (`_set_seed(seed)` before each frame) for temporal consistency

---

## 3. Server Migration: sun → moon

Training and inference moved from sun (8× A5000) to moon (2× RTX PRO 6000 Blackwell) due to DataParallel deadlocks on sun. Key differences:

| | sun | moon |
|---|---|---|
| GPUs | 8× A5000 (24 GB) | 2× RTX PRO 6000 Blackwell (96 GB) |
| CUDA | 12.2 | 13.0 |
| PyTorch | 2.5.1+cu121 | 2.11.0+cu128 |
| GPU arch | Ampere (sm_86) | Blackwell (sm_120) |

**moon-specific fixes:**
- Installed PyTorch 2.11.0+cu128 (cu128 wheel for CUDA 13.0 on Blackwell)
- xformers built for cu121 was incompatible → uninstalled
- DCVC CUDA kernels not compiled for sm_120 → automatic fallback to PyTorch (~50 pairs/sec)

---

## 4. Multi-GPU DataParallel Fixes

DataParallel caused repeated hangs and crashes throughout the week. The root causes and fixes:

| Problem | Root Cause | Fix |
|---|---|---|
| Training hang on sun after epoch start | All 8 GPUs visible, batch_size < device_count → empty splits | Added `--gpu` (single GPU) and `--gpus` (explicit device IDs) flags; guard: only wrap in DataParallel if `batch_size >= len(gpu_ids)` |
| Validation CUDA misaligned address | DataParallel + Blackwell GPU alignment issue | Unwrap DataParallel for validation (`model.module`); added `--no-val` flag |
| Checkpoint `module.` prefix | DataParallel saves state_dict with `module.` prefix | Strip prefix in `restorer.py` on checkpoint load |

---

## 5. Setup Script Fixes (`setup.sh`)

Multiple fixes to make `setup.sh` work on both sun and moon:

- **python_requires patch:** `sed -i 's/python_requires=">=3\.12"/python_requires=">=3.9"/'` on DCVC's setup.py
- **pybind11 visibility:** added `--no-build-isolation` to pip install of MLCodec extension
- **numpy ABI mismatch:** pinned `numpy<2.0` (numpy 2.x breaks packages compiled against 1.x)
- **S3Diff dependency conflict:** install S3Diff deps manually (skip its requirements.txt which downgrades PyTorch); reinstall correct torch last
- **CUDA version mapping:** added cu126 and cu124 index URLs; cu128 handled via manual override

---

## 6. Restoration Inference Optimization

First inference attempt on `bird1.mp4` (20 seconds, 627 frames, 1920×1080) was estimated at 1+ hour. Iterative optimizations:

| Change | Effect |
|---|---|
| Disabled tiling (`tile_size: 0`) | 15× speedup (from ~15 tiles/frame to 1) |
| Reduced DDIM steps 50 → 8 → 1 | 50× speedup on steps |
| Reduced `input_scale: 0.25` (1080p → 270p before restoration) | ~16× fewer pixels |
| Batched sliding windows (`batch_size: 8`) | ~4–8× GPU utilization improvement |
| Lower `t_start: 200` | Less noise to remove, fewer effective denoising steps |

Target: under 5 minutes for a 20-second video. Achieved in principle with these settings; full timing confirmation pending current pipeline run.

---

## 7. Pipeline End-to-End Run

Running the full pipeline on `bird1.mp4` for the first time:

- **Step 1 (Compression):** completed in 31.6s, 8.31 MB archive
- **Step 2 (Decompression):** completed in 16.4s, 627 frames
- **Step 3 (Restoration):** running — pending timing confirmation
- **Step 4 (S3Diff upscaling):** pending

---

## 8. S3Diff Integration Bugs Fixed

S3Diff had several compatibility issues with PyTorch 2.11+cu128:

| Error | Fix |
|---|---|
| `torch.utils._pytree._register_pytree_node` deprecated (xformers) | Uninstall xformers |
| `No module named 'torchvision.transforms.functional_tensor'` | `sed` patch in `S3Diff/basicsr/data/degradations.py` |
| `S3Diff.__init__()` unexpected `device` argument | Remove `device=` from constructor call |
| `deg_score` on CPU, `self.W` on CPU inside S3Diff forward | Patch `S3Diff/src/s3diff.py` to add `.to(deg_score.device)`; call `_net_sr.cuda()` after `set_eval()` |
| `vae_de_mlp` on CPU | `_net_sr = _net_sr.cuda()` after model load moves all submodules |

---

## 9. Paper Preparation

- **`PROJECT_NOTES.md`:** full technical reference document covering architecture, training details, inference configs, server setup, known issues, and run commands — for use by the paper-writing session
- **`requests.md`:** itemised list of all TODO items before paper submission (figures, tables, dataset description, references, timing measurements)
- **`visualizations/` folder:** five scripts created for all paper figures and metrics:
  - `pipeline_diagram.py` — Figure 1: system overview block diagram (runs without data)
  - `qualitative_comparison.py` — Figure 2: side-by-side frame comparison with 4× insets
  - `temporal_grid.py` — Figure 3: temporal coherence grid (3 methods × 4 frames)
  - `compute_metrics.py` — Tables 1 & 2: PSNR/SSIM/LPIPS with optional ROI masking
  - `temporal_consistency.py` — Section 5.4: optical-flow warping error metric

---

## Current State

- Pipeline actively running on moon (`bird1.mp4`, step 3/4)
- All code committed and pushed to GitHub
- Paper visualization scripts ready; pipeline diagram can be generated immediately

---

## Next Steps

### Immediate
1. Confirm pipeline completes and inspect `outputs/bird1_full.mp4` quality
2. Time the full run: `time python run_pipeline.py ...` for paper Section 6 timing claim
3. Run `python visualizations/pipeline_diagram.py` and add to paper as Figure 1

### Short-term
4. Run pipeline with `--save-intermediate` to capture frames at each stage for Figure 2
5. Run DCVC-Uniform baseline at matched bitrate for Table 1 comparison
6. Compute PSNR/SSIM/LPIPS using `visualizations/compute_metrics.py` with ROI masking

### Medium-term
7. Train or evaluate NCC-without-temporal-attention variant for Table 2 ablation
8. Compute optical-flow warping error for three methods (Table 2 / Section 5.4)
9. Fill in dataset statistics (clip count, total duration, species list, ROI area fraction)
10. Verify DCVC CVPR 2025 citation and MegaDetector v6 citation for bibliography
