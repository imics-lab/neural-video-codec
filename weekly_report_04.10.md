# Weekly Report — April 10, 2026

## Summary

This week focused on making the full end-to-end pipeline runnable: training Model R (RestoreUNet), fixing all blocking bugs, and running the first complete inference pass on `bird1.mp4` producing compressed → decompressed → restored → upscaled outputs.

---

## Pipeline Runs Completed

- First full pipeline run on `bird1.mp4` with all four stages:
  - **Compression**: DCVC dual-stream (ROI + BG) → 8.31 MB archive
  - **Decompression**: DCVC decode → 1080×720 frames
  - **Restoration**: RestoreUNet (Model R) → artifact reduction
  - **Upscaling**: S3Diff (one-step diffusion SR) → 2160×1440 final output
- Intermediate videos saved at each stage for qualitative comparison
- Total pipeline time: ~3671s for 627 frames (20s video)

---

## Model R Training

- Trained RestoreUNet (42.2M params) for 10 epochs on restoration pairs dataset
- Dataset: 420k+ pairs prepared via DCVC compress→decompress on all 122 source videos
- Training converged, checkpoints saved to `checkpoints/restore/best.pt`
- Training optimizations added: AMP (fp16), `--steps-per-epoch`, `--max-samples` flags

---

## Bug Fixes

### Resolution Pipeline
- Added `input_resolution: "1080x720"` to pipeline config — input videos are resized before DCVC compression so train/inference resolution matches
- Restoration model no longer rescales frames (preserves input resolution)
- S3Diff upscaling targets exactly 2160×1440 output
- Fixed output resolution bug where S3Diff was producing 5120×2880 (traced to dynamic scale miscomputation)

### Model Fixes
- **TemporalAttention** (`_blocks.py`): Fixed positional bias `attn_mask` expansion — was missing `* num_heads` factor, causing shape mismatch `(B*H*W, T, T)` vs expected `(B*H*W*num_heads, T, T)`
- **Restoration config**: Fixed wrong checkpoint path (`models/restoration.pth` → `checkpoints/restore/best.pt`)
- **S3Diff dtype mismatch**: VAE decoder received float32 input with float16 weights — fixed by using `torch.cuda.amp.autocast()` instead of `.half()` on the full model

### Training Data
- Data preparation script (`prepare_restoration.py`) updated to resize videos to 1080×720 before DCVC compression — ensures artifacts in training data match inference distribution
- Added skip-already-processed logic (resume on crash)
- Added `--max-frames-per-video 150` to limit DCVC time per video (~45s/video vs ~3min previously)

### VideoWriter Compatibility
- `video_assembler.py`: Added codec fallback chain `X264 → avc1 → mp4v → XVID` — moon server lacks hardware H.264 encoder (`h264_v4l2m2m`)

### S3Diff / torch.compile
- `torch.compile(mode="reduce-overhead")` uses CUDA graphs which break on `de_net.py`'s `torch.stack` across invocations — switched to `mode="default"`, then removed entirely
- Triton backend unsupported on Blackwell (sm_120) — `torch.compile` removed from upscaling path
- DCVC custom CUDA extensions cannot be compiled for sm_120 (CUDA 12.0 nvcc doesn't support Blackwell) — suppressed warning via `SUPPRESS_CUSTOM_KERNEL_WARNING=1`

---

## Quality Issues & Tuning

- Restored video showed heavy grain artifacts (color noise across entire frame)
- Root cause: `t_start=500` adds ~50% noise, `ddim_steps=20` insufficient to fully denoise
- Fix: reduced `t_start: 500 → 100` and `ddim_steps: 20 → 10` in `configs/gpu/restoration.yaml`
- At `t_start=100`, only ~10% noise injected — model stays much more faithful to input

---

## Next Steps

- Evaluate restored+upscaled output quality with new `t_start=100` settings
- Compare: original → decompressed → restored → upscaled side by side
- Compute PSNR/SSIM/LPIPS metrics for paper tables
- Generate qualitative figures using `visualizations/qualitative_comparison.py`
- Consider retraining Model R with more epochs if restoration quality still insufficient
