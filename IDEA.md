# Neural Video Codec — IDEA & Design Questions

This file captures open questions before we write a single line of code.
Answer each section, then we will iterate until the design is clear.

---

## Round 1 Questions

---

### A. Domain & Use Case

1. **What domain is this for?**
   The reference project targets wildlife monitoring. Is this for the same domain, or something else (security cameras, sports broadcast, medical imaging, autonomous driving, etc.)?

Same domain
2. **What should YOLO detect?**
   - Same animal classes as the reference? Custom classes? A specific COCO subset?
   - Should detection drive ROI selection (compress detected regions at higher quality), or is detection purely a downstream consumer of the decoded video?

Any animal, and yes ROI areas should be compressed least
3. **What is the end-to-end user story?**
   e.g. "Edge camera records raw video → compress on-device → send over low-bandwidth link → decompress → upscale → detect objects → save annotated video + JSON detections"

Edge camera records raw video → smart compress on-device (ROI regions are less compressed and bounding boxes and locations are sent to server to decompress and upscale) → send over low-bandwidth link → decompress → upscale →
---

### B. DCVC Compression / Decompression

4. **Same dual-stream ROI/BG separation as the reference?**
   The reference encodes detected regions at high quality (QP 63) and background at low quality (QP 25) in separate bitstreams. Do you want to keep this, simplify to a single stream, or change the quality split?

What whould be effect of single compared to separate?
5. **Frame removal / keyframe selection — keep it?**
   The reference uses a motion-based dual-timeline to drop redundant frames before compression. Do you want this, or should every frame be compressed?

For simplicity lets compress each frame
6. **Bitrate / quality targets?**
   Any constraints (e.g. "must fit 1 Mbps uplink", "target PSNR > 35 dB")?

No, but we want to see in the end how much compression we can add to be able to decompress+upscale it good enough.
7. **I-frame model vs P-frame model?**
   The reference uses both DCVC I-frame and P-frame models (CVPR 2025 checkpoints). Same checkpoints, or are you open to earlier ones?

Lets use same ones
8. **Compressed output format?**
   The reference packs everything into a ZIP archive with JSON metadata. Keep that, or prefer a container format like MKV, custom binary, etc.?

Zip is just fine
---

### C. YOLO Detection

9. **Which YOLO version?**
   The reference uses YOLOv9-c. Do you want v9, v8 (Ultralytics), v10, v11, or YOLO-World (open-vocabulary)?

Yolo 11
10. **Pre-trained or fine-tuned?**
    Use off-the-shelf COCO weights, a domain-specific checkpoint you already have, or is training pipeline included in scope?

COCO weights
11. **Detection timing in the pipeline — which of these?**
    - A. Detect → compress (detections inform ROI for encoding)
    - B. Compress → decompress → detect (detection is post-processing only)
    - C. Both: detect before compression for ROI selection, detect again after upscaling for final annotations
    - D. Other?

A
12. **Detection output format?**
    - Overlaid video (bounding boxes drawn)?
    - JSON/CSV sidecar file per video?
    - Both?
    - Embedded as metadata inside the compressed archive?

JSON with location and size
13. **Tracking?**
    The reference uses tracking (SORT/ByteTrack) to stabilise ROI masks across frames. Do you want per-frame detections only, or multi-object tracking with IDs?

MultiObject
---

### D. Diffusion Upscaling + Temporal Attention

14. **What exactly is "temporal attention" here? Pick the closest:**
    - A. **Frame-conditioned upscaling**: upscale frame N by attending to adjacent frames N-1, N+1 as extra context
    - B. **Temporal self-attention layers inside the UNet**: standard video diffusion approach (e.g. Align-Your-Latents, AnimateDiff style attention across a sliding window of frames)
    - C. **Cross-frame feature warp**: warp features from previous frame using optical flow, then fuse with attention
    - D. **Something else — please describe?**

B
15. **Upscaling factor?**
    4× (same as reference), 2×, or variable?

2x
16. **Input resolution range?**
    e.g. always 360p → 1440p, or variable (480p / 720p → 4K)?

360 -> 1440
17. **Temporal window size?**
    How many frames should the temporal attention look at simultaneously? (1 = no temporal, 3, 5, 7, …?)

3 but we may test different sizes, depending on which one would be fast enough
18. **DDIM inference steps?**
    The reference defaults to 8 steps for speed. Do you want fast inference (≤ 10 steps) or quality-first (50 steps)?

quality first but we will find out which one is the best
19. **Training the diffusion model — in scope?**
    - Yes: need training loop, dataset loading, loss functions, checkpointing
    - No: inference-only; assume a pretrained checkpoint will be provided
    - Maybe: inference first, training later

Yes
20. **If training is in scope:**
    - What dataset? (REDS, VIMEO-90K, custom, synthetic degradations?)
    - Supervised (HR/LR pairs) or self-supervised?
    - Loss function preferences? (pixel L1/L2, perceptual VGG, adversarial GAN, LPIPS?)

custom, 1 hour compilation of videos throughout one day of recording
21. **Memory / speed budget for upscaling?**
    The reference uses 768 px tile + 64 px overlap + Gaussian blend to avoid OOM. Same tiling, or is VRAM not a concern (A100-class GPU)?

tiling

---

### E. Architecture & Code Organisation

22. **Monorepo structure — follow the reference exactly?**
    `src/{compression, decompression, detection, upscaling, pipeline, ...}` with separate `run_*.py` entry points. Or a different layout?

Yes, same

23. **Should compression and detection be tightly coupled or separate modules?**
    Tight coupling: detection feeds directly into the encoder (ROI masks computed inside the compression phase).
    Loose coupling: they are independent modules connected by a config-driven pipeline.

Loose coupling

24. **Config format?**
    YAML (like the reference, with OmegaConf + schema validation), TOML, or plain Python dataclasses?

YAML

25. **Docker / containerisation?**
    - Same GPU Docker setup as the reference?
    - Or bare-metal install instructions only?

GPU docker

26. **Testing expectations?**
    - Unit tests per module (like the reference's `tests/` suite)?
    - Integration / end-to-end tests?
    - Minimal / none for now?

Unit tests per module
---

### F. Infrastructure & Hardware

27. **Target GPU?**
    RTX 3090/4090, A100, Jetson (edge), or cloud GPU (T4/L4)?

Cloud cluster of GPUS

28. **CUDA version?**
    The reference targets CUDA 12.6. Same, or different?

Any

29. **CPU fallback needed?**
    The reference is GPU-only. Is a CPU path required for any component?

If possible

30. **Operating system?**
    Windows (you appear to be on Windows 11), Linux, or both?

Both, or system ambigous 
---

### G. Scope & Priorities

31. **What is the MVP (minimum viable product)?**
    Which of these must work end-to-end first?
    - [ ] Compress a video with DCVC
    - [ ] Decompress and reconstruct
    - [ ] YOLO detection on raw video
    - [ ] YOLO detection on decompressed video
    - [ ] Diffusion upscaling (no temporal)
    - [ ] Diffusion upscaling (with temporal attention)
    - [ ] Full pipeline: compress → decompress → upscale → detect

Full pipeline

32. **Anything explicitly out of scope?**
    e.g. "no training code", "no Docker", "no Windows support", "no frame interpolation"

-
33. **Any existing code, checkpoints, or datasets you already have that we must reuse or integrate?**

Main goal is to have 2 distinct parts Compressor/Decompressor and upscaler and couple them together. In the end we want to train maybe even 2 models one that would map decompressed to original quality and another one that would just upscale(we don't have a ground truth for that(or we could simulate one by intentionally downscaling it prior))

34. **Timeline / urgency?**
    Any hard deadlines or milestone dates we should factor into the design (e.g. "need a demo by end of month")?

Need it by yesterday so yeah...
---

## How to Answer

Feel free to answer inline below each question, or replace the question with your answer.
Once you've replied, I'll do another round of clarifying questions if needed before writing any code.

---

## Round 2 Questions — Follow-ups on your answers

---

### A. Clarifications from Round 1

**A3 — user story was cut off.** It ends at "decompress → upscale →". What happens after upscaling?
Options:
- Save annotated video + JSON detections (run YOLO again post-upscale)?
- Save upscaled video only (no second detection pass)?
- Pass to downstream system?

Save upscaled video only
**A-new — edge device capability.** You said detection runs on the edge camera before compression. YOLO 11 needs a capable GPU. What is the edge device — a workstation with an RTX GPU, a Jetson Orin, or something else? This affects whether we can run YOLO 11 or need a lighter model on-device.

Jetson Orin

---

### B. Single vs Dual-Stream (Q4 follow-up)

Here is the trade-off you asked about:

| | Single stream | Dual stream (ROI + BG) |
|---|---|---|
| Compression ratio | Lower — whole frame encoded at one QP | Higher — BG encoded aggressively, ROI gets high quality |
| Bitrate savings | Modest | ~30–50% at same ROI quality |
| Complexity | Simple | Moderate — two bitstreams, needs ROI masks at decode time |
| Decompressor | One DCVC pass | Two DCVC passes + compositing |


**Q4b: Given this, do you want dual-stream (recommended, mirrors the reference, better bitrate) or single-stream (simpler code)?**

Then dual stream

---

### C. Resolution inconsistency (Q15 vs Q16)

You said **2× upscaling** (Q15) but **360p → 1440p** (Q16). That is actually **4×** (360 × 4 = 1440).

**Q15b: Which is the intended target?**
- A. 2× upscaling (360p → 720p, or 720p → 1440p)
- B. 4× upscaling (360p → 1440p)
- C. The upscaler should handle both stages: first 2× artifact-removal model, then 2× super-resolution model = effective 4×

This matters a lot for the model architecture.

Restoration no upscaling, SR upscale 2x

---

### D. Two-Model Architecture (Q33 follow-up)

You described two distinct models. Let me name them so we're aligned:

- **Model R (Restoration)**: input = decompressed frames (artifacts from DCVC), target = original pre-compression frames. Training pairs generated automatically from raw video (compress → decompress → pair with original).
- **Model S (Super-resolution)**: input = restoration output, target = HR frames. No real ground truth; simulate by intentionally downscaling originals, then train to recover them.

**D1: Should Model R and Model S be the same architecture (one UNet handles both tasks), or two separate models?**
- Same model: simpler, but task objectives differ (deblocking vs super-resolution)
- Two separate models: more params, more memory, but each specialises

Two separate models

**D2: Should both models have temporal attention (AnimateDiff-style sliding window)?**
Or only Model S (upscaler) gets temporal attention, and Model R (restoration) is frame-by-frame?

Restoration should have temporal attention and upscaler shouldn't

**D3: Pipeline order — which of these?**
- A. Decompress → Model R (restore) → Model S (upscale) → output
- B. Decompress → single combined model (restore + upscale) → output
- C. Decompress → Model S (upscale, Model R is optional / trained separately) → output

A

---

### E. Training Details

**E1: Raw video resolution from the camera?**
If the camera records 4K, we can generate:
- LR pairs for Model S by downscaling 4K → 1080p or 720p
- Restoration pairs for Model R by compressing/decompressing the footage
What is the native camera resolution?

1080p

**E2: Do you want the training script to auto-generate training pairs from raw video, or will you prepare the dataset externally?**
Auto-generation would: extract frames, simulate degradation (downscale for S, compress/decompress for R), save as HDF5 or folder of PNG pairs.

Auto-generation, I have a folder with a lot of videos and separate code (data_prep) should take path and create original and 

**E3: Multi-GPU training (DDP)?**
You said cloud cluster of GPUs. Do you want distributed training with `torch.distributed` / DDP from the start, or single-GPU training first with DDP as a later addition?

Single-GPU for starters

**E4: Loss function for Model R (restoration)?**
- Pixel loss only (L1/L2) — fast to train, tends to blur
- Pixel + perceptual (VGG) — sharper, more detail
- Pixel + perceptual + LPIPS — highest quality, slower
Any preference, or leave it configurable?

Pixel + perceptual (VGG)

---

### F. Scope confirmation

**F1: Frame interpolation — in or out?**
The reference includes AMT frame interpolation at decompression time to recover dropped frames. Since you said "compress every frame", there are no dropped frames to interpolate. But for future use, should the interpolation module still be included as a stub, or dropped entirely?

None frames are dropped so no interpolation is needed

**F2: Evaluation metrics — do you want automatic quality metrics computed after each pipeline run?**
e.g. PSNR, SSIM, LPIPS between output and original (when original is available for comparison). This is useful for the "how much compression can we add?" experiment you mentioned.

Yes

**F3: Do you want a benchmark/experiment script** that sweeps DCVC quality levels (QP values) and reports quality metrics, so you can find the best compression/quality operating point?

Yes

---

## Round 3 Questions — Final Blockers

Only 7 questions left. After these we code.

---

**R3-1: E2 answer was cut off.**
You wrote: "Auto-generation, I have a folder with a lot of videos and separate code (data_prep) should take path and create original and…"
Two sub-questions:
- a) What format should the generated pairs be saved as? Options: folder of PNGs (easy to inspect), HDF5 (fast I/O, one file), LMDB (fast random access for large datasets). Recommendation: PNG folder for flexibility.
- b) Is there existing `data_prep` code you want us to build on, or should we write it from scratch?

a) PNGs. b) No
---

**R3-2: Model S input/output resolution.**
Camera is 1080p. Pipeline is: compress 1080p → decompress 1080p → Model R (1080p → 1080p, artifact removal) → Model S (2× upscale).
So Model S output is **2160p (4K)**. Is that correct, or should Model S upscale from a lower resolution (e.g. the edge could also capture at 540p to save even more bandwidth, then upscale to 1080p)?

Yes, but also as we discussed we might consider downscaling resolution prior to feeding it to our pipeline so that we'll have ground truth
---

**R3-3: YOLO11 variant for Jetson Orin.**
YOLO11 comes in 5 sizes. Jetson Orin AGX has ~275 TOPS but still much slower than a desktop GPU:

| Variant | Params | Typical GPU fps | Jetson estimate |
|---|---|---|---|
| n (nano) | 2.6M | ~100 fps | ~30 fps |
| s (small) | 9.4M | ~80 fps | ~15 fps |
| m (medium) | 20M | ~50 fps | ~8 fps |

Which variant? Or should we make it configurable and document the trade-off?

Jetson will only run compression, decompression and later steps all happen on Cloud GPU

---

**R3-4: Loss function for Model S (super-resolution)?**
We defined Model R loss as Pixel + VGG. For Model S:
- Same (Pixel + VGG)?
- Add LPIPS on top (Pixel + VGG + LPIPS) — SR models typically benefit from LPIPS
- Adversarial (GAN) loss — can produce very sharp results but training is less stable
Recommendation: Pixel + VGG + LPIPS (no GAN) for stable training with good perceptual quality. OK?

Add LPIPS on top

---

**R3-5: Model R temporal attention — confirm the window is 3 frames (same as general setting in Q17)?**
Model R processes decompressed frames with temporal attention window=3 (current + 1 before + 1 after). Model S processes each frame independently (no temporal). Confirm?

Yes

---

**R3-6: ZIP transmission — any requirements?**
The edge camera produces a compressed ZIP that gets sent to the server.
- Any authentication / encryption needed (or out of scope)?
- Should the pipeline include an upload script (`run_upload.py`) or is transmission handled externally?
- Max archive size budget? (affects how we set default QP levels)

No, we test it on one machine for start

---

**R3-7: Jetson / edge Docker image — separate from server image?**
The server runs large models (DCVC + Model R + Model S) on cloud GPUs. The edge runs YOLO11 on Jetson Orin. These need different Docker images (ARM64 vs x86-64, different CUDA versions).
- Should the repo include both a `Dockerfile.edge` (Jetson, ARM64, YOLO only) and `Dockerfile.server` (cloud GPU, full pipeline)?
- Or is the edge deployment out of scope for this codebase (handled separately)?

Imagine we don't have jetson yet, we need to test our pipeline on prerecorded videos

---

## CONFIRMED DESIGN SPEC (locked — code starts here)

### Pipeline (single machine, prerecorded video)

```
run_compress.py
  └─ preprocessing:   extract frames from video
  └─ detection:       YOLO11 (configurable variant, default n) + ByteTrack → ROI masks + detections.json
  └─ roi_masking:     build per-frame binary masks from bounding boxes
  └─ compression:     DCVC dual-stream (ROI @ high QP, BG @ low QP) → ZIP archive

run_decompress.py
  └─ decompression:   unpack ZIP → DCVC dual-stream decode → composite frames

run_restore.py
  └─ restoration:     Model R — temporal-attention diffusion UNet, window=3, same resolution
                      Loss: Pixel (L1) + VGG perceptual

run_upscale.py
  └─ upscaling:       Model S — standard diffusion UNet, 2× SR (1080p → 2160p), frame-by-frame
                      Loss: Pixel (L1) + VGG perceptual + LPIPS

run_pipeline.py       chains all four above end-to-end
```

### Training

```
data_prep/
  prepare_restoration.py   videos folder → DCVC compress/decompress pairs → PNG pairs (degraded / original)
  prepare_sr.py            videos folder → bicubic downsample → PNG pairs (LR 540p / HR 1080p)

train_restoration.py       trains Model R (single GPU, DDP later)
train_sr.py                trains Model S (single GPU, DDP later)
```

Evaluation mode (ground truth available): pass `--downscale-input 0.5` to `run_compress.py` to simulate 540p input → pipeline outputs 1080p → compare against original.

### Evaluation / Benchmarking

```
eval_metrics.py       PSNR, SSIM, LPIPS between pipeline output and original
benchmark_qp.py       sweep DCVC QP levels, report metrics table
```

### Architecture

| Component | Architecture | Temporal | Resolution |
|---|---|---|---|
| Model R | Diffusion UNet + temporal self-attention | Yes, window=3 | 1:1 (no change) |
| Model S | Diffusion UNet (standard) | No | 2× upscale |

### Repo structure

```
src/
  compression/        dcvc_encoder.py, phase_compress.py
  decompression/      dcvc_decoder.py, phase_decompress.py
  detection/          yolo_detector.py (YOLO11 + ByteTrack)
  restoration/        _network.py, _diffusion.py, _blocks.py, restorer.py, phase_restore.py
  upscaling/          _network.py, _diffusion.py, _blocks.py, upscaler.py, phase_upscale.py
  roi_masking/        roi_masking.py
  preprocessing/      frame_extractor.py
  postprocessing/     video_assembler.py
  pipeline/           config_schema.py
data_prep/
training/
configs/gpu/
  compression.yaml, decompression.yaml, restoration.yaml, upscaling.yaml, pipeline.yaml
tests/
docker/
  Dockerfile.gpu, docker-compose.gpu.yaml, requirements.gpu.txt
models/               model weights (downloaded separately)
outputs/
scripts/              sanity checks, model download helpers
run_compress.py
run_decompress.py
run_restore.py
run_upscale.py
run_pipeline.py
eval_metrics.py
benchmark_qp.py
```

### Key decisions made

- No frame interpolation (every frame compressed)
- No upload script (single machine testing)
- One Docker image (GPU server, x86-64)
- YOLO11 variant configurable (default `yolo11n.pt`)
- Training patch size: 256×256
- DDIM steps: configurable (default 50 quality-first, with fast preset at 8)
- Temporal window size: configurable (default 3)
- CPU fallback: best-effort where PyTorch allows (DCVC C++ kernels are GPU-only)
- OS: paths use `pathlib.Path` throughout for Windows/Linux compatibility
