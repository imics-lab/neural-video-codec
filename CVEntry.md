# CV Entry — Neural Wildlife Camera Trap System

---

## Project Entry (résumé / one-liner)

**Neural Video Compression Pipeline for Edge-Deployed Wildlife Camera Traps** — end-to-end system running on NVIDIA Jetson that detects, compresses, and transmits wildlife footage at a fraction of raw bandwidth, with server-side AI restoration and upscaling.

---

## Full CV Entry (one page, technical role)

### Neural Wildlife Camera Trap with Edge AI Compression
*Personal / Research Project — Ranch Deployment*

Designed and deployed an end-to-end intelligent camera trap system combining edge inference, neural video compression, and generative AI restoration to capture and reconstruct high-quality wildlife footage over a bandwidth-constrained link.

**Edge node (NVIDIA Jetson + camera):**
- Ran YOLOv11 / MegaDetector v5a in real time to detect animal presence; triggered video recording only on positive detections, eliminating idle footage storage
- Built a ROI-aware neural video encoder using DCVC (Deep Contextual Video Compression): detected animal regions encoded at high fidelity (QP 63) while background encoded aggressively (QP 5), producing per-frame dual-stream ZIP archives with embedded bounding-box metadata
- Applied KLT optical-flow tracking between keyframes to maintain smooth ROI masks during rapid motion without re-running full detection every frame
- Packaged compressed archives and transmitted to remote server — achieving significant bandwidth reduction versus raw H.264 at equivalent visual quality on the subject

**Server-side pipeline (modular, each stage independently invocable):**
- **Decompression:** DCVC decode of ROI and background streams with alpha-composited reassembly using feathered soft masks
- **Restoration:** Custom RestoreUNet diffusion model with temporal-window attention; DDIM sampling with tiled inference for arbitrary-resolution frames; ROI-aware blending preserves animal detail while aggressively denoising background artifacts
- **Upscaling:** CogVideoX Video-to-Video (3D spatiotemporal attention, overlapping-chunk processing with linear-ramp blending) for temporally consistent super-resolution; original bicubic-resized frames composited back over ROI regions to prevent generative hallucination of the subject; Wan2.1 I2V available as alternative for generative hallucination comparison

**Key results / highlights:**
- Substantial reduction in data transmitted per event compared to raw video, enabling practical deployment on cellular or satellite uplinks
- Modular Python pipeline (`run_pipeline.py --stages compress decompress restore upscale-cogvideo`) allows any stage to be run independently or swapped
- Full Jupyter notebook demonstrating each stage with PSNR/SSIM metrics across pipeline stages

**Stack:** Python · PyTorch · NVIDIA Jetson · DCVC · Ultralytics YOLOv11 · MegaDetector · Diffusers (CogVideoX, Wan2.1) · OpenCV · HuggingFace Hub

---

## Short version (bullet points for CV body)

- Deployed NVIDIA Jetson camera trap on a working ranch; YOLOv11 triggers recording only on animal detection, eliminating idle storage
- Built ROI-aware DCVC neural video codec: high-fidelity subject stream + aggressively compressed background stream packed into a single archive, reducing transmission bandwidth significantly vs. raw H.264
- Server-side pipeline: DCVC decompression → RestoreUNet diffusion denoiser (temporal attention, tiled DDIM) → CogVideoX V2V super-resolution with ROI-preserving composite
- Entire pipeline modular and independently invocable per stage; benchmarked with PSNR/SSIM at each step

---

## Abstract (for a poster or project page)

Wildlife monitoring at scale is constrained by the cost of transmitting high-resolution video from remote edge devices. We present a neural camera trap system deployed on an NVIDIA Jetson at a ranch that uses on-device animal detection to trigger recording and ROI-aware neural video compression to minimise transmitted data. Detected subject regions are encoded at high quality while the background is compressed aggressively; the resulting dual-stream archive is transmitted to a server where a diffusion-based restoration model removes compression artifacts and a video-to-video upscaler (CogVideoX) reconstructs temporally consistent high-resolution footage. The subject region is protected from generative hallucination by compositing original bicubic-upscaled frames on top of the model output. The modular design allows each stage — compression, decompression, restoration, upscaling — to be run independently or replaced, enabling systematic ablation of each component's contribution to final quality.
