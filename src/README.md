# src/ - Source Modules

Each subdirectory is an independent, replaceable module. Stage scripts in the repo
root (`run_compress.py`, `run_decompress.py`, etc.) import from these modules;
`run_pipeline.py` chains them in order.

## Module overview

### compression/
Dual-stream DCVC encoder. Reads a video, runs detection to get ROI masks, then
encodes the ROI pixels and background pixels as two separate DCVC bitstreams at
different quality levels. Produces a ZIP archive containing both bitstreams, the ROI
masks, detection metadata, and a config snapshot.

Key files:
- `dcvc_encoder.py` - wraps the DCVC I/P-frame models for a single stream
- `phase_compress.py` - orchestrates detection, masking, dual-stream encoding, ZIP packing

### decompression/
Dual-stream DCVC decoder. Unpacks the ZIP, decodes both streams, and composites them
into full-resolution frames using the stored masks. Also supports ffmpeg-backed
decoding for H.264/HEVC/AV1 streams (used in codec comparison experiments).

Key files:
- `dcvc_decoder.py` / `ffmpeg_decoder.py` - stream-level decoders
- `reconstruct.py` - compositing logic (mask-weighted blend of ROI and BG frames)
- `roi_bg_decompress.py` - decompresses a single stream from the ZIP
- `common.py` - shared helpers (frame index selection, stream metadata)

### detection/
Animal detection and tracking. Runs MegaDetector (YOLOv9c backbone) on keyframes and
KLT optical-flow tracking between keyframes to generate per-frame bounding boxes with
track IDs. Outputs `detections.json` embedded in the archive.

Key files:
- `yolo_detector.py` - MegaDetector inference
- `tracker.py` - KLT/ByteTrack frame-to-frame box propagation

### roi_masking/
Converts detection bounding boxes to binary pixel masks. Applies padding, dilation,
and Gaussian feathering (for composite blending). Provides both hard masks (for stream
splitting during encoding) and soft masks (for compositing at decode time).

Key files:
- `roi_masking.py` - mask builder and feathering utilities

### restoration/
RestoreUNet: a temporal-attention diffusion U-Net that removes DCVC compression
artifacts. Processes a sequence of decompressed frames and outputs clean frames at
the same resolution. Uses img2img DDIM starting from t_start=50 (very low noise
level), so output is faithful to the input without hallucinating content.

Architecture: 3-level U-Net (channels 128/256/512), 4 residual blocks per level,
AnimateDiff-style temporal self-attention over a 3-frame window, trained with L1
noise-prediction loss only.

Key files:
- `_network.py` - RestoreUNet architecture
- `_diffusion.py` - GaussianDiffusion schedule (cosine, T=1000)
- `_blocks.py` - ResBlock, SpatialAttention, TemporalAttention
- `restorer.py` - inference wrapper (sequence batching, tiling, DDIM loop)
- `phase_restore.py` - stage entry point

### upscaling/
S3Diff 2x super-resolution. Takes restored frames (same resolution as input) and
outputs 2x upscaled frames. Frame-by-frame (no temporal attention in this stage).
Optional stage; can be skipped via pipeline config.

Key files:
- `upscaler.py` - inference wrapper
- `phase_upscale.py` - stage entry point

### preprocessing/
Extracts frames from a video file into a list of numpy arrays (or a temp folder of
PNGs). Used at the start of the compression stage.

### postprocessing/
Assembles a list of frames back into an MP4 or MKV video. Used at the end of each
stage that produces video output.

### pipeline/
Config schema (OmegaConf-based) and stage orchestration used by `run_pipeline.py`.
`config_schema.py` defines the full configuration dataclass so that YAML files are
validated on load.

## Module boundary: edge vs. server

```
Edge device (Jetson Orin Nano)     Server (cloud GPU)
--------------------------------   --------------------------------
src/preprocessing/                 src/decompression/
src/detection/                     src/restoration/
src/roi_masking/                   src/upscaling/
src/compression/                   src/postprocessing/
```

The ZIP archive is the interface. The edge produces it; the server consumes it. No
code changes are needed to deploy each side independently.
