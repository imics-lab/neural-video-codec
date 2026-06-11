# Neural Video Codec (NVC)

ROI-aware neural video compression with temporal-attention diffusion restoration.
Built for bandwidth-constrained wildlife camera traps; the paper "ROI-Aware Neural
Video Compression with Temporal-Attention Diffusion Restoration for Wildlife Camera
Traps" targets WACV 2026.

## What this system does

```
Edge device                            Server (cloud GPU)
-----------                            ------------------
raw video
  -> MegaDetector + KLT tracking
  -> dual-stream DCVC encode           ZIP archive
       ROI   stream (QP 63 = high)  ------------->  DCVC decode
       BG    stream (QP 5  = low)                   composite frames
  -> ZIP archive                                    RestoreUNet (artifact removal)
                                                    S3Diff upscale (optional 2x)
                                                    enhanced video
```

Animals occupy a small ROI (<10% of pixels). Sending the ROI at full quality and the
background aggressively saves significant bandwidth while preserving scientifically
relevant subject detail. RestoreUNet removes the blocking and ringing artifacts
introduced by aggressive background compression.

## Quick start

```bash
# 1. Install dependencies
pip install -r docker/requirements.gpu.txt

# 2. Download model weights
python scripts/download_models.py

# 3. Run the full pipeline on a video
python run_pipeline.py --video dataset/cow15.mp4 --config configs/gpu/pipeline.yaml

# 4. Run individual stages
python run_compress.py   --video my_video.mp4   --config configs/gpu/compression.yaml
python run_decompress.py --archive out/my_video.zip
python run_restore.py    --video  out/my_video_decompressed.mp4
python run_upscale.py    --video  out/my_video_restored.mp4
```

## Directory layout

```
src/                    Source modules (see src/README.md)
  compression/          DCVC dual-stream encoder, ROI/BG stream splitting
  decompression/        DCVC decoder, stream compositing
  detection/            MegaDetector + KLT tracking, ROI mask builder
  restoration/          RestoreUNet model, diffusion, DDIM inference
  upscaling/            S3Diff 2x super-resolution
  roi_masking/          Per-frame binary mask utilities
  preprocessing/        Frame extraction from video
  postprocessing/       Frame assembly back to video
  pipeline/             Config schema, stage orchestration

training/               Training scripts (see training/README.md)
  train_restoration.py  Train RestoreUNet (L1 diffusion loss)
  train_sr.py           Train super-resolution model

data_prep/              Dataset preparation (see data_prep/README.md)
  prepare_restoration.py  Auto-generate degraded/original pairs via DCVC
  prepare_sr.py           Auto-generate LR/HR pairs via bicubic downscale
  prepare_sr_enhanced.py  SR pair generation from restored output

configs/gpu/            YAML configuration files (one per stage + pipeline)
scripts/                Utility scripts (see scripts/README.md)
  download_models.py    Download DCVC and S3Diff checkpoints
  sanity_check.py       Verify installation
  gen_latex_tables.py   Generate paper tables from results/ CSVs
  gen_pipeline_fig.py   Generate pipeline overview figure (PDF + PNG)
  gen_qualitative_fig.py  Generate qualitative comparison figure

results/                Experiment output CSVs (created by run_experiments.py)
outputs/                Pipeline output videos (created at runtime)
checkpoints/            Model weight files (downloaded separately)
  restore/              Current VGG-trained restoration checkpoint
  restore_l1/           L1-only restoration checkpoint (recommended)
  sr/                   Super-resolution checkpoint
NCC__Neural_Compression_Codec/  LaTeX source for the WACV 2026 paper
DCVC/                   DCVC codec source (Microsoft, MIT license)
docker/                 Docker configuration for GPU server
tests/                  Unit tests (pytest)
```

## Running experiments (paper results)

```bash
# All experiments (takes several hours on a single GPU)
python run_experiments.py --all --video dataset/cow15.mp4

# Individual experiments
python run_experiments.py --stage-ablation
python run_experiments.py --qp-sweep
python run_experiments.py --ddim-ablation
python run_experiments.py --temporal
python run_experiments.py --rd
python run_experiments.py --codec-comparison

# View collected results
python run_experiments.py --summary

# Generate LaTeX tables from results
python scripts/gen_latex_tables.py
```

## Training the restoration model

```bash
# Generate training pairs (run once; requires DCVC)
python data_prep/prepare_restoration.py --videos /path/to/raw/clips --out data/restoration_pairs

# Train RestoreUNet with L1-only loss (recommended)
python training/train_restoration.py \
    --data      data/restoration_pairs \
    --config    configs/gpu/restoration.yaml \
    --ckpt-dir  checkpoints/restore_l1 \
    --epochs    200 \
    --batch-size 8 \
    --save-every 10 \
    --sample-every 20
```

## Configuration

All behaviour is controlled by YAML files in `configs/gpu/`.

| File | Controls |
|---|---|
| `compression.yaml` | DCVC QP levels, codec backend, detection settings |
| `restoration.yaml` | RestoreUNet checkpoint, t_start, DDIM steps, tiling |
| `upscaling.yaml` | S3Diff checkpoint, scale factor |
| `pipeline.yaml` | Which stages to run, I/O paths |

## FaunaScope edge deployment

This codebase is designed for the FaunaScope project: a Jetson Orin Nano-based
wildlife camera trap with solar power, IR camera, and cellular uplink. Only the
compression stage runs on the edge device; decompression, restoration, and upscaling
all run on the cloud server. See `src/README.md` for the module boundary.

## Docker

```bash
docker compose -f docker/docker-compose.gpu.yaml up
```

## Tests

```bash
pytest tests/ -v
```

## Citation

```bibtex
@inproceedings{Sakevych_2026_NVC,
  author    = {Mykhailo Sakevych and Vangelis Metsis},
  title     = {ROI-Aware Neural Video Compression with Temporal-Attention
               Diffusion Restoration for Wildlife Camera Traps},
  booktitle = {WACV},
  year      = {2026},
}
```
