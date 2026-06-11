# data_prep/ - Dataset Preparation

These scripts take raw wildlife video clips and produce training-ready PNG pair
datasets. Run them once before training.

## prepare_restoration.py

Generates (degraded, original) pairs for RestoreUNet training.

For each source video it:
1. Extracts frames as PNGs into `original/`.
2. Compresses the video with DCVC at the same settings used during inference
   (QP_ROI=63, QP_BG=5).
3. Decompresses the result and saves frames into `degraded/`.

Filenames in `original/` and `degraded/` are identical so the dataloader pairs them
by name.

```bash
python data_prep/prepare_restoration.py \
    --videos  /path/to/raw/clips \
    --out     data/restoration_pairs \
    --config  configs/gpu/compression.yaml
```

Options:
- `--videos PATH` - directory of MP4 files (scanned recursively)
- `--out PATH` - output directory (creates `original/` and `degraded/` subdirs)
- `--workers N` - parallel video processing (default: 2)
- `--max-frames N` - cap frames per clip to limit dataset size

Expected output size: roughly 120-150 KB per frame pair (1280x720 PNG).

## prepare_sr.py

Generates (LR, HR) pairs for super-resolution training by bicubic downscaling.

```bash
python data_prep/prepare_sr.py \
    --videos  /path/to/raw/clips \
    --out     data/sr_pairs \
    --scale   2
```

The `hr/` folder holds the original frames; `lr/` holds the 2x-downscaled versions.

## prepare_sr_enhanced.py

Like `prepare_sr.py` but uses the RestoreUNet output as the LR input instead of
bicubic downscaling. Use this to train the SR model on realistic restoration-stage
output rather than synthetic degradation.

```bash
python data_prep/prepare_sr_enhanced.py \
    --videos      /path/to/raw/clips \
    --out         data/sr_enhanced_pairs \
    --restore-cfg configs/gpu/restoration.yaml
```

## Data layout expected by training scripts

```
data/restoration_pairs/
  original/   frame_00000.png  frame_00001.png  ...
  degraded/   frame_00000.png  frame_00001.png  ...

data/sr_enhanced_pairs/
  hr/         frame_00000.png  frame_00001.png  ...
  lr/         frame_00000.png  frame_00001.png  ...
```

Filenames must match between paired directories. The dataloader sorts them
lexicographically and pairs by position; mismatches cause corrupted training pairs.
