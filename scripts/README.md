# Sanity-Check Scripts

These scripts verify the main pipeline stages on a small input. They are not a replacement for the full thesis evaluation.

## Model Download

Download all required models:

```bash
python scripts/download_models.py
```

Compression-only model download:

```bash
python scripts/download_compression_models.py --repo owner/repo --tag release-tag
```

Decompression-only model download:

```bash
python scripts/download_decompression_models.py --repo owner/repo --tag release-tag
```

Notes:

- scripts save model files into `models/` by default
- `download_models.py` verifies SHA256 checksums against `docs/model_checksums.sha256`
- if the GitHub repo is private, set `GITHUB_TOKEN`
- use a fixed release tag instead of `latest`

## 1. ROI Detection

```bash
python scripts/test_roi_detection.py --video data/test.mp4
```

Inspect:

- `outputs/sanity_checks/roi_detection/roi_overlay.mp4`
- whether animal boxes look plausible and temporally stable

## 2. Frame Removal

```bash
python scripts/test_frame_removal.py --video data/test.mp4
```

Inspect:

- `outputs/sanity_checks/frame_removal/frame_drop_overlay.mp4`
- `outputs/sanity_checks/frame_removal/roi_kept_preview.mp4`
- `outputs/sanity_checks/frame_removal/bg_kept_preview.mp4`
- whether dropped frames look redundant relative to retained anchors

## 3. Compression

```bash
python scripts/test_compression.py --video data/test.mp4 --repeat 2
```

Checks:

- archive is smaller than source video
- repeated runs with the same inputs produce stable outputs and summaries

## 4. Decompression

```bash
python scripts/test_decompression.py outputs/video.zip --config configs/gpu/decompression.yaml
```

Checks:

- decompression completes successfully
- reconstructed video is readable and non-empty
- output size and frame count match archive metadata expectations

## 5. Full Visual End-To-End Check

```bash
python run_compression.py data/test.mp4 --config configs/gpu/compression.yaml --output outputs/video.zip
python run_decompression.py outputs/video.zip --config configs/gpu/decompression.yaml --output outputs/sanity_checks/reconstructed/video.mp4
```

Inspect:

- reconstructed visual quality
- ROI coherence
- major temporal errors or reconstruction failures

## Typical Order

1. verify model downloads
2. run ROI detection
3. run frame removal
4. run compression
5. run decompression
6. inspect a small set of day and night clips visually

## Path Conventions

Run from the repository root.

Expected local directories:

- `data/`
- `models/`
- `outputs/`

The scripts write debug outputs under `outputs/sanity_checks/`.
