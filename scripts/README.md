# Phase 0 Sanity Checks

Run these checks before any experiment runs.

## 0) Model bootstrap

Compression-stage models:

```bash
python scripts/download_compression_models.py --repo owner/repo --tag artifact-v1
```

Decompression-stage models:

```bash
python scripts/download_decompression_models.py --repo owner/repo --tag artifact-v1
```

Notes:
- Both scripts save files into `gpu/models/` by default.
- If the GitHub repo is private, set `GITHUB_TOKEN` before running them.
- Release asset names must exactly match the filenames listed in `models/models.manifest.json`.
- For a paper artifact, use a fixed release tag, not `latest`.

## 1) ROI detection sanity

```bash
python scripts/test_roi_detection.py --video data/tune/video.mp4
```

What to inspect:
- `outputs/sanity_checks/roi_detection/roi_overlay.mp4`
- Animals should be detected with plausible bounding boxes.

## 2) Frame-removal sanity

```bash
python scripts/test_frame_removal.py --video data/tune/video.mp4
```

What to inspect:
- `outputs/sanity_checks/frame_removal/frame_drop_overlay.mp4`
- `outputs/sanity_checks/frame_removal/roi_kept_preview.mp4`
- `outputs/sanity_checks/frame_removal/bg_kept_preview.mp4`
- Dropped frames should look redundant relative to kept anchors.

## 3) Compression sanity + reproducibility

```bash
python scripts/test_compression.py --video data/tune/video.mp4 --repeat 2
```

Checks performed:
- Compressed archive size is smaller than source video.
- Same clip + same config run twice gives identical metrics/hashes.

If repeated runs differ, fix reproducibility before continuing.

## 4) Decompression sanity

```bash
python scripts/test_decompression.py outputs/video.zip --config configs/gpu/decompression.yaml
```

Checks performed:
- `run_decompression.py` succeeds with the provided archive/config.
- Reconstructed video is readable and non-empty.
- Output frame size matches archive metadata.
- Output frame count matches expected timeline (or `--max-frames` cap).
- Optional repeat mode validates deterministic reconstructed pixels.

## 5) Full end-to-end visual check

```bash
python run_compression.py data/tune/video.mp4 --config configs/gpu/compression.yaml --output outputs/video.zip
python run_decompression.py outputs/video.zip --config configs/gpu/decompression.yaml --output outputs/sanity_checks/reconstructed/video.mp4
```

What to inspect:
- Reconstructed video quality and temporal consistency.
- ROI regions should stay coherent without obvious artifacts.

## Recommended protocol

Repeat this process on 2-3 clips (day/night, single/multi-animal) before running real experiments.

## Custom config comparisons

For controlled profile comparisons, create additional compression YAML files by copying `configs/gpu/compression.yaml` and changing the dual-timeline intervals or QP values you want to test.

Example:

```bash
python scripts/test_frame_removal.py --config configs/gpu/compression_custom.yaml --video data/tune/video.mp4
```

## Notes

The scripts in this folder are developer sanity checks only. The public GPU runtime path is:
- `run_compression.py`
- `run_decompression.py`

Path handling:
- You can pass paths relative to `gpu/` like `outputs/...` and `data/...`.
- If you launch the scripts from the repo root, `gpu/...`-prefixed paths are also accepted.

Debug artifacts:
- `test_roi_detection.py` and `test_frame_removal.py` write visual debug outputs under `outputs/sanity_checks/...`.
- `test_compression.py` and `test_decompression.py` store captured runner output tails in each run record inside `summary.json`.
