# scripts/ - Utility Scripts

## download_models.py

Downloads all release-backed model files into `models/`.

```bash
python scripts/download_models.py
```

Downloads: MegaDetector YOLO weights, the deterministic `dcvc_int16` bundle,
and AMT interpolation weights from the configured GitHub Release.

For stage-specific downloads:

```bash
python scripts/download_compression_models.py
python scripts/download_decompression_models.py
```

## sanity_check.py

Verifies the installation: checks CUDA availability, imports all modules, runs a tiny
DCVC encode/decode on a synthetic frame, and checks that vmaf binary is in PATH.

```bash
python scripts/sanity_check.py
```

All checks should print "OK". If any fail, follow the printed instructions before
running experiments.

## gen_latex_tables.py

Reads CSVs from `results/` (written by `run_experiments.py`) and prints LaTeX
`tabular` environments ready to paste into the paper. Run after experiments finish.

```bash
python scripts/gen_latex_tables.py
python scripts/gen_latex_tables.py --results /path/to/results/
```

Generates tables for: stage ablation, rate-distortion baselines, QP sweep, temporal
window ablation, DDIM step ablation, codec comparison.

## gen_pipeline_fig.py

Generates the pipeline overview figure (`graphics/pipeline_overview.pdf` and `.png`)
used as Figure 1 in the paper. No external data needed; draws the diagram with
matplotlib.

```bash
python scripts/gen_pipeline_fig.py
python scripts/gen_pipeline_fig.py --out /custom/path/fig.pdf
```

## gen_qualitative_fig.py

Generates the qualitative comparison figure (Figure 3 in the paper). Needs three
video files: original, decompressed-only, and full-pipeline output.

```bash
python scripts/gen_qualitative_fig.py \
    --original  outputs/pipeline/bird1_original.mp4 \
    --decomp    outputs/pipeline/bird1_decompressed.mp4 \
    --restored  outputs/pipeline/bird1_restored.mp4 \
    --frame     30 \
    --roi       0.35 0.2 0.3 0.4
```

The `--roi` argument is `x y w h` as fractions of the frame (0.0-1.0). Adjust to
frame the animal tightly for the inset crops.
