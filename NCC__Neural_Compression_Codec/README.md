# ICTAI 2025 Paper - IEEEtran Format

This directory contains the LaTeX source for the paper submission to IEEE ICTAI 2025.

## Build

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Requires a standard TeX Live installation. The `IEEEtran.cls` file is included in this directory.

## Files

- `main.tex` - paper source (IEEEtran conference format)
- `preamble.tex` - shared macros (`\TODO`, `\todo`, `\red`)
- `References.bib` - bibliography
- `IEEEtran.cls` - IEEE conference class
- `graphics/` - figures (pipeline_overview.pdf, qualitative_comparison.pdf)

## Generating figures

```bash
python scripts/gen_pipeline_fig.py
python scripts/gen_qualitative_fig.py \
    --original  outputs/pipeline/original.mp4 \
    --decomp    outputs/pipeline/decompressed.mp4 \
    --restored  outputs/pipeline/restored.mp4 \
    --frame     30 --roi 0.35 0.2 0.3 0.4
```
