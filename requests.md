# Requests to Finalize the NCC Paper

Everything below is needed before the paper can be submitted. Items are grouped by section and roughly ordered by importance.

---

## 1. Figures (Blocking — Paper Cannot Compile Without These)

### 1.1 `graphics/pipeline.pdf` — System Overview Diagram
The paper references this in Figure 1 (caption: "NCC pipeline overview"). It should be a horizontal block diagram showing the four stages:

```
Video In → [ROI Detection: MegaDetector + KLT] → [Dual-Stream DCVC Encoder]
         → [Bitstream Archive: roi.bin + bg.bin + meta.json]
         → [DCVC Decoder + Feathered Blend] → [RestoreUNet DDIM] → [S3Diff SR] → Video Out
```

Show the soft mask M_t flowing from detection into both encoder and blending. Show QP_roi=63 and QP_bg=25 labels on the two streams. This can be a TikZ figure (like `graphics/icgd_overview.tex` was for the old paper) or exported from a diagram tool.

### 1.2 `graphics/qualitative.pdf` — Side-by-Side Frame Comparison
A wide figure (full `\linewidth`) showing one representative frame through the pipeline. Columns:
1. Original (uncompressed)
2. DCVC-Uniform at matched bitrate
3. NCC dual-stream (post-blend, pre-restoration)
4. NCC after RestoreUNet
5. NCC after S3Diff

**Also include a $4\times$ magnified crop inset** of the animal region in each column showing fine texture differences (fur, feathers). Pick a frame where the differences are clearly visible (compression blocking visible in column 2, restored detail in column 5).

Ideally add a second row with a second example clip (different species or lighting condition).

### 1.3 (Optional but Strongly Recommended) `graphics/temporal.pdf` — Temporal Consistency Plot
A figure showing 3–4 consecutive frames of the restored video to demonstrate temporal coherence. Could be a $3 \times 4$ grid of crops (rows = methods: DCVC-Uniform, NCC no-temporal-attn, NCC full; columns = time steps $t$, $t+1$, $t+2$, $t+3$).

---

## 2. Dataset Description (Section 4.1)

Fill in `\TODO{}` placeholders in Section 4.1 with:

- **Number of video clips** and **total duration** (minutes or hours of footage)
- **Source**: camera trap deployment, curated YouTube clips, specific public dataset, etc.
- **Resolution** (e.g., 1920×1080) and **frame rate** (e.g., 30 fps)
- **Animal species** represented (list the COCO animal classes or specific species)
- **Train / val / test split** (how many clips in each, any stratification by species/scene?)
- **Mean ROI area fraction** $\bar{\rho}$ (average fraction of frame area covered by animal bounding boxes across the test set — needed for the Discussion section bitrate overhead calculation)
- Whether any existing public dataset is used (e.g., iWildCam, Caltech Camera Traps) so it can be cited

---

## 3. Quantitative Results (Tables 1 and 2 — Core of the Paper)

Run the following evaluations and fill in both tables:

### Table 1: Main Results (`tab:main_results`)

For each of the 6 methods (HEVC, DCVC-Uniform, DCVC+Real-ESRGAN, NCC-no-restore, NCC-no-SR, NCC-full), at **matched effective bitrate** (same bpp), report:
- **Bitrate** (bpp or Mbps)
- **PSNR** (dB) on ROI region
- **SSIM** on ROI region
- **LPIPS** on ROI region
- **BD-rate** relative to HEVC (compute over 4 quality operating points per method using the `bd_rate` tool)

Ideally report mean ± std over the test clips.

Also run the **whole-frame** versions of PSNR/SSIM/LPIPS (can go in a supplementary table or a footnote).

### Table 2: Ablation (`tab:ablation`)

For the following configurations (all at the same target bitrate as Table 1), report PSNR, SSIM, LPIPS on ROI region:

| Configuration | Notes |
|---|---|
| Dual-stream only (no restoration, no SR) | Baseline for ablation |
| + Restoration, N_ddim=1 | Fast setting |
| + Restoration, N_ddim=50 | Quality setting |
| + SR (S3Diff) after N_ddim=50 | Full pipeline |
| Full pipeline, w/o temporal attention | Replace temporal attn with per-frame spatial attn; keep everything else |

The last row (w/o temporal attention) directly validates the temporal attention contribution and is important for the WACV reviewers.

---

## 4. Temporal Consistency Metric (Section 5.4)

Compute and report the **optical-flow warping error** (also called "temporal flickering" metric) between consecutive restored frames for:
- DCVC-Uniform (no restoration)
- NCC RestoreUNet without temporal attention
- NCC RestoreUNet with temporal attention (full)

Standard formula: for frames $\hat{F}_t$ and $\hat{F}_{t+1}$, compute optical flow $w_{t \to t+1}$, warp $\hat{F}_{t+1}$ back to frame $t$, and compute mean absolute error against $\hat{F}_t$ in non-occluded regions. A lower number = more temporally consistent. This can be computed with `raft-large` flow or even `cv2.calcOpticalFlowFarneback` for speed.

Report as mean over all consecutive frame pairs in the test set.

---

## 5. Reference Verification

The following references in `References.bib` need to be verified or completed:

- **`Li_2025_DCVC`**: The entry currently has a placeholder note `(TODO: verify exact title and page numbers)`. Find the correct CVPR 2025 citation for the DCVC model you are using (the one whose weights are in `models/cvpr2025_image.pth.tar` and `models/cvpr2025_video.pth.tar`). The paper title, authors, and page numbers must be accurate.

- **`Chadha_2020_ROI`**: This is a placeholder. Replace it with an actual reference for ROI-based neural or traditional video coding. Good candidates:
  - Xia et al., "Object-Based Coding of Video Using Robust Segmentation," or
  - A paper specifically on learned ROI video coding with detection masks.
  - If no good reference exists, the sentence citing it can be reworded to cite a standard overview.

- **`Beery_2019_MegaDetector`**: Confirm the arXiv number `1907.06772` is the correct MegaDetector paper. The v6 / YOLOv9c checkpoint may have a separate or updated citation — check if there is a newer MegaDetector v5/v6 paper or a GitHub/tech-report citation that covers the specific checkpoint used (`models/MDV6-yolov9-c.pt`).

---

## 6. Discussion Numbers (Section 6)

- **Mean ROI area fraction**: Fill `\TODO{$\rho$}\%` in the Discussion paragraph "Bitrate Overhead" with the actual measured value from the test set (see Dataset request above).
- **Timing**: Confirm the statement "approximately 3 minutes per 20-second video at N_ddim=1" with an actual timed run on the moon server (`time python run_pipeline.py --video dataset/bird1.mp4 ... --verbose`). Also record the N_ddim=50 time so the 50× claim can be verified.

---

## 7. Build Check

After filling in figures and running the bibliography, do a full build to catch any compilation errors:

```bash
cd NCC__Neural_Compression_Codec
pdflatex main.tex; bibtex main; pdflatex main.tex; pdflatex main.tex
```

Check that:
- No `\TODO{}` macros appear in the final PDF (they are colored red by the preamble — a quick visual scan suffices)
- All figures render (no missing file errors)
- Bibliography entries resolve without `?` citations
- Page count is within the WACV 8-page limit (references do not count toward the limit)

---

## 8. Paper Metadata (Low Priority, but Required Before Submission)

- **Paper ID**: Replace `\def\wacvPaperID{*****}` with the actual CMT-assigned Paper ID
- **Track**: Uncomment the correct line at the top of `main.tex`:
  - `\usepackage[review,applications]{wacv}` for the Applications track review version, or
  - Keep `\usepackage{wacv}` for camera-ready once accepted
