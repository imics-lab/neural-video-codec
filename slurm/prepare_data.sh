#!/usr/bin/env bash
#SBATCH --job-name=nvc_data_prep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/data_prep_%j.out
#SBATCH --error=logs/slurm/data_prep_%j.err

# ── Setup ────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."          # project root

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nvc

mkdir -p logs/slurm

echo "=============================="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=============================="

# ── Restoration data prep (slow — compress + decompress each video) ──────────
echo "[1/3] Preparing restoration pairs ..."
python data_prep/prepare_restoration.py \
    --videos dataset \
    --output data/restoration_pairs \
    --config configs/gpu/compression.yaml \
    --patch-size 256 \
    --patches-per-frame 2

# ── Enhanced SR pairs (degraded LR -> clean HR for upsample+denoise+sharpen) ─
echo "[2/3] Preparing enhanced SR pairs from restoration data ..."
python data_prep/prepare_sr_enhanced.py \
    --restoration-dir data/restoration_pairs \
    --output data/sr_enhanced_pairs \
    --scale 2

# ── Plain SR data prep (optional — clean bicubic pairs for comparison) ────────
echo "[3/3] Preparing plain SR pairs (optional baseline) ..."
python data_prep/prepare_sr.py \
    --videos dataset \
    --output data/sr_pairs \
    --scale 2 \
    --patch-size 256 \
    --patches-per-frame 2

echo "Data preparation complete."
