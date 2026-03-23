#!/usr/bin/env bash
#SBATCH --job-name=nvc_train_s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/train_s_%j.out
#SBATCH --error=logs/slurm/train_s_%j.err

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nvc

mkdir -p logs/slurm

echo "=============================="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "=============================="

python training/train_sr.py \
    --data data/sr_enhanced_pairs \
    --ckpt-dir checkpoints/sr \
    --config configs/gpu/upscaling.yaml \
    --workers 4 \
    ${RESUME:+--resume "$RESUME"}

echo "Training complete: $(date)"
