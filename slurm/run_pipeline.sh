#!/usr/bin/env bash
#SBATCH --job-name=nvc_pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/pipeline_%j.out
#SBATCH --error=logs/slurm/pipeline_%j.err

# Usage:
#   sbatch slurm/run_pipeline.sh --video dataset/bird1.mp4
#   sbatch --export=ALL,VIDEO=dataset/bird1.mp4 slurm/run_pipeline.sh

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

VIDEO="${VIDEO:-}"
if [[ -z "$VIDEO" ]]; then
    echo "ERROR: Set VIDEO env var, e.g.:"
    echo "  sbatch --export=ALL,VIDEO=dataset/bird1.mp4 slurm/run_pipeline.sh"
    exit 1
fi

python run_pipeline.py \
    --video "$VIDEO" \
    --config configs/gpu/compression.yaml \
    ${RESTORE_CONFIG:+--restore-config "$RESTORE_CONFIG"} \
    ${SR_CONFIG:+--sr-config "$SR_CONFIG"}

echo "Pipeline complete: $(date)"
