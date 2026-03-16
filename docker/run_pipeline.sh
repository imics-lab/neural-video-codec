#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_PATH="${WILDROI_ARCHIVE_PATH:-outputs/video.zip}"
RECON_PATH="${WILDROI_RECON_PATH:-outputs/video_reconstructed.mp4}"
INPUT_VIDEO="${WILDROI_INPUT_VIDEO:-video.mp4}"
COMPRESSION_CONFIG="${WILDROI_COMPRESSION_CONFIG:-configs/gpu/compression.yaml}"
DECOMPRESSION_CONFIG="${WILDROI_DECOMPRESSION_CONFIG:-configs/gpu/decompression.yaml}"

python run_compression.py "${INPUT_VIDEO}" --config "${COMPRESSION_CONFIG}" --output "${ARCHIVE_PATH}"
python run_decompression.py "${ARCHIVE_PATH}" --config "${DECOMPRESSION_CONFIG}" --output "${RECON_PATH}"
