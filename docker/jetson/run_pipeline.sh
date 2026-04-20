#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_PATH="${WILDROI_ARCHIVE_PATH:-outputs/video.zip}"
RECON_PATH="${WILDROI_RECON_PATH:-outputs/video_reconstructed.mp4}"
INPUT_VIDEO="${WILDROI_INPUT_VIDEO:-data/test.mp4}"
COMPRESSION_CONFIG="${WILDROI_COMPRESSION_CONFIG:-configs/jetson/compression.yaml}"
DECOMPRESSION_CONFIG="${WILDROI_DECOMPRESSION_CONFIG:-configs/jetson/decompression.yaml}"

bash docker/jetson/run_compression.sh
bash docker/jetson/run_decompression.sh
