#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_PATH="${WILDROI_ARCHIVE_PATH:-outputs/video.zip}"
RECON_PATH="${WILDROI_RECON_PATH:-outputs/video_reconstructed.mp4}"
DECOMPRESSION_CONFIG="${WILDROI_DECOMPRESSION_CONFIG:-configs/jetson/decompression.yaml}"

tmp_dir="$(mktemp -d /tmp/wildroi-decompress-XXXXXX)"
tmp_recon="${tmp_dir}/video_reconstructed.mp4"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

python run_decompression.py "${ARCHIVE_PATH}" --config "${DECOMPRESSION_CONFIG}" --output "${tmp_recon}"
mkdir -p "$(dirname "${RECON_PATH}")"
cp "${tmp_recon}" "${RECON_PATH}"
echo "[OK] copied reconstructed video to ${RECON_PATH}"
