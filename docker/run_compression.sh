#!/usr/bin/env bash
set -euo pipefail

INPUT_VIDEO="${WILDROI_INPUT_VIDEO:-data/test.mp4}"
ARCHIVE_PATH="${WILDROI_ARCHIVE_PATH:-outputs/video.zip}"
COMPRESSION_CONFIG="${WILDROI_COMPRESSION_CONFIG:-configs/gpu/compression.yaml}"

tmp_dir="$(mktemp -d /tmp/wildroi-compress-XXXXXX)"
tmp_archive="${tmp_dir}/archive.zip"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

python run_compression.py "${INPUT_VIDEO}" --config "${COMPRESSION_CONFIG}" --output "${tmp_archive}"

mkdir -p "$(dirname "${ARCHIVE_PATH}")"
cp "${tmp_archive}" "${ARCHIVE_PATH}"
echo "[OK] copied archive to ${ARCHIVE_PATH}"
