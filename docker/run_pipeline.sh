#!/usr/bin/env bash
set -euo pipefail

bash docker/run_compression.sh
bash docker/run_decompression.sh
