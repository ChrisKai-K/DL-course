#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="${1:-prj_4_clip.tar.gz}"

tar \
  --exclude="./outputs" \
  --exclude="./__pycache__" \
  --exclude="./src/__pycache__" \
  --exclude="./.DS_Store" \
  -czf "${ARCHIVE}" \
  README.md requirements.txt train.py evaluate.py zero_shot.py predict.py report_helper.py \
  configs scripts src reports notebooks

echo "Wrote ${ARCHIVE}"
