#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/data/cub_200_2011}"
EPOCHS="${2:-10}"
BATCH_SIZE="${3:-64}"

python train.py \
  --config configs/clip_lora_cub.yaml \
  --data-root "${DATA_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}"

python report_helper.py --run-dir outputs/clip_vitb32_lora_cub
