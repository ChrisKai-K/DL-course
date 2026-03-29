#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
python train.py --config configs/cub_resnet50.yaml "$@"
