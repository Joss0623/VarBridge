#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.." || exit 1
active_pid=${1:-0}
while [[ $active_pid -gt 0 ]] && kill -0 "$active_pid" 2>/dev/null; do sleep 60; done
bash scripts/run_encoder_matrix.sh
bash scripts/run_encoder_short_preprocess.sh
bash scripts/run_encoder_short.sh
printf "ALL_ENCODER_EXPERIMENTS_COMPLETE\n"
