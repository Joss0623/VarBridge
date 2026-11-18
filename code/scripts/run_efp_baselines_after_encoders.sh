#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.." || exit 1
until grep -q ALL_ENCODER_EXPERIMENTS_COMPLETE test_results/encoderplug_pipeline.log; do sleep 300; done
bash scripts/run_efp_baselines.sh
