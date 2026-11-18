#!/usr/bin/env bash
cd /home/home_new/qsmx/pycodes/GreyFerm || exit 1
grep -q '^ALL_COMMON_BOUND_ABLATIONS_COMPLETE' test_results/common_bound_ablation_pipeline.log 2>/dev/null && exit 0
flock -n test_results/common_bound_ablations/.watchdog.lock /home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python -u scripts/run_common_bound_ablations.py >> test_results/common_bound_ablation_pipeline.log 2>&1
