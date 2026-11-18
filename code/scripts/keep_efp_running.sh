#!/usr/bin/env bash
set -u
cd /home/home_new/qsmx/pycodes/GreyFerm || exit 1
exec 9>test_results/.efp_watchdog.lock
flock -n 9 || exit 0
export PYTHONDONTWRITEBYTECODE=1
alive() { pgrep -u "$(id -u)" -f "^bash scripts/$1( |$)" >/dev/null; }
launch() { nohup bash "scripts/$1" >>"test_results/$2" 2>&1 </dev/null & echo "$(date -Is) START $1 PID=$!" >>test_results/efp_watchdog.log; }
metric_count() { grep -h '^mse:' $1 2>/dev/null | wc -l; }
complete_encoder() {
 grep -q ALL_ENCODER_EXPERIMENTS_COMPLETE test_results/encoderplug_pipeline.log 2>/dev/null &&
 [[ $(metric_count 'test_results/encoderplug100_*_seed2021.log') -ge 48 ]] &&
 [[ $(metric_count 'test_results/encoderplug100_h*_seed2021.log') -ge 48 ]]
}
complete_numeric() {
 grep -q ALL_EFP_BASELINES_COMPLETE test_results/efp_baseline_pipeline.log 2>/dev/null &&
 [[ $(metric_count 'test_results/efp_main_baselines/*.log') -ge 108 ]]
}
complete_multi() {
 grep -q ALL_EFP_MULTIMODAL_COMPLETE test_results/efp_multimodal_pipeline.log 2>/dev/null &&
 [[ $(metric_count 'test_results/efp_multimodal/*.log') -ge 36 ]]
}
complete_ablation() {
 grep -q ALL_EFP_ABLATIONS_COMPLETE test_results/efp_ablation_pipeline.log 2>/dev/null &&
 [[ $(metric_count 'test_results/efp_ablations/*_seed2021.log') -ge 360 ]]
}

if ! complete_encoder; then
  # Time-LLM consumes nearly a full 24 GB GPU. Finish that queue before
  # resuming six-rank encoder training to avoid simultaneous CUDA OOM.
  if { ! complete_numeric || complete_multi; } &&
     ! alive run_encoder_pipeline.sh && ! alive run_encoder_matrix.sh &&
     ! alive run_encoder_short.sh && ! alive run_encoder_short_preprocess.sh; then
    launch run_encoder_pipeline.sh encoderplug_pipeline.log
  fi
fi
if ! complete_numeric; then
  if ! alive run_efp_baselines.sh; then
    launch run_efp_baselines.sh efp_baseline_pipeline.log
  fi
fi
if ! complete_multi; then
  if ! alive run_efp_multimodal.sh; then
    launch run_efp_multimodal.sh efp_multimodal_pipeline.log
  fi
fi
if [[ -f scripts/run_efp_ablations.sh ]] && complete_encoder && complete_numeric && complete_multi && ! complete_ablation; then
  if ! alive run_efp_ablations.sh; then
    launch run_efp_ablations.sh efp_ablation_pipeline.log
  fi
fi
