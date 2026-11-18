#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python
encoders=(distilgpt2 bert roberta)
datasets=(EFP_long_sub1 EFP_long_sub2 EFP_long_sub3)
run_one() {
 local gpu=$1 encoder=$2 ds=$3
 local log="test_results/encoderprep_${encoder}_${ds}.log"
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" preprocess_window_semantic.py --encoder "$encoder" --dataset "$ds" --gpu 0 >"$log" 2>&1
 local status=$?
 echo "$encoder $ds GPU=$gpu exit=$status"
 if [[ $status -ne 0 ]]; then tail -12 "$log"; fi
 return "$status"
}
status=0
for start in 0 6; do
 pids=()
 for offset in 0 1 2 3 4 5; do
  gpu=2
  i=$((start+gpu))
  if [[ $i -ge 9 ]]; then break; fi
  run_one "$gpu" "${encoders[$((i/3))]}" "${datasets[$((i%3))]}" & pids+=("$!")
 done
 for pid in "${pids[@]}"; do wait "$pid" || status=1; done
done
exit "$status"
