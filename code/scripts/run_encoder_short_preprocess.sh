#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python
pids=()
slot=0
for horizon in 6 12; do
 for encoder in gpt2 distilgpt2 bert roberta; do
  for ds in EFP_long_sub1 EFP_long_sub2 EFP_long_sub3; do
   cache="dataset/$ds/${ds}__${encoder}_window_semantic_h${horizon}.pt"
   if [[ -f "$cache" ]]; then echo "CACHED h=$horizon $encoder $ds"; continue; fi
   gpu=2$slot
   log="test_results/preprocess_h${horizon}_${encoder}_${ds}.log"
   nice -n 10 "$PY" preprocess_window_semantic.py --dataset "$ds" --encoder "$encoder" --horizon "$horizon" --gpu "$gpu" --batch_size 96 >"$log" 2>&1 &
   pids+=("$!")
   echo "START h=$horizon $encoder $ds GPU=$gpu"
   slot=$((slot+1))
   if [[ $slot -eq 8 ]]; then
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    if [[ $failed -ne 0 ]]; then echo "Preprocess failed; inspect test_results/preprocess_h*.log"; exit 1; fi
    pids=()
    slot=0
   fi
  done
 done
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
