#!/usr/bin/env bash
set -u
ROOT=/home/home_new/qsmx/pycodes
PROJ="$ROOT/GreyFerm"
PY=/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python
LOGDIR="$PROJ/test_results/efp_multimodal"
mkdir -p "$LOGDIR"
while ! grep -q 'ALL_EFP_BASELINES_COMPLETE' "$PROJ/test_results/efp_baseline_pipeline.log" 2>/dev/null; do
 sleep 60
done
models=(TimeCMA T3Time TimeLLM)
datasets=(EFP_long_sub1 EFP_long_sub2 EFP_long_sub3)
horizons=(6 12 24 48)
pids=()
slot=0
for ds in "${datasets[@]}"; do
 for h in "${horizons[@]}"; do
  for model in "${models[@]}"; do
   log="$LOGDIR/${ds}_h${h}_${model}.log"
   if [[ -f "$log" ]] && grep -q '^mse:' "$log" && ! grep -q 'Traceback' "$log"; then
    echo "SKIP $ds H=$h $model"
    continue
   fi
   gpu=$((slot+2))
   (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS=4
    export PYTHONDONTWRITEBYTECODE=1
    if [[ "$model" == "TimeLLM" ]]; then
     cd "$ROOT/Time-LLM" || exit 1
     "$PY" run_main.py --task_name long_term_forecast --is_training 1 \
       --model_id "${ds}_h${h}" --model_comment efp_multi10 --model TimeLLM \
       --data efp_batch --root_path "$PROJ/dataset/$ds" --data_path "$ds.csv" \
       --features M --seq_len 48 --label_len 24 --pred_len "$h" \
       --enc_in 5 --dec_in 5 --c_out 5 --d_model 16 --d_ff 32 --n_heads 4 \
       --patch_len 16 --stride 8 --prompt_domain 1 --llm_model GPT2 --llm_dim 768 --llm_layers 6 \
       --batch_size 32 --num_workers 2 --train_epochs 10 --patience 3 \
       --learning_rate 0.0001 --use_amp --lradj COS --des efp_multi10 \
       --checkpoints "$PROJ/checkpoints/efp_multimodal/TimeLLM"
    else
     cd "$PROJ" || exit 1
     "$PY" scripts/run_efp_multimodal.py --model "$model" --dataset "$ds" \
       --horizon "$h" --epochs 10 --patience 3 --seed 2021 --batch_size 32
    fi
   ) >"$log" 2>&1 &
   pids+=("$!")
   echo "START $ds H=$h $model GPU=$gpu"
   slot=$((slot+1))
   if [[ $slot -eq 6 ]]; then
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    if [[ $failed -ne 0 ]]; then echo "Multimodal group failed; inspect $LOGDIR"; exit 1; fi
    pids=()
    slot=0
   fi
  done
 done
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
printf 'ALL_EFP_MULTIMODAL_COMPLETE\n'
