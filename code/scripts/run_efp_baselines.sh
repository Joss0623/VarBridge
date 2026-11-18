#!/usr/bin/env bash
set -u
ROOT=/home/home_new/qsmx/pycodes
PY=/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python
LOGDIR="$ROOT/GreyFerm/test_results/efp_main_baselines"
mkdir -p "$LOGDIR"
models=(DLinear PatchTST TimesNet iTransformer TimeMixer Crossformer TiDE FEDformer xPatch)
datasets=(EFP_long_sub1 EFP_long_sub2 EFP_long_sub3)
horizons=(6 12 24 48)
pids=()
slot=0
for ds in "${datasets[@]}"; do
 for h in "${horizons[@]}"; do
  for model in "${models[@]}"; do
   log="$LOGDIR/${ds}_h${h}_${model}.log"
   if [[ -f "$log" ]] && grep -q ^mse: "$log" && ! grep -q Traceback "$log"; then
    echo "SKIP $ds H=$h $model"
    continue
   fi
   gpu=$((slot+2))
   (
    if [[ "$model" == xPatch ]]; then
     cd "$ROOT/xPatch" || exit 1
     CUDA_VISIBLE_DEVICES="$gpu" "$PY" run.py \
      --is_training 1 --model_id "${ds}_h${h}" --model xPatch --data efp_batch \
      --root_path "$ROOT/GreyFerm/dataset/$ds" --data_path "$ds.csv" --features M \
      --seq_len 48 --label_len 24 --pred_len "$h" --enc_in 5 \
      --patch_len 24 --stride 12 --ma_type ema --revin 1 \
      --batch_size 32 --num_workers 2 --train_epochs 10 --patience 3 \
      --learning_rate 0.0001 --gpu 0 --des efp_main10
    else
     cd "$ROOT/Time-Series-Library" || exit 1
     CUDA_VISIBLE_DEVICES="$gpu" "$PY" run.py \
      --task_name long_term_forecast --is_training 1 --with_curve 0 \
      --model_id "${ds}_h${h}" --model "$model" --data efp_batch \
      --root_path "$ROOT/GreyFerm/dataset/$ds" --data_path "$ds.csv" --features M \
      --seq_len 48 --label_len 24 --pred_len "$h" --enc_in 5 --dec_in 5 --c_out 5 \
      --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 --d_layers 1 \
      --down_sampling_layers 2 --down_sampling_window 2 --down_sampling_method avg \
      --batch_size 32 --num_workers 2 --train_epochs 10 --patience 3 \
      --learning_rate 0.0001 --gpu 0 --des efp_main10 --embed timeF
    fi
   ) >"$log" 2>&1 &
   pids+=("$!")
   echo "START $ds H=$h $model GPU=$gpu"
   slot=$((slot+1))
   if [[ $slot -eq 6 ]]; then
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    if [[ $failed -ne 0 ]]; then echo "Baseline group failed; inspect $LOGDIR"; exit 1; fi
    pids=()
    slot=0
   fi
  done
 done
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
printf "ALL_EFP_BASELINES_COMPLETE\n"
