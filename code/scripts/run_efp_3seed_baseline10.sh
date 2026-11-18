#!/usr/bin/env bash
set -u
cd /home/home_new/qsmx/pycodes/GreyFerm || exit 1
exec 9>test_results/.efp_3seed_baseline10.lock
flock -n 9 || exit 0
PY=/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python
TORCHRUN=/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/torchrun
LOGDIR=/home/home_new/qsmx/pycodes/GreyFerm/test_results/efp_3seed_baseline10
complete() {
 [[ -f "$1" ]] && [[ $(grep -c '^mse:' "$1") -eq "$2" ]] && ! grep -q Traceback "$1"
}
run_crossformer() {
 local pids=() slot=0 log seed ds h gpu failed pid
 for seed in 2021 2022 2023; do
  for ds in EFP_long_sub1 EFP_long_sub2 EFP_long_sub3; do
   for h in 6 12 24 48; do
    log="$LOGDIR/crossformer_${ds}_h${h}_seed${seed}.log"
    if complete "$log" 1; then echo "SKIP Crossformer $ds H=$h seed=$seed";continue;fi
    gpu=$((slot+2))
    if [[ -f "$log" ]];then mv "$log" "${log}.failed.$(date +%s)";fi
    (
     cd /home/home_new/qsmx/pycodes/Time-Series-Library || exit 1
     CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u run.py \
      --task_name long_term_forecast --is_training 1 --with_curve 0 \
      --model_id "${ds}_crossformer_3seed_h${h}_seed${seed}" --model Crossformer --data efp_batch \
      --root_path "/home/home_new/qsmx/pycodes/GreyFerm/dataset/$ds" --data_path "$ds.csv" --features M \
      --seq_len 48 --label_len 24 --pred_len "$h" --enc_in 5 --dec_in 5 --c_out 5 \
      --d_model 128 --d_ff 256 --n_heads 4 --e_layers 2 --d_layers 1 \
      --batch_size 126 --num_workers 2 --train_epochs 10 --patience 3 \
      --learning_rate 0.0001 --lradj sigmoid --loss huber --random_seed "$seed" \
      --gpu 0 --des efp_baseline10_20261011 --embed timeF
    ) >"$log" 2>&1 &
    pids+=("$!");echo "START Crossformer $ds H=$h seed=$seed GPU=$gpu";slot=$((slot+1))
    if [[ $slot -eq 6 ]];then
     failed=0;for pid in "${pids[@]}";do wait "$pid" || failed=1;done
     if [[ $failed -ne 0 ]];then echo 'Crossformer group failed';return 1;fi
     pids=();slot=0
    fi
   done
  done
 done
 failed=0;for pid in "${pids[@]}";do wait "$pid" || failed=1;done
 return "$failed"
}

until run_crossformer;do echo "RETRY $(date -Is)";sleep 120;done
echo ALL_EFP_3SEED_BASELINE10_COMPLETE
