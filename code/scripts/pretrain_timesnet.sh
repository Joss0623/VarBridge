#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python
run_one() {
 local gpu=$1 ds=$2
 local log="test_results/pretrain_timesnet_${ds}_seed2021.log"
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" run.py --task_name long_term_forecast --is_training 1 \
  --model_id "${ds}_noprompt_seed2021" --model TimesNet --data BatchMTSF \
  --root_path "dataset/$ds" --data_path "$ds.csv" --prompt_encoder gpt2 \
  --seq_len 48 --label_len 24 --token_len 24 \
  --test_seq_len 48 --test_label_len 24 --test_pred_len 48 --additional_test_pred_len 24 \
  --enc_in 5 --dec_in 5 --c_out 5 --patch_len 24 --stride 12 \
  --d_model 128 --n_heads 4 --e_layers 2 --d_ff 256 \
  --batch_size 128 --num_workers 2 --train_epochs 10 --patience 3 \
  --learning_rate 0.0001 --random_seed 2021 --gpu 0 --des plugpretrain20260930 >"$log" 2>&1
 local status=$?
 echo "$ds TimesNet GPU=$gpu exit=$status"
 grep '^mse:' "$log" | tail -2
 if [[ $status -ne 0 ]]; then tail -20 "$log"; fi
 return "$status"
}
status=0
pids=()
for i in 0 1 2; do run_one "$i" "EFP_long_sub$((i+1))" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
