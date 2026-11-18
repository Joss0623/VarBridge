#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python
run_one() {
 local gpu=$1 ds=$2 mode=$3
 local ckpts=(checkpoints/long_term_forecast_"$ds"_noprompt_seed2021_PatchTST_*/checkpoint.pth)
 local ckpt=${ckpts[0]}
 local log="test_results/m3early_${mode}_${ds}_seed2021.log"
 local early_mode=${mode%%_late}
 local late=()
 if [[ "$mode" == *_late ]]; then late=(--channel_interaction); fi
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" run.py --task_name long_term_forecast --is_training 1 \
  --model_id "${ds}_m3early_${mode}_seed2021" --model KIBR_PatchTST --data BatchMTSF \
  --root_path "dataset/$ds" --data_path "$ds.csv" \
  --prompt_encoder gpt2_window_horizon_correct --patchtst_ckpt "$ckpt" \
  --seq_len 48 --label_len 24 --token_len 24 \
  --test_seq_len 48 --test_label_len 24 --test_pred_len 48 --additional_test_pred_len 24 \
  --enc_in 5 --dec_in 5 --c_out 5 --patch_len 24 --stride 12 \
  --d_model 128 --n_heads 4 --e_layers 2 --d_ff 256 \
  --batch_size 128 --num_workers 2 --train_epochs 10 --patience 3 \
  --learning_rate 0.0001 --random_seed 2021 --gpu 0 --des "m3early${mode}20260929" \
  --mix_embeds "${late[@]}" --early_channel_mode "$early_mode" \
  --forecast_conditioned --tri_mmoe >"$log" 2>&1
 local status=$?
 echo "$ds $mode GPU=$gpu exit=$status"
 grep '^mse:' "$log" | tail -2
 if [[ $status -ne 0 ]]; then tail -12 "$log"; fi
 return "$status"
}
datasets=(EFP_long_sub1 EFP_long_sub2 EFP_long_sub3)
modes=(raw patch raw_late patch_late)
status=0
for start in 0 8; do
 pids=()
 for offset in 0 1 2 3 4 5 6 7; do
  i=$((start+offset))
  if [[ $i -ge 12 ]]; then break; fi
  ds=${datasets[$((i%3))]}
  mode=${modes[$((i/3))]}
  run_one "$offset" "$ds" "$mode" & pids+=("$!")
 done
 for pid in "${pids[@]}"; do wait "$pid" || status=1; done
done
exit "$status"
