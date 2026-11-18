#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python
run_one() {
 local gpu=$1 ds=$2 schedule=$3
 local ckpts=(checkpoints/long_term_forecast_"$ds"_noprompt_seed2021_PatchTST_*/checkpoint.pth)
 local log="test_results/dualsig100_${schedule}_${ds}_seed2021.log"
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" run.py --task_name long_term_forecast --is_training 1 \
  --model_id "${ds}_dualsig100_${schedule}_seed2021" --model KIBR_PatchTST --data BatchMTSF \
  --root_path "dataset/$ds" --data_path "$ds.csv" \
  --prompt_encoder gpt2_window_horizon_correct --patchtst_ckpt "${ckpts[0]}" \
  --seq_len 48 --label_len 24 --token_len 24 \
  --test_seq_len 48 --test_label_len 24 --test_pred_len 48 --additional_test_pred_len 24 \
  --enc_in 5 --dec_in 5 --c_out 5 --patch_len 24 --stride 12 \
  --d_model 128 --n_heads 4 --e_layers 2 --d_ff 256 \
  --batch_size 128 --num_workers 2 --train_epochs 100 --patience 100 \
  --learning_rate 0.0001 --lradj "$schedule" --random_seed 2021 --gpu 0 --des "dualsig100${schedule}20260929" \
  --mix_embeds --channel_interaction --early_channel_mode patch \
  --forecast_conditioned --tri_mmoe --dynamic_window_stats --separate_numeric_path >"$log" 2>&1
 local status=$?
 echo "$ds $schedule GPU=$gpu exit=$status"
 grep '^mse:' "$log" | tail -2
 if [[ $status -ne 0 ]]; then tail -20 "$log"; fi
 return "$status"
}
datasets=(EFP_long_sub1 EFP_long_sub2 EFP_long_sub3)
schedules=(sigmoid type1)
status=0
pids=()
for i in 0 1 2 3 4 5; do
 run_one "$i" "${datasets[$((i%3))]}" "${schedules[$((i/3))]}" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
