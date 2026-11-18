#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python
run_one() {
 local gpu=$1 ds=$2 variant=$3
 local ckpts=(checkpoints/long_term_forecast_"$ds"_noprompt_seed2021_PatchTST_*/checkpoint.pth)
 local flags=()
 if [[ "$variant" == dynamic || "$variant" == both ]]; then flags+=(--dynamic_window_stats); fi
 if [[ "$variant" == separate || "$variant" == both ]]; then flags+=(--separate_numeric_path); fi
 local log="test_results/fixpromexp_${variant}_${ds}_seed2021.log"
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" run.py --task_name long_term_forecast --is_training 1 \
  --model_id "${ds}_fixpromexp_${variant}_seed2021" --model KIBR_PatchTST --data BatchMTSF \
  --root_path "dataset/$ds" --data_path "$ds.csv" \
  --prompt_encoder gpt2_window_horizon_correct --patchtst_ckpt "${ckpts[0]}" \
  --seq_len 48 --label_len 24 --token_len 24 \
  --test_seq_len 48 --test_label_len 24 --test_pred_len 48 --additional_test_pred_len 24 \
  --enc_in 5 --dec_in 5 --c_out 5 --patch_len 24 --stride 12 \
  --d_model 128 --n_heads 4 --e_layers 2 --d_ff 256 \
  --batch_size 128 --num_workers 2 --train_epochs 10 --patience 3 \
  --learning_rate 0.0001 --random_seed 2021 --gpu 0 --des "fixpromexp${variant}20260929" \
  --mix_embeds --channel_interaction --early_channel_mode patch \
  --forecast_conditioned --tri_mmoe "${flags[@]}" >"$log" 2>&1
 local status=$?
 echo "$ds $variant GPU=$gpu exit=$status"
 grep '^mse:' "$log" | tail -2
 if [[ $status -ne 0 ]]; then tail -15 "$log"; fi
 return "$status"
}
datasets=(EFP_long_sub1 EFP_long_sub2 EFP_long_sub3)
variants=(dynamic separate both)
status=0
for start in 0 8; do
 pids=()
 for gpu in 0 1 2 3 4 5 6 7; do
  i=$((start+gpu))
  if [[ $i -ge 9 ]]; then break; fi
  run_one "$gpu" "${datasets[$((i%3))]}" "${variants[$((i/3))]}" & pids+=("$!")
 done
 for pid in "${pids[@]}"; do wait "$pid" || status=1; done
done
exit "$status"
