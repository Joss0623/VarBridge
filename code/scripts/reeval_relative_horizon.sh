#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python
run_one() {
 local gpu=$1 ds=$2 mode=$3 factor=$4
 local source="m3early_${mode}"
 local desc="m3early${mode}20260929"
 if [[ "$factor" != 0 ]]; then
  source="m3scale_${mode}_x${factor}"
  desc="m3scale${mode}x${factor}20260929"
 fi
 local trained=(checkpoints/long_term_forecast_"$ds"_"$source"_seed2021_KIBR_PatchTST_*/checkpoint.pth)
 if [[ ! -f "${trained[0]}" ]]; then echo "missing ${trained[0]}"; return 1; fi
 local test_dir=${trained[0]%/checkpoint.pth}
 test_dir=${test_dir#checkpoints/}
 local base=(checkpoints/long_term_forecast_"$ds"_noprompt_seed2021_PatchTST_*/checkpoint.pth)
 local early=${mode%%_late}
 local late=()
 if [[ "$mode" == *_late ]]; then late=(--channel_interaction); fi
 local log="test_results/hfix_${mode}_x${factor}_${ds}_seed2021.log"
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" run.py --task_name long_term_forecast --is_training 0 \
  --model_id "${ds}_hfix_${mode}_x${factor}_seed2021" --model KIBR_PatchTST --data BatchMTSF \
  --root_path "dataset/$ds" --data_path "$ds.csv" \
  --prompt_encoder gpt2_window_horizon_correct --patchtst_ckpt "${base[0]}" \
  --seq_len 48 --label_len 24 --token_len 24 \
  --test_seq_len 48 --test_label_len 24 --test_pred_len 48 \
  --enc_in 5 --dec_in 5 --c_out 5 --patch_len 24 --stride 12 \
  --d_model 128 --n_heads 4 --e_layers 2 --d_ff 256 \
  --batch_size 128 --num_workers 2 --train_epochs 10 --patience 3 \
  --learning_rate 0.0001 --random_seed 2021 --gpu 0 --des "${desc}_hfix" \
  --mix_embeds "${late[@]}" --early_channel_mode "$early" --coarse_factor "$factor" \
  --forecast_conditioned --tri_mmoe --test_dir "$test_dir" --test_file_name checkpoint.pth >"$log" 2>&1
 local status=$?
 echo "$ds $mode x$factor GPU=$gpu exit=$status"
 grep '^mse:' "$log" | tail -1
 if [[ $status -ne 0 ]]; then tail -15 "$log"; fi
 return "$status"
}
datasets=(EFP_long_sub1 EFP_long_sub2 EFP_long_sub3 EFP_long_sub1 EFP_long_sub2 EFP_long_sub3 EFP_long_sub2 EFP_long_sub3)
modes=(patch_late patch_late patch_late raw_late raw_late raw_late patch_late patch_late)
factors=(0 0 0 0 0 0 4 4)
status=0
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
 run_one "$gpu" "${datasets[$gpu]}" "${modes[$gpu]}" "${factors[$gpu]}" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
