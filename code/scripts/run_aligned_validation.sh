#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python
seed=2021
run_one() {
 local gpu=$1 ds=$2 variant=$3
 local ckpts=(checkpoints/long_term_forecast_"$ds"_noprompt_seed2021_PatchTST_*/checkpoint.pth)
 local ckpt=${ckpts[0]}
 local flags=()
 if [[ "$variant" == prompt || "$variant" == prompt_channel ]]; then flags+=(--mix_embeds); fi
 if [[ "$variant" == channel || "$variant" == prompt_channel ]]; then flags+=(--channel_interaction); fi
 local log="test_results/preserved_${ds}_${variant}_seed${seed}.log"
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" run.py --task_name long_term_forecast --is_training 1 \
  --model_id "${ds}_${variant}_preserved_seed${seed}" --model KIBR_PatchTST --data BatchMTSF \
  --root_path "dataset/$ds" --data_path "$ds.csv" --prompt_encoder gpt2 \
  --patchtst_ckpt "$ckpt" --seq_len 48 --label_len 24 --token_len 24 \
  --test_seq_len 48 --test_label_len 24 --test_pred_len 48 --additional_test_pred_len 24 \
  --enc_in 5 --dec_in 5 --c_out 5 --patch_len 24 --stride 12 \
  --d_model 128 --n_heads 4 --e_layers 2 --d_ff 256 \
  --batch_size 128 --num_workers 2 --train_epochs 10 --patience 3 \
  --learning_rate 0.0001 --random_seed "$seed" --gpu 0 --des preserved20260929 \
  "${flags[@]}" >"$log" 2>&1
 local status=$?
 echo "$ds $variant GPU=$gpu exit=$status"
 grep '^mse:' "$log" | tail -2
 if [[ $status -ne 0 ]]; then tail -10 "$log"; fi
 return "$status"
}
datasets=(EFP_long_sub1 EFP_long_sub2 EFP_long_sub3)
variants=(prompt channel prompt_channel)
pids=()
for i in 0 1 2 3 4 5 6 7; do
 ds=${datasets[$((i%3))]}
 variant=${variants[$((i/3))]}
 run_one "$i" "$ds" "$variant" & pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
run_one 0 EFP_long_sub3 prompt_channel || status=1
exit "$status"
