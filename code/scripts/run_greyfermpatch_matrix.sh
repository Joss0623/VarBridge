#!/usr/bin/env bash
# Reproduce GreyFerm-PatchTST: [all|preprocess|pretrain|forecast].
# Forecast mode uses GPUs 0-7 and requires prepared .pt files and four PatchTST checkpoints.
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p test_results
PY=/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
encoders=(gpt2 distilgpt2 bert roberta)
mode="${1:-forecast}"
preprocess_all() {
 local ds enc gpu log
 for ds in EFP_long_sub1 EFP_long_sub2 EFP_long_sub3; do
  for gpu in 0 1 2 3; do
   enc=${encoders[$gpu]}
   printf -v log 'test_results/preprocess_%s_%s.log' "$ds" "$enc"
   CUDA_VISIBLE_DEVICES="$gpu" "$PY" preprocess.py --dataset "$ds" --encoder "$enc" --gpu 0 >"$log" 2>&1 || return 1
  done
 done
}
pretrain_one() {
 local gpu=$1 ds=$2 seq=48 label=24 token=24 horizon=48 short=24 nvars=5 patch=24 stride=12 log
 printf -v log 'test_results/pretrain_%s.log' "$ds"
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" run.py --task_name long_term_forecast --is_training 1 \
  --model_id "${ds}_noprompt_seed2021" --model PatchTST --data BatchMTSF --root_path "dataset/$ds" --data_path "$ds.csv" \
  --prompt_encoder gpt2 --seq_len "$seq" --label_len "$label" --token_len "$token" \
  --test_seq_len "$seq" --test_label_len "$label" --test_pred_len "$horizon" --additional_test_pred_len "$short" \
  --enc_in "$nvars" --dec_in "$nvars" --c_out "$nvars" --patch_len "$patch" --stride "$stride" \
  --d_model 128 --n_heads 4 --e_layers 2 --d_ff 256 --batch_size 128 --num_workers 2 \
  --train_epochs 10 --patience 3 --learning_rate 0.0001 --random_seed 2021 --gpu 0 --des matrix20260929 >"$log" 2>&1
}
pretrain_all() {
 pretrain_one 0 EFP_long_sub1 & p0=$!
 pretrain_one 2 EFP_long_sub2 & p1=$!
 pretrain_one 4 EFP_long_sub3 & p2=$!
 wait "$p0" && wait "$p1" && wait "$p2"
}
run_one() {
 local gpu=$1 ds=$2 enc=$3
 local seq=48 label=24 token=24 horizon=48 short=24 nvars=5
 local ckpts=(checkpoints/long_term_forecast_"$ds"_noprompt_seed2021_PatchTST_*/checkpoint.pth)
 local ckpt="${ckpts[0]}"
 local tag="${ds}_${enc}_kibrpatch_seed2021"
 local log="test_results/grey_future_${tag}.log"
 local mix=()
 if [ "$enc" != noprompt ]; then mix=(--mix_embeds); fi
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" run.py --task_name long_term_forecast --is_training 1  --model_id "$tag" --model KIBR_PatchTST --data BatchMTSF --root_path "dataset/$ds" --data_path "$ds.csv"  --prompt_encoder "${enc/noprompt/gpt2}" --patchtst_ckpt "$ckpt" --seq_len "$seq" --label_len "$label" --token_len "$token"  --test_seq_len "$seq" --test_label_len "$label" --test_pred_len "$horizon" --additional_test_pred_len "$short"  --enc_in "$nvars" --dec_in "$nvars" --c_out "$nvars" --d_model 128 --n_heads 4 --e_layers 2 --d_ff 256  --batch_size 128 --num_workers 2 --train_epochs 10 --patience 3 --learning_rate 0.0001 --gpu 0  --random_seed 2021 --des futureonly20260929 "${mix[@]}" >"$log" 2>&1
 local code=$?
 echo "GPU=$gpu ds=$ds enc=$enc exit=$code"
 grep '^mse:' "$log" | tail -2
 if [ "$code" -ne 0 ]; then tail -12 "$log"; fi
}
run_gpu() {
 local gpu=$1 enc
 enc=${encoders[$((gpu%4))]}
 if [ "$gpu" -lt 4 ]; then
  run_one "$gpu" EFP_long_sub1 "$enc"
  run_one "$gpu" EFP_long_sub3 "$enc"
  if [ "$gpu" -eq 1 ]; then run_one "$gpu" EFP_long_sub1 noprompt; fi
  if [ "$gpu" -eq 2 ]; then run_one "$gpu" EFP_long_sub2 noprompt; fi
  if [ "$gpu" -eq 3 ]; then run_one "$gpu" EFP_long_sub3 noprompt; fi
 else
  run_one "$gpu" EFP_long_sub2 "$enc"
 fi
}
case "$mode" in
 preprocess) preprocess_all ;;
 pretrain) pretrain_all ;;
 forecast) for gpu in 0 1 2 3 4 5 6 7; do run_gpu "$gpu" & done; wait ;;
 all) preprocess_all && pretrain_all || exit 1
      for gpu in 0 1 2 3 4 5 6 7; do run_gpu "$gpu" & done; wait ;;
 *) echo 'Usage: run_greyfermpatch_matrix.sh [all|preprocess|pretrain|forecast]' >&2; exit 2 ;;
esac
