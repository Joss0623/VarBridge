#!/usr/bin/env bash
set -u
exec 9>test_results/.efp_embedding.lock
flock -n 9 || exit 0
cd "$(dirname "$0")/.." || exit 1
TORCHRUN=/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/torchrun
mkdir -p test_results/efp_embedding
SELECTED=$(/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python scripts/select_efp_text_encoder.py) || exit 1
echo "VALIDATION_SELECTED_TEXT=$SELECTED"
# Full-model encoder is chosen from validation MSE by select_efp_text_encoder.py.
# Each control below uses the same split, seed, training budget and evaluation.
variants=(embedding_add embedding_gate)
run_matrix() {
for variant in "${variants[@]}"; do
 for ds in EFP_long_sub1 EFP_long_sub2 EFP_long_sub3; do
  for h in 6 12 24; do
   log="test_results/efp_embedding/${SELECTED}_${variant}_${ds}_h${h}_seed2021.log"
   expected=1; [[ $h -eq 24 ]] && expected=2
   if [[ -f "$log" ]] && [[ $(grep -c ^mse: "$log") -eq $expected ]] && ! grep -q Traceback "$log"; then
    echo "SKIP $variant $ds H=$h"; continue
   fi
   ckpts=(checkpoints/long_term_forecast_"$ds"_noprompt_seed2021_PatchTST_*/checkpoint.pth)
   if [[ ! -f "${ckpts[0]}" ]]; then echo "MISSING PatchTST checkpoint $ds"; exit 1; fi
   prompt="${SELECTED}_window_horizon_correct"
   flags=(--mix_embeds --channel_interaction --early_channel_mode patch --forecast_conditioned --tri_mmoe --dynamic_window_stats --separate_numeric_path)
   extra=()
   case "$variant" in
    embedding_add) extra=(--embedding_fusion add);;
    embedding_gate) extra=(--embedding_fusion gate);;
   esac
   block=$h; eval_h=$h; additional=()
   if [[ $h -eq 24 ]]; then block=24; eval_h=48; additional=(--additional_test_pred_len 24); fi
   echo "START $variant $ds H=$h GPUs=2-7"
   CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 "$TORCHRUN" --standalone --nproc_per_node=6 run.py \
    --use_multi_gpu --task_name long_term_forecast --is_training 1 \
    --model_id "${ds}_embedding_${SELECTED}_${variant}_h${h}_seed2021" --model KIBR_PatchTST --data BatchMTSF \
    --root_path "dataset/$ds" --data_path "$ds.csv" \
    --prompt_encoder "$prompt" --numeric_encoder patchtst --numeric_ckpt "${ckpts[0]}" \
    --seq_len 48 --label_len 24 --token_len "$block" \
    --test_seq_len 48 --test_label_len 24 --test_pred_len "$eval_h" "${additional[@]}" \
    --enc_in 5 --dec_in 5 --c_out 5 --patch_len 24 --stride 12 \
    --d_model 128 --n_heads 4 --e_layers 2 --d_ff 256 \
    --batch_size 21 --num_workers 2 --train_epochs 100 --patience 100 \
    --learning_rate 0.0001 --lradj sigmoid --random_seed 2021 --gpu 0 \
    --des embedding10020261006 "${flags[@]}" "${extra[@]}" >"$log" 2>&1
   status=$?
   echo "END $variant $ds H=$h exit=$status"
   grep ^mse: "$log" | tail -2
   if [[ $status -ne 0 ]]; then tail -25 "$log"; return "$status"; fi
  done
 done
done
printf "ALL_EFP_EMBEDDING_COMPLETE\n"

}
until run_matrix; do echo "RETRY embedding queue $(date -Is)"; sleep 120; done
