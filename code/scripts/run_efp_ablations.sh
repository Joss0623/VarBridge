#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
TORCHRUN=/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/torchrun
mkdir -p test_results/efp_ablations
SELECTED=$(/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python scripts/select_efp_text_encoder.py) || exit 1
echo "VALIDATION_SELECTED_TEXT=$SELECTED"
# Full-model encoder is chosen from validation MSE by select_efp_text_encoder.py.
# Each control below uses the same split, seed, training budget and evaluation.
variants=(shuffled_mechanism no_mechanism generic_description random_semantic random_init_text learned_condition swap_to_trend cross_both no_prompt_self_attention no_knowledge no_channel no_early_channel early_raw no_dynamic_stats no_separate_numeric no_dynamic_gate no_initial_cross_attention sma_decomposition dema_decomposition no_forecast_conditioning additive_fusion dual_moe coarse_scale_2 coarse_scale_4)
for variant in "${variants[@]}"; do
 for ds in EFP_long_sub1 EFP_long_sub2 EFP_long_sub3; do
  for h in 6 12 24 48; do
   log="test_results/efp_ablations/${SELECTED}_${variant}_${ds}_h${h}_seed2021.log"
   expected=1; [[ $h -eq 48 ]] && expected=2
   if [[ -f "$log" ]] && [[ $(grep -c ^mse: "$log") -eq $expected ]] && ! grep -q Traceback "$log"; then
    echo "SKIP $variant $ds H=$h"; continue
   fi
   ckpts=(checkpoints/long_term_forecast_"$ds"_noprompt_seed2021_PatchTST_*/checkpoint.pth)
   if [[ ! -f "${ckpts[0]}" ]]; then echo "MISSING PatchTST checkpoint $ds"; exit 1; fi
   prompt="${SELECTED}_window_horizon_correct"
   flags=(--mix_embeds --channel_interaction --early_channel_mode patch --forecast_conditioned --tri_mmoe --dynamic_window_stats --separate_numeric_path)
   extra=()
   case "$variant" in
    shuffled_mechanism) prompt="${SELECTED}_window_horizon_shuffled";;
    no_mechanism) prompt="${SELECTED}_window_horizon_none";;
    generic_description) prompt="${SELECTED}_window_horizon_generic";;
    random_semantic) prompt="${SELECTED}_window_horizon_random";;
    random_init_text) prompt="${SELECTED}random_window_horizon_correct";;
    learned_condition) extra=(--prompt_control learned);;
    swap_to_trend) extra=(--initial_cross_target trend);;
    cross_both) extra=(--initial_cross_target both);;
    no_prompt_self_attention) extra=(--disable_prompt_self_attn);;
    no_knowledge) flags=(--channel_interaction --early_channel_mode patch);;
    no_channel) flags=(--mix_embeds --forecast_conditioned --tri_mmoe --dynamic_window_stats --separate_numeric_path);;
    no_early_channel) flags=(--mix_embeds --channel_interaction --forecast_conditioned --tri_mmoe --dynamic_window_stats --separate_numeric_path);;
    early_raw) flags=(--mix_embeds --channel_interaction --early_channel_mode raw --forecast_conditioned --tri_mmoe --dynamic_window_stats --separate_numeric_path);;
    no_dynamic_stats) flags=(--mix_embeds --channel_interaction --early_channel_mode patch --forecast_conditioned --tri_mmoe --separate_numeric_path);;
    no_separate_numeric) flags=(--mix_embeds --channel_interaction --early_channel_mode patch --forecast_conditioned --tri_mmoe --dynamic_window_stats);;
    no_dynamic_gate) extra=(--use_dynamic_gate 0);;
    no_initial_cross_attention) extra=(--use_cross_attn 0);;
    sma_decomposition) extra=(--ma_type sma);;
    dema_decomposition) extra=(--ma_type dema);;
    no_forecast_conditioning) flags=(--mix_embeds --channel_interaction --early_channel_mode patch);;
    additive_fusion) flags=(--mix_embeds --channel_interaction --early_channel_mode patch --forecast_conditioned);;
    dual_moe) flags=(--mix_embeds --channel_interaction --early_channel_mode patch --forecast_conditioned --direct_forecast --dual_moe);;
    coarse_scale_2) extra=(--coarse_factor 2);;
    coarse_scale_4) extra=(--coarse_factor 4);;
   esac
   block=$h; eval_h=$h; additional=()
   if [[ $h -eq 48 ]]; then block=24; eval_h=48; additional=(--additional_test_pred_len 24); fi
   if [[ "$variant" == random_init_text ]]; then
    cache_h=$h; [[ $h -eq 48 ]] && cache_h=24
    cache="dataset/$ds/${ds}__${SELECTED}random_window_semantic"
    [[ $cache_h -ne 24 ]] && cache="${cache}_h${cache_h}"
    if [[ ! -f "$cache.pt" ]]; then
     CUDA_VISIBLE_DEVICES=2 /home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python preprocess_window_semantic.py --dataset "$ds" --encoder "${SELECTED}random" --horizon "$cache_h" --gpu 0 --batch_size 96 >"test_results/efp_ablations/preprocess_random_${ds}_h${cache_h}.log" 2>&1 || exit 1
    fi
   fi
   echo "START $variant $ds H=$h GPUs=2-7"
   CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 "$TORCHRUN" --standalone --nproc_per_node=6 run.py \
    --use_multi_gpu --task_name long_term_forecast --is_training 1 \
    --model_id "${ds}_ablation_${SELECTED}_${variant}_h${h}_seed2021" --model KIBR_PatchTST --data BatchMTSF \
    --root_path "dataset/$ds" --data_path "$ds.csv" \
    --prompt_encoder "$prompt" --numeric_encoder patchtst --numeric_ckpt "${ckpts[0]}" \
    --seq_len 48 --label_len 24 --token_len "$block" \
    --test_seq_len 48 --test_label_len 24 --test_pred_len "$eval_h" "${additional[@]}" \
    --enc_in 5 --dec_in 5 --c_out 5 --patch_len 24 --stride 12 \
    --d_model 128 --n_heads 4 --e_layers 2 --d_ff 256 \
    --batch_size 21 --num_workers 2 --train_epochs 100 --patience 100 \
    --learning_rate 0.0001 --lradj sigmoid --random_seed 2021 --gpu 0 \
    --des ablation10020260930 "${flags[@]}" "${extra[@]}" >"$log" 2>&1
   status=$?
   echo "END $variant $ds H=$h exit=$status"
   grep ^mse: "$log" | tail -2
   if [[ $status -ne 0 ]]; then tail -25 "$log"; exit "$status"; fi
  done
 done
done
printf "ALL_EFP_ABLATIONS_COMPLETE\n"
