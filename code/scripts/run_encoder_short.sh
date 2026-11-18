#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
TORCHRUN=/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/torchrun
datasets=(EFP_long_sub1 EFP_long_sub2 EFP_long_sub3)
texts=(gpt2 distilgpt2 bert roberta)
numerics=(patchtst timesnet)
for horizon in 6 12; do
 for numeric in "${numerics[@]}"; do
  for text in "${texts[@]}"; do
   for ds in "${datasets[@]}"; do
    if [[ "$numeric" == patchtst ]]; then model_name=PatchTST; else model_name=TimesNet; fi
    ckpts=(checkpoints/long_term_forecast_"$ds"_noprompt_seed2021_"$model_name"_*/checkpoint.pth)
    if [[ ! -f "${ckpts[0]}" ]]; then echo "Missing $numeric checkpoint for $ds" >&2; exit 1; fi
    log="test_results/encoderplug100_h${horizon}_${numeric}_${text}_${ds}_seed2021.log"
    if [[ -f "$log" ]] && [[ $(grep -c '^mse:' "$log") -eq 1 ]] && ! grep -q Traceback "$log"; then
     echo "SKIP completed H=$horizon $numeric $text $ds"
     continue
    fi
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 "$TORCHRUN" --standalone --nproc_per_node=6 \
     run.py --use_multi_gpu --task_name long_term_forecast --is_training 1 \
     --model_id "${ds}_encoderplug100_h${horizon}_${numeric}_${text}_seed2021" --model KIBR_PatchTST --data BatchMTSF \
     --root_path "dataset/$ds" --data_path "$ds.csv" \
     --prompt_encoder "${text}_window_horizon_correct" --numeric_encoder "$numeric" --numeric_ckpt "${ckpts[0]}" \
     --seq_len 48 --label_len 24 --token_len "$horizon" \
     --test_seq_len 48 --test_label_len 24 --test_pred_len "$horizon" \
     --enc_in 5 --dec_in 5 --c_out 5 --patch_len 24 --stride 12 \
     --d_model 128 --n_heads 4 --e_layers 2 --d_ff 256 \
     --batch_size 21 --num_workers 2 --train_epochs 100 --patience 100 \
     --learning_rate 0.0001 --lradj sigmoid --random_seed 2021 --gpu 0 \
     --des encoderplug10020260930 \
     --mix_embeds --channel_interaction --early_channel_mode patch \
     --forecast_conditioned --tri_mmoe --dynamic_window_stats --separate_numeric_path >"$log" 2>&1
    status=$?
    echo "H=$horizon $numeric $text $ds GPUs=2-7 exit=$status"
    grep '^mse:' "$log" | tail -1
    if [[ $status -ne 0 ]]; then tail -25 "$log"; exit "$status"; fi
   done
  done
 done
done
