#!/bin/bash

echo "=================================================================="
echo "  🚀 Start Zero-Shot Forecasting for Multiple Models"
echo "  📏 Prediction Lengths: 24, 48"
echo "=================================================================="

# ==================================================================
# 1. KIBR_Gpt2
# ==================================================================
# model_name=KIBR_Gpt2
# ma_type=ema
# alpha=0.3
# beta=0.3 

# echo -e "\n>>> Running Model: $model_name <<<"
# for pred_len in 24 48
# do
#     echo "--> Current pred_len & token_len: $pred_len"
#     python -u run.py \
#       --task_name zero_shot_forecast \
#       --is_training 1 \
#       --root_path ./dataset/ \
#       --data_path DYG_source1/DYG_source1.csv \
#       --test_data_path DYG_target1/DYG_target1.csv \
#       --model_id "DYG_transfer_forecast_${pred_len}" \
#       --model $model_name \
#       --data DYG \
#       --llm_ckp_dir /home/home_new/qsmx/.cache/modelscope/hub/models/AI-ModelScope/gpt2 \
#       --seq_len 48 \
#       --label_len 0 \
#       --token_len $pred_len \
#       --test_seq_len 48 \
#       --test_label_len 0 \
#       --test_pred_len $pred_len \
#       --learning_rate 0.001 \
#       --lradj 'sigmoid'\
#       --ma_type $ma_type \
#       --alpha $alpha \
#       --beta $beta \
#       --mlp_hidden_dim 256 \
#       --mlp_hidden_layers 3 \
#       --train_epochs 500 \
#       --patience 20 \
#       --batch_size 256 \
#       --des 'Exp' \
#       --itr 1 \
#       --loss 'SMAPE' \
#       --use_amp \
#       --cosine \
#       --tmax 10 \
#       --val_set_shuffle
# done

# # ==================================================================
# # 2. AutoTimes_Gpt2
# # ==================================================================
# model_name=AutoTimes_Gpt2

# echo -e "\n>>> Running Model: $model_name <<<"
# for pred_len in 24 48
# do
#     echo "--> Current pred_len & token_len: $pred_len"
#     python -u run.py \
#       --task_name zero_shot_forecast \
#       --is_training 1 \
#        --root_path ./dataset/ \
#       --data_path DYG_source1/DYG_source1.csv \
#       --test_data_path DYG_target1/DYG_target1.csv \
#       --model_id "DYG_transfer_forecast_${pred_len}" \
#       --model $model_name \
#       --data DYG \
#       --llm_ckp_dir /home/home_new/qsmx/.cache/modelscope/hub/models/AI-ModelScope/gpt2 \
#       --seq_len 48 \
#       --label_len 0 \
#       --token_len $pred_len \
#       --test_seq_len 48 \
#       --test_label_len 0 \
#       --test_pred_len $pred_len \
#       --learning_rate 0.001 \
#       --mlp_hidden_dim 128 \
#       --mlp_hidden_layers 3 \
#       --train_epochs 100 \
#       --patience 10 \
#       --batch_size 256 \
#       --des 'Exp' \
#       --itr 1 \
#       --loss 'SMAPE' \
#       --use_amp \
#       --cosine \
#       --tmax 10 \
#       --val_set_shuffle
# done

# ==================================================================
# 3. TimesNet
# ==================================================================
model_name=TimesNet

echo -e "\n>>> Running Model: $model_name <<<"
for pred_len in 24 48
do
    echo "--> Current pred_len & token_len: $pred_len"
    python -u run.py \
      --task_name zero_shot_forecast \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path DYG_source1/DYG_source1.csv \
      --test_data_path DYG_target1/DYG_target1.csv \
      --model_id "DYG_transfer_forecast_${pred_len}" \
      --model $model_name \
      --data DYG \
      --top_k 3 \
      --freq w \
      --features M \
      --target kla \
      --enc_in 4 \
      --dec_in 4 \
      --c_out 4 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --d_model 16 \
      --d_ff 16 \
      --seq_len 48 \
      --label_len 0 \
      --token_len $pred_len \
      --test_seq_len 48 \
      --test_label_len 0 \
      --test_pred_len $pred_len \
      --learning_rate 0.0001 \
      --train_epochs 100 \
      --patience 5 \
      --batch_size 256 \
      --des 'Exp' \
      --itr 1 \
      --loss 'SMAPE' \
      --use_amp \
      --val_set_shuffle
done

# ==================================================================
# 4. PatchTST
# ==================================================================
model_name=PatchTST

echo -e "\n>>> Running Model: $model_name <<<"
for pred_len in 24 48
do
    echo "--> Current pred_len & token_len: $pred_len"
    python -u run.py \
      --task_name zero_shot_forecast \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path DYG_source1/DYG_source1.csv \
      --test_data_path DYG_target1/DYG_target1.csv \
      --model_id "DYG_transfer_forecast_${pred_len}" \
      --model $model_name \
      --data DYG \
      --llm_ckp_dir /home/home_new/qsmx/.cache/modelscope/hub/models/AI-ModelScope/gpt2 \
      --seq_len 48 \
      --label_len 0 \
      --token_len $pred_len \
      --test_seq_len 48 \
      --test_label_len 0 \
      --test_pred_len $pred_len \
      --learning_rate 0.0001 \
      --train_epochs 100 \
      --patience 5 \
      --batch_size 256 \
      --des 'Exp' \
      --itr 1 \
      --loss 'SMAPE' \
      --use_amp \
      --val_set_shuffle
done

# ==================================================================
# 5. TiDE
# ==================================================================
model_name=TiDE

echo -e "\n>>> Running Model: $model_name <<<"
for pred_len in 24 48
do
    echo "--> Current pred_len & token_len: $pred_len"
    python -u run.py \
      --task_name zero_shot_forecast \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path DYG_source1/DYG_source1.csv \
      --test_data_path DYG_target1/DYG_target1.csv \
      --model_id "DYG_transfer_forecast_${pred_len}" \
      --model $model_name \
      --data DYG \
      --freq w \
      --features M \
      --target kla \
      --enc_in 4 \
      --dec_in 4 \
      --c_out 4 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --d_model 16 \
      --d_ff 16 \
      --seq_len 48 \
      --label_len 0 \
      --token_len $pred_len \
      --test_seq_len 48 \
      --test_label_len 0 \
      --test_pred_len $pred_len \
      --learning_rate 0.0001 \
      --train_epochs 100 \
      --patience 5 \
      --batch_size 256 \
      --des 'Exp' \
      --itr 1 \
      --loss 'SMAPE' \
      --use_amp \
      --val_set_shuffle
done

echo -e "\n🎉 All models and lengths finished successfully!"