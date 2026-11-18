#!/bin/bash

model_name=KIBR_Gpt2
llm_ckp_dir="/home/home_new/qsmx/.cache/modelscope/hub/models/AI-ModelScope/gpt2"
seq_len=672
label_len=576
token_len=96
learning_rate=0.0001
train_epochs=500
patience=20
batch_size=256
gpu=7

ma_type=ema
alpha=0.3
beta=0.3

echo "=================================================================="
echo "  🚀 Start Pipeline: $model_name"
echo "  📊 Mode: Train once (len 672) -> Test multiple lengths (96, 192)"
echo "=================================================================="


echo -e "\n>>> [1/2] Processing Dataset: DYG_new_sub1..."
root_path='./dataset/DYG_new_sub1/' 
data_path='DYG_new_sub1.csv'
data_name='custom'
target='nn'
model_id="DYG_sub1_${seq_len}_${token_len}"
enc_in=3

test_dir_2="long_term_forecast_${model_id}_${model_name}_${data_name}_sl${seq_len}_ll${label_len}_tl${token_len}_lr${learning_rate}_bt${batch_size}_wd0_hd256_hl3_cosFalse_mixTrue_test_0"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id \
  --model $model_name \
  --data $data_name \
  --target $target \
  --features M \
  --llm_ckp_dir $llm_ckp_dir \
  --enc_in $enc_in \
  --seq_len $seq_len \
  --label_len $label_len \
  --token_len $token_len \
  --test_seq_len $seq_len \
  --test_label_len $label_len \
  --test_pred_len $token_len \
  --batch_size $batch_size \
  --mlp_hidden_layers 3 \
  --train_epochs $train_epochs \
  --patience $patience \
  --use_amp \
  --gpu $gpu \
  --tmax 10 \
  --mix_embeds \
  --drop_last \
  --learning_rate $learning_rate \
  --lradj 'sigmoid' \
  --ma_type $ma_type \
  --alpha $alpha \
  --beta $beta

# for test_pred_len in 96 192 
# do
#   echo "--> Testing DYG_new_sub1 | Pred Len: $test_pred_len"
#   python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 0 \
#     --root_path $root_path \
#     --data_path $data_path \
#     --model_id $model_id \
#     --model $model_name \
#     --data $data_name \
#     --target $target \
#     --features M \
#     --llm_ckp_dir $llm_ckp_dir \
#     --enc_in $enc_in \
#     --seq_len $seq_len \
#     --label_len $label_len \
#     --token_len $token_len \
#     --test_seq_len $seq_len \
#     --test_label_len $label_len \
#     --test_pred_len $test_pred_len \
#     --batch_size $batch_size \
#     --mlp_hidden_layers 3 \
#     --use_amp \
#     --gpu $gpu \
#     --tmax 10 \
#     --mix_embeds \
#     --drop_last \
#     --learning_rate $learning_rate \
#     --lradj 'sigmoid' \
#     --ma_type $ma_type \
#     --alpha $alpha \
#     --beta $beta \
#     --test_dir $test_dir_2
# done


echo -e "\n>>> [2/2] Processing Dataset: DYG_new_sub2..."
root_path='./dataset/DYG_new_sub2/'  
data_path='DYG_new_sub2.csv'
data_name='custom'
target='hx'
model_id="DYG_sub2_${seq_len}_${token_len}"
enc_in=3

test_dir_3="long_term_forecast_${model_id}_${model_name}_${data_name}_sl${seq_len}_ll${label_len}_tl${token_len}_lr${learning_rate}_bt${batch_size}_wd0_hd256_hl3_cosFalse_mixTrue_test_0"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id \
  --model $model_name \
  --data $data_name \
  --target $target \
  --features M \
  --llm_ckp_dir $llm_ckp_dir \
  --enc_in $enc_in \
  --seq_len $seq_len \
  --label_len $label_len \
  --token_len $token_len \
  --test_seq_len $seq_len \
  --test_label_len $label_len \
  --test_pred_len $token_len \
  --batch_size $batch_size \
  --mlp_hidden_layers 3 \
  --train_epochs $train_epochs \
  --patience $patience \
  --use_amp \
  --gpu $gpu \
  --tmax 10 \
  --mix_embeds \
  --drop_last \
  --learning_rate $learning_rate \
  --lradj 'sigmoid' \
  --ma_type $ma_type \
  --alpha $alpha \
  --beta $beta

# for test_pred_len in 96 192 
# do
#   echo "--> Testing DYG_new_sub2 | Pred Len: $test_pred_len"
#   python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 0 \
#     --root_path $root_path \
#     --data_path $data_path \
#     --model_id $model_id \
#     --model $model_name \
#     --data $data_name \
#     --target $target \
#     --features M \
#     --llm_ckp_dir $llm_ckp_dir \
#     --enc_in $enc_in \
#     --seq_len $seq_len \
#     --label_len $label_len \
#     --token_len $token_len \
#     --test_seq_len $seq_len \
#     --test_label_len $label_len \
#     --test_pred_len $test_pred_len \
#     --batch_size $batch_size \
#     --mlp_hidden_layers 3 \
#     --use_amp \
#     --gpu $gpu \
#     --tmax 10 \
#     --mix_embeds \
#     --drop_last \
#     --learning_rate $learning_rate \
#     --lradj 'sigmoid' \
#     --ma_type $ma_type \
#     --alpha $alpha \
#     --beta $beta \
#     --test_dir $test_dir_3
# done

echo "=================================================================="
echo "  🎉 All experiments finished successfully!"
echo "=================================================================="