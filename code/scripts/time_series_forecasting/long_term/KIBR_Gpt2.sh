model_name=Decomp_Gpt2
ma_type=ema
alpha=0.3
beta=0.3

# training one model with a context length
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/DYG_new_sub2/ \
  --data_path DYG_new_sub2.csv \
  --model_id DYG_new_sub2_672_96 \
  --model $model_name \
  --data custom \
  --llm_ckp_dir /home/home_new/qsmx/.cache/modelscope/hub/models/AI-ModelScope/gpt2 \
  --enc_in 3 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --batch_size 256 \
  --mlp_hidden_layers 0 \
  --train_epochs 100 \
  --patience 10 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --drop_last \
  --ma_type $ma_type \
  --alpha $alpha \
  --beta $beta \

# testing the model on all forecast lengths
for test_pred_len in 96 192
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/DYG_new_sub2/ \
  --data_path DYG_new_sub2.csv \
  --model_id DYG_new_sub2_672_96 \
  --model $model_name \
  --data custom \
  --llm_ckp_dir /home/home_new/qsmx/.cache/modelscope/hub/models/AI-ModelScope/gpt2 \
  --enc_in 3 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len $test_pred_len \
  --batch_size 256 \
  --learning_rate 0.0005 \
  --mlp_hidden_layers 0 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --drop_last \
  --ma_type $ma_type \
  --alpha $alpha \
  --beta $beta \
  --test_dir long_term_forecast_DYG_new_sub2_672_96_Decomp_Gpt2_custom_sl672_ll576_tl96_lr0.0001_bt256_wd0_hd256_hl0_cosFalse_mixTrue_test_0
done
