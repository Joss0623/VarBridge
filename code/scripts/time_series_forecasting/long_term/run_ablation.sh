#!/bin/bash

# ==================================================================
# 🌐 全局基础超参数设定 (Global Hyperparameters)
# ==================================================================
model_name=KIBR_Gpt2
llm_ckp_dir="/home/home_new/qsmx/.cache/modelscope/hub/models/AI-ModelScope/gpt2"
seq_len=672
label_len=576
token_len=96
learning_rate=0.0001
train_epochs=100
patience=10
batch_size=256
gpu=7

alpha=0.3
beta=0.3

# 定义我们需要跑的 4 组实验
# ablation_names=("wo_Knowledge" "wo_EMA" "wo_Gate" "wo_CrossAttn")
ablation_names=("wo_EMA")

echo "=================================================================="
echo "  🚀 Start Comprehensive Ablation Studies"
echo "  📊 Experiments: 4 Ablations across 2 Datasets"
echo "=================================================================="

# ==================================================================
# 🛠️ 核心执行函数 (Core Experiment Function)
# 接收参数: 数据集别名, root_path, data_path, data_name, target, enc_in
# ==================================================================
run_experiment() {
    local ds_alias=$1
    local root_path=$2
    local data_path=$3
    local data_name=$4
    local target=$5
    local enc_in=$6

    echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "  🌟 [Dataset Entering]: $ds_alias"
    echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

    for ab_name in "${ablation_names[@]}"; do
        echo -e "\n=================================================================="
        echo "  🔥 Current Run: [Dataset: $ds_alias] | [Ablation: $ab_name]"
        echo "=================================================================="

        # 根据消融名称，动态生成专属的超参数指令
        # 注意：因为 mix_embeds 是 store_true，启用时带上该指令，关闭时直接不写即可。
        case $ab_name in
            # "wo_Knowledge") 
                #flags="--ma_type ema --use_dynamic_gate 1 --use_cross_attn 1" ;;
            "wo_EMA")       
                flags="--ma_type sma --mix_embeds --use_dynamic_gate 1 --use_cross_attn 1" ;;
            #"wo_Gate")      
                #flags="--ma_type ema --mix_embeds --use_dynamic_gate 0 --use_cross_attn 1" ;;
            #"wo_CrossAttn") 
                #flags="--ma_type ema --mix_embeds --use_dynamic_gate 1 --use_cross_attn 0" ;;
        esac

        # 构建专有的 model_id 确保权重分离
        model_id="${ds_alias}_${seq_len}_${token_len}_${ab_name}"

        # 拼接基础通用指令
        base_cmd="python -u run.py \
            --task_name long_term_forecast \
            --root_path $root_path \
            --data_path $data_path \
            --model_id $model_id \
            --model $model_name \
            --data $data_name \
            --llm_ckp_dir $llm_ckp_dir \
            --enc_in $enc_in \
            --seq_len $seq_len \
            --label_len $label_len \
            --token_len $token_len \
            --batch_size $batch_size \
            --mlp_hidden_layers 3 \
            --use_amp \
            --gpu $gpu \
            --tmax 10 \
            --drop_last \
            --learning_rate $learning_rate \
            --lradj 'sigmoid' \
            --alpha $alpha \
            --beta $beta"

        # 如果是 DYG 等 custom 数据集，追加 target 和 features 指令
        if [ "$data_name" == "custom" ]; then
            base_cmd="$base_cmd --target $target --features M"
        fi

        # ---------------------------------------------------------
        # [步骤 1]: 执行训练 (Train)
        # ---------------------------------------------------------
        echo ">>> [Phase 1/2] Training Model..."
        eval "$base_cmd --is_training 1 --train_epochs $train_epochs --patience $patience $flags"

        # ---------------------------------------------------------
        # [步骤 2]: 执行多尺度测试 (Test 96 & 192)
        # ---------------------------------------------------------
        for test_pred_len in 96 192 
        do
            echo ">>> [Phase 2/2] Testing | Pred Len: $test_pred_len"
            eval "$base_cmd --is_training 0 \
                --test_seq_len $seq_len \
                --test_label_len $label_len \
                --test_pred_len $test_pred_len \
                $flags"
        done
        
        echo "✅ Finished: [$ab_name] on [$ds_alias]"
    done
}

# ==================================================================
# 🚦 启动流水线 (Pipeline Execution)
# 参数顺序: 数据集别名, root_path, data_path, data_name, target, enc_in
# ==================================================================

# 2. 跑 红霉素 发酵子集 1
run_experiment "DYG_sub1" "./dataset/DYG_new_sub1/" "DYG_new_sub1.csv" "custom" "nn" 3

# 3. 跑 红霉素 发酵子集 2
run_experiment "DYG_sub2" "./dataset/DYG_new_sub2/" "DYG_new_sub2.csv" "custom" "hx" 3

echo -e "\n=================================================================="
echo "  🎉 🎉 All Ablation Experiments Completed Successfully! 🎉 🎉"
echo "=================================================================="