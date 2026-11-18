import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # 设备配置
        self.device = configs.gpu
        print(f"Initializing GPT-2 on device: {self.device}")
        
        # 1. 加载 GPT-2 模型
        self.gpt2 = GPT2Model.from_pretrained(
            configs.llm_ckp_dir,
            torch_dtype=torch.float16, # 保持半精度以节省显存
        ).to(self.device)
        
        # 2. 加载 Tokenizer 并设置 Pad Token
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(configs.llm_ckp_dir)
        # GPT-2 默认没有 pad_token，必须设置，否则 padding=True 会报错
        if self.gpt2_tokenizer.pad_token is None:
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        
        self.hidden_dim_of_gpt2 = self.gpt2.config.n_embd 
        
        # 3. 冻结 GPT-2 参数
        for param in self.gpt2.parameters():
            param.requires_grad = False

    def forecast(self, x_mark_enc):        
        """
        x_mark_enc: 字符串列表，例如 ["This is series from 24.0h to 43.0h", ...]
        """
        # 1. 批量分词并自动进行 Padding 和 Truncation
        # padding=True 确保 Batch 内所有序列长度一致，解决 RuntimeError
        inputs = self.gpt2_tokenizer(
            x_mark_enc, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512 # 防止极端长序列 OOM
        ).to(self.device)
        
        # 2. 获取词嵌入 (Word Token Embeddings)
        # 也可以写成 self.gpt2.get_input_embeddings()(inputs['input_ids'])
        inputs_embeds = self.gpt2.wte(inputs['input_ids'])
        
        # 3. 通过 GPT-2 提取特征
        # 必须传入 attention_mask，让模型忽略 Padding 部分
        gpt_outputs = self.gpt2(
            inputs_embeds=inputs_embeds, 
            attention_mask=inputs['attention_mask']
        )[0] # [batch, seq_len, hidden_dim]
        
        # 4. 提取最后一个有效 token 的输出作为该时间戳的表征 (TE_i)
        # 由于有 Padding，最后一个 token 的位置 = attention_mask 的和 - 1
        last_token_indexes = inputs['attention_mask'].sum(dim=1) - 1
        batch_size = gpt_outputs.size(0)
        
        # 提取每个样本对应的真实结尾向量
        text_outputs = gpt_outputs[torch.arange(batch_size), last_token_indexes]
        
        return text_outputs # [batch, hidden_dim]
    
    def forward(self, x_mark_enc):
        return self.forecast(x_mark_enc)