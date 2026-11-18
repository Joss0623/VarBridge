import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.decomp import DECOMP 
import os
import numpy as np

class PromptEncoder(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, prompt_embeds):
        attn_out, _ = self.self_attn(prompt_embeds, prompt_embeds, prompt_embeds)
        return self.norm(prompt_embeds + attn_out)

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, ts_query, prompt_kv):
        attn_out, attn_weights = self.cross_attn(query=ts_query, key=prompt_kv, value=prompt_kv)
        return self.norm(ts_query + attn_out), attn_weights

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len
        self.seq_len = configs.seq_len 
        self.hidden_dim_of_gpt2 = 768
    
        self.dataset_name = os.path.basename(configs.data_path).split('.')[0]
        self.device = f"cuda:{configs.local_rank}" if configs.use_multi_gpu else f"cuda:{configs.gpu}"
            
        # LLM 基座
        self.gpt2 = GPT2Model.from_pretrained(configs.llm_ckp_dir) 
        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False
            
        # =======================================================
        # Ablation 1 & 4: w/o Knowledge & w/o Cross Attention
        # =======================================================
        self.mix = configs.mix_embeds
        self.use_cross_attn = getattr(configs, 'use_cross_attn', True)
        
        if self.mix:
            self.LLMtodim = nn.Linear(self.hidden_dim_of_gpt2, self.hidden_dim_of_gpt2)
            self.Prompt_encoder = PromptEncoder(self.hidden_dim_of_gpt2, dropout=configs.dropout)
            if self.use_cross_attn:
                self.cross_attention = CrossAttentionFusion(self.hidden_dim_of_gpt2, dropout=configs.dropout)
            self.add_scale = nn.Parameter(torch.ones([]) * 0.1) 

        # =======================================================
        # Ablation 2: w/o EMA
        # =======================================================
        safe_kernel_size = getattr(configs, 'kernel_size', 25)

        self.decomp = DECOMP(ma_type=configs.ma_type, alpha=configs.alpha, beta=configs.beta, kernel_size=safe_kernel_size)
        
        self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_gpt2)
        self.trend_encoder = nn.Linear(self.token_len, self.hidden_dim_of_gpt2)
        self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.token_len)
        
        # =======================================================
        # Ablation 3: w/o Dynamic Gate
        # =======================================================
        self.use_dynamic_gate = getattr(configs, 'use_dynamic_gate', True)
        if self.use_dynamic_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim_of_gpt2 * 2, 128),
                nn.ReLU(),
                nn.Linear(128, self.hidden_dim_of_gpt2),
                nn.Sigmoid()
            )
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1. 实例归一化
        means = x_enc.mean(1, keepdim=True).detach()    
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        bs, L, n_vars = x_enc.shape 
        
        # 2. 序列分解
        seasonal_init, trend_init = self.decomp(x_enc) 
        s = seasonal_init.permute(0, 2, 1).reshape(bs * n_vars, L)
        t = trend_init.permute(0, 2, 1).reshape(bs * n_vars, L)

        # 3. 时域分块
        s_fold = s.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        t_fold = t.unfold(dimension=-1, size=self.token_len, step=self.token_len)

        # 4. 特征编码
        s_embeds = self.encoder(s_fold) 
        
        # --- 消融执行区: w/o Knowledge & w/o Cross Attention ---
        if self.mix:
            if x_mark_enc.shape[0] != s_embeds.shape[0]:
                n_vars_repeat = s_embeds.shape[0] // x_mark_enc.shape[0]
                x_mark_enc = x_mark_enc.repeat_interleave(n_vars_repeat, dim=0)
            
            prompt_out = self.LLMtodim(x_mark_enc)
            prompt_out = self.Prompt_encoder(prompt_out)
            
            if self.use_cross_attn:
                # 原始策略：交叉注意力动态对齐
                cross_out, attn_weights = self.cross_attention(ts_query=s_embeds, prompt_kv=prompt_out)
                s_embeds = s_embeds + self.add_scale * cross_out
            else:
                # 退化策略：朴素静态均值叠加
                naive_prompt = prompt_out.mean(dim=1, keepdim=True)
                s_embeds = s_embeds + self.add_scale * naive_prompt

        # LLM 前向传播
        s_out = self.gpt2(inputs_embeds=s_embeds).last_hidden_state
        t_out = self.trend_encoder(t_fold) 

        # --- 消融执行区: w/o Dynamic Gate ---
        if self.use_dynamic_gate:
            s_pooled = s_out.mean(dim=1, keepdim=True) 
            t_pooled = t_out.mean(dim=1, keepdim=True) 
            gating_input = torch.cat([s_pooled, t_pooled], dim=-1)
            g = self.gate_mlp(gating_input) 
            combined_out = g * s_out + (1 - g) * t_out
        else:
            combined_out = 0.5 * s_out + 0.5 * t_out

        # 5. 可视化数据保存
        if not self.training and not hasattr(self, '_has_saved_vis_data'):
            vis_dir = './vis_results'
            tsne_dir = os.path.join(vis_dir, 'tsne_data')
            os.makedirs(vis_dir, exist_ok=True)
            os.makedirs(tsne_dir, exist_ok=True)

            orig_data = (x_enc[0] * stdev[0] + means[0]).detach().cpu().numpy()
            trend_data = (trend_init[0] * stdev[0] + means[0]).detach().cpu().numpy()
            seasonal_data = (seasonal_init[0] * stdev[0]).detach().cpu().numpy()

            np.save(os.path.join(vis_dir, f'{self.dataset_name}_orig_data.npy'), orig_data)
            np.save(os.path.join(vis_dir, f'{self.dataset_name}_trend_data.npy'), trend_data)
            np.save(os.path.join(vis_dir, f'{self.dataset_name}_seasonal_data.npy'), seasonal_data)

            if self.mix and self.use_cross_attn and 'attn_weights' in locals():
                attn_w = attn_weights[0].detach().cpu().numpy()
                np.save(os.path.join(vis_dir, f'{self.dataset_name}_cross_attn_weights.npy'), attn_w)

            ts_emb = s_out.mean(dim=1).detach().cpu().numpy()
            fore_emb = combined_out.mean(dim=1).detach().cpu().numpy()
            if self.mix:
                prompt_emb = prompt_out.mean(dim=1).detach().cpu().numpy()
            else:
                prompt_emb = np.zeros_like(ts_emb)

            np.save(os.path.join(tsne_dir, f'{self.dataset_name}_ts.npy'), ts_emb)
            np.save(os.path.join(tsne_dir, f'{self.dataset_name}_prompt.npy'), prompt_emb)
            np.save(os.path.join(tsne_dir, f'{self.dataset_name}_forecast.npy'), fore_emb)

            self._has_saved_vis_data = True 
        
        # 6. 解码与反归一化
        dec_out = self.decoder(combined_out) 
        dec_out = dec_out.reshape(bs, n_vars, -1).permute(0, 2, 1) 
        
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)