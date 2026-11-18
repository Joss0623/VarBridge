import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy
from layers.decomp import DECOMP
from models.PatchTST import Model as PatchTST
from models.TimesNet import Model as TimesNet

class PromptEncoder(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, prompt_embeds):
        out, _ = self.self_attn(prompt_embeds, prompt_embeds, prompt_embeds, need_weights=False)
        return self.norm(prompt_embeds + out)

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, ts_query, prompt_kv):
        out, weights = self.cross_attn(ts_query, prompt_kv, prompt_kv, need_weights=False)
        return self.norm(ts_query + out), weights

class Model(nn.Module):
    """CHO audit variant: target-bound mechanism memory, separate numerical statistics."""
    def __init__(self, configs):
        super().__init__()
        self.numeric_encoder_name = getattr(configs, 'numeric_encoder', 'patchtst')
        if self.numeric_encoder_name not in ('patchtst', 'timesnet'):
            raise ValueError('numeric_encoder must be patchtst or timesnet')
        checkpoint = getattr(configs, 'numeric_ckpt', '') or configs.patchtst_ckpt
        if not checkpoint:
            raise ValueError('--numeric_ckpt (or --patchtst_ckpt) is required')
        base_config = copy(configs)
        base_config.mix_embeds = False
        base_config.token_len = 24  # Pretrained numerical backbones forecast 24 steps.
        numeric_class = PatchTST if self.numeric_encoder_name == 'patchtst' else TimesNet
        self.patch_backbone = numeric_class(base_config)
        self.patch_backbone.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
        self.patch_backbone.requires_grad_(False)
        self.patch_backbone.eval()
        self.token_len, self.patch_len, self.stride = configs.token_len, configs.patch_len, configs.stride
        self.hidden_dim = configs.d_model
        if self.numeric_encoder_name == 'timesnet':
            self.timesnet_channel_adapter = nn.Linear(self.patch_len, self.hidden_dim)
        self.mix = configs.mix_embeds
        self.forecast_conditioned = bool(getattr(configs, 'forecast_conditioned', False))
        self.moe_fusion = bool(getattr(configs, 'moe_fusion', False))
        self.direct_forecast = bool(getattr(configs, 'direct_forecast', False))
        self.dual_moe = bool(getattr(configs, 'dual_moe', False))
        self.tri_mmoe = bool(getattr(configs, 'tri_mmoe', False))
        self.embedding_fusion = getattr(configs, 'embedding_fusion', 'output')
        if self.embedding_fusion != 'output' and not self.tri_mmoe:
            raise ValueError('Embedding fusion requires the forecast-conditioned full configuration')
        self.dynamic_window_stats = bool(getattr(configs, 'dynamic_window_stats', False))
        self.separate_numeric_path = bool(getattr(configs, 'separate_numeric_path', False))
        if self.forecast_conditioned and not self.mix:
            raise ValueError('Forecast-conditioned alignment requires --mix_embeds')
        if self.moe_fusion and not self.forecast_conditioned:
            raise ValueError('MoE fusion requires --forecast_conditioned')
        if self.direct_forecast and not self.forecast_conditioned:
            raise ValueError('Direct forecast requires --forecast_conditioned')
        if self.dual_moe and (not self.direct_forecast or self.moe_fusion):
            raise ValueError('Dual MoE requires --direct_forecast and excludes --moe_fusion')
        if self.tri_mmoe and (self.direct_forecast or self.dual_moe or self.moe_fusion or not self.forecast_conditioned):
            raise ValueError('Three-way MMoE requires a frozen base prediction and forecast-conditioned prompts')
        if (self.dynamic_window_stats or self.separate_numeric_path) and not self.tri_mmoe:
            raise ValueError('Dynamic window statistics and separate numeric path require --tri_mmoe')
        self.prompt_control = getattr(configs, 'prompt_control', 'text')
        self.initial_cross_target = getattr(configs, 'initial_cross_target', 'seasonal')
        self.disable_prompt_self_attn = bool(getattr(configs, 'disable_prompt_self_attn', False))
        self.use_cross_attn = bool(getattr(configs, 'use_cross_attn', True))
        self.use_dynamic_gate = bool(getattr(configs, 'use_dynamic_gate', True))
        self.use_channel = bool(getattr(configs, 'channel_interaction', False))
        self.early_channel_mode = getattr(configs, 'early_channel_mode', 'none')
        if self.early_channel_mode not in ('none', 'raw', 'patch'):
            raise ValueError('early_channel_mode must be none, raw, or patch')
        if self.early_channel_mode != 'none':
            early_dim = configs.seq_len if self.early_channel_mode == 'raw' else self.hidden_dim
            self.early_channel_attention = nn.MultiheadAttention(
                early_dim, configs.n_heads, dropout=configs.dropout, batch_first=True)
            self.early_channel_norm = nn.LayerNorm(early_dim)
            self.early_channel_scale = nn.Parameter(torch.tensor(0.1))
        self.coarse_factor = int(getattr(configs, 'coarse_factor', 0))
        if self.coarse_factor not in (0, 2, 4):
            raise ValueError('coarse_factor must be 0, 2, or 4')
        if self.coarse_factor:
            if configs.seq_len % self.coarse_factor or self.patch_len % self.coarse_factor or self.stride % self.coarse_factor:
                raise ValueError('Coarse factor must divide seq_len, patch_len and stride')
            coarse_patch_len = self.patch_len // self.coarse_factor
            self.coarse_season_encoder = nn.Linear(coarse_patch_len, self.hidden_dim)
            self.coarse_trend_encoder = nn.Linear(coarse_patch_len, self.hidden_dim)
            self.coarse_season_scale = nn.Parameter(torch.tensor(0.1))
            self.coarse_trend_scale = nn.Parameter(torch.tensor(0.1))
        self.decomp = DECOMP(ma_type=configs.ma_type, alpha=configs.alpha,
                             beta=configs.beta, kernel_size=getattr(configs, 'kernel_size', 25))
        self.encoder = nn.Linear(self.patch_len, self.hidden_dim)
        self.trend_encoder = nn.Linear(self.patch_len, self.hidden_dim)
        if self.mix:
            self.LLMtodim = nn.Linear(getattr(configs, 'text_embed_dim', 768), self.hidden_dim)
            self.Prompt_encoder = PromptEncoder(self.hidden_dim, configs.n_heads, configs.dropout)
            self.mechanism_projection = nn.Sequential(
                nn.LayerNorm(getattr(configs, 'text_embed_dim', 768)),
                nn.Linear(getattr(configs, 'text_embed_dim', 768), self.hidden_dim),
                nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim))
            self.stats_scale = nn.Parameter(torch.tensor(0.1))
            if self.prompt_control == 'learned':
                self.learned_prompt = nn.Parameter(torch.randn(configs.enc_in, configs.enc_in + 3, self.hidden_dim) * 0.02)
            if self.dynamic_window_stats:
                self.current_stats_encoder = nn.Sequential(
                    nn.Linear(10, self.hidden_dim), nn.GELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim))
                self.current_stats_role = nn.Parameter(torch.zeros(self.hidden_dim))
            if self.use_cross_attn:
                self.cross_attention = CrossAttentionFusion(self.hidden_dim, configs.n_heads, configs.dropout)
            self.add_scale = nn.Parameter(torch.tensor(0.1))
            if self.forecast_conditioned:
                self.target_embedding = nn.Embedding(configs.enc_in, self.hidden_dim)
                self.horizon_embedding = nn.Embedding(2 * self.token_len + 1, self.hidden_dim)
                self.query_state = nn.Linear(self.hidden_dim, self.hidden_dim)
                self.query_norm = nn.LayerNorm(self.hidden_dim)
                self.semantic_attention = nn.MultiheadAttention(
                    self.hidden_dim, configs.n_heads, dropout=configs.dropout, batch_first=True)
                self.semantic_delta = nn.Linear(self.hidden_dim, 1)
                self.semantic_gate = nn.Linear(2 * self.hidden_dim, 1)
                nn.init.normal_(self.semantic_delta.weight, std=0.02)
                nn.init.zeros_(self.semantic_delta.bias)
                nn.init.zeros_(self.semantic_gate.weight)
                nn.init.constant_(self.semantic_gate.bias, -1.0)
                if self.moe_fusion:
                    self.fusion_gate = nn.Linear(2 * self.hidden_dim, 3)
                    nn.init.zeros_(self.fusion_gate.weight)
                    with torch.no_grad():
                        self.fusion_gate.bias.copy_(torch.tensor([0.0, 2.0, 0.0]))
                    self.fusion_log_scale = nn.Parameter(torch.tensor(0.54132485))
                if self.embedding_fusion != 'output':
                    self.condition_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
                    if self.embedding_fusion == 'gate':
                        self.embedding_gate = nn.Sequential(
                            nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.GELU(),
                            nn.Linear(self.hidden_dim, self.hidden_dim), nn.Sigmoid())
                if (self.dual_moe or self.tri_mmoe) and self.embedding_fusion == 'output':
                    def expert(in_dim):
                        net = nn.Sequential(nn.Linear(in_dim, self.hidden_dim), nn.GELU(),
                                            nn.Linear(self.hidden_dim, 1))
                        nn.init.normal_(net[-1].weight, std=0.01)
                        nn.init.zeros_(net[-1].bias)
                        return net
                    self.numeric_experts = nn.ModuleList([expert(self.hidden_dim) for _ in range(2)])
                    self.semantic_experts = nn.ModuleList([expert(2 * self.hidden_dim) for _ in range(2)])
                    numeric_router_dim = self.hidden_dim if self.separate_numeric_path else 2 * self.hidden_dim
                    self.numeric_router = nn.Linear(numeric_router_dim, 3)
                    self.semantic_router = nn.Linear(2 * self.hidden_dim, 3)
                    nn.init.zeros_(self.numeric_router.weight)
                    nn.init.zeros_(self.semantic_router.weight)
                    nn.init.zeros_(self.semantic_router.bias)
                    with torch.no_grad():
                        self.numeric_router.bias.copy_(torch.tensor([2.0, 0.0, 0.0]))
                    if self.dual_moe:
                        self.semantic_mix_gate = nn.Linear(2 * self.hidden_dim, 1)
                        nn.init.zeros_(self.semantic_mix_gate.weight)
                        nn.init.constant_(self.semantic_mix_gate.bias, -1.0)
                    else:
                        self.tri_router = nn.Linear(2 * self.hidden_dim, 3)
                        nn.init.zeros_(self.tri_router.weight)
                        with torch.no_grad():
                            self.tri_router.bias.copy_(torch.tensor([1.0, 0.0, 0.0]))
        if self.use_channel:
            self.channel_attention = nn.MultiheadAttention(self.hidden_dim, configs.n_heads,
                                                            dropout=configs.dropout, batch_first=True)
            self.channel_norm = nn.LayerNorm(self.hidden_dim)
        if self.use_dynamic_gate:
            self.gate_mlp = nn.Sequential(nn.Linear(self.hidden_dim * 2, 128), nn.ReLU(),
                                          nn.Linear(128, self.hidden_dim), nn.Sigmoid())
        self.decoder = nn.Linear(self.hidden_dim, self.token_len)
        patch_count = (self.patch_backbone.head_nf // self.hidden_dim if self.numeric_encoder_name == 'patchtst'
                       else (configs.seq_len - self.patch_len) // self.stride + 2)
        self.residual_head = nn.Linear(patch_count * self.token_len, self.token_len)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def train(self, mode=True):
        super().train(mode)
        self.patch_backbone.eval()
        return self

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        normed = x_enc - means
        stdev = torch.sqrt(torch.var(normed, dim=1, keepdim=True, unbiased=False) + 1e-5)
        normed = normed / stdev
        bs, length, n_vars = normed.shape
        decomp_input = normed
        if self.early_channel_mode == 'raw':
            raw_channels = normed.transpose(1, 2)
            raw_cross, _ = self.early_channel_attention(
                raw_channels, raw_channels, raw_channels, need_weights=False)
            decomp_input = normed + self.early_channel_scale * self.early_channel_norm(raw_cross).transpose(1, 2)
        with torch.no_grad():
            if self.numeric_encoder_name == 'patchtst':
                patches, _ = self.patch_backbone.patch_embedding(normed.permute(0, 2, 1))
                frozen, _ = self.patch_backbone.encoder(patches)
                n_patches = frozen.shape[1]
                if not self.direct_forecast:
                    base_features = frozen.reshape(bs, n_vars, n_patches, self.hidden_dim).permute(0, 1, 3, 2)
                    base_norm = self.patch_backbone.head(base_features).permute(0, 2, 1)
            else:
                full_forecast, temporal_features = self.patch_backbone.forecast_with_features(
                    normed, None, None, None)
                base_norm = full_forecast[:, -self.token_len:, :]
                global_patches = F.pad(temporal_features.transpose(1, 2), (0, self.stride),
                                       mode='replicate').unfold(-1, self.patch_len, self.stride)
                global_patches = global_patches.mean(-1).transpose(1, 2)
                n_patches = global_patches.shape[1]
                channel_patches = F.pad(normed.transpose(1, 2), (0, self.stride),
                                        mode='replicate').unfold(-1, self.patch_len, self.stride)
            if not self.direct_forecast:
                base_norm = base_norm[:, :self.token_len, :]
                base = base_norm * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)
        if self.numeric_encoder_name == 'timesnet':
            frozen = (global_patches[:, None, :, :] +
                      self.timesnet_channel_adapter(channel_patches)).reshape(
                          bs * n_vars, n_patches, self.hidden_dim)
        residual_frozen = frozen
        if self.early_channel_mode == 'patch':
            patch_channels = frozen.reshape(bs, n_vars, n_patches, self.hidden_dim).transpose(1, 2)
            patch_channels = patch_channels.reshape(bs * n_patches, n_vars, self.hidden_dim)
            patch_cross, _ = self.early_channel_attention(
                patch_channels, patch_channels, patch_channels, need_weights=False)
            patch_channels = patch_channels + self.early_channel_scale * self.early_channel_norm(patch_cross)
            residual_frozen = patch_channels.reshape(bs, n_patches, n_vars, self.hidden_dim)
            residual_frozen = residual_frozen.transpose(1, 2).reshape(bs * n_vars, n_patches, self.hidden_dim)
        seasonal, trend = self.decomp(decomp_input)
        def make_patches(x):
            x = x.permute(0, 2, 1).reshape(bs * n_vars, length)
            return F.pad(x, (0, self.stride), mode='replicate').unfold(-1, self.patch_len, self.stride)
        s_embeds = residual_frozen + self.encoder(make_patches(seasonal))
        t_embeds = self.trend_encoder(make_patches(trend))
        if self.coarse_factor:
            factor = self.coarse_factor
            coarse = F.avg_pool1d(decomp_input.transpose(1, 2), factor, stride=factor).transpose(1, 2)
            coarse_season, coarse_trend = self.decomp(coarse)
            coarse_len = length // factor
            coarse_patch = self.patch_len // factor
            coarse_stride = self.stride // factor
            def coarse_patches(x):
                x = x.permute(0, 2, 1).reshape(bs * n_vars, coarse_len)
                return F.pad(x, (0, coarse_stride), mode='replicate').unfold(-1, coarse_patch, coarse_stride)
            s_embeds = s_embeds + self.coarse_season_scale * self.coarse_season_encoder(coarse_patches(coarse_season))
            t_embeds = t_embeds + self.coarse_trend_scale * self.coarse_trend_encoder(coarse_patches(coarse_trend))
        def channel_mix(embeds):
            by_patch = embeds.reshape(bs, n_vars, n_patches, self.hidden_dim).transpose(1, 2)
            by_patch = by_patch.reshape(bs * n_patches, n_vars, self.hidden_dim)
            cross, _ = self.channel_attention(by_patch, by_patch, by_patch, need_weights=False)
            cross = self.channel_norm(by_patch + cross)
            return cross.reshape(bs, n_patches, n_vars, self.hidden_dim).transpose(1, 2).reshape(
                bs * n_vars, n_patches, self.hidden_dim)
        if self.use_channel and self.separate_numeric_path:
            s_embeds = channel_mix(s_embeds)
        numeric_s_embeds = s_embeds
        if self.mix:
            if x_mark_enc.shape[-1] != self.LLMtodim.in_features:
                raise ValueError('Prompt embedding dimension does not match --text_embed_dim')
            if x_mark_enc.ndim == 4:
                if x_mark_enc.shape[:2] != (bs, n_vars):
                    raise ValueError('Expected window prompts [batch, variables, tokens, 768]')
                if self.forecast_conditioned and x_mark_enc.shape[2] != n_vars + 3:
                    raise ValueError('Expected window, one mechanism token per variable, stage and operation')
                prompt = x_mark_enc.reshape(bs * n_vars, x_mark_enc.shape[2], x_mark_enc.shape[3])
            else:
                prompt = x_mark_enc.repeat_interleave(n_vars, dim=0)
            # Raw slots: observed window, five variable mechanisms, stage, operation.
            # Explicit diagonal binding: target j can retrieve only mechanism j.
            variable_ids = torch.arange(bs * n_vars, device=prompt.device) % n_vars
            own_mechanism = prompt[torch.arange(bs * n_vars, device=prompt.device), 1 + variable_ids]
            bound_knowledge = self.mechanism_projection(own_mechanism) + self.target_embedding(variable_ids)
            # General context never self-attends to mechanism memory.
            prompt = self.LLMtodim(torch.cat((prompt[:, :1], prompt[:, 1+n_vars:]), dim=1))
            if self.prompt_control == 'learned':
                prompt = self.learned_prompt.unsqueeze(0).expand(bs, -1, -1, -1).reshape(
                    bs * n_vars, n_vars + 3, self.hidden_dim)
            if self.dynamic_window_stats:
                # Token 0 is the fixed initial-window text. This extra token is
                # recomputed from the actual input window on every recursive call.
                current = x_enc.transpose(1, 2).reshape(bs * n_vars, length)
                centered = current - current.mean(1, keepdim=True)
                variance = centered.square().mean(1).clamp_min(1e-6)
                ac1 = (centered[:, :-1] * centered[:, 1:]).mean(1) / variance
                ac12 = (centered[:, :-12] * centered[:, 12:]).mean(1) / variance
                stats = torch.stack((current.min(1).values, current.max(1).values,
                                     current.median(1).values, current.mean(1),
                                     variance.sqrt(), current[:, 0], current[:, -1],
                                     current[:, -1] - current[:, 0], ac1, ac12), dim=-1)
                current_token = self.current_stats_encoder(stats) + self.current_stats_role
                # Numerical statistics affect numerical seasonal representation only.
                # They are never keys/values in either text attention memory.
                s_embeds = s_embeds + self.stats_scale * current_token[:, None, :]
            if not self.disable_prompt_self_attn:
                prompt = self.Prompt_encoder(prompt)
            if self.use_cross_attn:
                if self.initial_cross_target in ('seasonal', 'both'):
                    cross, _ = self.cross_attention(s_embeds, prompt)
                    s_embeds = s_embeds + self.add_scale * cross
                if self.initial_cross_target in ('trend', 'both'):
                    cross, _ = self.cross_attention(t_embeds, prompt)
                    t_embeds = t_embeds + self.add_scale * cross
            else:
                s_embeds = s_embeds + self.add_scale * prompt.mean(1, keepdim=True)
        if self.use_channel and not self.separate_numeric_path:
            s_embeds = channel_mix(s_embeds)
        if not self.separate_numeric_path:
            numeric_s_embeds = s_embeds
        if self.use_dynamic_gate:
            pooled = torch.cat([numeric_s_embeds.mean(1, keepdim=True), t_embeds.mean(1, keepdim=True)], -1)
            gate = self.gate_mlp(pooled)
            combined = gate * numeric_s_embeds + (1 - gate) * t_embeds
        else:
            combined = 0.5 * numeric_s_embeds + 0.5 * t_embeds
        decoded = self.decoder(combined).reshape(bs, n_vars, -1)
        residual = self.residual_head(decoded).permute(0, 2, 1)
        if self.forecast_conditioned:
            state_embeds = s_embeds if self.separate_numeric_path else residual_frozen
            state = (state_embeds + t_embeds).mean(1).reshape(bs, n_vars, self.hidden_dim)
            target = self.target_embedding(torch.arange(n_vars, device=x_enc.device))
            # The model is trained on one token_len-step block. Recursive calls
            # therefore reuse its trained relative positions instead of unseen
            # absolute positions (token_len+1 .. 2*token_len).
            steps = torch.arange(1, self.token_len + 1, device=x_enc.device)
            if int(steps[-1]) >= self.horizon_embedding.num_embeddings:
                raise ValueError('Forecast horizon exceeds the configured horizon embedding')
            horizon = self.horizon_embedding(steps)
            query = self.query_norm(self.query_state(state)[:, :, None, :]
                                    + target[None, :, None, :] + horizon[None, None, :, :])
            query = query.reshape(bs * n_vars, self.token_len, self.hidden_dim)
            knowledge = bound_knowledge[:, None, :] # one bound mechanism per target; no numerical tokens
            context, _ = self.semantic_attention(query, knowledge, knowledge, need_weights=False)
            fusion_input = torch.cat([query, context], -1)
            if self.embedding_fusion != 'output':
                # Variable- and horizon-conditioned numerical representation.
                numeric_state = combined.mean(1).reshape(bs, n_vars, self.hidden_dim)
                z = self.query_norm(self.query_state(numeric_state)[:, :, None, :]
                                    + target[None, :, None, :] + horizon[None, None, :, :])
                z = z.reshape(bs * n_vars, self.token_len, self.hidden_dim)
                condition = self.condition_projection(context)
                if self.embedding_fusion == 'gate':
                    condition = self.embedding_gate(torch.cat((z, context), -1)) * condition
                fused = z + condition
                # Reuse the existing decoder; horizon h uses its h-th output.
                correction = self.decoder(fused).diagonal(dim1=1, dim2=2)
                correction = correction.reshape(bs, n_vars, self.token_len).permute(0, 2, 1)
                final_norm = base_norm + correction
            elif self.dual_moe or self.tri_mmoe:
                trend_state = t_embeds.mean(1).reshape(bs, n_vars, self.hidden_dim)
                seasonal_state = numeric_s_embeds.mean(1).reshape(bs, n_vars, self.hidden_dim)
                trend_query = trend_state[:, :, None, :] + horizon[None, None, :, :]
                seasonal_query = seasonal_state[:, :, None, :] + horizon[None, None, :, :]
                trend_out = self.numeric_experts[0](trend_query).squeeze(-1).permute(0, 2, 1)
                seasonal_out = self.numeric_experts[1](seasonal_query).squeeze(-1).permute(0, 2, 1)
                num_choices = torch.stack((residual, trend_out, seasonal_out), dim=-1)
                if self.separate_numeric_path:
                    numeric_state = (numeric_s_embeds + t_embeds).mean(1).reshape(bs, n_vars, self.hidden_dim)
                    numeric_query = self.query_norm(self.query_state(numeric_state)[:, :, None, :]
                                                    + target[None, :, None, :] + horizon[None, None, :, :])
                    numeric_router_input = numeric_query.reshape(bs * n_vars, self.token_len, self.hidden_dim)
                else:
                    numeric_router_input = fusion_input
                num_weights = torch.softmax(self.numeric_router(numeric_router_input), dim=-1)
                num_weights = num_weights.reshape(bs, n_vars, self.token_len, 3).permute(0, 2, 1, 3)
                numeric = (num_weights * num_choices).sum(-1)
                sem_choices = torch.cat((self.semantic_delta(context),
                                         self.semantic_experts[0](fusion_input),
                                         self.semantic_experts[1](fusion_input)), dim=-1)
                sem_weights = torch.softmax(self.semantic_router(fusion_input), dim=-1)
                sem_mix = (sem_weights * sem_choices).sum(-1)
                sem_mix = sem_mix.reshape(bs, n_vars, self.token_len).permute(0, 2, 1)
                if self.tri_mmoe:
                    tri_weights = torch.softmax(self.tri_router(fusion_input), dim=-1)
                    tri_weights = tri_weights.reshape(bs, n_vars, self.token_len, 3).permute(0, 2, 1, 3)
                    candidates = torch.stack((base_norm, numeric, base_norm + sem_mix), dim=-1)
                    final_norm = (tri_weights * candidates).sum(-1)
                else:
                    mix_gate = torch.sigmoid(self.semantic_mix_gate(fusion_input))
                    mix_gate = mix_gate.reshape(bs, n_vars, self.token_len).permute(0, 2, 1)
                    residual = numeric + mix_gate * sem_mix
            elif self.moe_fusion:
                weights = torch.softmax(self.fusion_gate(fusion_input), dim=-1)
                weights = weights.reshape(bs, n_vars, self.token_len, 3).permute(0, 2, 1, 3)
                semantic = self.semantic_delta(context).reshape(bs, n_vars, self.token_len).permute(0, 2, 1)
                scale = F.softplus(self.fusion_log_scale)
                residual = scale * (weights[..., 1] * residual + weights[..., 2] * semantic)
            else:
                alpha = torch.sigmoid(self.semantic_gate(fusion_input))
                semantic = alpha * self.semantic_delta(context)
                residual = residual + semantic.reshape(bs, n_vars, self.token_len).permute(0, 2, 1)
        if self.direct_forecast:
            return means + residual * stdev[:, 0, :].unsqueeze(1)
        if self.tri_mmoe:
            return means + final_norm * stdev[:, 0, :].unsqueeze(1)
        return base + residual * stdev[:, 0, :].unsqueeze(1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
