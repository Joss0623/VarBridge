"""Isolated Time-LLM-style numerical front end for VarBridge.

No GPT-2 transformer, Prompt-as-Prefix, or replacement text encoder.
Original frozen numerical forecast and downstream fusion/readout are retained.
Time-LLM TokenEmbedding/ReprogrammingLayer below are copied from the
server's official implementation; only a final 768->d_model bridge is added.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from pathlib import Path
from models.KIBR_PatchTST_BoundMechanism import Model as OriginalModel

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class NumericInput(nn.Module):
    def __init__(self, cfg, word_embeddings):
        super().__init__()
        self.patch_len, self.stride = cfg.patch_len, cfg.stride
        self.register_buffer("word_embeddings", word_embeddings.detach().clone())
        vocab, llm_dim = word_embeddings.shape
        self.num_prototypes = int(getattr(cfg, "input_num_prototypes", 1000))
        if not 1 <= self.num_prototypes <= vocab:
            raise ValueError("Invalid input_num_prototypes")
        # Exact Time-LLM vocabulary-axis mapping, freshly initialized; never
        # reuse the trained TimeLLM mapping or reprogramming parameters.
        self.mapping_layer = nn.Linear(vocab, self.num_prototypes)
        self.patch_embeddings = nn.ModuleDict({
            role: TokenEmbedding(self.patch_len, cfg.d_model)
            for role in ("raw", "seasonal", "trend")})
        self.reprogramming = nn.ModuleDict({
            role: ReprogrammingLayer(cfg.d_model, cfg.n_heads,
                                     cfg.d_ff, llm_dim)
            for role in ("raw", "seasonal", "trend")})
        self.bridges = nn.ModuleDict({
            role: nn.Linear(llm_dim, cfg.d_model)
            for role in ("raw", "seasonal", "trend")})
        self.dropout = nn.Dropout(cfg.dropout)

    def prototypes(self):
        return self.mapping_layer(self.word_embeddings.T).T

    def forward(self, values, prototypes, role):
        # [B,L,C] -> [B*C,P,patch_len], same replication padding as Time-LLM.
        patches = F.pad(values.transpose(1, 2), (0, self.stride),
                        mode="replicate").unfold(-1, self.patch_len, self.stride)
        patches = patches.reshape(-1, patches.shape[-2], self.patch_len)
        embedded = self.dropout(self.patch_embeddings[role](patches))
        aligned = self.reprogramming[role](embedded, prototypes, prototypes)
        return self.bridges[role](aligned)

class Model(OriginalModel):
    def __init__(self, configs):
        if getattr(configs, "numeric_encoder", "patchtst") != "patchtst":
            raise ValueError("This input-only variant currently supports PatchTST")
        if getattr(configs, "coarse_factor", 0):
            raise ValueError("Coarse-scale input needs a separately specified adapter")
        if not getattr(configs, "tri_mmoe", False):
            raise ValueError("Use the adopted tri_mmoe configuration")
        super().__init__(configs)
        # Read only the frozen GPT-2 word table from the already verified
        # baseline checkpoint. No fine-tuned baseline parameters are imported.
        default = Path(__file__).resolve().parents[1] / "checkpoints/efficiency8_4090/TimeLLM_CHO_h24_seed2021_100epochs.pth"
        path = Path(getattr(configs, "input_vocabulary_checkpoint", default))
        state = torch.load(path, map_location="cpu")
        table = state["llm_model.wte.weight"]
        if table.shape != (50257, 768) or not torch.equal(table, state["word_embeddings"]):
            raise ValueError("Unexpected GPT-2 vocabulary table")
        del self.encoder
        del self.trend_encoder
        self.numeric_input = NumericInput(configs, table)
        self.input_vocabulary_checkpoint = str(path)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        normed = x_enc - means
        stdev = torch.sqrt(torch.var(normed, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
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
        # Keep the frozen backbone forecast, replace its fusion-facing input features.
        prototypes = self.numeric_input.prototypes()
        frozen = self.numeric_input(normed, prototypes, 'raw')
        if frozen.shape != (bs * n_vars, n_patches, self.hidden_dim):
            raise ValueError('Reprogrammed patches must match the fusion interface')
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
        s_embeds = residual_frozen + self.numeric_input(seasonal, prototypes, 'seasonal')
        t_embeds = self.numeric_input(trend, prototypes, 'trend')
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
