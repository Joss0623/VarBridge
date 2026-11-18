import argparse
import hashlib
import json
import os
import random
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from preprocess import ENCODERS, RULES

NAMES = {'jn':'biomass concentration', 'nd':'viscosity', 'bcmt':'cumulative alcohol feed',
         'ht':'reducing sugar', 'zt':'total sugar', 'hx':'chemical efficiency',
         'doz':'dissolved oxygen', 'phzx':'online pH', 'fi':'airflow rate'}
VARIANTS = ('correct', 'shuffled', 'none', 'generic', 'random')
MECHANISM_UNITS = {
    'jn': 'Biomass growth consumes available substrate and can increase broth viscosity.',
    'nd': 'Increased broth viscosity can hinder mixing and oxygen transfer.',
    'bcmt': 'Cumulative alcohol feed records dosing; it is not the instantaneous cellular uptake rate.',
    'ht': 'Reducing sugar is a usable carbon substrate that can be replenished by feeding and consumed by cells.',
    'zt': 'Total sugar reflects the remaining sugar pool, which depends on feed input and consumption.',
    'hx': 'Product formation depends on metabolic activity and suitable substrate and oxygen availability.',
    'doz': 'Dissolved oxygen reflects the balance between oxygen supply and biological consumption.',
    'phzx': 'Broth pH can influence enzyme activity and metabolic conversion.',
    'fi': 'Airflow contributes to oxygen supply, with its effect modified by mixing and broth viscosity.',
}
STAGE_KNOWLEDGE = 'Only observed batch age is available; no biological stage label is assigned.'
OPERATION_CONTEXT = {
    'EFP_long_sub1': 'Feed additions and substrate consumption affect the carbon pool; viscosity can change as biomass grows.',
    'EFP_long_sub2': 'Aeration, dissolved oxygen and pH jointly characterize the operating environment for conversion.',
    'EFP_long_sub3': 'Substrate availability, biomass, viscosity and pH jointly characterize the fermentation state.',
}
RANDOM_SEMANTICS = (
    'Urban traffic flow can depend on road capacity and signal timing.',
    'Cloud cover and wind can affect the local solar energy output.',
    'Retail inventory changes with deliveries and customer purchases.',
    'River water level can vary after precipitation in a watershed.',
    'Computer network latency depends on queue load and routing.',
)

def top_lags(values, count=12):
    scores = []
    for lag in range(1, 25):
        a, b = values[:-lag], values[lag:]
        a, b = a - a.mean(), b - b.mean()
        denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
        if denom < 1e-8:
            corr = 0.0
        else:
            corr = float(np.dot(a, b) / denom)
        scores.append((lag, corr))
    scores.sort(key=lambda item: (-abs(item[1]), item[0]))
    return ', '.join(f'{lag}:{score:+.2f}' for lag, score in scores[:count])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=RULES)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=96)
    ap.add_argument('--style', choices=['original', 'structured'], default='original')
    ap.add_argument('--encoder', choices=tuple(ENCODERS) + tuple(x+'random' for x in ('gpt2','distilgpt2','bert','roberta')), default='gpt2')
    ap.add_argument('--horizon', type=int, choices=(6, 12, 24), default=24)
    args = ap.parse_args()
    path = os.path.join('dataset', args.dataset, args.dataset + '.csv')
    raw = pd.read_csv(path)
    features = [c for c in raw.columns if c not in ('date','batch_id','hh')]
    if len(features) != 5:
        raise ValueError(f'Expected 5 variables, got {features}')
    data = raw[features].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    hh = pd.to_numeric(raw.hh, errors='raise').to_numpy(dtype=np.float64)
    if not np.isfinite(data).all() or not np.isfinite(hh).all():
        raise ValueError('Nonfinite source values')
    bids = raw.batch_id.to_numpy()
    bounds = np.r_[0, np.flatnonzero(bids[1:] != bids[:-1]) + 1, len(raw)]
    if len(np.unique(bids)) != len(bounds) - 1:
        raise ValueError('Noncontiguous batch IDs')
    starts = []
    prompts = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi - lo < 96:
            continue
        for start in range(lo, hi - 48 - args.horizon + 1):
            starts.append(start)
            window = data[start:start+48]
            age0, age1 = hh[start]-hh[lo], hh[start+47]-hh[lo]
            for channel, name in enumerate(features):
                vals = window[:, channel]
                recent = ', '.join(f'{v:.3f}' for v in vals[-12:])
                delta = vals[-1] - vals[0]
                direction = 'upward' if delta > 1e-8 else 'downward' if delta < -1e-8 else 'flat'
                if args.style == 'structured':
                    prompts.append(
                        f'[TASK] Forecast {args.horizon} hourly values of {NAMES[name]} ({name}) from 48 observed values. '
                        f'[STATS] min={vals.min():.3f}; max={vals.max():.3f}; median={np.median(vals):.3f}; '
                        f'last={vals[-1]:.3f}; change={delta:+.3f}; direction={direction}. '
                        f'[LAGS] {top_lags(vals, 5)}. '
                        f'[CONTEXT] Erythromycin fermentation; batch age={age0:.1f}–{age1:.1f} h.'
                    )
                    continue
                prompts.append(
                    f'Data description: Erythromycin fermentation, variable {NAMES[name]} ({name}). '
                    f'Task: forecast the next {args.horizon} hourly steps from the previous 48 observations. '
                    f'Input window batch age: {age0:.1f} to {age1:.1f} hours. '
                    f'Input statistics: minimum {vals.min():.3f}; maximum {vals.max():.3f}; '
                    f'median {np.median(vals):.3f}; mean {vals.mean():.3f}; '
                    f'first {vals[0]:.3f}; last {vals[-1]:.3f}; '
                    f'trend {direction}, change {delta:+.3f}. '
                    f'Top 12 autocorrelation lags (lag:correlation): {top_lags(vals)}. '
                    f'Last 12 observed values: {recent}.'
                )
    desc, _ = RULES[args.dataset]
    shuffled = [MECHANISM_UNITS[features[(j+1) % len(features)]] for j in range(len(features))]
    random_units = list(RANDOM_SEMANTICS)
    random.Random(2021).shuffle(random_units)
    rule_prompts = []
    knowledge_prompts = []
    variants = ('correct',) if args.style == 'structured' else VARIANTS
    for variant in variants:
        for j, name in enumerate(features):
            if variant == 'correct':
                mechanism = MECHANISM_UNITS[name]
            elif variant == 'shuffled':
                mechanism = shuffled[j]
            elif variant == 'none':
                mechanism = 'No variable-specific mechanism is supplied.'
            elif variant == 'generic':
                mechanism = 'This process variable changes over time during fermentation.'
            else:
                mechanism = random_units[j]
            rule_prompts.append(
                f'Data description: {desc} Variable: {NAMES[name]} ({name}). '
                f'Knowledge set K. Mechanism unit: {mechanism} '
                f'Stage knowledge: {STAGE_KNOWLEDGE} '
                f'Operation context: {OPERATION_CONTEXT[args.dataset]}'
            )
            if args.style == 'structured':
                knowledge_prompts.extend((
                    f'[MECHANISM] {NAMES[name]}: {mechanism}',
                    f'[STAGE] {NAMES[name]}: {STAGE_KNOWLEDGE}',
                    f'[CONTEXT] {NAMES[name]}: {OPERATION_CONTEXT[args.dataset]}'
                ))
            else:
                knowledge_prompts.extend((
                    f'Mechanism for {NAMES[name]}: {mechanism}',
                    f'Stage knowledge for {NAMES[name]}: {STAGE_KNOWLEDGE}',
                    f'Operation context for {NAMES[name]}: {OPERATION_CONTEXT[args.dataset]}'
                ))
    base_encoder = args.encoder[:-6] if args.encoder.endswith('random') else args.encoder
    source = ENCODERS[base_encoder]
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.encoder.endswith('random'):
        torch.manual_seed(2021)
        model = AutoModel.from_config(AutoConfig.from_pretrained(source)).to(f'cuda:{args.gpu}').eval()
    else:
        model = AutoModel.from_pretrained(source).to(f'cuda:{args.gpu}').eval()
    model.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats(args.gpu)
    started = time.time()
    def encode(texts, mean_pool=False):
        chunks = []
        with torch.no_grad():
            for offset in range(0, len(texts), args.batch_size):
                inputs = tokenizer(texts[offset:offset+args.batch_size], padding=True,
                                   truncation=False, return_tensors='pt').to(f'cuda:{args.gpu}')
                if inputs['input_ids'].shape[1] > 320:
                    raise ValueError('Prompt exceeds 320 tokens; shorten the template instead of truncating')
                hidden = model(**inputs).last_hidden_state
                if mean_pool:
                    mask = inputs['attention_mask'].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1)
                else:
                    end = inputs['attention_mask'].sum(1)-1
                    pooled = hidden[torch.arange(len(end), device=hidden.device), end]
                chunks.append(pooled.to(torch.float16).cpu())
        return torch.cat(chunks)
    causal = base_encoder in ('gpt2', 'distilgpt2')
    rule_embeddings = encode(rule_prompts, mean_pool=not causal).reshape(len(variants), len(features), 768)
    knowledge_embeddings = encode(knowledge_prompts, mean_pool=True).reshape(len(variants), len(features), 3, 768).float()
    knowledge_embeddings -= knowledge_embeddings.mean(dim=(0, 1), keepdim=True)
    knowledge_embeddings = (knowledge_embeddings / knowledge_embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                            * 768**0.5).half()
    window_embeddings = encode(prompts, mean_pool=not causal).reshape(len(starts), len(features), 768)
    cache_name = args.encoder + ('_window_structured' if args.style == 'structured' else '_window_semantic') + (f'_h{args.horizon}' if args.horizon != 24 else '')
    out = os.path.join('dataset', args.dataset, args.dataset + '__' + cache_name + '.pt')
    torch.save({'starts':torch.tensor(starts, dtype=torch.int32),
                'window_embeddings':window_embeddings,
                'rule_embeddings':rule_embeddings,
                'knowledge_embeddings':knowledge_embeddings,
                'variants':variants, 'features':features}, out)
    meta = {'dataset':args.dataset,'encoder':args.encoder,'input_length':48,'horizon':args.horizon,
            'windows':len(starts),'variables':features,'variants':variants,
            'knowledge_set':{'mechanism_units':MECHANISM_UNITS,
                             'stage_knowledge':STAGE_KNOWLEDGE,
                             'operation_context':OPERATION_CONTEXT[args.dataset]},
            'prompt_style':args.style,
            'statistics':(['min','max','median','last','change','trend','top5_autocorrelation_lags']
                          if args.style == 'structured' else
                          ['min','max','median','mean','trend','top12_autocorrelation_lags','last12_values']),
            'embedding_shape':list(window_embeddings.shape),
            'knowledge_embedding_shape':list(knowledge_embeddings.shape),
            'knowledge_pooling':'attention_masked_mean',
            'knowledge_normalization':'per_role_centered_unit_norm_times_sqrt_768',
            'source_sha256':hashlib.sha256(open(path,'rb').read()).hexdigest(),
            'elapsed_seconds':round(time.time()-started,2),
            'peak_gpu_mb':round(torch.cuda.max_memory_allocated(args.gpu)/2**20,2)}
    with open(out+'.json','w') as f:
        json.dump(meta,f,indent=2)
    print(json.dumps(meta))

if __name__ == '__main__':
    main()
