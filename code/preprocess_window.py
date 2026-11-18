import argparse
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from preprocess import ENCODERS, RULES

NAMES = {'jn':'biomass concentration', 'nd':'viscosity', 'bcmt':'cumulative alcohol feed',
         'ht':'reducing sugar', 'zt':'total sugar', 'hx':'chemical efficiency',
         'doz':'dissolved oxygen', 'phzx':'online pH', 'fi':'airflow rate'}
VARIANTS = ('full', 'numeric', 'wrong')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=RULES)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=96)
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
        for start in range(lo, hi - 48 - 24 + 1):
            starts.append(start)
            window = data[start:start+48]
            age0, age1 = hh[start]-hh[lo], hh[start+47]-hh[lo]
            for channel, name in enumerate(features):
                vals = window[:, channel]
                recent = ', '.join(f'{v:.3f}' for v in vals[-12:])
                prompts.append(
                    f'Erythromycin fermentation. Variable: {NAMES[name]} ({name}). '
                    f'Observed 48-hour input window, batch age {age0:.1f} to {age1:.1f} hours. '
                    f'First {vals[0]:.3f}; last {vals[-1]:.3f}; minimum {vals.min():.3f}; '
                    f'maximum {vals.max():.3f}; mean {vals.mean():.3f}; '
                    f'last-minus-first trend {vals[-1]-vals[0]:.3f}. '
                    f'Last 12 observed values: {recent}. Forecast the next 24 hours.'
                )
    desc, rules = RULES[args.dataset]
    other = {'EFP_long_sub1':'EFP_long_sub2',
             'EFP_long_sub2':'EFP_long_sub3',
             'EFP_long_sub3':'EFP_long_sub1'}[args.dataset]
    rule_prompts = []
    for variant in VARIANTS:
        for name in features:
            base = f'Data description: {desc} Variable: {NAMES[name]} ({name}). Task: forecast from the observed 48-hour window.'
            if variant == 'full':
                base += f' Process knowledge: {rules}'
            elif variant == 'wrong':
                base += f' Process knowledge: {RULES[other][1]}'
            rule_prompts.append(base)
    source = ENCODERS['gpt2']
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(source).to(f'cuda:{args.gpu}').eval()
    model.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats(args.gpu)
    started = time.time()
    def encode(texts):
        chunks = []
        with torch.no_grad():
            for offset in range(0, len(texts), args.batch_size):
                inputs = tokenizer(texts[offset:offset+args.batch_size], padding=True,
                                   truncation=True, max_length=256, return_tensors='pt').to(f'cuda:{args.gpu}')
                hidden = model(**inputs).last_hidden_state
                end = inputs['attention_mask'].sum(1)-1
                pooled = hidden[torch.arange(len(end), device=hidden.device), end]
                chunks.append(pooled.to(torch.float16).cpu())
        return torch.cat(chunks)
    rule_embeddings = encode(rule_prompts).reshape(len(VARIANTS), len(features), 768)
    window_embeddings = encode(prompts).reshape(len(starts), len(features), 768)
    out = os.path.join('dataset', args.dataset, args.dataset + '__gpt2_window.pt')
    torch.save({'starts':torch.tensor(starts, dtype=torch.int32),
                'window_embeddings':window_embeddings,
                'rule_embeddings':rule_embeddings,
                'variants':VARIANTS, 'features':features}, out)
    meta = {'dataset':args.dataset,'encoder':'gpt2','input_length':48,
            'windows':len(starts),'variables':features,'variants':VARIANTS,
            'embedding_shape':list(window_embeddings.shape),
            'source_sha256':hashlib.sha256(open(path,'rb').read()).hexdigest(),
            'elapsed_seconds':round(time.time()-started,2),
            'peak_gpu_mb':round(torch.cuda.max_memory_allocated(args.gpu)/2**20,2)}
    with open(out+'.json','w') as f:
        json.dump(meta,f,indent=2)
    print(json.dumps(meta))

if __name__ == '__main__':
    main()
