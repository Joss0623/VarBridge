import argparse, hashlib, json, os, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ENCODERS = {
    'gpt2': '/home/home_new/qsmx/.cache/modelscope/hub/models/AI-ModelScope/gpt2',
    'distilgpt2': 'distilbert/distilgpt2',
    'bert': 'google-bert/bert-base-uncased',
    'roberta': 'FacebookAI/roberta-base',
}
RULES = {
    'EFP_long_sub1': ('Erythromycin fermentation: biomass, viscosity, cumulative alcohol feed, reducing sugar, and total sugar.',
                      'Biomass growth can raise viscosity; carbon feeds replenish substrate while growth and product formation consume sugar. These are context-dependent tendencies, not enforced constraints.'),
    'EFP_long_sub2': ('Erythromycin fermentation: chemical efficiency, biomass, dissolved oxygen, online pH, and airflow.',
                      'Aeration supplies oxygen, while biomass consumes it; pH and oxygen availability can affect metabolic conversion. These tendencies depend on operating conditions.'),
    'EFP_long_sub3': ('Erythromycin fermentation: chemical efficiency, biomass, viscosity, online pH, and total sugar.',
                      'Sugar availability supports growth and product synthesis. Rising biomass can increase viscosity, potentially limiting oxygen transfer. These relations are qualitative priors.')
}
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True,choices=RULES)
    p.add_argument('--encoder',required=True,choices=ENCODERS)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--batch_size',type=int,default=64)
    a=p.parse_args()
    root = os.path.join('dataset',a.dataset)
    path=os.path.join(root,a.dataset+'.csv')
    raw=pd.read_csv(path)
    bids=raw.batch_id.to_numpy()
    reset=np.flatnonzero(bids[1:]!=bids[:-1])+1
    bounds=np.r_[0,reset,len(raw)]
    if len(np.unique(bids)) != len(bounds)-1: raise ValueError('Noncontiguous batch IDs')
    hh=pd.to_numeric(raw.hh,errors='raise').to_numpy()
    token_len=24
    desc,rules=RULES[a.dataset]
    prompts=[]
    for start,end in zip(bounds[:-1],bounds[1:]):
        for i in range(start,end):
            patch_end=min(i+token_len-1,end-1)
            age=float(hh[i]-hh[start])
            prompts.append(f'Data description: {desc} Task: multivariate forecasting from the preceding 48 observations. Input patch age: {age:.1f} to {float(hh[patch_end]-hh[start]):.1f} hours within one batch. Process knowledge: {rules}')
    unique=list(dict.fromkeys(prompts))
    index={s:i for i,s in enumerate(unique)}
    ids=torch.tensor([index[s] for s in prompts],dtype=torch.long)
    source=ENCODERS[a.encoder]
    tok=AutoTokenizer.from_pretrained(source)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModel.from_pretrained(source).to(f'cuda:{a.gpu}').eval()
    for param in model.parameters(): param.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats(a.gpu)
    start_time=time.time()
    outputs=[]
    with torch.no_grad():
        for offset in range(0,len(unique),a.batch_size):
            batch=unique[offset:offset+a.batch_size]
            inputs=tok(batch,padding=True,truncation=True,max_length=256,return_tensors='pt').to(f'cuda:{a.gpu}')
            hidden=model(**inputs).last_hidden_state
            mask=inputs['attention_mask']
            if a.encoder in ('gpt2','distilgpt2'):
                pooled=hidden[torch.arange(len(batch),device=hidden.device),mask.sum(1)-1]
            else:
                pooled=(hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
            outputs.append(pooled.float().cpu())
    embeddings=torch.cat(outputs)[ids].contiguous()
    out=os.path.join(root,f'{a.dataset}__{a.encoder}.pt')
    torch.save(embeddings,out)
    meta={'dataset':a.dataset,'encoder':a.encoder,'model_source':source,'rows':len(raw),
          'unique_prompts':len(unique),'embedding_shape':list(embeddings.shape),
          'source_sha256':hashlib.sha256(open(path,'rb').read()).hexdigest(),
          'elapsed_seconds':round(time.time()-start_time,3),
          'peak_gpu_mb':round(torch.cuda.max_memory_allocated(a.gpu)/2**20,2),
          'prompt_sha256':hashlib.sha256('\n'.join(prompts).encode()).hexdigest()}
    with open(out+'.json','w') as f: json.dump(meta,f,indent=2)
    print(json.dumps(meta))
if __name__=='__main__': main()
