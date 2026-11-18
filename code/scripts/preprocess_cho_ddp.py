"""Frozen text encoders distributed over six ranks; observed-only window statistics."""
import os,random
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoModel,AutoTokenizer
from cho_data import ROOT,DATA,FEATURES,NAMES,MECHANISMS

def main():
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK']);assert world==6 and os.environ['CUDA_VISIBLE_DEVICES']=='2,3,4,5,6,7'
 torch.cuda.set_device(local);dist.init_process_group('nccl');torch.set_num_threads(4);print('CHO_TEXT_RANK',rank,'LOCAL',local,flush=True)
 raw=pd.read_csv(DATA/'CHO.csv');m=pd.read_csv(DATA/'batch_manifest.csv');values=raw[FEATURES].to_numpy(float)
 def top_lags(v):
  scores=[]
  for lag in range(1,25):
   a=v[:-lag]-v[:-lag].mean();b=v[lag:]-v[lag:].mean();den=float(np.sqrt((a*a).sum()*(b*b).sum()));scores.append((lag,float((a*b).sum()/den) if den>1e-8 else 0.))
  scores.sort(key=lambda x:(-abs(x[1]),x[0]));return ', '.join(f'{lag}:{s:+.2f}' for lag,s in scores[:12])
 for encoder,source in [('roberta','FacebookAI/roberta-base'),('gpt2','/home/home_new/qsmx/.cache/modelscope/hub/models/AI-ModelScope/gpt2')]:
  if all((DATA/f'CHO__{encoder}_h{h}.pt').exists() for h in [6,12,24]):continue
  tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
  if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
  model=AutoModel.from_pretrained(source,local_files_only=True).cuda().eval();model.requires_grad_(False)
  def encode(texts):
   chunks=[]
   with torch.no_grad():
    for off in range(0,len(texts),64):
     inp=tokenizer(texts[off:off+64],padding=True,truncation=True,max_length=256,return_tensors='pt').to(local)
     hidden=model(**inp).last_hidden_state;mask=inp['attention_mask']
     if encoder=='gpt2':pooled=hidden[torch.arange(len(hidden),device=hidden.device),mask.sum(1)-1]
     else:pooled=(hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
     chunks.append(pooled.float().cpu())
   return torch.cat(chunks) if chunks else torch.empty(0,768)
  knowledge=[]
  for name,mechanism in zip(NAMES,MECHANISMS):knowledge.extend([f'Mechanism for {name}: {mechanism}',f'Stage knowledge for {name}: Only observed culture age is used; no biological stage label is assigned.',f'Operation context for {name}: CHO-K1 cell cultivation under batch or fed-batch operation. Oxygen, pH, optical and dielectric signals characterize the culture; no future measurements are supplied.'])
  k=encode(knowledge).reshape(5,3,768)
  shuffled=knowledge.copy()
  for j,name in enumerate(NAMES):shuffled[j*3]=f'Mechanism for {name}: {MECHANISMS[(j+1)%5]}'
  sk=encode(shuffled).reshape(5,3,768);none=encode(['No variable-specific mechanism is supplied.'])[0]
  for h in [6,12,24]:
   out=DATA/f'CHO__{encoder}_h{h}.pt'
   if out.exists():continue
   starts=[];ages=[];modes=[]
   for row in m[m.eligible].itertuples():
    ix=np.flatnonzero(raw.batch_id.to_numpy()==row.batch_id)
    for s in range(ix[0],ix[-1]-48-h+2):starts.append(s);ages.append(float(raw.hh.iloc[s]));modes.append(row.mode)
   local_ids=list(range(rank,len(starts),world));texts=[]
   for i in local_ids:
    window=values[starts[i]:starts[i]+48]
    for j,name in enumerate(NAMES):
     v=window[:,j];delta=v[-1]-v[0];direction='upward' if delta>1e-8 else 'downward' if delta<-1e-8 else 'flat';recent=', '.join(f'{z:.3f}' for z in v[-12:])
     texts.append(f'Data description: CHO-K1 {modes[i]} cultivation, variable {name}. Task: forecast the next {h} hourly steps from the previous 48 observations. Input window culture age: {ages[i]:.1f} to {ages[i]+47:.1f} hours. Input statistics: minimum {v.min():.3f}; maximum {v.max():.3f}; median {np.median(v):.3f}; mean {v.mean():.3f}; first {v[0]:.3f}; last {v[-1]:.3f}; trend {direction}, change {delta:+.3f}. Top 12 autocorrelation lags (lag:correlation): {top_lags(v)}. Last 12 observed values: {recent}.')
   embeddings=encode(texts).reshape(len(local_ids),5,768)
   # Rank files are temporary shards, deleted after verified atomic cache assembly.
   shard=DATA/f'.{encoder}_h{h}_rank{rank}.pt';torch.save({'ids':local_ids,'embeddings':embeddings},shard);dist.barrier()
   if rank==0:
    full=torch.empty(len(starts),5,768);seen=[]
    for r in range(world):
     part=torch.load(DATA/f'.{encoder}_h{h}_rank{r}.pt',map_location='cpu');full[part['ids']]=part['embeddings'];seen+=part['ids']
    assert sorted(seen)==list(range(len(starts))) and torch.isfinite(full).all()
    tmp=out.with_suffix('.tmp');torch.save({'features':FEATURES,'starts':torch.tensor(starts),'window_embeddings':full,'knowledge_embeddings':k,'shuffled_knowledge_embeddings':sk,'no_mechanism_embedding':none},tmp);tmp.replace(out)
    for r in range(world):(DATA/f'.{encoder}_h{h}_rank{r}.pt').unlink()
    print('CHO_TEXT_CACHE_COMPLETE',encoder,h,len(starts),flush=True)
   dist.barrier()
  del model;torch.cuda.empty_cache()
 if rank==0:print('ALL_CHO_TEXT_CACHES_COMPLETE',flush=True)
 dist.destroy_process_group()
if __name__=='__main__':main()
