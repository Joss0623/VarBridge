"""Only new plugin caches; preserve original CHO/EFP caches and literal causal prompts."""
import argparse,hashlib
import numpy as np,pandas as pd,torch
from transformers import AutoModel,AutoTokenizer
from current_bound_plugins import ROOT,CACHE
from cho_data import DATA,FEATURES,NAMES,MECHANISMS
from preprocess import ENCODERS

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--encoder',choices=['bert','gpt2'],required=True);args=ap.parse_args();out=CACHE/f'CHO_{args.encoder}_h24.pt'
 if out.exists():
  q=torch.load(out,map_location='cpu');assert q['features']==FEATURES and q['window_embeddings'].shape==(len(q['starts']),5,768) and q['knowledge_embeddings'].shape==(5,3,768)
  assert torch.isfinite(q['window_embeddings']).all() and torch.isfinite(q['knowledge_embeddings']).all()
  print('PLUGIN_TEXT_CACHE_REUSED',out,flush=True);print('ALL_PLUGIN_TEXT_COMPLETE',args.encoder,len(q['starts']),flush=True);return
 torch.set_num_threads(2);torch.manual_seed(2021);source=ENCODERS[args.encoder];tok=AutoTokenizer.from_pretrained(source,local_files_only=True)
 if tok.pad_token is None:tok.pad_token=tok.eos_token
 net=AutoModel.from_pretrained(source,local_files_only=True).cuda().eval().requires_grad_(False)
 def encode(texts,mechanism=False):
  ans=[]
  with torch.no_grad():
   for off in range(0,len(texts),32):
    inp=tok(texts[off:off+32],padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda');h=net(**inp).last_hidden_state;mask=inp['attention_mask']
    if args.encoder=='gpt2' and not mechanism:z=h[torch.arange(len(h),device=h.device),mask.sum(1)-1]
    else:z=(h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
    ans.append(z.float().cpu())
    if off%2048==0:print('PLUGIN_TEXT_PROGRESS',args.encoder,off,len(texts),flush=True)
  return torch.cat(ans)
 raw=pd.read_csv(DATA/'CHO.csv');manifest=pd.read_csv(DATA/'batch_manifest.csv');values=raw[FEATURES].to_numpy(float);starts=[];ages=[];modes=[]
 for row in manifest[manifest.eligible].itertuples():
  ix=np.flatnonzero(raw.batch_id.to_numpy()==row.batch_id)
  for s in range(ix[0],ix[-1]-48-24+2):starts.append(s);ages.append(float(raw.hh.iloc[s]));modes.append(row.mode)
 def top_lags(v):
  scores=[]
  for lag in range(1,25):
   a=v[:-lag]-v[:-lag].mean();b=v[lag:]-v[lag:].mean();den=float(np.sqrt((a*a).sum()*(b*b).sum()));scores.append((lag,float((a*b).sum()/den) if den>1e-8 else 0.))
  scores.sort(key=lambda x:(-abs(x[1]),x[0]));return ', '.join(f'{lag}:{s:+.2f}' for lag,s in scores[:12])
 if args.encoder=='gpt2':
  old=torch.load(DATA/'CHO__gpt2_h24.pt',map_location='cpu');assert old['features']==FEATURES and old['starts'].tolist()==starts;windows=old['window_embeddings'];assert torch.isfinite(windows).all()
 else:
  texts=[]
  for i,s in enumerate(starts):
   window=values[s:s+48]
   for j,name in enumerate(NAMES):
    v=window[:,j];delta=v[-1]-v[0];direction='upward' if delta>1e-8 else 'downward' if delta<-1e-8 else 'flat';recent=', '.join(f'{z:.3f}' for z in v[-12:])
    texts.append(f'Data description: CHO-K1 {modes[i]} cultivation, variable {name}. Task: forecast the next 24 hourly steps from the previous 48 observations. Input window culture age: {ages[i]:.1f} to {ages[i]+47:.1f} hours. Input statistics: minimum {v.min():.3f}; maximum {v.max():.3f}; median {np.median(v):.3f}; mean {v.mean():.3f}; first {v[0]:.3f}; last {v[-1]:.3f}; trend {direction}, change {delta:+.3f}. Top 12 autocorrelation lags (lag:correlation): {top_lags(v)}. Last 12 observed values: {recent}.')
  windows=encode(texts).reshape(len(starts),5,768)
 knowledge=[]
 for name,mechanism in zip(NAMES,MECHANISMS):knowledge.extend([f'Mechanism for {name}: {mechanism}',f'Stage knowledge for {name}: Only observed culture age is used; no biological stage label is assigned.',f'Operation context for {name}: CHO-K1 cell cultivation under batch or fed-batch operation. Oxygen, pH, optical and dielectric signals characterize the culture; no future measurements are supplied.'])
 static=encode(knowledge,mechanism=True).reshape(5,3,768)
 assert windows.shape==(len(starts),5,768) and torch.isfinite(windows).all() and torch.isfinite(static).all()
 torch.save(dict(features=FEATURES,starts=torch.tensor(starts),window_embeddings=windows,knowledge_embeddings=static,source=source,observed_readout='last_valid_token' if args.encoder=='gpt2' else 'masked_mean',static_readout='masked_mean',prompt_recipe='CHO original observed-only H24; unchanged literal stage/operation/mechanisms'),out.with_suffix('.tmp'));out.with_suffix('.tmp').replace(out)
 print('ALL_PLUGIN_TEXT_COMPLETE',args.encoder,len(starts),flush=True)
if __name__=='__main__':main()
