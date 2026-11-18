import torch,numpy as np
from cho_data import NAMES
class TextEncoder:
 def __init__(self,device):
  from transformers import AutoTokenizer,AutoModel
  self.tok=AutoTokenizer.from_pretrained('FacebookAI/roberta-base',local_files_only=True);self.net=AutoModel.from_pretrained('FacebookAI/roberta-base',local_files_only=True,add_pooling_layer=False).to(device).eval().requires_grad_(False);self.device=device
 @torch.no_grad()
 def encode(self,texts):
  values=[]
  for k in range(0,len(texts),32):
   inp=self.tok(texts[k:k+32],padding=True,truncation=True,max_length=256,return_tensors='pt').to(self.device);h=self.net(**inp).last_hidden_state;mask=inp['attention_mask'];values.append(((h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]).float().cpu())
  return torch.cat(values)
def texts_for_window(data,x):
 window=data['raw_window'];texts=[]
 for j,name in enumerate(NAMES):
  v=window[:,j];delta=v[-1]-v[0];direction='upward' if delta>1e-8 else 'downward' if delta< -1e-8 else 'flat';recent=', '.join(f'{a:.3f}' for a in v[-12:]);scores=[]
  for lag in range(1,25):
   a=v[:-lag]-v[:-lag].mean();b=v[lag:]-v[lag:].mean();den=float(np.sqrt((a*a).sum()*(b*b).sum()));scores.append((lag,float((a*b).sum()/den) if den>1e-8 else 0.))
  scores.sort(key=lambda v:(-abs(v[1]),v[0]));lags=', '.join(f'{lag}:{c:+.2f}' for lag,c in scores[:12]);age=data['age'][0]
  texts.append(f"Data description: CHO-K1 {data['modes'][0]} cultivation, variable {name}. Task: forecast the next 24 hourly steps from the previous 48 observations. Input window culture age: {age:.1f} to {age+47:.1f} hours. Input statistics: minimum {v.min():.3f}; maximum {v.max():.3f}; median {np.median(v):.3f}; mean {v.mean():.3f}; first {v[0]:.3f}; last {v[-1]:.3f}; trend {direction}, change {delta:+.3f}. Top 12 autocorrelation lags (lag:correlation): {lags}. Last 12 observed values: {recent}.")
 return texts
