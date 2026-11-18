"""CHO protocol fixed before model fitting; EXP-disjoint, causal hourly measurements."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'dataset/CHO'
FEATURES=['dco2','transmission','permittivity','ph','do']
NAMES=['dissolved carbon dioxide (% saturation)','optical transmission (arbitrary units)','permittivity (pF/cm)','pH','dissolved oxygen (% air saturation)']
MECHANISMS=[
 'Dissolved carbon dioxide reflects production, gas exchange and the supplied gas composition; it is not a direct cell-count measurement.',
 'Optical transmission is affected by light scattering and absorption in the culture; it is not a calibrated biomass concentration.',
 'Permittivity reflects polarization of the culture and may relate to viable cell volume, with dependence on cell physiology and sensor conditions.',
 'Culture pH reflects acid-base balance and process control; its dynamics depend on metabolism, gas exchange and base addition.',
 'Dissolved oxygen reflects the balance between oxygen transfer, cellular demand and operating control.']
SPLITS={**{i:'train' for i in [1,2,3,4,5,6,7,10]},8:'val',11:'val',9:'test',12:'test'}

def prepare():
 raw=DATA/'raw/Individual raw data tables'
 frames=[]; manifest=[]; audit=[]
 for f in sorted(raw.glob('*.xlsx')):
  exp=int(f.name[3:6]);mode='fed-batch' if 'fed-batch' in f.name else 'batch'
  sheets=['inline_CO2','inline_Turbidity','inline_Permittivity','inline_ProcessParameters']
  requested=[r+'_'+s for r in ['R01','R02'] for s in sheets]
  book=pd.read_excel(f,sheet_name=requested)
  for reactor in ['R01','R02']:
   specs=[('inline_CO2','Dissolved CO2 (%-sat)'),('inline_Turbidity','Transmission (arb. Unit)'),('inline_Permittivity','Permittivity (pF/cm)'),('inline_ProcessParameters','pH'),('inline_ProcessParameters','Dissolved O2 (% air sat)')]
   streams=[]
   for suffix,col in specs:
    d=book[reactor+'_'+suffix][['Time (h)',col]].apply(pd.to_numeric,errors='coerce').dropna().sort_values('Time (h)').drop_duplicates('Time (h)',keep='last')
    assert np.isfinite(d.to_numpy()).all()
    streams.append(d)
   end=int(np.floor(min(d.iloc[-1,0] for d in streams)));grid=np.arange(0,end+1,dtype=float)
   vals=[];valid=np.ones(len(grid),dtype=bool)
   for d in streams:
    t=d.iloc[:,0].to_numpy();v=d.iloc[:,1].to_numpy();idx=np.searchsorted(t,grid,side='right')-1
    ok=idx>=0;idx=np.maximum(idx,0);age=grid-t[idx]
    ok &= (age>=-1e-8)&(age<=2.0)
    valid &= ok;vals.append(v[idx])
   values=np.stack(vals,axis=1)
   # Split at unavailable/stale observations; never bridge a missing span.
   good=np.flatnonzero(valid);groups=np.split(good,np.flatnonzero(np.diff(good)>1)+1) if len(good) else []
   for segment,ix in enumerate(groups):
    if not len(ix):continue
    bid=exp*100+int(reactor[-2:])*10+segment
    frame=pd.DataFrame(values[ix],columns=FEATURES);frame.insert(0,'hh',grid[ix]);frame.insert(0,'batch_id',bid);frame.insert(0,'date',grid[ix])
    frames.append(frame);manifest.append(dict(batch_id=bid,experiment=exp,reactor=reactor,mode=mode,split=SPLITS[exp],source=f.name,start_hour=float(grid[ix[0]]),end_hour=float(grid[ix[-1]]),points=len(ix),eligible=len(ix)>=96))
   audit.append(dict(experiment=exp,reactor=reactor,end_hour=end,valid_hourly_points=int(valid.sum()),stale_or_missing_hourly_points=int((~valid).sum())))
 assert len(audit)==24
 pd.concat(frames,ignore_index=True).to_csv(DATA/'CHO.csv',index=False)
 m=pd.DataFrame(manifest);m.to_csv(DATA/'batch_manifest.csv',index=False);pd.DataFrame(audit).to_csv(DATA/'sampling_audit.csv',index=False)
 pd.DataFrame({'feature':FEATURES,'description':NAMES,'mechanism':MECHANISMS}).to_csv(DATA/'variables.csv',index=False)
 eligible=m[m.eligible];assert all(eligible.groupby('experiment').split.nunique()==1)
 for split in ['train','val','test']:
  q=eligible[eligible.split==split];print(split,'EXP',q.experiment.unique().tolist(),'trajectories',len(q),'rows',q.points.sum(),'windows',{h:int((q.points-48-h+1).sum()) for h in [6,12,24,48]},flush=True)
 print('SHORT',m[~m.eligible].to_dict('records'),flush=True)
 print('AUDIT',audit,flush=True)

class CHOWindows(Dataset):
 def __init__(self,split,horizon,encoder=None):
  raw=pd.read_csv(DATA/'CHO.csv');m=pd.read_csv(DATA/'batch_manifest.csv');eligible=m[m.eligible]
  train_ids=eligible.loc[eligible.split=='train','batch_id'];train=raw[raw.batch_id.isin(train_ids)][FEATURES].to_numpy(float)
  self.mean=train.mean(0);self.std=train.std(0);self.std[self.std==0]=1
  self.values=torch.tensor(((raw[FEATURES].to_numpy(float)-self.mean)/self.std).astype('float32'))
  self.starts=[];self.ids=[];self.horizon=horizon
  for bid in eligible.loc[eligible.split==split,'batch_id']:
   ix=np.flatnonzero(raw.batch_id.to_numpy()==bid);assert ix[-1]-ix[0]+1==len(ix)
   for s in range(ix[0],ix[-1]-48-horizon+2):self.starts.append(s);self.ids.append(int(bid))
  self.cache=None
  if encoder:
   ch=24 if horizon==48 else horizon
   self.cache=torch.load(DATA/f'CHO__{encoder}_h{ch}.pt',map_location='cpu')
   self.lookup={int(x):i for i,x in enumerate(self.cache['starts'])}
   assert self.cache['features']==FEATURES
  assert len(self.starts)>0
 def __len__(self):return len(self.starts)
 def __getitem__(self,i):
  s=self.starts[i];x=self.values[s:s+48];y=self.values[s+48:s+48+self.horizon]
  if self.cache:
   pos=self.lookup[s];window=self.cache['window_embeddings'][pos]
   knowledge=self.cache['knowledge_embeddings']
   bank=knowledge[:,0,:].unsqueeze(0).expand(5,-1,-1)
   mark=torch.cat([window[:,None,:],bank,knowledge[:,1:,:]],dim=1).float()
  else:mark=torch.zeros(5,8,768)
  return x,y,mark,torch.tensor(self.ids[i])
if __name__=='__main__':prepare()
