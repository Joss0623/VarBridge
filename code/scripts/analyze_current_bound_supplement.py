"""Checkpoint-only nine-part fusion analysis. No fitting or adopted-source edits."""
import os,sys,time,math,csv,copy,hashlib,gc,argparse,inspect
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from common_bound_supplement import ROOT,DATASETS,make_config,SupplementModel,main_checkpoint,main_log,seed_all
from common_bound_ablations import Windows,configure
OUT=ROOT/'test_results/current_bound_supplement'
FAULTS=['noise_0p05','noise_0p1','noise_0p2','drop_1','drop_2','stuck_8','stuck_16','delay_1','delay_2','delay_4']+[f'channel_{j}' for j in range(5)]+['attention_high','attention_low']
def atomic_save(obj,path):
 path=Path(path);tmp=path.with_name(path.name+'.tmp');torch.save(obj,tmp);tmp.replace(path)
def write_csv(rows,path):
 if not rows:return
 path=Path(path);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 tmp.replace(path)
def samples(dataset):
 dest=OUT/dataset;dest.mkdir(exist_ok=True)
 path=dest/'samples.pt'
 if path.exists():return torch.load(path,map_location='cpu')
 ds=Windows(dataset,'test',48,'roberta','full');base=ds.base
 raw=pd.read_csv(ROOT/'dataset'/dataset/(dataset+'.csv'));features=ds.feature_names;bids=raw.batch_id.to_numpy();starts=base.starts if dataset=='CHO' else base.valid_indices
 groups={}
 for i,s in enumerate(starts):groups.setdefault(int(bids[s]),[]).append(i)
 indices=[]
 for bid,ix in groups.items():indices.extend(ix[j] for j in np.unique(np.linspace(0,len(ix)-1,min(32,len(ix))).astype(int)))
 rows=[ds[i] for i in indices];ss=np.asarray([starts[i] for i in indices]);x=torch.stack([a[0] for a in rows]);mark=torch.stack([a[2] for a in rows]);y=torch.stack([a[1] for a in rows])
 means=base.mean if dataset=='CHO' else base.scaler.mean_;std=base.std if dataset=='CHO' else base.scaler.scale_
 age=[];age_end=[];modes=[]
 manifest=pd.read_csv(ROOT/'dataset/CHO/batch_manifest.csv') if dataset=='CHO' else None
 for s in ss:
  lo=np.flatnonzero(bids==bids[s])[0];age.append(float(raw.hh.iloc[s]) if dataset=='CHO' else float(raw.hh.iloc[s]-raw.hh.iloc[lo]));age_end.append(float(raw.hh.iloc[s+47]) if dataset=='CHO' else float(raw.hh.iloc[s+47]-raw.hh.iloc[lo]));modes.append(str(manifest.loc[manifest.batch_id==bids[s],'mode'].iloc[0]) if dataset=='CHO' else 'fermentation')
 rawvals=raw[features].to_numpy(float);trids=manifest.loc[(manifest.eligible)&(manifest.split=='train'),'batch_id'].tolist() if dataset=='CHO' else base.batch_ids['train']
 tr=(rawvals[np.isin(bids,trids)]-means)/std;bins=np.quantile(tr,[.25,.5,.75],axis=0)
 bank=mark[0,0,1:6].clone();banks={'correct':bank,'mismatched':torch.roll(bank,-1,0)}
 if dataset=='CHO':
  controls=torch.load(ROOT/'dataset/CHO/CHO_ablation_mechanisms.pt',map_location='cpu');banks['generic']=controls['generic_description']['embeddings'];banks['placeholder']=controls['no_mechanism']['embeddings'];banks['mismatched']=base.cache['shuffled_knowledge_embeddings'][:,0,:].float().clone()
 else:
  c=torch.load(ROOT/'dataset'/dataset/(dataset+'__roberta_window_semantic.pt'),map_location='cpu',mmap=True)
  for name,key in [('generic','generic'),('placeholder','none'),('mismatched','shuffled')]:banks[name]=c['knowledge_embeddings'][c['variants'].index(key),:,0,:].float().clone()
 result=dict(dataset=dataset,x=x,y=y,mark=mark,indices=torch.tensor(indices),starts=torch.tensor(ss),bids=torch.tensor([int(bids[s]) for s in ss]),age=age,age_end=age_end,modes=modes,features=features,mean=torch.tensor(means),std=torch.tensor(std),banks=banks,value_bins=torch.tensor(bins),selection='at most32 evenly spaced H48-eligible test origins per trajectory')
 atomic_save(result,path);print('FEATURE_SAMPLES',dataset,len(x),'trajectories',len(groups),flush=True);return result
class Capture:
 def __init__(self,net):
  self.net=net;self.data={};self.handles=[]
  names=['patch_backbone.encoder','patch_backbone.head','cross_attention','channel_norm','numeric_router','semantic_router','tri_router','residual_head','numeric_experts.0','numeric_experts.1','semantic_delta','semantic_experts.0','semantic_experts.1','channel_attention']
  for name,m in net.named_modules():
   if name in names:self.handles.append(m.register_forward_hook(self.hook(name)))
 def hook(self,name):
  def f(module,args,out):
   value=out[0] if isinstance(out,tuple) else out
   self.data[name]=value.detach().cpu()
   if name in ['cross_attention','channel_attention']:
    a=module.cross_attn if name=='cross_attention' else module
    q,k,v=(args[0],args[1],args[1]) if name=='cross_attention' else args[:3]
    if name=='cross_attention':self.data['numeric_statistics_features']=q.detach().cpu()
    w=a.in_proj_weight;b=a.in_proj_bias;d=a.embed_dim;h=a.num_heads
    qq=F.linear(q,w[:d],None if b is None else b[:d]);kk=F.linear(k,w[d:2*d],None if b is None else b[d:2*d])
    qq=qq.reshape(len(q),q.shape[1],h,d//h).transpose(1,2);kk=kk.reshape(len(k),k.shape[1],h,d//h).transpose(1,2)
    self.data[name+'_weights']=torch.softmax(qq@kk.transpose(-1,-2)/math.sqrt(d//h),-1).mean(1).cpu()
  return f
 def close(self):
  for h in self.handles:h.remove()
def load_model(dataset,variant='full',device='cuda:0'):
 cfg=make_config(dataset)
 if variant!='full':
  cfg=type(cfg)(**configure(vars(cfg).copy(),variant));from models.KIBR_PatchTST_BoundAblation import Model
  net=Model(cfg);path=ROOT/'checkpoints/common_bound_ablations'/f'{dataset}_h24_seed2021_{variant}.pth'
 else:net=SupplementModel(cfg);path=main_checkpoint(dataset)
 net.load_state_dict(torch.load(path,map_location='cpu'),strict=True);return net.to(device).eval()
@torch.no_grad()
def predict(net,x,mark,h=24,trace=False):
 outputs=[];traces={};cap=Capture(net) if trace else None
 device=next(net.parameters()).device
 for start in range(0,len(x),32):
  xx=x[start:start+32].to(device);mm=mark[start:start+32].to(device);pred=[]
  for _ in range(math.ceil(h/24)):
   yy=net(xx[:,-48:],mm,None,mm)[:,-24:];pred.append(yy);xx=torch.cat([xx,yy],1)
  outputs.append(torch.cat(pred,1)[:,:h].cpu())
  if cap:
   n=len(mm);pp=cap.data['patch_backbone.encoder'].shape[1]
   if h==24:
    base=cap.data['patch_backbone.head'].permute(0,2,1)
    nc=torch.stack([cap.data['residual_head'].permute(0,2,1),cap.data['numeric_experts.0'].squeeze(-1).permute(0,2,1),cap.data['numeric_experts.1'].squeeze(-1).permute(0,2,1)],-1)
    nr=cap.data['numeric_router'].reshape(n,5,24,3).permute(0,2,1,3).softmax(-1);numeric=(nc*nr).sum(-1)
    mc=torch.cat([cap.data[k] for k in ['semantic_delta','semantic_experts.0','semantic_experts.1']],-1).reshape(n,5,24,3).permute(0,2,1,3)
    mr=cap.data['semantic_router'].reshape(n,5,24,3).permute(0,2,1,3).softmax(-1);corrected=base+(mc*mr).sum(-1)
    tw=cap.data['tri_router'].reshape(n,5,24,3).permute(0,2,1,3).softmax(-1)
    xx0=x[start:start+32];mean=xx0.mean(1,keepdim=True);std=(xx0.var(1,keepdim=True,unbiased=False)+1e-5).sqrt()
    recon=(torch.stack([base,numeric,corrected],-1)*tw).sum(-1)*std+mean
    assert torch.allclose(recon,outputs[-1],rtol=2e-5,atol=2e-6),float((recon-outputs[-1]).abs().max())
    for key,value in [('candidate_backbone',base),('candidate_numeric',numeric),('candidate_corrected',corrected)]:traces.setdefault(key,[]).append(value*std+mean)
   # Flattened per-variable modules and batch-by-patch modules have distinct layouts.
   for key,value in cap.data.items():
    if key=='channel_norm':value=value.reshape(n,pp,5,128).transpose(1,2).mean(2)
    elif key=='channel_attention_weights':value=value.reshape(n,pp,5,5).mean(1)
    elif key=='patch_backbone.encoder':value=value.reshape(n,5,pp,128).mean(2)
    elif key=='numeric_statistics_features':value=value.reshape(n,5,pp,128).mean(2)
    elif key=='cross_attention_weights':value=value.reshape(n,5,pp,3).mean(2)
    elif key in ['numeric_router','semantic_router','tri_router']:value=value.reshape(n,5,24,3).permute(0,2,1,3).softmax(-1)
    else:continue
    traces.setdefault(key,[]).append(value)
 if cap:cap.close()
 return torch.cat(outputs),{k:torch.cat(v) for k,v in traces.items()}
def metrics(dataset,group,condition,model,pred,y,rows):
 delta=pred-y[:,:pred.shape[1]]
 assert torch.isfinite(delta).all()
 rows.append(dict(dataset=dataset,group=group,condition=condition,model=model,horizon=pred.shape[1],windows=len(pred),mse=float(delta.square().mean()),mae=float(delta.abs().mean())))
 return delta
class TextEncoder:
 def __init__(self,device):
  from transformers import AutoTokenizer,AutoModel
  self.tok=AutoTokenizer.from_pretrained('FacebookAI/roberta-base',local_files_only=True);self.net=AutoModel.from_pretrained('FacebookAI/roberta-base',local_files_only=True).to(device).eval().requires_grad_(False);self.device=device
 @torch.no_grad()
 def encode(self,texts):
  values=[]
  for k in range(0,len(texts),32):
   inp=self.tok(texts[k:k+32],padding=True,truncation=True,max_length=256,return_tensors='pt').to(self.device);h=self.net(**inp).last_hidden_state;mask=inp['attention_mask'];values.append(((h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]).float().cpu())
  return torch.cat(values)
def texts_for_window(data,x):
 from preprocess_window_semantic import NAMES,top_lags
 from cho_data import NAMES as CHO_NAMES
 raw=x.double()*data['std']+data['mean'];texts=[]
 for i,window in enumerate(raw.numpy()):
  for j,f in enumerate(data['features']):
   v=window[:,j];delta=v[-1]-v[0];direction='upward' if delta>1e-8 else 'downward' if delta< -1e-8 else 'flat';recent=', '.join(f'{a:.3f}' for a in v[-12:]);a=data['age'][i];end_age=data['age_end'][i]
   intro=(f"Data description: CHO-K1 {data['modes'][i]} cultivation, variable {CHO_NAMES[j]}. Task: forecast the next 24 hourly steps from the previous 48 observations. Input window culture age: {a:.1f} to {end_age:.1f} hours. " if data['dataset']=='CHO' else f'Data description: Erythromycin fermentation, variable {NAMES[f]} ({f}). Task: forecast the next 24 hourly steps from the previous 48 observations. Input window batch age: {a:.1f} to {end_age:.1f} hours. ')
   texts.append(intro+f'Input statistics: minimum {v.min():.3f}; maximum {v.max():.3f}; median {np.median(v):.3f}; mean {v.mean():.3f}; first {v[0]:.3f}; last {v[-1]:.3f}; trend {direction}, change {delta:+.3f}. Top 12 autocorrelation lags (lag:correlation): {top_lags(v)}. Last 12 observed values: {recent}.')
 return texts
def perturb(data,name,importance):
 x=data['x'].clone();g=torch.Generator().manual_seed(2021);n=len(x);perm=torch.stack([torch.randperm(5,generator=g) for _ in range(n)])
 if name.startswith('noise_'):x+=float(name.split('_')[1].replace('p','.'))*torch.randn(x.shape,generator=g)
 else:
  for i in range(n):
   if name.startswith('channel_'):ids=[int(name.split('_')[1])]
   elif name=='attention_high':ids=[int(importance[i].argmax())]
   elif name=='attention_low':ids=[int(importance[i].argmin())]
   else:ids=perm[i,:int(name.split('_')[1]) if name.startswith('drop_') else 1].tolist()
   for j in ids:
    if name.startswith(('drop_','channel_','attention_')):x[i,:,j]=0
    elif name.startswith('stuck_'):
     k=int(name.split('_')[1]);x[i,-k:,j]=x[i,-k-1,j]
    else:
     lag=int(name.split('_')[1]);old=x[i,:,j].clone();x[i,lag:,j]=old[:-lag];x[i,:lag,j]=old[0]
 return x
def condition_mark(data,name,x,encoder,importance):
 path=OUT/data['dataset']/('observation_'+name+'.pt')
 if path.exists():obs=torch.load(path,map_location='cpu')
 else:
  obs=encoder.encode(texts_for_window(data,x)).reshape(len(x),5,768)
  # Match the original EFP cache storage precision; CHO cache is float32.
  if data['dataset']!='CHO':obs=obs.half().float()
  atomic_save(obs,path)
 mark=data['mark'].clone();mark[:,:,0]=obs;return mark
def model_text_banks(data,encoder):
 path=OUT/data['dataset']/'text_banks.pt'
 if path.exists():return torch.load(path,map_location='cpu')
 from preprocess_window_semantic import MECHANISM_UNITS,NAMES
 from cho_data import MECHANISMS,NAMES as CN
 originals=MECHANISMS if data['dataset']=='CHO' else [MECHANISM_UNITS[f] for f in data['features']]
 names=CN if data['dataset']=='CHO' else [f'{NAMES[f]} ({f})' for f in data['features']]
 if data['dataset']=='CHO':
  rewrite=['Carbon dioxide production, gas exchange and inlet gas composition affect dissolved carbon dioxide; this signal does not directly count cells.','Culture transmission varies with light absorption and scattering and does not constitute a calibrated biomass measurement.','Polarization and viable cell volume can influence permittivity, depending on physiological state and sensor conditions.','Acid-base balance and process control determine culture pH dynamics alongside metabolism, gas exchange and base addition.','Oxygen transfer, cellular demand and process control together determine dissolved oxygen.']
  from preprocess_cho_bound_mechanisms import CONTRADICTORY
  false=CONTRADICTORY
 else:
  rewrites={'hx':'Metabolic activity together with suitable substrate and oxygen availability influences product formation.','jn':'Growth of biomass uses available substrate and may raise broth viscosity.','doz':'The dissolved oxygen level reflects oxygen supply relative to biological consumption.','phzx':'Enzyme activity and metabolic conversion can be affected by broth pH.','fi':'Airflow supplies oxygen; mixing and broth viscosity modify its influence.'}
  wrong={'hx':'Product formation is independent of metabolic activity, substrate availability and oxygen availability.','jn':'Biomass growth consumes no substrate and cannot affect broth viscosity.','doz':'Dissolved oxygen does not depend on oxygen supply or biological consumption.','phzx':'Broth pH cannot influence enzyme activity or metabolic conversion.','fi':'Airflow does not contribute to oxygen supply and has no relationship to mixing or broth viscosity.'}
  rewrite=[rewrites[f] for f in data['features']];false=[wrong[f] for f in data['features']]
 if data['dataset']=='CHO':concise=['Dissolved carbon dioxide reflects production and gas exchange, not direct cell counts.','Optical transmission reflects absorption and scattering, not calibrated biomass.','Permittivity relates to polarization and viable cell volume.','pH reflects acid-base balance, metabolism and process control.','Dissolved oxygen reflects transfer, cellular demand and control.']
 else:
  short={'hx':'Product formation relates to metabolism, substrate and oxygen.','jn':'Biomass growth consumes substrate and can increase viscosity.','doz':'Dissolved oxygen balances oxygen supply and biological consumption.','phzx':'Broth pH affects enzyme activity and metabolic conversion.','fi':'Airflow supplies oxygen; mixing and viscosity modify its effect.'}
  concise=[short[f] for f in data['features']]
 phrases={'rewrite':[f'Mechanism for {name}: {t}' for name,t in zip(names,rewrite)],'concise':[f'Mechanism for {name}: {t}' for name,t in zip(names,concise)],'wrong_target_name':[f'Mechanism for {names[(j+1)%5]}: {t}' for j,t in enumerate(originals)],'contradictory':[f'Mechanism for {name}: {t}' for name,t in zip(names,false)]}
 banks={k:encoder.encode(v) for k,v in phrases.items()};result=dict(embeddings=banks,texts=phrases,description='Contradictory descriptions are intentionally false experimental controls; rewrites use existing mechanism statements only.')
 atomic_save(result,path);return result
def analyze(dataset,device):
 dest=OUT/dataset;data=samples(dataset);x=data['x'];mark=data['mark'];y=data['y'];rows=[];seed_all();net=load_model(dataset,device=device)
 clean,tr=predict(net,x,mark,trace=True);atomic_save(dict(prediction=clean,trace=tr),dest/'clean_analysis.pt');metrics(dataset,'clean','clean','full',clean,y,rows)
 # Hooks must not change numerical predictions.
 audit_n=min(32,len(x));plain,_=predict(net,x[:audit_n],mark[:audit_n]);assert torch.equal(plain,clean[:audit_n]) or torch.allclose(plain,clean[:audit_n],rtol=1e-5,atol=1e-6)
 for name in ['backbone','numeric','corrected']:
  metrics(dataset,'candidates','clean',name,tr['candidate_'+name],y,rows)
 importance=tr['channel_attention_weights'].mean(1);encoder=TextEncoder(device)
 binding=[]
 for j in range(5):
  mm=mark.clone();mm[:,:,1+j]=data['banks']['correct'][(j+1)%5];p,_=predict(net,x,mm);metrics(dataset,'binding',str(j),'full',p,y,rows)
  for k in range(5):binding.append(dict(dataset=dataset,input_variable=data['features'][j],output_variable=data['features'][k],prediction_rms=float((p[:,:,k]-clean[:,:,k]).square().mean().sqrt())))
  unaffected=[k for k in range(5) if k!=j];assert torch.allclose(p[:,:,unaffected],clean[:,:,unaffected],rtol=1e-5,atol=1e-6)
 write_csv(binding,dest/'binding_response.csv')
 for kind,bank in {**data['banks'],**model_text_banks(data,encoder)['embeddings']}.items():
  mm=mark.clone();mm[:,:,1:6]=bank[None,None];p,_=predict(net,x,mm);metrics(dataset,'text',kind,'full',p,y,rows)
 # Uniform and cross-window-shuffled routing, keeping learned experts and all data fixed.
 for router in ['numeric_router','semantic_router','tri_router']:
  for intervention in ['uniform','shuffle']:
   def hook(module,args,out):
    if intervention=='uniform':return torch.zeros_like(out)
    v=out.reshape(-1,5,24,3);return v[torch.randperm(len(v),generator=torch.Generator().manual_seed(4071+len(v))).to(v.device)].reshape_as(out)
   handle=getattr(net,router).register_forward_hook(hook);p,_=predict(net,x,mark);handle.remove();metrics(dataset,'routing',router+'_'+intervention,'full',p,y,rows)
 control=load_model(dataset,'no_channel',device)
 nochan,_=predict(control,x,mark);metrics(dataset,'clean','clean','no_channel',nochan,y,rows)
 for condition in FAULTS:
  xx=perturb(data,condition,importance);mm=condition_mark(data,condition,xx,encoder,importance)
  p,_=predict(net,xx,mm);metrics(dataset,'sensor',condition,'full',p,y,rows)
  pc,_=predict(control,xx,mm);metrics(dataset,'sensor',condition,'no_channel',pc,y,rows)
  if condition in ['noise_0p1','noise_0p2']:
   for kind,bank in data['banks'].items():
    m2=mm.clone();m2[:,:,1:6]=bank[None,None];q,_=predict(net,xx,m2);metrics(dataset,'joint_quality',condition+'_'+kind,'full',q,y,rows)
  print('FEATURE_CONDITION_COMPLETE',dataset,condition,flush=True);write_csv(rows,dest/'feature_metrics.csv')
 del encoder,control;gc.collect();torch.cuda.empty_cache()
 # H48 comparisons retain original observed text and trained relative positions.
 step_rows=[]
 for variant in ['full','no_dynamic_stats','no_forecast_conditioning']:
  nn=net if variant=='full' else load_model(dataset,variant,device)
  p,_=predict(nn,x,mark,48);metrics(dataset,'recursive','H48',variant,p,y,rows)
  for h in range(48):step_rows.append(dict(dataset=dataset,model=variant,step=h+1,mse=float((p[:,h]-y[:,h]).square().mean()),mae=float((p[:,h]-y[:,h]).abs().mean())))
  if variant!='full':del nn
 write_csv(step_rows,dest/'recursive_steps.csv')
 # Per-variable/value-bin summaries accompany the joint plots.
 details=[]
 for j,f in enumerate(data['features']):
  for q in range(4):
   binids=torch.bucketize(y[:,:24,j].contiguous(),data['value_bins'][:,j].float());sel=binids==q;delta=(clean-y[:,:24])[:,:,j][sel]
   if len(delta):details.append(dict(dataset=dataset,variable=f,training_value_quartile=q+1,elements=len(delta),mse=float(delta.square().mean()),mae=float(delta.abs().mean())))
 write_csv(details,dest/'conditional_errors.csv');write_csv(rows,dest/'feature_metrics.csv');(dest/'INFERENCE_COMPLETE').write_text('checkpoint inference ready for nine-family plots; no retraining\n');print('CURRENT_SUPPLEMENT_FEATURES_COMPLETE',dataset,flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=DATASETS,required=True);ap.add_argument('--gpu',type=int,default=2);args=ap.parse_args();assert args.gpu in range(2,8)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2' and args.gpu==2;torch.set_num_threads(2);analyze(args.dataset,'cuda:0')
if __name__=='__main__':main()
