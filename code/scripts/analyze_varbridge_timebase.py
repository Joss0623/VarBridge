"""Read-only checkpoint analyses for TimeBase-style evidence; no fitting."""
import os,sys,csv,re,json,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common_bound_supplement import main_checkpoint,main_log
OUT=ROOT/'test_results/varbridge_timebase_analysis';OUT.mkdir(exist_ok=True)
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
TASKS=['EFP_long_sub1','EFP_long_sub2','EFP_long_sub3','CHO']
LABELS={TASKS[0]:['φ','η','γ','ρ','σ'],TASKS[1]:['α','φ','ω','θ','ψ'],TASKS[2]:['α','φ','η','θ','σ'],'CHO':['χ','τ','ε','θ','ω']}
COL=['#26857c','#df8651','#547eaa','#ac78a2','#8d984e'];BRANCH=['Backbone','Numerical expert','Corrected backbone','Final fusion']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12,'axes.titlesize':12,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':400})
def csvout(name,rows):
 if not rows:return
 p=OUT/(name+'.csv');tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(fig,name):
 fig.savefig(FIG/(name+'.png'),dpi=400,bbox_inches='tight',facecolor='white');plt.close(fig)
def raw_task(task):
 raw=pd.read_csv(ROOT/'dataset'/task/(task+'.csv'));features=[c for c in raw if c not in ['date','hh','batch_id']]
 if task=='CHO':
  manifest=pd.read_csv(ROOT/'dataset/CHO/batch_manifest.csv');manifest=manifest[manifest.eligible].copy()
  ids={s:manifest.loc[manifest.split==s,'batch_id'].tolist() for s in ['train','val','test']}
 else:
  bid=raw.batch_id;eligible=[b for b in bid.drop_duplicates() if int((bid==b).sum())>=96];n=len(eligible);a=int(.7*n);b=int(.1*n)
  ids={'train':eligible[:a],'val':eligible[a:a+b],'test':eligible[a+b:]};manifest=None
 train=raw[raw.batch_id.isin(ids['train'])][features].to_numpy(float);mean=train.mean(0);std=train.std(0);std[std==0]=1
 return raw,features,ids,mean,std,manifest

def data_analysis():
 # Diagnostic text geometry uses ONE preprocessing for both tasks, without replacing model caches.
 from preprocess_window_semantic import NAMES,MECHANISM_UNITS
 from transformers import AutoTokenizer,AutoModel
 texts={}
 for task in TASKS:
  _,features,_,_,_,_=raw_task(task)
  if task=='CHO':texts[task]=torch.load(ROOT/'dataset/CHO/CHO_bound_mechanisms.pt',map_location='cpu')['correct']['texts']
  else:texts[task]=[f'Mechanism for {NAMES[f]}: {MECHANISM_UNITS[f]}' for f in features]
 corpus=list(dict.fromkeys(t for task in TASKS for t in texts[task]))
 tok=AutoTokenizer.from_pretrained('FacebookAI/roberta-base',local_files_only=True)
 enc=AutoModel.from_pretrained('FacebookAI/roberta-base',local_files_only=True).eval().requires_grad_(False)
 inp=tok(corpus,padding=True,truncation=True,max_length=256,return_tensors='pt')
 with torch.no_grad():
  hidden=enc(**inp).last_hidden_state;mask=inp['attention_mask'];emb=(hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
 center=emb.mean(0,keepdim=True);diag={t:emb[i]-center[0] for i,t in enumerate(corpus)}
 torch.save(dict(texts=corpus,raw_embeddings=emb,shared_centroid=center,recipe='Frozen RoBERTa FP32 CPU evaluation; masked mean; subtract shared centroid of all unique correct descriptions across EFP/CHO; cosine only for diagnostic geometry; adopted caches untouched'),OUT/'diagnostic_text_geometry.pt')
 del enc
 stats=[];matrows=[];trajectory=[]
 for task in TASKS:
  raw,features,ids,mean,std,manifest=raw_task(task);levels=[];diffs=[]
  for bid in ids['train']:
   q=raw[raw.batch_id==bid];z=(q[features].to_numpy(float)-mean)/std
   levels.append(z-z.mean(0));diffs.append(np.diff(z,axis=0))
  corr=np.corrcoef(np.concatenate(levels),rowvar=False);dcorr=np.corrcoef(np.concatenate(diffs),rowvar=False)
  v=torch.stack([diag[t] for t in texts[task]]).numpy();cos=(v@v.T)/np.outer(np.linalg.norm(v,axis=1),np.linalg.norm(v,axis=1))
  for kind,m in [('within_batch_level_pearson',corr),('within_batch_difference_pearson',dcorr),('shared_centroid_diagnostic_description_cosine',cos)]:
   assert np.isfinite(m).all()
   for i,a in enumerate(features):
    for j,b in enumerate(features):matrows.append(dict(dataset=task,matrix=kind,row=a,column=b,value=float(m[i,j]),partition='train' if 'pearson' in kind else 'static_correct_descriptions'))
  fig,ax=plt.subplots(2,2,figsize=(8.5,5.5),layout='constrained');ax=ax.ravel()
  # Predetermined first eligible training trajectory, never selected by prediction quality.
  q=raw[raw.batch_id==ids['train'][0]];z=(q[features].to_numpy(float)-mean)/std;t=q.hh.to_numpy(float);t=t-t[0]
  for j,f in enumerate(features):
   ax[0].plot(t,z[:,j],color=COL[j],label=LABELS[task][j],lw=1.1)
   for tt,value in zip(t,z[:,j]):trajectory.append(dict(dataset=task,partition='train',batch_id=int(ids['train'][0]),variable=f,time_hours=float(tt),training_standardized_value=float(value)))
  ax[0].set(xlabel='Time (h)',ylabel='Standardized value',title='(a) First training trajectory');ax[0].legend(ncol=5,loc='upper center',bbox_to_anchor=(.5,1),handlelength=.7,columnspacing=.6);ax[0].grid(alpha=.15)
  for k,(m,title,lo,hi,cmap) in enumerate([(corr,'(b) Within-batch levels',-1,1,'RdBu_r'),(dcorr,'(c) Hourly changes',-1,1,'RdBu_r'),(cos,'(d) Centered text similarity',-1,1,'RdBu_r')],1):
   im=ax[k].imshow(m,vmin=lo,vmax=hi,cmap=cmap);ax[k].set_xticks(range(5),LABELS[task]);ax[k].set_yticks(range(5),LABELS[task]);ax[k].set_title(title,fontsize=11)
   for i in range(5):
    for j in range(5):ax[k].text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',fontsize=10,color='white' if abs(m[i,j])>.65 else 'black')
   fig.colorbar(im,ax=ax[k],fraction=.045,pad=.03)
  save(fig,'data_characteristics_'+task)
  for split,bids in ids.items():
   q=raw[raw.batch_id.isin(bids)];lengths=[int((q.batch_id==b).sum()) for b in bids]
   stats.append(dict(dataset=task,split=split,trajectories=len(bids),observations=len(q),variables=len(features),sampling_hours=1,lookback=48,H24_windows=sum(n-48-24+1 for n in lengths),H48_windows=sum(n-48-48+1 for n in lengths),minimum_trajectory_points=min(lengths),maximum_trajectory_points=max(lengths)))
 csvout('dataset_statistics',stats);csvout('data_relationships',matrows);csvout('representative_training_trajectories',trajectory)
 print('DATA_CHARACTERISTICS_COMPLETE',flush=True)

def checkpoint_analysis(device):
 from common_bound_ablations import Windows
 from analyze_current_bound_supplement import load_model,predict
 audit=[];batches=[];groups=[];candidates=[];corrrows=[];wins=[];plotdata=[]
 for task in ['EFP_long_sub2','CHO']:
  log=main_log(task);txt=log.read_text();epochs=sorted(set(map(int,re.findall(r'Epoch:\s*(\d+)',txt))))
  assert epochs==list(range(1,101)) and 'completed_epochs=100' in txt
  ck=main_checkpoint(task);h=sha(ck);audit.append(dict(dataset=task,checkpoint=str(ck),checkpoint_sha256=h,source_log=str(log),completed_epochs=100,seed=2021,checkpoint_selection='validation_MSE',horizon=24))
  ds=Windows(task,'test',24);raw,features,ids,mean,std,manifest=raw_task(task)
  starts=np.asarray(ds.base.starts if task=='CHO' else ds.base.valid_indices);bids=raw.batch_id.to_numpy()[starts]
  net=load_model(task,device=device);allpred=[];ally=[];traces=[]
  # Bounded inference chunks; every H24 test window is retained.
  for a in range(0,len(ds),128):
   items=[ds[i] for i in range(a,min(a+128,len(ds)))];x=torch.stack([q[0] for q in items]);y=torch.stack([q[1] for q in items]);m=torch.stack([q[2] for q in items])
   with torch.no_grad():p,tr=predict(net,x,m,trace=True)
   allpred.append(p);ally.append(y);traces.append({k:tr[k] for k in ['candidate_backbone','candidate_numeric','candidate_corrected','tri_router']})
  y=torch.cat(ally).numpy();pred=torch.cat(allpred).numpy();trace={k:torch.cat([t[k] for t in traces]).numpy() for k in traces[0]}
  pp=np.stack([trace['candidate_backbone'],trace['candidate_numeric'],trace['candidate_corrected'],pred],-1);assert np.isfinite(pp).all()
  err=pp-y[...,None];nr=trace['tri_router'];recon=(pp[...,:3]*nr).sum(-1);assert np.allclose(recon,pred,atol=3e-6,rtol=3e-5)
  source_re=re.search(r'(?:EFP_BOUND_RESULT|CHO_RESULT)[^\n]*h=24[^\n]*mse=([\d.eE+-]+)[^\n]*mae=([\d.eE+-]+)[^\n]*test_windows=(\d+)',txt)
  if source_re:
   assert int(source_re[3])==len(ds);assert abs(float((err[...,-1]**2).mean())-float(source_re[1]))<1e-5
  torch.save(dict(dataset=task,starts=starts,batch_ids=bids,features=features,target=torch.from_numpy(y),candidates=torch.from_numpy(pp),final_weights=torch.from_numpy(nr),candidate_names=BRANCH,checkpoint_sha256=h,selection='ALL H24 test windows'),OUT/(task+'_fulltest_branches.pt'))
  for c,name in enumerate(BRANCH):
   candidates.append(dict(dataset=task,branch=name,windows=len(ds),mse=float((err[...,c]**2).mean()),mae=float(abs(err[...,c]).mean()),checkpoint_sha256=h))
   for j,f in enumerate(features):candidates.append(dict(dataset=task,branch=name+' / '+f,windows=len(ds),mse=float((err[:,:,j,c]**2).mean()),mae=float(abs(err[:,:,j,c]).mean()),checkpoint_sha256=h))
  corr=np.corrcoef(err.reshape(-1,4),rowvar=False)
  for i in range(4):
   for j in range(4):corrrows.append(dict(dataset=task,branch1=BRANCH[i],branch2=BRANCH[j],signed_error_pearson=float(corr[i,j])))
  winners=np.argmin(err[...,:3]**2,axis=-1)
  for i in range(3):wins.append(dict(dataset=task,branch=BRANCH[i],scalar_lowest_squared_error_fraction=float((winners==i).mean()),mean_final_router_weight=float(nr[...,i].mean()),interpretation='descriptive hindsight candidate win rate; router weights are not modality proportions'))
  train_vol=[]
  for bid in ids['train']:
   z=(raw.loc[raw.batch_id==bid,features].to_numpy(float)-mean)/std;train_vol.append(abs(np.diff(z,axis=0)).mean())
  boundaries=np.quantile(train_vol,[.25,.5,.75]);batchvol={};batchgroup={}
  for bid in ids['test']:
   z=(raw.loc[raw.batch_id==bid,features].to_numpy(float)-mean)/std;vol=float(abs(np.diff(z,axis=0)).mean());batchvol[bid]=vol
   if task=='CHO':
    mr=manifest[manifest.batch_id==bid].iloc[0];batchgroup[bid]=f'EXP{int(mr.experiment):03d} / {mr["mode"]}'
   else:batchgroup[bid]='Q'+str(int(np.searchsorted(boundaries,vol,side='right'))+1)
  for bid in np.unique(bids):
   sel=bids==bid
   for c,name in enumerate(BRANCH):batches.append(dict(dataset=task,batch_id=int(bid),group=batchgroup[bid],branch=name,windows=int(sel.sum()),mse=float((err[sel,...,c]**2).mean()),mae=float(abs(err[sel,...,c]).mean()),trajectory_volatility=batchvol[bid],group_rule='CHO original EXP/mode; EFP whole-trajectory mean absolute hourly difference with TRAIN-batch quartile cutpoints; retrospective descriptive grouping'))
  for group in sorted(set(batchgroup.values()))+['ALL']:
   sel=np.ones(len(bids),dtype=bool) if group=='ALL' else np.array([batchgroup[b]==group for b in bids]);unique=np.unique(bids[sel])
   if not len(unique):continue
   for c,name in enumerate(BRANCH):
    vals=np.array([float((err[bids==b,...,c]**2).mean()) for b in unique]);groups.append(dict(dataset=task,group=group,branch=name,trajectories=len(unique),windows=int(sel.sum()),window_weighted_mse=float((err[sel,...,c]**2).mean()),window_weighted_mae=float(abs(err[sel,...,c]).mean()),equal_trajectory_mse=float(vals.mean())))
  fig,ax=plt.subplots(1,3,figsize=(9.5,3.3),layout='constrained',gridspec_kw={'width_ratios':[1,1,1.4]})
  vals=[float((err[...,i]**2).mean()) for i in range(4)];short=['Base','Numeric','Corrected','Fusion']
  ax[0].bar(range(4),vals,color=[COL[2],COL[0],COL[3],'#d65c4f']);ax[0].set_xticks(range(4),short,rotation=25);ax[0].set(ylabel='Full-test MSE',title='(a) Candidate forecasts',ylim=(0,max(vals)*1.2))
  for i,v in enumerate(vals):ax[0].text(i,v,f'{v:.4f}',ha='center',va='bottom',fontsize=9)
  im=ax[1].imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');ax[1].set_xticks(range(4),short,rotation=35);ax[1].set_yticks(range(4),short);ax[1].set_title('(b) Prediction-error correlation')
  for i in range(4):
   for j in range(4):ax[1].text(j,i,f'{corr[i,j]:.2f}',ha='center',va='center',fontsize=10,color='white' if abs(corr[i,j])>.65 else 'black')
  fig.colorbar(im,ax=ax[1],fraction=.045)
  # First chronological test origin and fixed variable (potency / permittivity).
  j=0 if task!='CHO' else 2;t=np.arange(1,25);ax[2].plot(t,y[0,:,j],color='black',lw=2,label='Observed')
  for c in range(4):ax[2].plot(t,pp[0,:,j,c],color=[COL[2],COL[0],COL[3],'#d65c4f'][c],lw=1.4,ls=['--',':','-.','-'][c],label=short[c])
  ax[2].set(xlabel='Forecast step (h)',ylabel='Standardized value',title='(c) First test origin: '+LABELS[task][j]);ax[2].legend(ncol=3,fontsize=10,loc='best');ax[2].grid(alpha=.15);save(fig,'branch_complementarity_'+task)
  fig,ax=plt.subplots(1,2,figsize=(9.5,3.3),layout='constrained')
  batchorder=list(np.unique(bids));x=np.arange(len(batchorder))
  for c in [0,1,2,3]:
   vals=[float((err[bids==b,...,c]**2).mean()) for b in batchorder]
   ax[0].plot(x,vals,marker='o',markersize=3,lw=1,color=[COL[2],COL[0],COL[3],'#d65c4f'][c],label=short[c])
  ax[0].set(xlabel='Test trajectory (original order)',ylabel='Trajectory MSE',title='(a) All test trajectories');ax[0].legend(ncol=2,fontsize=8);ax[0].grid(alpha=.15)
  if task=='CHO':ax[0].set_xticks(x,[str(b) for b in batchorder])
  gg=[r for r in groups if r['dataset']==task and r['group']!='ALL'];gn=sorted(set(r['group'] for r in gg));xx=np.arange(len(gn));w=.18
  for c,name in enumerate(BRANCH):
   vals=[next(r['window_weighted_mse'] for r in gg if r['branch']==name and r['group']==g) for g in gn];ax[1].bar(xx+(c-1.5)*w,vals,width=w,color=[COL[2],COL[0],COL[3],'#d65c4f'][c],label=short[c])
  ax[1].set_xticks(xx,gn,rotation=15);ax[1].set(ylabel='Window-weighted MSE',title='(b) Test trajectory groups');ax[1].grid(axis='y',alpha=.15);save(fig,'batch_groups_'+task)
  assert sha(ck)==h
  del net,ds;torch.cuda.empty_cache();print('FULL_TEST_BRANCH_BATCH_COMPLETE',task,len(bids),flush=True)
 csvout('checkpoint_audit',audit);csvout('candidate_metrics',candidates);csvout('branch_error_correlations',corrrows);csvout('candidate_win_rates',wins);csvout('per_test_trajectory',batches);csvout('trajectory_group_results',groups)
 print('BRANCH_AND_BATCH_ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['data','checkpoint'],required=True);a=ap.parse_args();torch.set_num_threads(2)
 if a.part=='data':data_analysis()
 else:checkpoint_analysis('cuda:0')
