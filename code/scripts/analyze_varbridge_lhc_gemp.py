"""Read-only reuse of original results and training-only descriptive plots."""
import sys,re,csv,json,hashlib
from pathlib import Path
ROOT=Path('/home/home_new/qsmx/pycodes/GreyFerm');sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import skew,kurtosis,t as student_t,f as fdist
from varbridge_lhc_gemp import OUT,guard,writecsv
from analyze_varbridge_timebase import raw_task,LABELS,TASKS
torch.set_num_threads(2)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':12,'axes.labelsize':12,'axes.titlesize':12,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False})
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
COL=['#278e86','#ca984d','#608bab','#9b7ab6','#d25a50']
def save(fig,name):
 fig.savefig(FIG/(name+'.png'),dpi=360,bbox_inches='tight',facecolor='white');plt.close(fig)
guard()
rows=[];bindings=[];relationrows=[]
for task in TASKS:
 raw,features,ids,mean,std,_=raw_task(task);train=raw[raw.batch_id.isin(ids['train'])];vals=train[features].to_numpy(float)
 fig,ax=plt.subplots(1,5,figsize=(12.0,2.8),layout='constrained')
 for j,feature in enumerate(features):
  v=vals[:,j];q=np.quantile(v,[.25,.5,.75])
  rows.append(dict(dataset=task,variable=feature,symbol=LABELS[task][j],partition='train',n=len(v),mean=v.mean(),median=q[1],sample_sd=v.std(ddof=1),q25=q[0],q75=q[2],minimum=v.min(),maximum=v.max(),skew=skew(v,bias=False),excess_kurtosis=kurtosis(v,bias=False)))
  ax[j].hist(v,bins='fd',density=True,color=COL[j],alpha=.6,edgecolor='white',linewidth=.3)
  ax[j].set_title(LABELS[task][j]+' / '+feature);ax[j].set_xlabel('Original measurement');ax[j].tick_params(labelsize=9);ax[j].grid(axis='y',alpha=.12)
  if j==0:ax[j].set_ylabel('Density')
  bindings.append(dict(dataset=task,variable=feature,symbol=LABELS[task][j],mechanism_slot=j,target_index=j,sampling_hours=1,split_rule='Original eligible batch/EXP membership; no resegmentation into physiological stages'))
 save(fig,'training_variable_distributions_'+task)
 # TRAIN-only lag2 diagnostic: regress target on own lags versus own+other lags.
 # No transition across trajectory boundaries, no Granger-to-causality claim.
 xx=[];yy=[]
 for bid in ids['train']:
  q=train[train.batch_id==bid][features].to_numpy(float);z=(q-mean)/std
  xx.append(np.concatenate([z[1:-1],z[:-2]],axis=1));yy.append(z[2:])
 x=np.concatenate(xx);y=np.concatenate(yy);n=len(x);mat=np.zeros((5,5))
 for target in range(5):
  base=np.column_stack([np.ones(n),x[:,target],x[:,target+5]])
  resid=y[:,target]-base@np.linalg.lstsq(base,y[:,target],rcond=None)[0];sse0=resid@resid
  for source in range(5):
   if source==target:continue
   full=np.column_stack([base,x[:,source],x[:,source+5]])
   resid=y[:,target]-full@np.linalg.lstsq(full,y[:,target],rcond=None)[0];sse1=resid@resid
   F=max(0.,(sse0-sse1)/2/(sse1/(n-5)));p=fdist.sf(F,2,n-5);mat[target,source]=np.log1p(F)
   relationrows.append(dict(dataset=task,target=features[target],source=features[source],lag=2,train_pairs=n,F_statistic=F,p_value=p,log1p_F=np.log1p(F),interpretation='Linear lagged predictive association only; not a causal relation'))
 fig,ax=plt.subplots(figsize=(4.7,3.9),layout='constrained');im=ax.imshow(mat,cmap='YlGnBu')
 ax.set_xticks(range(5),LABELS[task]);ax.set_yticks(range(5),LABELS[task]);ax.set(xlabel='Lagged source variable',ylabel='Predicted variable',title='Training-only lagged predictability')
 for i in range(5):
  for j in range(5):ax.text(j,i,'—' if i==j else f'{mat[i,j]:.1f}',ha='center',va='center',color='white' if mat[i,j]>.6*mat.max() else 'black',fontsize=10)
 fig.colorbar(im,ax=ax,label='log(1 + F statistic)',fraction=.045);save(fig,'training_lagged_associations_'+task)
writecsv(OUT/'descriptive_statistics.csv',rows);writecsv(OUT/'target_bindings.csv',bindings);writecsv(OUT/'training_lagged_associations.csv',relationrows)
routingrows=[]
for task in ['EFP_long_sub2','CHO']:
 d=ROOT/'test_results/current_bound_supplement'/task
 sample=torch.load(d/'samples.pt',map_location='cpu');clean=torch.load(d/'clean_analysis.pt',map_location='cpu');tr=clean['trace']
 weights=tr['tri_router'].numpy();bids=sample['bids'].numpy();age=np.array(sample['age_end'],float)
 # Predetermined first chronological test trajectory, no target/error-based selection.
 bid=bids[0];ix=np.flatnonzero(bids==bid);ix=ix[np.argsort(age[ix])]
 w=weights[ix,:,0,:].mean(1);xx=sample['x'].numpy()[ix,-1,0]
 fig,ax=plt.subplots(2,1,figsize=(8.8,4.3),sharex=True,layout='constrained')
 for k,name in enumerate(['Backbone','Numerical','Corrected backbone']):ax[0].plot(age[ix],w[:,k],color=[COL[2],COL[0],COL[4]][k],label=name,lw=1.8)
 ax[0].set(ylabel='Mean fusion weight',ylim=(-.02,1.02),title=f'{task.replace("EFP_long_","EFP-")} / first test trajectory {int(bid)}');ax[0].legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.03),fontsize=10)
 ax[1].plot(age[ix],xx,color=COL[2],label='Latest observed value',lw=1.8);ax[1].set(xlabel='Observed culture / batch age (h)',ylabel='Standardized value');ax[1].legend(loc='best',fontsize=10)
 for a in ax:a.grid(alpha=.16)
 save(fig,'routing_sensor_alignment_'+task)
 fig,ax=plt.subplots(1,3,figsize=(11.4,3.25),layout='constrained')
 for a,key,title in zip(ax,['numeric_router','semantic_router','tri_router'],['Numerical routing','Correction routing','Final routing']):
  m=tr[key].numpy().mean((0,2)).T
  im=a.imshow(m,vmin=0,vmax=1,cmap='BuGn',aspect='auto',extent=[.5,24.5,2.5,-.5]);a.set(xlabel='Relative forecast step (h)',yticks=[0,1,2],yticklabels=['1','2','3'],title=title);a.set_ylabel('Candidate index')
 fig.colorbar(im,ax=ax.ravel().tolist(),fraction=.02,pad=.02,label='Mean weight')
 save(fig,'routing_by_forecast_step_'+task)
 for key in ['numeric_router','semantic_router','tri_router']:
  z=tr[key].numpy()
  for j,f in enumerate(sample['features']):
   for h in range(24):
    p=z[:,h,j,:].mean(0)
    routingrows.append(dict(dataset=task,router=key,variable=f,horizon_step=h+1,weight_1=p[0],weight_2=p[1],weight_3=p[2],entropy=float(-(p*np.log(p.clip(1e-12))).sum()),origins=len(z)))
 # Three context attention tokens / actual variable interactions, not fabricated semantic retrieval.
 fig,ax=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
 for a,key,labels,title in [(ax[0],'channel_attention_weights',LABELS[task],'Cross-variable interaction'),(ax[1],'cross_attention_weights',['Observed summary','Stage context','Operating context'],'General-context interaction')]:
  m=tr[key].numpy().mean(0);im=a.imshow(m,cmap='BuGn',vmin=0,vmax=1,aspect='auto')
  a.set_xticks(range(len(labels)),labels,rotation=20 if len(labels)==3 else 0);a.set_yticks(range(5),LABELS[task]);a.set_title(title);a.set_ylabel('Target variable')
  for i in range(5):
   for j in range(len(labels)):a.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',fontsize=10,color='white' if m[i,j]>.6 else 'black')
 fig.colorbar(im,ax=ax.ravel().tolist(),fraction=.023,pad=.02,label='Attention weight')
 save(fig,'observed_attention_maps_'+task)
writecsv(OUT/'routing_diagnostics.csv',routingrows)
# Current-model distinct-seed summary from the audited624-row log assembly.
verified=OUT/'deliverables/main_three_seed_statistics.csv'
assert verified.exists(),'Assemble current reference tables first'
stat=pd.read_csv(verified).to_dict('records')
writecsv(OUT/'existing_three_seed_intervals.csv',stat)
# Existing full fitted ablations, use separate tasks and real single-seed values, no SD.
report=(ROOT/'test_results/common_bound_ablations/final_report.md').read_text()
parsed=[]
for line in report.splitlines():
 a=line.split('|')
 if len(a)==10 and a[1] in ['EFP','CHO']:
  parsed.append(dict(dataset=a[1],variant=a[2],mse=float(a[3]),mae=float(a[5])))
if parsed:
 r=pd.DataFrame(parsed)
 choose=['full','shuffled_mechanism','generic_description','learned_condition','random_init_text','no_channel','no_dynamic_stats','no_forecast_conditioning','additive_fusion']
 for task,q in r.groupby('dataset'):
  q=q.set_index('variant');sel=[v for v in choose if v in q.index];fig,ax=plt.subplots(1,2,figsize=(10.8,3.4),layout='constrained')
  for a,metric in zip(ax,['mse','mae']):
   vals=[q.loc[v,metric] for v in sel];a.bar(np.arange(len(sel)),vals,color=[COL[4]]+[COL[2]]*(len(sel)-1),width=.72);a.axhline(q.loc['full',metric],ls='--',lw=1,color=COL[4]);a.set_xticks(range(len(sel)),['Full','Shuffled','Generic','Learned','Rand. encoder','No variable attn','No statistics','No joint condition','Additive'],rotation=50,ha='right',fontsize=9);a.set_ylabel(metric.upper());a.set_title(task+' / '+metric.upper());a.grid(axis='y',alpha=.15)
  save(fig,'existing_ablation_bars_'+task)
 writecsv(OUT/'existing_ablation_aggregates.csv',parsed)
guard()
(OUT/'READONLY_ANALYSES_COMPLETE').write_text('Training-only distributions/statistics/lagged diagnostics; original checkpoint routing; real distinct-seed intervals. No neural training.\n')
print('READONLY_ANALYSES_COMPLETE',len(list(FIG.glob('*.png'))),'PNGs',flush=True)

