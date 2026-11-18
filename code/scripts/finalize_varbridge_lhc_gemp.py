"""Audit and render completed LHC/GEMP-inspired supplements, without paper edits."""
import sys,time,json,csv,math
from pathlib import Path
ROOT=Path('/home/home_new/qsmx/pycodes/GreyFerm');sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from varbridge_lhc_gemp import OUT,CK,VARIANTS,TASKS,guard,writecsv,sha
from varbridge_lhc_gemp_lookback import B,LENGTHS
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':12,'axes.labelsize':12,'axes.titlesize':12,'legend.fontsize':10,'axes.spines.top':False,'axes.spines.right':False})
COL=['#608bab','#278e86','#d25a50']
def save(fig,name):fig.savefig(OUT/'figures'/(name+'.png'),dpi=360,bbox_inches='tight',facecolor='white');plt.close(fig)
def context_results():
 rows=[]
 for task in TASKS:
  for v in VARIANTS:
   marker=OUT/(task+'_'+v+'_COMPLETE')
   if not marker.exists():continue
   r=json.loads(marker.read_text());assert r['epochs']==100 and r['seed']==2021
   curve=pd.read_csv(OUT/(task+'_'+v+'_learning.csv'))
   assert curve.epoch.tolist()==list(range(1,101))
   assert r['best_epoch']==int(curve.loc[curve.validation_mse.idxmin(),'epoch'])
   assert sha(r['checkpoint'])==r['checkpoint_sha256']
   assert np.isfinite([r['mse'],r['mae'],r['validation_mse']]).all()
   assert r['test_windows']==(381 if task=='CHO' else 4431)
   for e in [0,1,5,10,25,50,75,100]:
    d=torch.load(OUT/'traces'/f'{task}_{v}_epoch{e:03d}.pt',map_location='cpu');assert d['epoch']==e
    for z in d['weights'].values():assert torch.isfinite(z).all() and torch.allclose(z.sum(-1),torch.ones_like(z[...,0]),atol=2e-6)
   rows.append(r)
 writecsv(OUT/'context_control_results.csv',rows)
 if len(rows)!=16:return False
 df=pd.DataFrame(rows)
 for task in TASKS:
  q=df[df.dataset==task].set_index('variant')
  fig,axs=plt.subplots(1,2,figsize=(9.4,3.4),layout='constrained')
  for ax,metric in zip(axs,['mse','mae']):
   vs=['group_1','group_2','group_3','group_4','own_target'];vals=[q.loc[v,metric] for v in vs]
   ax.plot(range(1,6),vals,marker='o',color=COL[0],label='Group-pooled context')
   ax.scatter([5],[vals[-1]],marker='*',s=150,color=COL[2],label='Own-target context',zorder=4)
   ax.axhline(vals[-1],color=COL[2],ls='--',lw=1);ax.set(xticks=range(1,6),xlabel='Number of conditioning groups',ylabel=metric.upper(),title=task.replace('EFP_long_','EFP-'));ax.grid(alpha=.15)
  axs[0].legend(fontsize=9);save(fig,'binding_group_count_'+task)
  fig,axs=plt.subplots(1,3,figsize=(11.4,3.5),layout='constrained')
  for v,color,label in [('own_target',COL[0],'Correct target'),('contradictory',COL[2],'Contradictory text'),('noisy_text','#9b7ab6','Noisy embedding'),('group_1','#ca984d','Global pooled context')]:
   c=pd.read_csv(OUT/(task+'_'+v+'_learning.csv'));axs[0].plot(c.epoch,c.validation_mse,color=color,label=label)
   ww=[]
   for e in range(101):
    d=torch.load(OUT/'traces'/f'{task}_{v}_epoch{e:03d}.pt',map_location='cpu');z=d['weights']['tri_router'];ww.append(z.mean((0,1,2)).numpy())
   ww=np.array(ww);axs[1].plot(range(101),ww[:,2],color=color,label=label);ent=-(ww*np.log(ww.clip(1e-12))).sum(-1);axs[2].plot(range(101),ent,color=color,label=label)
  axs[0].set(ylabel='Validation MSE',xlabel='Epoch',title='Learning under text interventions')
  axs[1].set(ylabel='Corrected-backbone weight',xlabel='Epoch',title='Fixed training probe')
  axs[2].set(ylabel='Entropy of mean routing',xlabel='Epoch',title='Routing concentration')
  for a in axs:a.grid(alpha=.15)
  axs[0].legend(fontsize=8);save(fig,'context_learning_'+task)
  for v in ['own_target','contradictory','noisy_text','group_1']:
   fig,axs=plt.subplots(1,3,figsize=(10.5,3.1),layout='constrained')
   cube=[]
   for e in range(101):
    d=torch.load(OUT/'traces'/f'{task}_{v}_epoch{e:03d}.pt',map_location='cpu');cube.append(d['weights']['tri_router'].mean((0,2)).numpy())
   z=np.array(cube)
   for k,a in enumerate(axs):
    im=a.imshow(z[:,:,k].T,aspect='auto',origin='lower',vmin=0,vmax=1,cmap='BuGn',extent=[-.5,100.5,.5,24.5]);a.set(xlabel='Epoch',ylabel='Relative forecast step',title=['Backbone','Numerical','Corrected backbone'][k])
   fig.colorbar(im,ax=axs.ravel().tolist(),label='Mean weight',fraction=.02,pad=.02);save(fig,'routing_training_heatmap_'+task+'_'+v)
 return True
def status():
 ctx=sum((OUT/(t+'_'+v+'_COMPLETE')).exists() for t in TASKS for v in VARIANTS)
 look=list(B.glob('*_COMPLETE'));look=[p for p in look if '_seed2021_COMPLETE' in p.name]
 api=OUT/'api/progress.json';j=json.loads(api.read_text()) if api.exists() else {}
 return dict(context_fits_completed=ctx,context_fits_expected=16,lookback_fits_completed=len(look),lookback_fits_expected=40,llm_calls_received=j.get('received',0),llm_calls_expected=120,readonly_plots=len(list((OUT/'figures').glob('*.png'))),all_formal_sources_unchanged=True)
context_plotted=False
while True:
 guard();s=status();(OUT/'completion_status.json').write_text(json.dumps(s,indent=2));context_ok=False
 if s['context_fits_completed']==16:
  if not context_plotted:context_plotted=context_results()
  context_ok=context_plotted
 api_done=(OUT/'api/analysis_status.json').exists();look_done=(B/'ALL_LOOKBACK_COMPLETE').exists()
 if context_ok and look_done and api_done:break
 time.sleep(30)
# Explicit final audit: no marker alone suffices.
rows=[]
for p in B.glob('*_result.csv'):
 r=list(csv.DictReader(p.open()))[0];curve=pd.read_csv(B/(p.name.replace('_result.csv','_learning.csv')))
 expected=100 if r['model'] in ['VarBridge','VarBridgeBackbone'] else 10
 assert int(r['epochs'])==expected and curve.epoch.tolist()==list(range(1,expected+1))
 assert sha(r['checkpoint'])==r['checkpoint_sha256']
 rows.append(r)
assert len(rows)==40
for task in TASKS:
 values=[r for r in rows if r['dataset']==task]
 assert len({r['test_windows'] for r in values})==1
api=json.loads((OUT/'api/analysis_status.json').read_text())
assert api['reference_not_expert_truth'] and api['new_training']==False
if not api['complete']:
 (OUT/'PARTIAL_API_AUDIT').write_text(json.dumps(api,indent=2))
 raise RuntimeError('API incomplete: do not mark all experiments complete')
guard()
formal=json.loads((OUT/'formal_result_hashes.json').read_text())
assert all(sha(p)==h for p,h in formal.items()),'Formal result/checkpoint mutation'
s=status();s['fully_audited']=True;s['no_manuscript_or_main_results_modified']=True
(OUT/'FINAL_AUDIT_PASS.json').write_text(json.dumps(s,indent=2));print('ALL_LHC_GEMP_SUPPLEMENTS_COMPLETE',json.dumps(s),flush=True)

