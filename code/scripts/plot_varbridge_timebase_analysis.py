"""TimeBase-inspired small multiples and objective result tables; server outputs only."""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'test_results/varbridge_timebase_analysis';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
assert (OUT/'ALL_CHECKPOINT_ANALYSES_COMPLETE').exists()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':13,'axes.labelsize':12,'legend.fontsize':11,'axes.spines.top':False,'axes.spines.right':False})
r=pd.read_csv(OUT/'efficiency/results.csv');r=r.set_index('model').loc[['DLinear','PatchTST','iTransformer','Crossformer','TimeMixer','TimesNet','TimeLLM','VarBridge']].reset_index()
assert len(r)==8 and r.physical_gpu.nunique()==1 and (r.epochs==100).all()
colors=['#5785a6','#4d9c91','#a0a64b','#8c75a9','#d39856','#78a8b7','#a76f9a','#d34f43'];markers=['o','s','^','D','v','P','X','*']
fig,ax=plt.subplots(1,2,figsize=(10,4.0),layout='constrained')
# Area is linear in GPU memory; parameter axis is explicitly logarithmic in panel b.
# Fixed label offsets avoid data-dependent hiding of models.
offsets={'DLinear':(9,6),'PatchTST':(-8,-18),'iTransformer':(5,7),'Crossformer':(-8,11),'TimeMixer':(8,-12),'TimesNet':(-8,13),'TimeLLM':(9,8),'VarBridge':(9,-1)}
for k,row in r.iterrows():
 name=row.model;x=float(row.latency_median_ms);mse=float(row.mse);params=float(row.resident_parameters);memory=float(row.peak_allocated_MiB);label='Time-LLM' if name=='TimeLLM' else name
 ax[0].scatter(x,mse,s=memory*2,c=colors[k],alpha=.65,edgecolors=colors[k],marker='o',zorder=3)
 dx,dy=offsets[name];ax[0].annotate(label,(x,mse),xytext=(dx,dy),textcoords='offset points',ha='right' if dx<0 else 'left',fontsize=11,color=colors[k],fontweight='bold' if name=='VarBridge' else 'normal')
 ax[1].scatter(params,mse,s=90 if name!='VarBridge' else 150,c=colors[k],marker=markers[k],label=label,zorder=3)
 ax[1].annotate(label,(params,mse),xytext=(8,5 if name!='PatchTST' else -16),textcoords='offset points',fontsize=11,color=colors[k],ha='left')
for a in ax:a.set_xscale('log');a.grid(alpha=.17);a.set_ylabel('Full-test MSE');a.margins(x=.25,y=.2)
ax[0].set(xlabel='Cached prediction latency (ms, log scale)',title='(a) Accuracy and inference cost');ax[0].text(.02,.02,'Bubble area ∝ peak allocated GPU memory',transform=ax[0].transAxes,fontsize=11)
ax[1].set(xlabel='Resident predictor parameters (log scale)',title='(b) Accuracy and predictor size');ax[1].set_xlim(r.resident_parameters.min()*.45,r.resident_parameters.max()*5)
fig.savefig(FIG/'efficiency_timebase_overview.png',dpi=400,bbox_inches='tight',facecolor='white');plt.close(fig)
# Compact table, native measurements, no training-time comparison from heterogeneous fits.
headers=['Model','MSE','MAE','Resident\nparams','Median\n(ms)','P95\n(ms)','Peak GPU\n(MiB)','CPU cache\n(KiB/window)']
body=[]
for row in r.itertuples():
 p=row.resident_parameters;ps=f'{p/1e6:.2f}M' if p>=1e6 else f'{p/1e3:.2f}K'
 body.append(['Time-LLM' if row.model=='TimeLLM' else row.model,f'{row.mse:.4f}',f'{row.mae:.4f}',ps,f'{row.latency_median_ms:.3f}',f'{row.latency_p95_ms:.3f}',f'{row.peak_allocated_MiB:.1f}',f'{row.mean_CPU_cache_bytes/1024:.1f}'])
fig,ax=plt.subplots(figsize=(9.5,3.4));ax.axis('off');t=ax.table(cellText=body,colLabels=headers,cellLoc='center',loc='center',colWidths=[.17,.09,.09,.12,.12,.12,.13,.16]);t.auto_set_font_size(False);t.set_fontsize(11);t.scale(1,1.65)
for (i,j),cell in t.get_celld().items():
 cell.set_edgecolor('#cccccc');cell.set_linewidth(.4)
 if i==0:cell.set_facecolor('#e8edf0');cell.set_text_props(weight='bold');cell.set_height(cell.get_height()*1.25)
 elif i==8:cell.set_facecolor('#fceae7');cell.set_text_props(color='#b84438',weight='bold')
 elif i%2==0:cell.set_facecolor('#f4f4f4')
fig.savefig(FIG/'efficiency_timebase_table.png',dpi=400,bbox_inches='tight',facecolor='white');plt.close(fig)
# Side-by-side, same units: cached predictor parameter and frozen offline encoder count.
v=r[r.model=='VarBridge'].iloc[0]
# TimeBase-style compact dataset and grouped-result tables.
def table_png(body,headers,name,width):
 fig,ax=plt.subplots(figsize=(width,.44*(len(body)+2)));ax.axis('off')
 tab=ax.table(cellText=body,colLabels=headers,cellLoc='center',loc='center');tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,1.6)
 for (i,j),cell in tab.get_celld().items():
  cell.set_edgecolor('#cccccc');cell.set_linewidth(.4)
  if i==0:cell.set_facecolor('#e8edf0');cell.set_text_props(weight='bold')
  elif i%2==0:cell.set_facecolor('#f3f3f3')
 fig.savefig(FIG/(name+'.png'),dpi=400,bbox_inches='tight',facecolor='white');plt.close(fig)
stats=pd.read_csv(OUT/'dataset_statistics.csv');body=[]
for task in stats.dataset.unique():
 q=stats[stats.dataset==task].set_index('split');body.append([task.replace('EFP_long_','EFP-'),5,'1 h',' / '.join(str(int(q.loc[z,'trajectories'])) for z in ['train','val','test']),int(q.loc['test','H24_windows']),int(q.loc['test','H48_windows'])])
table_png(body,['Task','Variables','Sampling','Train / val / test\ntrajectories','H24 test\nwindows','H48 test\nwindows'],'dataset_statistics_table',9)
gg=pd.read_csv(OUT/'trajectory_group_results.csv');body=[]
for task in gg.dataset.unique():
 for group in gg.loc[(gg.dataset==task)&(gg.group!='ALL'),'group'].unique():
  q=gg[(gg.dataset==task)&(gg.group==group)].set_index('branch');body.append([task.replace('EFP_long_','EFP-'),group,int(q.iloc[0].trajectories),*[f'{q.loc[b,"window_weighted_mse"]:.4f}' for b in ['Backbone','Numerical expert','Corrected backbone','Final fusion']]])
table_png(body,['Task','Test group','Trajectories','Backbone\nMSE','Numeric\nMSE','Corrected\nMSE','Fusion\nMSE'],'trajectory_groups_table',9.5)
notes=[]
notes.append('# VarBridge checkpoint-only TimeBase-style analyses\n')
notes.append('All VarBridge adaptation checkpoints used here completed100 epochs; seed2021; selected by validation MSE. Existing numerical-backbone preparation is unchanged. No new training or model-source edit.\n')
notes.append('## Data characteristics\nFour task views; training-only trajectory-centered Pearson levels/first-difference matrices; diagnostic frozen-description cosine after shared centering over unique correct EFP/CHO descriptions. The identical diagnostic encoding is separate from unchanged task-specific adopted caches. Similarity is descriptive, not evidence for causal semantics or low-rank structure. First training trajectory shown without outcome selection.\n')
c=pd.read_csv(OUT/'candidate_metrics.csv');co=pd.read_csv(OUT/'branch_error_correlations.csv');wr=pd.read_csv(OUT/'candidate_win_rates.csv');gr=pd.read_csv(OUT/'trajectory_group_results.csv')
notes.append('## Full-test candidate forecasts\n')
notes.append('Variable-specialization heatmaps annotate absolute MSE; their colors encode log10(MSE/backbone MSE), with a symmetric diverging range. Efficiency bubble areas scale linearly with peak allocated GPU memory and use identical circular markers.\n')
for task in ['EFP_long_sub2','CHO']:
 q=c[(c.dataset==task)&(~c.branch.str.contains(' / '))]
 notes.append(task+': '+ '; '.join(f'{x.branch} MSE={x.mse:.6f}, MAE={x.mae:.6f}' for x in q.itertuples())+'.\n')
 cc=co[(co.dataset==task)&(co.branch1=='Backbone')&(co.branch2=='Corrected backbone')].iloc[0]
 notes.append(f'Backbone versus corrected-backbone signed-error correlation: {cc.signed_error_pearson:.6f}. These forecasts share a base path; correlation is not an independent-modality estimate.\n')
 for x in wr[wr.dataset==task].itertuples():notes.append(f'{x.branch}: hindsight scalar win fraction {x.scalar_lowest_squared_error_fraction:.4f}; mean learned final weight {x.mean_final_router_weight:.4f}.\n')
batch_metrics=pd.read_csv(OUT/'per_test_trajectory.csv')
for task in ['EFP_long_sub2','CHO']:
 pivot=batch_metrics[batch_metrics.dataset==task].pivot(index='batch_id',columns='branch',values='mse')
 notes.append(f'{task}: final fusion has lower trajectory MSE than its frozen backbone in {int((pivot["Final fusion"]<pivot["Backbone"]).sum())}/{len(pivot)} test trajectories.\n')
notes.append('## Test trajectory groups\nEFP quartiles use training-batch volatility boundaries; full-trajectory test volatility is a retrospective description, not a forecasting input. CHO is grouped by original EXP and culture mode; parallel reactors are not independent experiments. Both window-weighted and equal-trajectory MSE are provided.\n')
for x in gr[(gr.branch=='Final fusion')].itertuples():notes.append(f'{x.dataset} {x.group}: {x.trajectories} trajectories, {x.windows} windows; window MSE={x.window_weighted_mse:.6f}, equal-trajectory MSE={x.equal_trajectory_mse:.6f}.\n')
notes.append('## Fresh eight-model efficiency\n')
notes.append(f'One physical RTX4090 GPU{int(v.physical_gpu)}; FP32,batch1,two CPUthreads;20 warmups,32 fixed windows x5 repeats;381-window errors. No oldGPU1 timings merged. CPU text-feature cache for VarBridge; CPU native prompt input-embedding cache for TimeLLM, all GPT2 prefix/patch processing retained. Latency excludes text cache construction/model load/static preparation.\n')
for x in r.itertuples():notes.append(f'{x.model}: MSE={x.mse:.6f}; median={x.latency_median_ms:.3f}ms; P95={x.latency_p95_ms:.3f}ms; peak={x.peak_allocated_MiB:.1f}MiB; resident={x.resident_parameters}; training-trainable={x.trainable_at_training_parameters}; offline-text={x.offline_text_encoder_parameters}; CPUcache={x.mean_CPU_cache_bytes:.0f}bytes/window; counted matmul/conv MACs={x.operator_macs_per_sample:.0f}.\n')
notes.append('Operator MACs exclude FFT, softmax, normalization, elementwise operations and text preparation. They are an explicitly scoped supplementary operator count, not total end-to-end compute, and are therefore not used as an overall-efficiency ranking.\n')
notes.append('No across-seed SD or significance claim is derived from inference repetitions or overlapping windows. Figures use final PNGs only; code/raw outputs stay on the server.\n')
(OUT/'analysis_summary.md').write_text('\n'.join(notes))

# Resolve candidate complementarity at the sensor level, using all test windows.
import torch
sensor_rows=[]
for task,symbols in [('EFP_long_sub2',['α','φ','ω','θ','ψ']),('CHO',['χ','τ','ε','θ','ω'])]:
 d=torch.load(OUT/(task+'_fulltest_branches.pt'),map_location='cpu');pred=d['candidates'].numpy();target=d['target'].numpy();weights=d['final_weights'].numpy()
 errors=pred.astype(np.float64)-target.astype(np.float64)[...,None];mse=(errors**2).mean(axis=(0,1));relative=np.log10(mse/mse[:,[0]])
 route=weights.mean(axis=(0,1),dtype=np.float64);sdroute=weights.std(axis=(0,1),dtype=np.float64);pairs=[(0,1),(0,2),(1,2)]
 errorcorr=np.array([[np.corrcoef(errors[:,:,j,a].ravel(),errors[:,:,j,b].ravel())[0,1] for a,b in pairs] for j in range(5)])
 for j,f in enumerate(d['features']):
  for k,branch in enumerate(d['candidate_names']):sensor_rows.append(dict(dataset=task,variable=f,branch=branch,mse=float(mse[j,k]),log10_mse_ratio_to_backbone=float(relative[j,k]),mean_final_weight=float(route[j,k]) if k<3 else '',SD_final_weight_over_windows_and_steps=float(sdroute[j,k]) if k<3 else ''))
 fig,ax=plt.subplots(1,3,figsize=(10,3.6),layout='constrained',gridspec_kw={'width_ratios':[1.3,1,1]})
 mats=[relative,route,errorcorr];titles=['(a) MSE; color = log ratio','(b) Mean final fusion weights','(c) Candidate-error correlations'];columns=[['Base','Numeric','Corrected','Fusion'],['Base','Numeric','Corrected'],['Base–Num.','Base–Corr.','Num.–Corr.']]
 for k,(matrix,title,cols) in enumerate(zip(mats,titles,columns)):
  if k==0:
   from matplotlib.colors import TwoSlopeNorm
   span=max(.1,float(abs(matrix).max()));im=ax[k].imshow(matrix,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-span,vcenter=0,vmax=span));clabel='log10(MSE / backbone MSE)'
  else:
   im=ax[k].imshow(matrix,cmap='YlGnBu' if k==1 else 'RdBu_r',vmin=0 if k==1 else -1,vmax=1);clabel='Mean weight' if k==1 else 'Pearson r'
  ax[k].set_xticks(range(matrix.shape[1]),cols,rotation=25);ax[k].set_yticks(range(5),symbols);ax[k].set_title(title,fontsize=11);fig.colorbar(im,ax=ax[k],fraction=.045,pad=.04,label=clabel)
  for j in range(5):
   for c0 in range(matrix.shape[1]):
    value=float(matrix[j,c0]);label=f'{mse[j,c0]:.3f}' if k==0 else f'{value:.2f}'
    ax[k].text(c0,j,label,ha='center',va='center',fontsize=11,color='white' if (abs(value)>(span*.65 if k==0 else .65)) else 'black')
 fig.savefig(FIG/('variable_branch_specialization_'+task+'.png'),dpi=400,bbox_inches='tight',facecolor='white');plt.close(fig)
with (OUT/'variable_branch_specialization.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(sensor_rows[0]));writer.writeheader();writer.writerows(sensor_rows)

provenance={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')}
(OUT/'figure_provenance.json').write_text(json.dumps({'source_hashes':provenance,'efficiency_source_sha256':hashlib.sha256((OUT/'efficiency/results.csv').read_bytes()).hexdigest(),'palette':colors,'seed':2021,'no_training':True},indent=2))
(OUT/'PLOTS_AND_SUMMARY_COMPLETE').write_text('TimeBase-style PNG figures and explicit result summary; all source data preserved on server\n')
print('TIMEBASE_ANALYSIS_PLOTS_COMPLETE',len(list(FIG.glob('*.png'))),flush=True)
