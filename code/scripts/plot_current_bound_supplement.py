"""Fixed multi-panel figure recipes. Reads canonical server artifacts only."""
from pathlib import Path
import argparse,inspect
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common_bound_supplement import ROOT,DATASETS,DEFAULTS,GRIDS
OUT=ROOT/'test_results/current_bound_supplement';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
DISPLAY={'EFP_long_sub2':'EFP-sub2','CHO':'CHO'}
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7'];plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,name):
 fig.savefig(FIG/(name+'.pdf'));fig.savefig(FIG/(name+'.png'),dpi=240);plt.close(fig)
def sensitivity():
 d=pd.read_csv(OUT/'sensitivity_results.csv');assert len(d)==24,'No partial sensitivity figures'
 for ds in DATASETS:
  fig,axs=plt.subplots(2,3,figsize=(10,5.8),layout='constrained');fig.suptitle(('EFP-sub2' if ds=='EFP_long_sub2' else 'CHO')+' | H24 | seed 2021',fontsize=12)
  for j,(key,grid) in enumerate(GRIDS.items()):
   for i,metric in enumerate(['mse','mae']):
    q=d[(d.dataset==ds)&((d.variant=='full')|d.variant.str.startswith(key+'_'))].sort_values(key);assert len(q)==len(grid)
    ax=axs[i,j];ax.plot(q[key],q[metric],marker='o',color=COLORS[i]);ax.axvline(DEFAULTS[key],ls='--',color='gray',lw=1);ax.set(xlabel={'alpha':'EMA coefficient','fusion_heads':'Fusion attention heads','expert_width':'Expert hidden width'}[key],ylabel=metric.upper(),title=f'({chr(97+i*3+j)}) '+{'alpha':'EMA coefficient','fusion_heads':'Attention heads','expert_width':'Expert width'}[key]);ax.set_xticks(grid)
    default=q[q[key]==DEFAULTS[key]].iloc[0];ax.scatter([DEFAULTS[key]],[default[metric]],s=60,facecolors='white',edgecolors=COLORS[i],zorder=5,label='Default (reused)');ax.legend(fontsize=10,frameon=False)
  save(fig,'sensitivity_'+ds)
 print('SENSITIVITY_FIGURES_COMPLETE',flush=True)
def features():
 metrics={ds:pd.read_csv(OUT/ds/'feature_metrics.csv') for ds in DATASETS}
 data={ds:torch.load(OUT/ds/'samples.pt',map_location='cpu') for ds in DATASETS};clean={ds:torch.load(OUT/ds/'clean_analysis.pt',map_location='cpu') for ds in DATASETS}
 # Four sensor fault families, two process rows.
 fig,axs=plt.subplots(2,4,figsize=(9,5.5),layout='constrained')
 families=[('Noise',['clean','noise_0p05','noise_0p1','noise_0p2'],[0,.05,.1,.2]),('Dropped channels',['clean','drop_1','drop_2'],[0,1,2]),('Stuck segment',['clean','stuck_8','stuck_16'],[0,8,16]),('Delayed channel',['clean','delay_1','delay_2','delay_4'],[0,1,2,4])]
 for r,ds in enumerate(DATASETS):
  for c,(name,conditions,xx) in enumerate(families):
   ax=axs[r,c]
   for k,model in enumerate(['full','no_channel']):
    yy=[float(metrics[ds][(metrics[ds].condition==cond)&(metrics[ds].model==model)].iloc[0].mse) for cond in conditions];ax.plot(xx,yy,marker=['o','s'][k],color=COLORS[k],label={'full':'VarBridge','no_channel':'w/o cross-variable\nattention'}[model])
   ax.set(xlabel=name,ylabel='MSE',title=DISPLAY[ds])
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=2,fontsize=11,frameon=False)
 save(fig,'sensor_reliability')
 fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
 for ax,ds in zip(axs,DATASETS):
  kinds=['correct','generic','mismatched','placeholder'];values=[]
  for noise in ['clean','noise_0p1','noise_0p2']:
   row=[]
   for kind in kinds:
    q=metrics[ds];q=q[(q.group=='text')&(q.condition==kind)] if noise=='clean' else q[(q.group=='joint_quality')&(q.condition==noise+'_'+kind)]
    row.append(float(q.iloc[0].mse))
   values.append(row)
  im=ax.imshow(values,cmap='viridis',vmin=min(float(metrics[k].loc[(metrics[k].group=='text')|(metrics[k].group=='joint_quality'),'mse'].min()) for k in DATASETS),vmax=max(float(metrics[k].loc[(metrics[k].group=='text')|(metrics[k].group=='joint_quality'),'mse'].max()) for k in DATASETS));ax.set_xticks(range(4),kinds,rotation=25,ha='right');ax.set_yticks(range(3),['0','0.1','0.2']);ax.set(ylabel='Noise SD',title=DISPLAY[ds])
  for i in range(3):
   for j in range(4):ax.text(j,i,f'{values[i][j]:.3f}',ha='center',va='center',color='black' if im.norm(values[i][j])>.55 else 'white')
  fig.colorbar(im,ax=ax,label='MSE')
 save(fig,'numeric_text_quality')
 # Attention matrices and context weights.
 fig,axs=plt.subplots(2,2,figsize=(9,7),layout='constrained')
 for r,ds in enumerate(DATASETS):
  tr=clean[ds]['trace'];names=data[ds]['features']
  for c,(key,cols) in enumerate([('channel_attention_weights',names),('cross_attention_weights',['Observed','Stage','Operation'])]):
   values=tr[key].mean(0).numpy();ax=axs[r,c];im=ax.imshow(values,cmap='Blues',vmin=0,vmax=1);ax.set_xticks(range(len(cols)),cols,rotation=30,ha='right');ax.set_yticks(range(5),names);ax.set_title(DISPLAY[ds]+' / '+('sensor attention' if c==0 else 'process context'));fig.colorbar(im,ax=ax,label='Mean attention')
 save(fig,'attention_structure')
 fig,axs=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
 for ax,ds in zip(axs,DATASETS):
  q=pd.read_csv(OUT/ds/'binding_response.csv');names=data[ds]['features'];mat=q.pivot(index='input_variable',columns='output_variable',values='prediction_rms').loc[names,names];im=ax.imshow(mat,cmap='magma');ax.set_xticks(range(5),names,rotation=25,ha='right');ax.set_yticks(range(5),names);ax.set(xlabel='Forecast target',ylabel='Perturbed mechanism slot',title=DISPLAY[ds]);fig.colorbar(im,ax=ax,label='Prediction RMS change')
 save(fig,'variable_binding_response')
 fig,axs=plt.subplots(2,3,figsize=(10,5.5),layout='constrained')
 for r,ds in enumerate(DATASETS):
  for c,(router,title) in enumerate([('numeric_router','Numerical experts'),('semantic_router','Mechanism corrections'),('tri_router','Final candidates')]):
   weights=clean[ds]['trace'][router].mean((0,2));ax=axs[r,c]
   for k in range(3):ax.plot(np.arange(1,25),weights[:,k],color=COLORS[k],ls=['-','--',':'][k],label=str(k+1))
   ax.set(xlabel='Forecast step',ylabel='Routing weight',ylim=(0,1),title=DISPLAY[ds]+' / '+title);ax.legend(ncol=3,fontsize=7)
 save(fig,'expert_routing')
 fig,axs=plt.subplots(2,2,figsize=(9,6),layout='constrained')
 for r,ds in enumerate(DATASETS):
  q=metrics[ds];names=['clean','attention_high','attention_low','drop_1'];vals=[]
  for cond in names:vals.append(float(q[(q.condition==cond)&(q.model=='full')].iloc[0].mse))
  axs[r,0].bar(['Clean','High attention','Low attention','Random'],vals,color=COLORS);axs[r,0].tick_params(axis='x',rotation=20);axs[r,0].set(ylabel='MSE',title=DISPLAY[ds]+' / sensor occlusion')
  z=q[q.group=='routing'];axs[r,1].bar(np.arange(len(z)),z.mse,color=COLORS[0]);axs[r,1].axhline(vals[0],color='gray',ls='--');axs[r,1].set_xticks(np.arange(len(z)),[v.replace('numeric_router_','Numerical / ').replace('semantic_router_','Mechanism / ').replace('tri_router_','Final / ').replace('shuffle','shuffled') for v in z.condition],rotation=50,ha='right',fontsize=10);axs[r,1].set(ylabel='MSE',title=DISPLAY[ds]+' / routing interventions')
 save(fig,'attention_routing_interventions')
 fig,axs=plt.subplots(2,2,figsize=(9,6),layout='constrained')
 for r,ds in enumerate(DATASETS):
  q=metrics[ds];q=q[q.group=='text'];axs[r,0].bar(np.arange(len(q)),q.mse,color=COLORS[0]);axs[r,0].set_xticks(np.arange(len(q)),q.condition,rotation=55,ha='right',fontsize=7);axs[r,0].set(ylabel='MSE',title=ds+' / description interventions')
  e=pd.read_csv(OUT/ds/'conditional_errors.csv')
  for j,name in enumerate(data[ds]['features']):
   sub=e[e.variable==name];axs[r,1].plot(sub.training_value_quartile,sub.mse,marker='o',label=name)
  axs[r,1].set(xlabel='Training-defined value quartile',ylabel='MSE',title=ds+' / conditional errors');axs[r,1].legend(fontsize=6,ncol=2)
 save(fig,'text_and_value_conditions')
 fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
 for ax,ds in zip(axs,DATASETS):
  q=pd.read_csv(OUT/ds/'recursive_steps.csv')
  for j,name in enumerate(q.model.unique()):
   z=q[q.model==name];ax.plot(z.step,z.mse,color=COLORS[j],label=name)
  ax.axvline(24.5,color='gray',ls='--');ax.set(xlabel='Forecast step',ylabel='MSE',title=DISPLAY[ds]);ax.legend(fontsize=10)
 save(fig,'recursive_error_propagation')
 # Joint distributions compare the full model with its frozen-backbone and numerical candidates.
 from scipy.stats import gaussian_kde
 fig=plt.figure(figsize=(9,6.5),layout='constrained');subfigs=fig.subfigures(2,3)
 for r,ds in enumerate(DATASETS):
  features=data[ds]['features'];selected=['phzx','doz','hx'] if ds!='CHO' else ['ph','do','permittivity'];truth=data[ds]['y'][:,:24]
  for c,name in enumerate(selected):
   j=features.index(name);grid=subfigs[r,c].add_gridspec(2,2,width_ratios=[5,1],height_ratios=[1,5],hspace=.03,wspace=.03);top=subfigs[r,c].add_subplot(grid[0,0]);ax=subfigs[r,c].add_subplot(grid[1,0]);side=subfigs[r,c].add_subplot(grid[1,1]);y=truth[:,:,j].flatten().numpy();ix=np.unique(np.linspace(0,len(y)-1,min(4000,len(y))).astype(int));y=y[ix]
   numerical='Crossformer' if ds!='CHO' else 'PatchTST';predictions=[clean[ds]['prediction'][:,:,j],torch.load(OUT/ds/(numerical+'_benchmark_predictions.pt'),map_location='cpu')['prediction'][:,:,j],torch.load(OUT/ds/'T3Time_benchmark_predictions.pt',map_location='cpu')['prediction'][:,:,j]]
   allvals=[y]+[p.flatten().numpy()[ix] for p in predictions];lo=min(v.min() for v in allvals);hi=max(v.max() for v in allvals);line=np.linspace(lo,hi,150)
   top.plot(line,gaussian_kde(y)(line),color='black',label='Observed');side.plot(gaussian_kde(y)(line),line,color='black')
   for k,(pr,label) in enumerate(zip(predictions,['VarBridge',numerical,'T3Time'])):
    pred=pr.flatten().numpy()[ix];ax.scatter(y,pred,s=2,alpha=.15,color=COLORS[k],rasterized=True,label=label);side.plot(gaussian_kde(pred)(line),line,color=COLORS[k],ls=['-','--',':'][k])
   ax.plot([lo,hi],[lo,hi],ls='--',color='gray');ax.set(xlabel='Observed',ylabel='Predicted',xlim=(lo,hi),ylim=(lo,hi));top.set_title(DISPLAY[ds]+' / '+{'phzx':'pH','doz':'Dissolved oxygen','hx':'Potency','ph':'pH','do':'Dissolved oxygen','permittivity':'Permittivity'}[name]);top.axis('off');side.axis('off');
   if c==2:
    legend=ax.legend(loc='upper left',fontsize=10,markerscale=3,framealpha=.9)
    for handle in legend.legend_handles:handle.set_alpha(1.)
 save(fig,'Multi_Dataset_Current_JointPlot')
 # One joint projection per dataset for three matched representation stages.
 from sklearn.decomposition import PCA
 from sklearn.manifold import TSNE
 fig,axs=plt.subplots(2,3,figsize=(9,6),layout='constrained')
 for r,ds in enumerate(DATASETS):
  tr=clean[ds]['trace'];n=len(data[ds]['x']);ix=np.unique(np.linspace(0,n-1,min(300,n)).astype(int));keys=['patch_backbone.encoder','numeric_statistics_features','channel_norm'];arr=[tr[k][ix].reshape(-1,128).numpy() for k in keys];npoints=len(arr[0]);joint=np.concatenate(arr);joint=PCA(n_components=min(50,joint.shape[1]),random_state=2021).fit_transform(joint)
  opts=dict(n_components=2,perplexity=30,init='pca',learning_rate='auto',random_state=2021);opts['max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter']=1000;embedding=TSNE(**opts).fit_transform(joint);labels=np.tile(np.arange(5),len(ix))
  for c,key in enumerate(keys):
   z=embedding[c*npoints:(c+1)*npoints];axs[r,c].scatter(z[:,0],z[:,1],c=labels,cmap='tab10',s=3,alpha=.6);axs[r,c].set(title=DISPLAY[ds]+' / '+['Backbone','Numerical statistics','Process-text interaction'][c],xticks=[],yticks=[],xlim=(embedding[:,0].min()-2,embedding[:,0].max()+2),ylim=(embedding[:,1].min()-2,embedding[:,1].max()+2))
  from matplotlib.lines import Line2D
  axs[r,2].legend(handles=[Line2D([],[],marker='o',ls='',color=plt.get_cmap('tab10')(j/4),label={'hx':'Potency','jn':'Biomass','doz':'DO','phzx':'pH','fi':'Airflow','dco2':'dCO2','transmission':'Transmission','permittivity':'Permittivity','ph':'pH','do':'DO'}[name]) for j,name in enumerate(data[ds]['features'])],fontsize=10,loc='best')
  np.savez_compressed(OUT/ds/'representation_projection.npz',projection=embedding,sample_indices=ix,variable_labels=labels)
 save(fig,'representation_tsne')
 (OUT/'FEATURE_FIGURES_COMPLETE').write_text('Fixed figure recipes applied; all panels retain actual outcomes.\n');print('FEATURE_FIGURES_COMPLETE',flush=True)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['sensitivity','features'],required=True);args=ap.parse_args()
 {'sensitivity':sensitivity,'features':features}[args.stage]()
if __name__=='__main__':main()
