"""Paper PNG from audited GPU1 online remeasurement only; no training/model edits."""
from pathlib import Path
import csv,math,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FuncFormatter
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'test_results/efficiency8_4090/cached_text_remeasure'
rows=list(csv.DictReader((D/'paper_results.csv').open()))
assert len(rows)==8 and all(r['physical_gpu']=='1' and r['batch_size']=='1' for r in rows)
assert 'FINAL_EFFICIENCY8_CACHED_AUDIT_PASS' in (D/'final_audit.log').read_text()
positions={'DLinear':(0,32,'center'),'PatchTST':(-8,-36,'right'),'iTransformer':(0,27,'center'),'Crossformer':(0,21,'center'),'TimeMixer':(0,27,'center'),'TimesNet':(-16,19,'right'),'TimeLLM':(20,18,'left'),'VarBridge':(14,-3,'left')}
with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'axes.spines.top':False,'axes.spines.right':False}):
 fig,ax=plt.subplots(figsize=(7.1,4.1))
 fig.subplots_adjust(left=.10,right=.98,bottom=.15,top=.96)
 for row in rows:
  m=row['model'];x=float(row['latency_median_ms']);y=float(row['mse']);p=int(row['resident_parameters'])
  c='#C43C39' if m=='VarBridge' else '#7953A1' if m=='TimeLLM' else '#31708E'
  ax.scatter(x,y,s=60*math.log10(p),c=c,alpha=.58,edgecolors=c,linewidths=1.1,zorder=3)
  dx,dy,ha=positions[m];params=f'{p/1e6:.2f}M' if p>=1e6 else f'{p/1e3:.2f}K'
  ax.annotate(('Time-LLM' if m=='TimeLLM' else m)+'\n'+params,(x,y),xytext=(dx,dy),textcoords='offset points',ha=ha,va='center',fontsize=10,color=c,fontweight='bold' if m=='VarBridge' else 'normal',zorder=4)
 ax.set_xscale('log');ax.set_xlim(.27,90);ax.set_ylim(.105,.60)
 ax.xaxis.set_major_locator(FixedLocator([.3,1,3,10,30]));ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
 ax.set_xlabel('Cached-text prediction latency (ms, log scale)')
 ax.set_ylabel('Test MSE')
 ax.grid(axis='y',color='#DDE2E6',linewidth=.6);ax.set_axisbelow(True)
 ax.text(.02,.97,r'Bubble area $\propto\log_{10}$(resident parameters)',transform=ax.transAxes,va='top',fontsize=9,color='#444444')
 fig.savefig(D/'efficiency8_cached_bubble.png',dpi=600,facecolor='white')
 plt.close(fig)
(D/'cached_figure_provenance.txt').write_text('Source: audited paper_results.csv (TimeLLM input-embedding cache explicitly selected by user).\nCPU precomputed text cache; include normalization/cache lookup/H2D/model/D2H; exclude cache generation/load/fixed context. All100epoch checkpoints, GPU1 FP32 batch1 2CPU threads,160timings and381testwindows. X median latency logscale; Y fulltestMSE; area60*log10(resident predictor parameters). VarBridge1.032M predictor excludes offlineRoBERTa; TimeLLM132.474M includes required GPT2. Cache memory is not a parameter count. Original online/KV timings are not used.\nSource SHA256 '+hashlib.sha256((D/'paper_results.csv').read_bytes()).hexdigest()+'\n')
print(D/'efficiency8_cached_bubble.png')
