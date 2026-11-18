"""Final source, window, epoch, timing, cache and figure audit. Read-only to adopted assets."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np,pandas as pd,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
OUT=ROOT/'test_results/varbridge_timebase_analysis'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
a=pd.read_csv(OUT/'source_hashes_before.csv');b=pd.read_csv(OUT/'source_hashes_after.csv');assert a.equals(b)
for r in a.itertuples():assert sha(r.path)==r.sha256
print('ORIGINAL_SOURCE_AND_CHECKPOINT_HASHES_UNCHANGED',len(a),flush=True)
for task,n,bn in [('EFP_long_sub2',4431,80),('CHO',381,3)]:
 d=torch.load(OUT/(task+'_fulltest_branches.pt'),map_location='cpu');p=d['candidates'];y=d['target'];w=d['final_weights']
 assert p.shape==(n,24,5,4) and w.shape==(n,24,5,3) and len(set(d['starts']))==n and len(set(d['batch_ids']))==bn
 assert torch.isfinite(p).all() and torch.isfinite(y).all() and torch.isfinite(w).all()
 assert torch.allclose(w.sum(-1),torch.ones(n,24,5),atol=1e-6)
 assert torch.allclose((p[...,:3]*w).sum(-1),p[...,-1],atol=3e-6,rtol=3e-5)
 tab=pd.read_csv(OUT/'candidate_metrics.csv');q=tab[(tab.dataset==task)&(tab.branch=='Final fusion')].iloc[0]
 assert abs(float(((p[...,-1]-y)**2).mean())-q.mse)<1e-6
 print('FULL_TEST_CANDIDATES_RECONSTRUCTION_PASS',task,n,bn,flush=True)
r=pd.read_csv(OUT/'efficiency/results.csv');assert len(r)==8 and r.model.nunique()==8
assert (r.epochs==100).all() and (r.seed==2021).all() and (r.test_windows==381).all() and (r.batch_size==1).all() and (r.cpu_threads==2).all()
assert r.physical_gpu.nunique()==1 and r.precision.nunique()==1 and r.precision.iloc[0]=='float32'
for row in r.itertuples():
 t=pd.read_csv(OUT/'efficiency'/(row.model+'_cached_samples.csv'))
 assert len(t)==160 and not t.duplicated(['repeat','window_index']).any() and t.window_index.nunique()==32 and t.repeat.nunique()==5
 assert np.isfinite(t.latency_ms).all() and (t.latency_ms>0).all() and abs(t.latency_ms.median()-row.latency_median_ms)<1e-9
 assert sha(row.checkpoint)==row.checkpoint_sha256 and row.max_prediction_abs_error<3e-5 and row.profiling_prediction_max_abs_error<3e-5
 if row.model=='TimeLLM':assert 'native prompt input embeddings' in row.cache_kind
assert len({tuple(pd.read_csv(OUT/'efficiency'/(m+'_cached_samples.csv')).query('repeat==0').start) for m in r.model})==1
print('EIGHT_100EPOCH_SOURCES_SAME_GPU_1280_TIMINGS_381WINDOWS_PASS',flush=True)
# Persist raw data provenance and exact retrospective group cutpoints.
from analyze_varbridge_timebase import raw_task
cut=[];sources=[]
for task in ['EFP_long_sub1','EFP_long_sub2','EFP_long_sub3','CHO']:
 raw,fs,ids,mean,std,manifest=raw_task(task);v=[]
 for bid in ids['train']:
  z=(raw.loc[raw.batch_id==bid,fs].to_numpy(float)-mean)/std;v.append(abs(np.diff(z,axis=0)).mean())
 for q,x in zip([.25,.5,.75],np.quantile(v,[.25,.5,.75])):cut.append(dict(dataset=task,training_batch_quantile=q,mean_absolute_hourly_difference=float(x)))
 path=ROOT/'dataset'/task/(task+'.csv');sources.append(dict(path=str(path),sha256=sha(path)))
pd.DataFrame(cut).to_csv(OUT/'training_volatility_cutpoints.csv',index=False);pd.DataFrame(sources).to_csv(OUT/'dataset_source_hashes.csv',index=False)
figs=list((OUT/'figures').glob('*.png'));assert len(figs)==14
for path in figs:
 im=Image.open(path);assert im.width>=1500 and im.height>=700 and min(im.info.get('dpi',(0,0)))>=399
print('FOURTEEN_PNG_METADATA_PASS',flush=True)
# All material here is analysis-only; source file inspection catches accidental optimizer use.
for name in ['analyze_varbridge_timebase.py','measure_varbridge_timebase_efficiency.py','run_varbridge_timebase_analysis.py']:
 text=(ROOT/'scripts'/name).read_text()
 assert '.backward(' not in text and 'optimizer.step(' not in text
print('NO_BACKWARD_OR_OPTIMIZER_UPDATES',flush=True)
(OUT/'FINAL_AUDIT_PASS').write_text('Validated 2026-11-14: unchanged adopted sources/checkpoints;100-epoch checkpoint origins;all4431/381 H24 windows;exact candidate reconstruction;same exclusive GPU0 eight-model160timings;native TimeLLM embedding-cache;14PNG;no training.\n')
print('VARBRIDGE_TIMEBASE_FINAL_AUDIT_PASS',flush=True)
