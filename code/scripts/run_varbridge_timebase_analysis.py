"""Exclusive-GPU checkpoint-only analysis coordinator. Never starts training."""
import os,sys,time,csv,subprocess,hashlib,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'test_results/varbridge_timebase_analysis';OUT.mkdir(exist_ok=True)
PY='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python';MODELS=['DLinear','PatchTST','iTransformer','Crossformer','TimeMixer','TimesNet','TimeLLM','VarBridge']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def csvwrite(p,rows):
 with p.with_suffix('.tmp').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 p.with_suffix('.tmp').replace(p)
def resources():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 g=[dict(zip(['index','uuid','free','util'],[x.strip() for x in line.split(',')])) for line in raw.splitlines()]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True);pp={x['uuid']:set() for x in g}
 for l in apps.splitlines():
  z=l.split(',')
  if len(z)==2:pp[z[0].strip()].add(int(z[1]))
 return g,pp
lock=(OUT/'.analysis.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'coordinator.pid').write_text(str(os.getpid()))
tracked=[ROOT/'models/KIBR_PatchTST_BoundMechanism.py',ROOT/'models/KIBR_PatchTST_BoundAblation.py',ROOT/'scripts/train_efp_bound_mechanism_ddp.py',ROOT/'scripts/train_cho_bound_mechanism_ddp.py',ROOT/'scripts/common_bound_supplement.py']
training=list(csv.DictReader((ROOT/'test_results/efficiency8_4090/all_training_sources.csv').open()))
tracked += [Path(r['checkpoint']) for r in training];tracked += [ROOT/'checkpoints/efp_bound_mechanism/EFP_long_sub2_h24_seed2021_correct.pth']
before={str(p):sha(p) for p in tracked};csvwrite(OUT/'source_hashes_before.csv',[dict(path=p,sha256=h) for p,h in before.items()])
env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
def job(name,cmd,gpu=None):
 log=OUT/(name+'.log')
 if (OUT/(name+'.done')).exists():print('REUSE_COMPLETE',name,flush=True);return
 if log.exists():log.rename(str(log)+'.technical.'+str(time.time_ns()))
 e=env.copy()
 if gpu is not None:e.update(CUDA_VISIBLE_DEVICES=gpu['index'],EFFICIENCY_GPU=gpu['index'])
 else:e['CUDA_VISIBLE_DEVICES']=''
 with log.open('w') as f:
  proc=subprocess.Popen([PY,'-u']+cmd,cwd=ROOT,env=e,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  (OUT/'worker.pid').write_text(str(proc.pid));print('START',name,'pid',proc.pid,flush=True)
  while proc.poll() is None:
   if gpu:
    _,pp=resources()
    if pp[gpu['uuid']]-{proc.pid}:proc.terminate();proc.wait();raise RuntimeError('GPU contention: own analysis stopped, other users untouched')
   time.sleep(2)
  assert proc.returncode==0,(name,proc.returncode)
 (OUT/(name+'.done')).write_text('completed read-only analysis\n');print('COMPLETE',name,flush=True)
job('data_characteristics',['scripts/analyze_varbridge_timebase.py','--part','data'])
selected_path=OUT/'selected_gpu.csv'
if selected_path.exists():chosen=next(csv.DictReader(selected_path.open()))
else:
 chosen=None
 for attempt in range(3):
  gg,pp=resources();idle=[g for g in gg if not pp[g['uuid']] and int(g['free'])>=20000 and int(g['util'])<=5]
  if not idle:raise RuntimeError('No exclusive idle GPU; stop safely without scheduling training')
  if chosen is None:chosen=idle[0]
  assert any(g['uuid']==chosen['uuid'] for g in idle),'Selected GPU became busy';time.sleep(2)
 csvwrite(selected_path,[chosen])
gg,pp=resources();assert not pp[chosen['uuid']]
print('FIXED_GPU',chosen,flush=True)
job('branch_batch',['scripts/analyze_varbridge_timebase.py','--part','checkpoint'],chosen)
env['EFFICIENCY_GPU']=chosen['index'];env['CUDA_VISIBLE_DEVICES']=chosen['index']
subprocess.run([PY,'-c','from measure_efficiency8_online_anygpu import audit_training; audit_training()'],cwd=ROOT,env={**env,'PYTHONPATH':str(ROOT/'scripts')},check=True)
(OUT/'efficiency').mkdir(exist_ok=True)
for m in MODELS:job('efficiency_'+m,['scripts/measure_varbridge_timebase_efficiency.py','--model',m],chosen)
rows=[next(csv.DictReader((OUT/'efficiency'/(m+'_cached.csv')).open())) for m in MODELS]
assert len({r['physical_gpu'] for r in rows})==1
assert all(int(r['epochs'])==100 and int(r['test_windows'])==381 and int(r['samples'])==160 for r in rows)
csvwrite(OUT/'efficiency/results.csv',rows)
assert before=={p:sha(p) for p in before}
csvwrite(OUT/'source_hashes_after.csv',[dict(path=p,sha256=h) for p,h in before.items()])
(OUT/'ALL_CHECKPOINT_ANALYSES_COMPLETE').write_text('100-epoch origins; full H24 tests; same-GPU eight-model cached efficiency; original source/checkpoint hashes unchanged; no training\n');print('ALL_CHECKPOINT_ANALYSES_COMPLETE',flush=True)
