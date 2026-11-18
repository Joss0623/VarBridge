"""Coordinate semantic supplemental fits alongside preserved RoBERTa fine-tunes."""
import os,sys,time,json,csv,subprocess,signal,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'test_results/varbridge_semantic_refinement';F=ROOT/'test_results/varbridge_text_finetune';ROLE=sys.argv[1];PY=sys.executable;TRAIN=ROOT/'scripts/run_varbridge_semantic_refinement.py'
JOBS=[('CHO','llm_static',1),('CHO','llm_repair',1),('CHO','neutral_repair',1)] if ROLE=='4090' else [('EFP_long_sub2','llm_static',0),('EFP_long_sub2','llm_repair',1),('EFP_long_sub2','neutral_repair',0)]
def write(p,z):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(z,indent=2));t.replace(p)
def running():
 out={}
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:
   if d.stat().st_uid!=os.getuid():continue
   q=(d/'cmdline').read_bytes().decode().split('\0');family=next((n for n in ['run_varbridge_semantic_refinement.py','run_varbridge_text_finetune.py'] if any(a.endswith(n) for a in q)),None)
   if family is None or 'train' not in q:continue
   k=q.index('train');task,mode=q[k+1:k+3];env=dict(a.split('=',1) for a in (d/'environ').read_bytes().decode().split('\0') if '=' in a);gpu=int(env['CUDA_VISIBLE_DEVICES']);key=(family,task,mode);assert key not in out,'DUPLICATE_ACTIVE_JOB';out[key]={'pid':int(d.name),'gpu':gpu,'family':family,'task':task,'mode':mode}
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 return out
def resources():
 z=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True);a=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],universal_newlines=True);host=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1024;out={}
 for line in z.splitlines():
  i,u,f,v=[s.strip() for s in line.split(',')];out[int(i)]={'free':int(f),'util':int(v),'pids':{int(l.split(',')[1]) for l in a.splitlines() if l.split(',')[0].strip()==u},'host':host}
 return out
lock=(R/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert (R/'PREFLIGHT_PASS.json').exists();(R/'queue.pid').write_text(str(os.getpid()));initial=running();write(R/'preserved_existing_workers.json',{'role':ROLE,'time':time.time(),'workers':list(initial.values())});print('PRESERVED_EXISTING',list(initial.values()),flush=True);seen={};failed=set();children={};times={};last_launch=0
while True:
 live=running();res=resources();semantic={key:q for key,q in live.items() if key[0]=='run_varbridge_semantic_refinement.py'}
 for key,q in list(seen.items()):
  if key not in live and not (R/(q['task']+'_'+q['mode']+'_COMPLETE.json')).exists() and key not in failed:
   failed.add(key);write(R/(q['task']+'_'+q['mode']+'_FAILURE.json'),{'time':time.time(),'pid':q['pid'],'reason':'Exited without completion; full epoch recovery state retained; no duplicate or blind retry'});print('TECHNICAL_FAILURE',q,flush=True)
 seen.update(semantic)
 for t,m,g in JOBS:
  key=('run_varbridge_semantic_refinement.py',t,m)
  if key in live or key in failed or (R/(t+'_'+m+'_COMPLETE.json')).exists():continue
  q=res[g];allowed={v['pid'] for v in live.values() if v['gpu']==g}
  if q['free']<14000 or q['host']<50000 or q['pids']-allowed or time.time()-last_launch<20:continue
  good=True
  for _ in range(3):
   time.sleep(2);q=resources()[g];allowed={v['pid'] for v in running().values() if v['gpu']==g}
   if q['free']<14000 or q['host']<50000 or q['pids']-allowed:good=False;break
  if not good:continue
  assert key not in running();log=R/'logs'/(t+'_'+m+'.log');assert not log.exists(),'Existing partial log requires reviewed recovery, not fresh restart'
  with log.open('w') as f:p=subprocess.Popen([PY,'-u',str(TRAIN),'train',t,m],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  record={'pid':p.pid,'gpu':g,'family':key[0],'task':t,'mode':m};seen[key]=record;children[p.pid]=p;times[p.pid]=time.time();last_launch=time.time();print('LAUNCH',record,flush=True);break
 live=running();res=resources()
 for g in {q[2] for q in JOBS}:
  own={v['pid'] for v in live.values() if v['gpu']==g};foreign=res[g]['pids']-own;new=[v for v in live.values() if v['gpu']==g and v['family']=='run_varbridge_semantic_refinement.py']
  if new and (foreign or res[g]['free']<4000):
   q=max(new,key=lambda v:times.get(v['pid'],0));print('PRESSURE_STOP_NEW_OWN_ONLY',q,'foreign',list(foreign),'free',res[g]['free'],flush=True);os.killpg(q['pid'],signal.SIGTERM)
 progress=[]
 for family,path in [('run_varbridge_semantic_refinement.py',R),('run_varbridge_text_finetune.py',F)]:
  for q in live.values():
   if q['family']!=family:continue
   stem=q['task']+'_'+q['mode'];p=path/(stem+'_learning.csv');rows=list(csv.DictReader(p.open())) if p.exists() else [];progress.append(dict(q,epochs=len(rows),last_epoch_seconds=float(rows[-1]['seconds']) if rows else None))
 write(R/'combined_progress.json',{'time':time.time(),'role':ROLE,'active':progress,'failed':[list(x) for x in failed]})
 for p in children.values():p.poll()
 finished=all((R/(t+'_'+m+'_COMPLETE.json')).exists() or ('run_varbridge_semantic_refinement.py',t,m) in failed for t,m,g in JOBS)
 if finished and not live:break
 time.sleep(10)
print('ALL_LOCAL_TASKS_FINISHED',ROLE,'failed',len(failed),flush=True)
