"""Share only our own GPU1 RoBERTa fits; never change the immutable trainer."""
import os,sys,time,json,subprocess,fcntl,signal,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'test_results/varbridge_text_finetune';PY='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python';TRAIN=ROOT/'scripts/run_varbridge_text_finetune.py';TASKS=['CHO'];MODES=['layernorm','layernorm_position'];GPU=1

def write(p,x):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def alive(p):
 try:
  s=Path('/proc/%d/stat'%p).read_text();return s.split(') ',1)[1][0]!='Z'
 except FileNotFoundError:return False
def running():
 result={}
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:
   if d.stat().st_uid!=os.getuid():continue
   z=(d/'cmdline').read_bytes().decode().split('\0')
   if 'train' not in z or not any(a.endswith('run_varbridge_text_finetune.py') for a in z):continue
   k=z.index('train');task,mode=z[k+1:k+3];env=dict(a.split('=',1) for a in (d/'environ').read_bytes().decode().split('\0') if '=' in a);assert env.get('CUDA_VISIBLE_DEVICES')=='1',env.get('CUDA_VISIBLE_DEVICES');assert (task,mode) not in result,'DUPLICATE FIT';result[task,mode]=int(d.name)
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 return result
def resource():
 z=subprocess.check_output(['nvidia-smi','-i',str(GPU),'--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True).strip().split(',');uuid=z[0].strip();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],universal_newlines=True);pids={int(l.split(',')[1]) for l in apps.splitlines() if l.split(',')[0].strip()==uuid};host=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1024
 return int(z[1]),host,pids
lock=(R/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(R/'takeover_record.json',{'time':time.time(),'old_queue':1402547,'preserved_worker':1484030,'reason':'User authorized concurrent own fits; GPU1 only, no sharing foreign jobs','trainer_unchanged':True});(R/'queue.pid').write_text(str(os.getpid()));(R/'accelerator.pid').write_text(str(os.getpid()));seen=running();launch_times={p:0 for p in seen.values()};children={};failed=set();last_launch=0
print('ADOPTED',seen,flush=True)
while True:
 live=running()
 for job,p in list(seen.items()):
  if job not in live and not (R/('%s_%s_COMPLETE.json'%job)).exists():
   failed.add(job);write(R/('%s_%s_FAILURE.json'%job),{'pid':p,'time':time.time(),'reason':'Worker exited without completion; preserve log/checkpoint; no automatic re-fit.'});print('FAILED',job,p,flush=True)
 seen.update(live)
 done={(t,m) for t in TASKS for m in MODES if (R/('%s_%s_COMPLETE.json'%(t,m))).exists()};pending=[(t,m) for t in TASKS for m in MODES if (t,m) not in live and (t,m) not in done and (t,m) not in failed]
 free,host,pids=resource();own=set(live.values());foreign=pids-own
 if foreign or free<4000:
  if live:
   newest=max(live.values(),key=lambda p:launch_times.get(p,0));print('RESOURCE_PRESSURE_STOP_OWN_ONLY',newest,'foreign',list(foreign),'free',free,flush=True);os.killpg(newest,signal.SIGTERM)
 elif pending and len(live)<4 and free>=12000 and host>=50000 and time.time()-last_launch>=30:
  good=True
  for _ in range(3):
   f,h,ps=resource()
   if f<12000 or h<50000 or ps-set(running().values()):good=False;break
   time.sleep(2)
  if good:
   t,m=pending[0];assert not (R/('%s_%s_COMPLETE.json'%(t,m))).exists();assert (t,m) not in running();log=R/'logs'/('%s_%s.log'%(t,m));assert not log.exists(),'Preserve partial run: manual recovery required'
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
   with log.open('w') as f:p=subprocess.Popen([PY,'-u',str(TRAIN),'train',t,m],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   seen[t,m]=p.pid;children[p.pid]=p;launch_times[p.pid]=time.time();last_launch=time.time();print('LAUNCH',t,m,'GPU1',p.pid,flush=True)
 rows=[]
 for t in TASKS:
  for m in MODES:
   path=R/('%s_%s_learning.csv'%(t,m));q=list(csv.DictReader(path.open())) if path.exists() else [];rows.append({'task':t,'mode':m,'epochs':len(q),'pid':running().get((t,m)),'complete':(t,m) in done,'recent_epoch_seconds':sum(float(a['seconds']) for a in q[-3:])/len(q[-3:]) if q else None})
 write(R/'concurrent_progress.json',rows)
 for p in list(children.values()):p.poll()
 if not live and not pending:break
 time.sleep(10)
print('ALL_TEXT_FINETUNE_COMPLETE' if not failed else 'TECHNICAL_RECOVERY_REQUIRED',flush=True)
