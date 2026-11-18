"""Two independent single4090 fits, each exact global126 with six logical21-sample streams."""
import os,sys,time,json,subprocess,fcntl,threading,queue,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from summarize_common_bound_ablations_30 import guard,record,collect,R
PY='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python';TR='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/torchrun'
GPUS=[0,3,2,4,5,6,7];lock=(R/'queue.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('QUEUE_ALREADY_LIVE')
guard();(R/'queue.pid').write_text(str(os.getpid()));plan=json.loads((R/'plan.json').read_text());todo=queue.Queue();mutex=threading.Lock();active={};failures=[]
def live(j=None):
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   a=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
   if not any(Path(x).name in ['train_common_bound_ablation_30_ddp.py','train_common_bound_ablation_30_single.py'] for x in a if x):continue
   if j is None or all(k in a and a[a.index(k)+1]==str(v) for k,v in [('--dataset',j['dataset']),('--horizon',j['horizon']),('--variant',j['variant'])]):found.append(int(p.name))
  except (OSError,IndexError,ValueError):pass
 return found
def gpu_state(g):
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 line=next(l for l in rows.splitlines() if int(l.split(',')[0])==g);_,uuid,free,util=[x.strip() for x in line.split(',')]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True);pids=[]
 for l in apps.splitlines():
  a,b=[x.strip() for x in l.split(',')]
  if a!=uuid:continue
  p=Path('/proc')/b
  try:
   state=(p/'stat').read_text().rsplit(')',1)[1].split()[0]
   if state not in ['Z','X']:pids.append(int(b))
  except (FileNotFoundError,ProcessLookupError):pass
 return dict(gpu=g,uuid=uuid,free=int(free),util=int(util),pids=pids)
def admission(g):
 try:
  host=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1024;q=gpu_state(g)
  return q['free']>=20000 and host>=50000 and (g in (0,3) or (not q['pids'] and q['util']<=5))
 except (ValueError,subprocess.CalledProcessError):return False
def status():
 with mutex:
  z={'time':time.time(),'actual_gpu_workers':GPUS,'global_batch':126,'actual_ranks_per_fit':1,'logical_shards':6,'reused':plan['reused'],'pending_queue':todo.qsize(),'active':dict(active),'failures':list(failures)}
 tmp=R/'queue_progress.tmp';tmp.write_text(json.dumps(z,indent=2));tmp.replace(R/'queue_progress.json')
def worker(g):
 while True:
  while not admission(g):
   if todo.empty():return
   time.sleep(5)
  try:j=todo.get_nowait()
  except queue.Empty:return
  if record(j):todo.task_done();continue
  success=False
  for attempt in range(1,4):
   while not admission(g):time.sleep(15)
   ok=True
   for _ in range(3):
    time.sleep(2)
    if not admission(g):ok=False;break
   while not ok:
    time.sleep(15)
    ok=all(admission(g) and not time.sleep(2) for _ in range(3))
   guard();assert not live(j),'DUPLICATE_TRAINING'
   log=R/(j['stem']+'.log')
   if log.exists():log.rename(str(log)+'.technical_history.'+str(time.time_ns()))
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
   cmd=[TR,'--master_port='+str(29820+g),'--nproc_per_node=1','scripts/train_common_bound_ablation_30_single.py','--dataset',j['dataset'],'--model','Fusion','--horizon',str(j['horizon']),'--seed','2021','--variant',j['variant'],'--epochs','30']
   f=log.open('w');p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   with mutex:active[str(g)]=dict(stem=j['stem'],pid=p.pid,attempt=attempt)
   print('SINGLE_START',g,j['stem'],'PID',p.pid,flush=True)
   rc=p.wait();f.close()
   if rc==0 and record(j):success=True;print('SINGLE_DONE',g,j['stem'],flush=True);break
   print('TECHNICAL_FAILURE',g,j['stem'],rc,flush=True);time.sleep(15)
  with mutex:
   active.pop(str(g),None)
   if not success:failures.append(j['stem'])
  todo.task_done()
  if not success:return
while live():time.sleep(5)
for j in plan['jobs']:
 if not record(j):todo.put(j)
print('SINGLE_QUEUE_START',dict(gpus=GPUS,pending=todo.qsize(),global_batch=126,microbatch_max=21,actual_ranks=1),flush=True)
threads=[threading.Thread(target=worker,args=(g,),name='single_gpu_'+str(g)) for g in GPUS]
for t in threads:t.start()
while any(t.is_alive() for t in threads):status();time.sleep(10)
for t in threads:t.join()
status()
if failures:raise RuntimeError('Technical recovery required: '+str(failures))
d=collect(False);(R/'ALL_BUDGET30_COMPLETE.json').write_text(json.dumps({k:v for k,v in d.items() if k not in ['raw','summary','validation']},indent=2))
print('ALL_COMMON_BOUND_ABLATIONS_30_COMPLETE',flush=True)
