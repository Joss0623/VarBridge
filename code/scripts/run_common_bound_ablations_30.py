"""Isolated fixed30-epoch queue. Original100 results and existing live jobs remain untouched."""
import os,sys,time,json,subprocess,fcntl,threading,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from summarize_common_bound_ablations_30 import guard,record,collect,R
PY='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python';TR='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/torchrun'
def ready():
 try:
  q=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True);free={int(a):int(b) for a,b in [l.split(',') for l in q.splitlines()]}
  host=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1024
  return min(free[i] for i in range(2,8))>=14000 and host>=50000
 except (ValueError,subprocess.CalledProcessError):return False
def live(j=None):
 out=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   a=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
   if not any(x.endswith('/train_common_bound_ablation_30_ddp.py') or x=='scripts/train_common_bound_ablation_30_ddp.py' for x in a):continue
   if j is None or all(k in a and a[a.index(k)+1]==str(v) for k,v in [('--dataset',j['dataset']),('--horizon',j['horizon']),('--variant',j['variant'])]):out.append(int(p.name))
  except (OSError,ValueError,IndexError):pass
 return out
lock=(R/'queue.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('QUEUE_ALREADY_LIVE')
guard();(R/'queue.pid').write_text(str(os.getpid()));plan=json.loads((R/'plan.json').read_text());startlock=threading.Lock()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3,4,5,6,7',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
while live():print('WAIT_EXISTING_BUDGET30_WORKER',live(),flush=True);time.sleep(30)
def run(j):
 if record(j):return
 for attempt in range(1,4):
  while True:
   if record(j):return
   if ready():break
   time.sleep(30)
  with startlock:
   if record(j):return
   assert not live(j),'DUPLICATE_JOB'
   while not ready():time.sleep(30)
   guard();log=R/(j['stem']+'.log')
   if log.exists():log.rename(str(log)+'.technical_history.'+str(time.time_ns()))
   slot=int(threading.current_thread().name.rsplit('_',1)[-1]);cmd=[TR,'--master_port='+str(29771+slot),'--nproc_per_node=6','scripts/train_common_bound_ablation_30_ddp.py','--dataset',j['dataset'],'--model','Fusion','--horizon',str(j['horizon']),'--seed','2021','--variant',j['variant'],'--epochs','30']
   f=log.open('w');p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print('START',j['stem'],'PID',p.pid,'ATTEMPT',attempt,flush=True);time.sleep(5)
  rc=p.wait();f.close()
  if rc==0 and record(j):print('DONE',j['stem'],flush=True);return
  print('TECHNICAL_FAILURE',j['stem'],rc,flush=True);time.sleep(20)
 raise RuntimeError('Three technical failures: '+j['stem'])
print('BUDGET30_QUEUE',len(plan['jobs']),'REUSE',plan['reused'],'NEW',plan['new_fits'],'MAX_CONCURRENT',2,'CURRENT_ADMISSION',ready(),flush=True)
with ThreadPoolExecutor(max_workers=2,thread_name_prefix='budget30') as pool:
 tasks=[pool.submit(run,j) for j in plan['jobs'] if not j['reuse_prefix_checkpoint']]
 for f in tasks:f.result()
d=collect(False);(R/'ALL_BUDGET30_COMPLETE.json').write_text(json.dumps({k:v for k,v in d.items() if k not in ['raw','summary','validation']},indent=2))
print('ALL_COMMON_BOUND_ABLATIONS_30_COMPLETE',flush=True)
