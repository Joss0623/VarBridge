"""Resource-gated, resumable supplementary queue; never starts duplicate workers."""
import os,re,sys,time,fcntl,subprocess,threading,csv,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from common_bound_supplement import ROOT,DATASETS,TASKS,DEFAULTS,main_log,main_checkpoint
OUT=ROOT/'test_results/current_bound_supplement';OUT.mkdir(exist_ok=True,parents=True)
PY='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python';TR=str(Path(PY).with_name('torchrun'))
ENV=os.environ.copy();ENV.update(CUDA_VISIBLE_DEVICES='2,3,4,5,6,7',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
START=threading.Lock()
SUMMARY=threading.Lock()
def ready():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
 free={int(a):int(b) for a,b in (line.split(',') for line in rows)}
 host=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1024
 return min(free[i] for i in range(2,8))>=14000 and host>=50000,dict(min_gpu_free_MiB=min(free[i] for i in range(2,8)),host_available_MiB=round(host))
def adopted_unchanged():
 text=(OUT/'protocol.md').read_text()
 for name,digest in re.findall(r'^- ((?:models|scripts)/[^:]+): ([0-9a-f]{64})$',text,re.M):
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,('adopted source changed',name)
def live_workers():
 names={'train_current_bound_supplement_ddp.py','check_current_bound_supplement.py','analyze_current_bound_supplement.py','prepare_current_supplement_text.py'}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid!=os.getuid():continue
   args=(p/'cmdline').read_bytes().decode(errors='replace').split('\x00')
   if args and any(Path(a).name in names for a in args[1:]) and '--distributed' in args or args and any(Path(a).name=='train_current_bound_supplement_ddp.py' for a in args[1:]):return True
  except OSError:pass
 return False
def complete(path):
 if not path.exists():return False
 t=path.read_text(errors='replace')
 ep=re.findall(r'^Epoch: (\d+) ',t,re.M);ranks=set(re.findall(r'CURRENT_SUPPLEMENT_DDP_RANK (\d+)',t))
 results=re.findall(r'^CURRENT_SUPPLEMENT_RESULT (.+)$',t,re.M)
 return 'CURRENT_SUPPLEMENT_JOB_COMPLETE' in t and ep==[str(i) for i in range(1,101)] and ranks==set('012345') and len(results)==1 and 'completed_epochs=100' in results[0] and all(float(re.search(k+r'=([^ ]+)',results[0])[1])>=0 for k in ['mse','mae'])
def wait_ready(name):
 previous=None;last=0
 while True:
  ok,state=ready()
  if ok:return
  signature=(state['min_gpu_free_MiB']//2000,state['host_available_MiB']//10000)
  if signature!=previous or time.time()-last>=3600:print('WAIT_RESOURCE',name,state,flush=True);previous=signature;last=time.time()
  time.sleep(60)
def run(ds,variant,slot):
 path=OUT/f'{ds}_h24_seed2021_{variant}.log'
 if complete(path):print('SKIP_COMPLETE',path.name,flush=True);return
 for attempt in range(1,4):
  with START:
   wait_ready(path.name);adopted_unchanged()
   if complete(path):return
   if path.exists():path.rename(str(path)+'.failed.'+str(time.time_ns()))
   cmd=[TR,'--master_port='+str(29701+slot),'--nproc_per_node=6','scripts/train_current_bound_supplement_ddp.py','--dataset',ds,'--model','Fusion','--horizon','24','--seed','2021','--variant',variant]
   f=open(path,'w');proc=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
   print('START_DDP6',ds,variant,'PID',proc.pid,'ATTEMPT',attempt,flush=True);time.sleep(8)
  code=proc.wait();f.close()
  if code==0 and complete(path):print('COMPLETE_FIT',ds,variant,flush=True);return
  print('TECHNICAL_FAILURE',ds,variant,'EXIT',code,flush=True);time.sleep(30)
 raise RuntimeError('Three technical failures '+str(path))
def summarize_unlocked():
 rows=[]
 for ds in DATASETS:
  t=main_log(ds).read_text();rank=set(re.findall(r'(?:EFP|CHO)_DDP_RANK (\d+)',t));epochs=re.findall(r'^Epoch: (\d+) ',t,re.M)
  assert rank==set('012345') and epochs==[str(i) for i in range(1,101)]
  candidates=re.findall(r'^(?:EFP_BOUND_RESULT|CHO_RESULT) (.+)$',t,re.M)
  row=next(x for x in candidates if ' h=24 ' in ' '+x+' ' and 'completed_epochs=100' in x)
  rows.append(dict(dataset=ds,variant='full',seed=2021,horizon=24,alpha=.3,fusion_heads=4,expert_width=128,reused=True,mse=float(re.search(r'mse=([^ ]+)',row)[1]),mae=float(re.search(r'mae=([^ ]+)',row)[1]),log=str(main_log(ds)),checkpoint=str(main_checkpoint(ds))))
  for v,changes in TASKS.items():
   path=OUT/f'{ds}_h24_seed2021_{v}.log'
   if not complete(path):continue
   row=re.search(r'^CURRENT_SUPPLEMENT_RESULT (.+)$',path.read_text(),re.M)[1];params=dict(DEFAULTS);params.update(changes)
   rows.append(dict(dataset=ds,variant=v,seed=2021,horizon=24,**params,reused=False,mse=float(re.search(r'mse=([^ ]+)',row)[1]),mae=float(re.search(r'mae=([^ ]+)',row)[1]),log=str(path),checkpoint=str(ROOT/'checkpoints/current_bound_supplement'/f'{ds}_h24_seed2021_{v}.pth')))
 tmp=OUT/'sensitivity_results.csv.tmp'
 with tmp.open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 tmp.replace(OUT/'sensitivity_results.csv');print('SUMMARY',len(rows),'of24 unique training records',flush=True)
 return len(rows)
def summarize():
 with SUMMARY:return summarize_unlocked()
def main():
 lock=open(OUT/'.queue.lock','w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:print('QUEUE_ALREADY_LIVE',flush=True);return
 adopted_unchanged();summarize()
 while live_workers():print('WAIT_EXISTING_WORKERS',flush=True);time.sleep(60)
 cpu=OUT/'preflight_cpu.log';assert cpu.exists() and 'ALL_CURRENT_SUPPLEMENT_CPU_PREFLIGHT_COMPLETE' in cpu.read_text()
 pre=OUT/'preflight_ddp6.log'
 if not pre.exists() or 'ALL_CURRENT_SUPPLEMENT_DDP6_PREFLIGHT_COMPLETE' not in pre.read_text():
  wait_ready('six_rank_preflight')
  with pre.open('w') as f:subprocess.run([TR,'--master_port=29700','--nproc_per_node=6','scripts/check_current_bound_supplement.py','--distributed'],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
  assert len(set(re.findall(r'SUPPLEMENT_PREFLIGHT_RANK (\d+)',pre.read_text())))==6
 tasks=[(ds,v) for v in TASKS for ds in DATASETS]
 # Slots are sequential within each worker, guaranteeing unique rendezvous ports.
 def work(slot):
  for ds,v in tasks[slot::2]:run(ds,v,slot);summarize()
 with ThreadPoolExecutor(max_workers=2) as pool:
  for f in [pool.submit(work,slot) for slot in range(2)]:f.result()
 adopted_unchanged();assert summarize()==24
 print('ALL_CURRENT_BOUND_SENSITIVITY_COMPLETE',flush=True)
if __name__=='__main__':main()
