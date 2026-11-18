"""Sequential full-six-GPU queue with flock, memory gate, and technical retries."""
from pathlib import Path
import fcntl,os,re,subprocess,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'test_results/efp_bound_mechanism';OUT.mkdir(parents=True,exist_ok=True)
lock=open(OUT/'.queue.lock','w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('EFP bound queue already active')
PY='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python';TR='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/torchrun'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2,3,4,5,6,7',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
def memory_ready():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines();free={int(a):int(b) for a,b in [s.split(',') for s in rows]}
 return all(free[i]>=14000 for i in range(2,8))
def run(name,cmd,marker):
 log=OUT/(name+'.log')
 if log.exists() and marker in log.read_text(errors='replace'):print('SKIP',name,flush=True);return
 for attempt in range(1,4):
  while not memory_ready():print('WAIT_MEMORY_ALL6',name,flush=True);time.sleep(60)
  if log.exists():log.rename(str(log)+f'.failed.{int(time.time())}')
  print('START_DDP6',name,'ATTEMPT',attempt,'GPUS2-7',flush=True)
  with open(log,'w') as handle:
   proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT);code=proc.wait()
  if code==0 and marker in log.read_text(errors='replace'):print('END_SUCCESS',name,flush=True);return
  print('TECHNICAL_FAILURE',name,'EXIT',code,flush=True);time.sleep(20)
 raise RuntimeError('EFP bound job failed after three technical retries: '+name)

launch=[TR,'--master_port=29683','--nproc_per_node=6']
run('preflight',launch+['scripts/check_efp_bound_mechanisms.py'],'ALL_EFP_BOUND_PREFLIGHT_COMPLETE')
for dataset in ['EFP_long_sub1','EFP_long_sub2','EFP_long_sub3']:
 for seed in [2021,2022,2023]:
  for h in [6,12,24]:
   name=f'{dataset}_h{h}_seed{seed}_correct'
   run(name,launch+['scripts/train_efp_bound_mechanism_ddp.py','--dataset',dataset,'--model','Fusion','--horizon',str(h),'--seed',str(seed),'--variant','correct'],'EFP_BOUND_JOB_COMPLETE')
print('ALL_EFP_BOUND_MECHANISM_COMPLETE',flush=True)
