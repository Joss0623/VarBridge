"""Wait for any idle physical4090, bind one GPU for all eight online measurements."""
from pathlib import Path
import subprocess,os,time,fcntl,csv
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'test_results/efficiency8_4090'
OUT=BASE/'online_remeasure';OUT.mkdir(exist_ok=True)
PY='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python'
def snapshot():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 occupied={l.split(',')[0].strip() for l in apps.splitlines() if l.strip()}
 result={}
 for line in raw.splitlines():
  idx,uid,name,free,util=[s.strip() for s in line.split(',')]
  if '4090' in name and uid not in occupied and int(free)>=20000 and int(util)<=5:result[int(idx)]=uid
 return result
def main():
 f=(OUT/'.launcher.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (OUT/'launcher.pid').write_text(str(os.getpid()))
 selected=OUT/'selected_gpu.csv'
 if selected.exists():
  row=next(csv.DictReader(selected.open()));idx=int(row['physical_gpu']);uid=row['uuid']
  print('PRESERVE_SELECTED_GPU',idx,uid,flush=True)
 else:
  print('WAIT_FOR_ANY_EXCLUSIVE_IDLE_4090',flush=True)
  while True:
   a=snapshot()
   if a:
    idx=min(a);uid=a[idx];stable=True
    for _ in range(2):
     time.sleep(3)
     if snapshot().get(idx)!=uid:stable=False;break
    if stable:break
   time.sleep(15)
  with selected.open('w') as file:
   writer=csv.DictWriter(file,fieldnames=['physical_gpu','uuid','selected_at_server']);writer.writeheader();writer.writerow(dict(physical_gpu=idx,uuid=uid,selected_at_server=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
  print('SELECTED_ONE_GPU_FOR_ALL8',idx,uid,flush=True)
 env=os.environ.copy();env.update(EFFICIENCY_GPU=str(idx),PYTHONDONTWRITEBYTECODE='1')
 with (OUT/'pipeline.log').open('a') as log:
  p=subprocess.Popen([PY,'-u',str(ROOT/'scripts/measure_efficiency8_online_anygpu.py')],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  (OUT/'queue.pid').write_text(str(p.pid));print('STARTED_BOUND_GPU_QUEUE',idx,p.pid,flush=True)
  code=p.wait()
 assert code==0,('bound GPU inference queue failed; preserve diagnostics and selected GPU',idx,code)
 assert (OUT/'ALL8_ONLINE_INFERENCE_COMPLETE').exists()
 print('ALL8_REMEASURE_COMPLETE',flush=True)
if __name__=='__main__':main()
