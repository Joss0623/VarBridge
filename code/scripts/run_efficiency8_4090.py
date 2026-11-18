"""Current8model study: reuse2 genuine4090 fits; train6 missing100epoch fits only."""
from pathlib import Path
import subprocess,os,time,csv,re,fcntl,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'test_results/efficiency8_4090';PY='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python';MODELS=['DLinear','PatchTST','iTransformer','Crossformer','TimeMixer','TimesNet']
ENV=os.environ.copy();ENV.update(CUDA_VISIBLE_DEVICES='6',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
def gpu():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);idx,uid,mem,util=next([v.strip() for v in l.split(',')] for l in raw.splitlines() if l.split(',')[0].strip()=='6')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True);pids={int(l.split(',')[1]) for l in apps.splitlines() if l.split(',')[0].strip()==uid};return uid,int(mem),int(util),pids
def admit():
 for _ in range(1):
  uid,mem,util,pids=gpu();assert not pids and mem>=20000 and util<=5,('GPU6 not exclusive/idle',mem,util,pids)
 return uid
def job(m,preflight=False):
 admit();name=m+('_preflight' if preflight else '');log=OUT/(name+'.log')
 if log.exists():log.rename(str(log)+'.technical.'+str(time.time_ns()))
 cmd=[PY,'-u','scripts/train_efficiency8_4090.py','--model',m]+(['--preflight-only'] if preflight else [])
 print('START',name,flush=True)
 with log.open('w') as f:
  p=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);(OUT/'worker.pid').write_text(str(p.pid))
  while p.poll() is None:
   uid,mem,util,pids=gpu()
   if pids-{p.pid}:p.terminate();p.wait();raise RuntimeError(('ForeignGPUprocess; invalid timing retained as technical history',m,pids))
   time.sleep(1)
  assert p.returncode==0,('Failed',name,p.returncode)
 s=log.read_text()
 if preflight:assert 'EFF8_PREFLIGHT_COMPLETE '+m in s
 else:assert 'EFFICIENCY100_JOB_COMPLETE '+m in s and re.findall(r'^EFF100_EPOCH (\d+) ',s,re.M)==[str(i) for i in range(1,101)]
 print('COMPLETE',name,flush=True)
def main():
 lock=(OUT/'.queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'coordinator.pid').write_text(str(os.getpid()));assert len(list(csv.DictReader((OUT/'reused_training_sources.csv').open())))==2
 for m in MODELS:
  if not (OUT/(m+'_preflight.log')).exists() or 'EFF8_PREFLIGHT_COMPLETE '+m not in (OUT/(m+'_preflight.log')).read_text():job(m,True)
  log=OUT/(m+'.log')
  if log.exists() and 'EFFICIENCY100_JOB_COMPLETE '+m in log.read_text() and (OUT/(m+'.csv')).exists():print('SKIP_EXISTING_NEW100',m,flush=True);continue
  job(m)
 rows=list(csv.DictReader((OUT/'reused_training_sources.csv').open()))
 for m in MODELS:
  p=OUT/(m+'.csv');row=next(csv.DictReader(p.open()));row.update(source_csv=str(p),source_log=str(OUT/(m+'.log')),origin='native4090_new_singleGPU6_100');rows.append(row)
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/'all_training_sources.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 (OUT/'ALL8_TRAINING_RECORDS_COMPLETE').write_text('2reused4090+6new4090 complete. Await uniform online inference, no training time comparisons for reused fits.\n');print('ALL8_TRAINING_RECORDS_COMPLETE',flush=True)
if __name__=='__main__':main()
