"""Serial guarded plugin-experiment queue; independent of main/efficiency jobs."""
from pathlib import Path
import os,sys,subprocess,time,fcntl,re,csv
from current_bound_plugins import *
PY='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python';TR='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/torchrun'
lock=open(OUT/'.queue.lock','w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('Current plugin queue already active')
INITIAL=source_hashes();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='4,6',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
def ready():
 gpu={int(a):int(b) for a,b in [line.split(',') for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()]}
 mem=int(next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')).split()[1])//1024
 return min(gpu[4],gpu[6])>=14000 and mem>=50000,gpu,mem
def orphans():
 p=subprocess.run(['pgrep','-u',str(os.getuid()),'-f','scripts/train_current_bound_plugins.py'],capture_output=True,text=True)
 return p.stdout.strip()
def job(name,cmd,marker,checkpoint_path=None):
 log=OUT/(name+'.log')
 if log.exists() and marker in log.read_text(errors='replace'):print('PLUGIN_SKIP_COMPLETE',name,flush=True);return
 for attempt in range(1,4):
  while True:
   ok,gpu,mem=ready();pids=orphans()
   if ok and not pids:break
   print('PLUGIN_WAIT_RESOURCES',name,'free',gpu,'host',mem,'orphan',pids,flush=True);time.sleep(30)
  assert source_hashes()==INITIAL
  if log.exists():log.rename(str(log)+'.technical_failure.'+str(int(time.time())))
  if checkpoint_path and checkpoint_path.exists():checkpoint_path.rename(str(checkpoint_path)+'.technical_partial.'+str(int(time.time())))
  print('PLUGIN_START',name,'ATTEMPT',attempt,'GPU4,6',flush=True)
  with log.open('w') as f:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT);code=p.wait()
  if code==0 and marker in log.read_text(errors='replace'):print('PLUGIN_END_SUCCESS',name,flush=True);return
  print('PLUGIN_TECHNICAL_FAILURE',name,code,flush=True);time.sleep(15)
 raise RuntimeError('Plugin job failed after three technical attempts: '+name)
def ddp(ds,num,text,phase,stage='adapt'):
 name='_'.join([ds,num,text,stage,phase]);cp=(numerical_checkpoint(ds,num) if stage=='pretrain' else checkpoint(ds,num,text)) if phase=='train' else None
 cmd=[TR,'--nproc_per_node=2','--master_port=29721','scripts/train_current_bound_plugins.py','--dataset',ds,'--numeric',num,'--text',text,'--phase',phase,'--stage',stage]
 job(name,cmd,'PLUGIN_JOB_COMPLETE',cp)
print('PLUGIN_QUEUE_PROTOCOL two datasets / H24 / seed2021 / 12 fusion cells / 10 new adaptation fits / matched references',flush=True)
for ds in DATASETS:ddp(ds,'patchtst','roberta','reuse')
for num in NUMERICS:ddp('EFP_long_sub2',num,'roberta','reference')
for num,text in [('timesnet','roberta'),('patchtst','bert'),('patchtst','gpt2'),('timesnet','bert'),('timesnet','gpt2')]:
 ddp('EFP_long_sub2',num,text,'preflight');ddp('EFP_long_sub2',num,text,'train')
for text in ['bert','gpt2']:job('CHO_'+text+'_text',[PY,'-u','scripts/prepare_current_bound_plugin_text.py','--encoder',text],'ALL_PLUGIN_TEXT_COMPLETE')
ddp('CHO','timesnet','roberta','preflight','pretrain');ddp('CHO','timesnet','roberta','train','pretrain')
for num in NUMERICS:ddp('CHO',num,'roberta','reference')
for num,text in [('patchtst','bert'),('patchtst','gpt2'),('timesnet','roberta'),('timesnet','bert'),('timesnet','gpt2')]:
 ddp('CHO',num,text,'preflight');ddp('CHO',num,text,'train')
rows=[]
for log in sorted(OUT.glob('*.log')):
 for line in log.read_text(errors='replace').splitlines():
  if line.startswith(('PLUGIN_RESULT ','PLUGIN_REFERENCE ')):
   d=dict(re.findall(r'(\w+)=([^\s]+)',line));d['kind']='fusion' if line.startswith('PLUGIN_RESULT ') else 'backbone';d['text']=d.get('text','none');d['seed']='2021';d['horizon']='24';d['log']=str(log.relative_to(ROOT));rows.append(d)
assert len([r for r in rows if r['kind']=='fusion'])==12 and len([r for r in rows if r['kind']=='backbone'])==4
assert len({(r['dataset'],r['numeric'],r['text'],r['kind']) for r in rows})==16
assert source_hashes()==INITIAL
keys=['dataset','numeric','text','kind','seed','horizon','mse','mae','windows','epochs','reused','best_epoch','checkpoint','log']
with (OUT/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
(OUT/'ALL_CURRENT_BOUND_PLUGINS_COMPLETE').write_text('12 fusion results, 4 matched frozen-backbone references; seed2021 H24; current architecture unchanged.\n')
print('ALL_CURRENT_BOUND_PLUGINS_COMPLETE',flush=True)
