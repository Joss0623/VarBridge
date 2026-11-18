"""Unified 24-variant EFP/CHO ablations. Fixed settings, resumable, six-rank sequential queue."""
from pathlib import Path
import fcntl,os,re,subprocess,time
from common_bound_ablations import ROOT,VARIANTS
OUT=ROOT/'test_results/common_bound_ablations';OUT.mkdir(parents=True,exist_ok=True)
PY='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python';TR='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/torchrun'
lock=open(OUT/'.queue.lock','w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit(0)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2,3,4,5,6,7',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
def memory_ready():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
 free={int(a):int(b) for a,b in [s.split(',') for s in rows]}
 live_names={'prepare_common_bound_ablation_text.py','check_common_bound_ablations.py','train_common_bound_ablation_ddp.py'}
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if proc.stat().st_uid!=os.getuid():continue
   args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\x00')
   if args and 'python' in Path(args[0]).name and any(Path(a).name in live_names for a in args[1:]):return False
  except (OSError,ProcessLookupError):pass
 return all(free[i]>=14000 for i in range(2,8))
def complete(log,marker,training=False):
 if not log.exists():return False
 text=log.read_text(errors='replace')
 if marker not in text:return False
 if not training:return True
 epochs=re.findall(r'^Epoch: (\d+) ',text,re.M)
 ranks=set(re.findall(r'COMMON_ABLATION_DDP_RANK (\d+) ',text))
 results=re.findall(r'^COMMON_ABLATION_RESULT (.+)$',text,re.M)
 return epochs==[str(i) for i in range(1,101)] and ranks==set('012345') and len(results)==(2 if '_h24_' in log.name else 1) and all('completed_epochs=100' in x for x in results)
def compatible(log,dataset,h):
 text=log.read_text(errors='replace');epochs=re.findall(r'^Epoch: (\d+) ',text,re.M)
 ranks=set(re.findall(r'(?:CHO|EFP)_DDP_RANK (\d+) ',text))
 marker='EFP_BOUND_JOB_COMPLETE' if dataset!='CHO' else 'CHO_BOUND_JOB_COMPLETE'
 assert marker in text and epochs==[str(i) for i in range(1,101)] and ranks==set('012345'),str(log)
 rows=re.findall(r'^(?:EFP_BOUND_RESULT|CHO_RESULT) (.+)$',text,re.M)
 assert len(rows)==(2 if h==24 else 1) and all('completed_epochs=100' in x for x in rows),str(log)
def link(name,source):
 target=OUT/(name+'.log');assert source.exists(),source
 if target.is_symlink():assert target.resolve()==source.resolve()
 elif target.exists():raise RuntimeError('Refusing to overwrite '+str(target))
 else:target.symlink_to(source)
def run(name,cmd,marker,training=False):
 log=OUT/(name+'.log')
 if complete(log,marker,training):print('SKIP_COMPLETE',name,flush=True);return
 for attempt in range(1,4):
  while not memory_ready():print('WAIT_MEMORY_ALL6',name,flush=True);time.sleep(60)
  if complete(log,marker,training):print('SKIP_RECOVERED',name,flush=True);return
  if log.exists():log.rename(str(log)+'.failed.'+str(time.time_ns()))
  print('START_DDP6',name,'ATTEMPT',attempt,'GPUS2-7',flush=True)
  with open(log,'w') as f:
   proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
   print('LIVE_CHILD',proc.pid,name,flush=True);code=proc.wait()
  if code==0 and complete(log,marker,training):print('END_SUCCESS',name,flush=True);return
  print('TECHNICAL_FAILURE',name,'EXIT',code,flush=True);time.sleep(20)
 raise RuntimeError('Three technical failures: '+name)
launch=[TR,'--master_port=29687','--nproc_per_node=6']
# Reuse only logs from the current architecture, exact batches and fixed recipe.
for dataset in ['CHO','EFP_long_sub1','EFP_long_sub2','EFP_long_sub3']:
 for seed in [2021,2022,2023]:
  for h in [6,12,24]:
   source=(ROOT/'test_results/cho_bound_mechanism'/f'Fusion_h{h}_seed{seed}_correct.log' if dataset=='CHO' else ROOT/'test_results/efp_bound_mechanism'/f'{dataset}_h{h}_seed{seed}_correct.log')
   compatible(source,dataset,h);link(f'{dataset}_h{h}_seed{seed}_full',source)
for seed in [2021,2022,2023]:
 for h in [6,12,24]:
  source=ROOT/'test_results/cho_bound_mechanism'/f'Fusion_h{h}_seed{seed}_unrelated.log'
  compatible(source,'CHO',h);link(f'CHO_h{h}_seed{seed}_random_semantic',source)
print('COMPATIBLE_REUSE 36 full fits plus9 CHO unrelated fits=45 training logs /60 testcells',flush=True)
run('text_preparation',launch+['scripts/prepare_common_bound_ablation_text.py'],'ALL_COMMON_ABLATION_TEXT_COMPLETE')
run('preflight',launch+['scripts/check_common_bound_ablations.py'],'ALL_COMMON_BOUND_ABLATION_PREFLIGHT_COMPLETE')
for variant in VARIANTS:
 for dataset in ['CHO','EFP_long_sub1','EFP_long_sub2','EFP_long_sub3']:
  for seed in [2021,2022,2023]:
   for h in [6,12,24]:
    name=f'{dataset}_h{h}_seed{seed}_{variant}'
    if dataset=='CHO' and variant=='random_semantic':
     compatible(OUT/(name+'.log'),dataset,h);print('SKIP_COMPATIBLE',name,flush=True);continue
    run(name,launch+['scripts/train_common_bound_ablation_ddp.py','--dataset',dataset,'--model','Fusion','--horizon',str(h),'--seed',str(seed),'--variant',variant],'COMMON_ABLATION_JOB_COMPLETE',True)
subprocess.run([PY,'scripts/summarize_common_bound_ablations.py','--summary-only','--report'],cwd=ROOT,env=env,check=True)
print('ALL_COMMON_BOUND_ABLATIONS_COMPLETE',flush=True)
