"""After LN completion, migrate surviving position run to the same idle4090."""
import os,time,json,subprocess,shlex,hashlib,signal
from pathlib import Path
ROOT=Path('/home/home_new/qsmx/pycodes/GreyFerm');OLD='/home/qsmx/VarBridge_supplement/pycodes/GreyFerm';PY='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python';KEY='/home/home_new/qsmx/.ssh/varbridge_3090_sync';SSH=['ssh','-o','BatchMode=yes','-i',KEY,'-p','51700','qsmx@10.15.198.98'];R=ROOT/'test_results/varbridge_text_finetune';D=R/'acceleration_diagnostics';stem='EFP_long_sub2_layernorm_position';env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
def remote(code):return subprocess.check_output(SSH+['python3 -c '+shlex.quote(code)],text=True,timeout=40)
def copy(rel):
 dst=ROOT/rel;dst.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['rsync','-a','-e','ssh -i '+KEY+' -p 51700','qsmx@10.15.198.98:'+OLD+'/'+rel,str(dst)],check=True)
 src=subprocess.check_output(SSH+['sha256sum '+OLD+'/'+rel],text=True).split()[0];assert hashlib.sha256(dst.read_bytes()).hexdigest()==src
while not (R/'EFP_long_sub2_layernorm_COMPLETE.json').exists():time.sleep(60)
# Wait until completion worker and any foreign context leave GPU1; never displace them.
uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
while True:
 rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 if not any(l.split(',')[0].strip()==uuid for l in rows.splitlines()):break
 time.sleep(30)
if remote("from pathlib import Path;print((Path(%r)/'test_results/varbridge_text_finetune/%s_COMPLETE.json').exists())"%(OLD,stem)).strip()=='True':print('POSITION_ALREADY_COMPLETE; NO_RESTART',flush=True);raise SystemExit
for suffix in ['frozen_components_v3.pth','latest_resume_v3.pth']:copy('checkpoints/varbridge_text_finetune/'+stem+'_'+suffix)
# Short benchmark has no saved optimizer updates. It must pass before source is paused.
subprocess.run([PY,'-u','scripts/resume_varbridge_text_v3_position_4090.py','benchmark'],cwd=ROOT,env=env,check=True)
b=json.loads((D/'4090_migration_position_benchmark.json').read_text());assert max(b['batch_seconds'])<8.,'No useful speedup; source continues unchanged'
pid=2420691
remote("import os,signal; p=%d; q=open('/proc/%%d/cmdline'%%p,'rb').read();assert b'run_varbridge_text_finetune_fast_v3.py' in q and b'layernorm_position' in q;os.kill(p,signal.SIGSTOP)"%pid)
try:
 for suffix in ['_latest_resume_v3.pth','.pth']:copy('checkpoints/varbridge_text_finetune/'+stem+suffix)
except BaseException:
 remote('import os,signal;os.kill(%d,signal.SIGCONT)'%pid);raise
remote('import os,signal;os.kill(%d,signal.SIGTERM);os.kill(%d,signal.SIGCONT)'%(pid,pid));time.sleep(2);assert remote('from pathlib import Path;print(Path("/proc/%d").exists())'%pid).strip()=='False'
log=R/'logs'/(stem+'_4090_resume.log')
with log.open('a') as f:p=subprocess.Popen([PY,'-u','scripts/resume_varbridge_text_v3_position_4090.py','train'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out={'time':time.time(),'mode':'layernorm_position','target_pid':p.pid,'source_pid':pid,'target_gpu':1,'benchmark':b};(D/'migration_position_4090.json').write_text(json.dumps(out,indent=2));print('POSITION_MIGRATED',json.dumps(out),flush=True)
