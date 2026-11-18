"""Relocate a private supplementary replica; never modifies adopted model files."""
from pathlib import Path
import os,sys,hashlib,re,ast,subprocess
ROOT=Path('/home/qsmx/VarBridge_supplement/pycodes/GreyFerm');OUT=ROOT/'test_results/current_bound_supplement';PY='/home/qsmx/VarBridge_supplement/anaconda3/envs/AutoTimes/bin/python'
# All modifications are restricted to supplementary launchers and copied baseline path configuration.
base=(ROOT/'scripts/run_current_bound_supplement_ddp2.py').read_text()
base=base.replace('/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python',PY)
base=base.replace("TR=str(Path(PY).with_name('torchrun'))","TR=PY")
base=base.replace("cmd=[TR,'--master_port='", "cmd=[TR,'-m','torch.distributed.run','--master_port='")
base=base.replace("[TR,'--master_port=29700'", "[TR,'-m','torch.distributed.run','--master_port=29700'")
base=base.replace("HF_HUB_OFFLINE='1',", "HF_HUB_OFFLINE='1',HF_HOME='/home/qsmx/VarBridge_supplement/.cache/huggingface',")
base=base.replace("if remote_owned(ds,variant):print('RESERVED_3090',ds,variant,flush=True);return", "if not remote_owned(ds,variant):return")
base=base.replace("while summarize()!=24:","while summarize()!=4:").replace("print('WAIT_3090_RESULTS'", "print('WAIT_OWN3090_RESULTS'")
base=base.replace('ALL_CURRENT_BOUND_SENSITIVITY_COMPLETE','ALL_3090_ASSIGNED_SENSITIVITY_COMPLETE')
base=base.replace("pipeline.log","pipeline_3090.log")
ast.parse(base);(ROOT/'scripts/run_assigned_supplement_3090.py').write_text(base)
for name in ['analyze_current_bound_supplement.py','benchmark_current_bound_supplement.py']:
 p=ROOT/'scripts'/name;s=p.read_text().replace(',mmap=True','').replace("k.removeprefix('module.')", "(k[7:] if k.startswith('module.') else k)")
 s=s.replace("os.environ.get('CUDA_VISIBLE_DEVICES')=='2'","os.environ.get('CUDA_VISIBLE_DEVICES')=='0'")
 s=s.replace("args.gpu in range(2,8)","args.gpu==0").replace("os.environ.get('CUDA_VISIBLE_DEVICES')=='2' and args.gpu==2","os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and args.gpu==0")
 s=s.replace(" and args.gpu==2", " and args.gpu==0")
 ast.parse(s);p.write_text(s)
# Exact same GPT2 assets in the transferred HuggingFace snapshot replace an absolute source-host path.
p=ROOT.parent/'Time-LLM/models/TimeLLM.py';s=p.read_text();snapshot=next(Path('/home/qsmx/VarBridge_supplement/.cache/huggingface/hub/models--gpt2/snapshots').iterdir());s=s.replace('/home/home_new/qsmx/.cache/modelscope/hub/models/AI-ModelScope/gpt2',str(snapshot));p.write_text(s)
for name,digest in re.findall(r'^- ((?:models|scripts)/[^:]+): ([0-9a-f]{64})$',(OUT/'protocol.md').read_text(),re.M):assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
print('PROTECTED_SOURCES_EXACT',flush=True)
# Import the copied environment before launching persistent work.
code="import sys,torch,transformers,numpy,pandas,scipy,sklearn,matplotlib,seaborn,einops; print('ENV_READY',sys.version,torch.__version__,transformers.__version__,torch.cuda.is_available(),flush=True)"
subprocess.run([PY,'-c',code],check=True)
# Record hardware separately from historical4090 measurements.
with (OUT/'hardware_3090.txt').open('w') as f:subprocess.run(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv'],stdout=f,check=True)
with (OUT/'protocol.md').open('a') as f:f.write('\n3090 replica: exact copied AutoTimes environment(Python3.8,torch2.0.1+cu117,transformers4.35.2), seed2021. Supplementary analysis loader omits unsupported mmap keyword; benchmark strips module prefix with Python3.8-compatible slicing. Adopted models/data/checkpoints unchanged. All efficiency rows are3090 only; dedicated GPU0, quiet admission.\n')
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',HF_HOME='/home/qsmx/VarBridge_supplement/.cache/huggingface',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
for script,log,pidfile in [('run_assigned_supplement_3090.py','pipeline_3090.log','coordinator_3090.pid'),('run_analysis_supplement_3090.py','analysis_pipeline_3090.log','analysis_3090.pid')]:
 with (OUT/log).open('a') as f:p=subprocess.Popen([PY,'-u','scripts/'+script],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (OUT/pidfile).write_text(str(p.pid)+'\n');print('STARTED',script,p.pid,flush=True)
print('BOOTSTRAP_3090_COMPLETE',flush=True)
