"""Budget-labelled same-GPU cached inference using existing formal checkpoints."""
import os,sys,time,json,csv,subprocess,signal,fcntl,hashlib,argparse,re
from pathlib import Path
ROOT=Path("/home/home_new/qsmx/pycodes/GreyFerm");sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
R=ROOT/"test_results/varbridge_lhc_gemp";OUT=R/"budget_efficiency"
PY="/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python"
MODELS=["DLinear","iTransformer","TimeLLM","VarBridge"]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def available():
 rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free,utilization.gpu","--format=csv,noheader,nounits"],universal_newlines=True)
 apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],universal_newlines=True)
 occupied={x.split(",")[0].strip() for x in apps.splitlines()}
 return [(int(a),b.strip()) for a,b,c,d in [x.split(",") for x in rows.splitlines()] if int(c)>=20000 and int(d)<=5 and b.strip() not in occupied]
def worker(model):
 source=ROOT/"scripts/measure_timellm_embedding_cache.py"
 s=source.read_text().split("if __name__=='__main__':")[0]
 s=re.sub(r"OUT=SOURCE/[^;\n]+","OUT=ROOT/'test_results/varbridge_lhc_gemp/budget_efficiency'",s,count=1)
 s=s.replace("(SOURCE/'all_training_sources.csv')","(OUT/'sources.csv')").replace("(SOURCE/'online_remeasure/online_inference_results.csv')","(OUT/'sources.csv')")
 s=s.replace("epochs=100,test_windows=381","epochs=int(row['epochs']),test_windows=381")
 s=s.replace("by causal prefix KV","by cached native prompt input embeddings")
 needle=" net=net.to('cuda:0').eval().requires_grad_(False)"
 extra="""
 # Reconstruct the original formal evaluation separately; its TimeLLM precision
 # and DDP-strided batch4 padding are preserved. Timed comparison remains FP32/batch1.
 original_precision_mse=None;original_precision_mae=None
"""
 assert s.count(needle)==1;s=s.replace(needle,extra+needle)
 needle=" contexts={};context_meta=[];max_error=0.;se=ae=0.;prep_seconds=0."
 audit="""
 if model=='TimeLLM':
  se_original=ae_original=0.;n_original=0
  with torch.no_grad():
   for rank in range(6):
    ids=list(range(rank,len(ds),6))
    for start in range(0,len(ids),4):
     ix=ids[start:start+4];xx=torch.cat([normalized(i) for i in ix]).to('cuda:0')
     dec=torch.cat([xx[:,-24:],torch.zeros(len(ix),24,5,device='cuda:0')],1)
     with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
      pred=type(net).forecast(net,xx,torch.zeros(len(ix),48,4,device='cuda:0'),dec,torch.zeros(len(ix),48,4,device='cuda:0'))[:,-24:].float().cpu()
     target=torch.stack([ds[i][1] for i in ix]);dd=(pred-target).double();se_original+=float(dd.square().sum());ae_original+=float(dd.abs().sum());n_original+=dd.numel()
  original_precision_mse=se_original/n_original;original_precision_mae=ae_original/n_original
  assert abs(original_precision_mse-float(refrow['mse']))<5e-6 and abs(original_precision_mae-float(refrow['mae']))<5e-6,('original BF16/batch4 checkpoint audit mismatch',original_precision_mse,refrow['mse'])
  del xx,dec,pred,target,dd
  torch.cuda.empty_cache()
  print('FORMAL_TIMELLM_EVALUATION_REPRODUCED',original_precision_mse,original_precision_mae,flush=True)
"""
 assert s.count(needle)==1;s=s.replace(needle,audit+needle)
 old=" assert abs(mse-float(refrow['mse']))<5e-6 and abs(mae-float(refrow['mae']))<5e-6,('full381 accuracy changed',model,mse,mae)"
 new=" if model!='TimeLLM':\n "+old+"\n # TimeLLM FP32/batch1 MSE is paired with FP32/batch1 latency, not historical BF16/batch4 accuracy."
 assert old in s;s=s.replace(old,new)
 s=s.replace("result=dict(model=model,scenario=","result=dict(original_formal_test_mse=float(refrow['mse']),original_precision_reproduced_mse=original_precision_mse,original_precision_reproduced_mae=original_precision_mae,actual_training_epochs=int(row['epochs']),training_budget='100/10 epoch counterpart; existing checkpoints',model=model,scenario=")
 if os.environ.get("VB_REFERENCE_COMPILE_ONLY")=="1":
  compile(s,str(source),"exec");print("DYNAMIC_WORKER_SYNTAX_PASS",model);return
 namespace={"__name__":"budget_measure_worker","__file__":str(Path(__file__).resolve())}
 exec(compile(s,str(source),"exec"),namespace);namespace["worker"](model)
def queue():
 lk=(OUT/"queue.lock").open("w");fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
 selected=OUT/"selected_gpu.csv"
 while True:
  if selected.exists():
   pick=list(csv.DictReader(selected.open()))[0];chosen=(int(pick["physical_gpu"]),pick["uuid"]);break
  first=available();time.sleep(5);second=available();time.sleep(5);third=available();common=[p for p in first if p in second and p in third]
  if common:
   chosen=common[0]
   with selected.open("w") as f:w=csv.writer(f);w.writerow(["physical_gpu","uuid"]);w.writerow(chosen)
   print("SELECTED_QUIET_GPU",chosen,flush=True);break
  print("WAIT_NO_EXCLUSIVE_IDLE_GPU",flush=True);time.sleep(45)
 gpu,uuid=chosen
 for model in MODELS:
  result=OUT/(model+"_cached.csv")
  if result.exists():
   x=list(csv.DictReader(result.open()))[0];assert digest(x["checkpoint"])==x["checkpoint_sha256"];continue
  while chosen not in available():time.sleep(45)
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),EFFICIENCY_GPU=str(gpu),OMP_NUM_THREADS="2",HF_HUB_OFFLINE="1",TOKENIZERS_PARALLELISM="false",PYTHONDONTWRITEBYTECODE="1")
  with (OUT/(model+".log")).open("a") as f:
   p=subprocess.Popen([PY,"-u",str(Path(__file__).resolve()),"--model",model],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   print("START",model,p.pid,flush=True)
   while p.poll() is None:
    time.sleep(10);apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],universal_newlines=True)
    foreign=[int(x.split(",")[1]) for x in apps.splitlines() if x.split(",")[0].strip()==uuid and int(x.split(",")[1])!=p.pid]
    if foreign:
     os.killpg(p.pid,signal.SIGTERM);p.wait();raise RuntimeError("Contention: stopped only own worker; retain diagnostics and wait for same GPU")
   if p.wait()!=0:raise RuntimeError("Technical measurement failure: "+model)
  assert result.exists(),result
 rows=[]
 for model in MODELS:
  # Existing measurement helpers name their result files as below.
  candidates=[OUT/(model+"_cached.csv"),OUT/(model+"_cached.csv")]
  p=next((p for p in candidates if p.exists()),None);assert p
  x=list(csv.DictReader(p.open()))[0];rows.append(x)
  assert int(x["samples"])==160 and int(x["physical_gpu"])==gpu and int(x["epochs"])==(100 if model=="VarBridge" else 10)
 with (OUT/"results.csv").open("w") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (OUT/"ALL_BUDGET_INFERENCE_COMPLETE").write_text("Four existing formal checkpoints; same quiet GPU, FP32 batch1, native prompt-input embedding cache; no training.\n")
 subprocess.run([PY,str(ROOT/"scripts/finalize_varbridge_reference_efficiency.py")],cwd=ROOT,check=True)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--model",choices=MODELS);args=ap.parse_args()
 if args.model:worker(args.model)
 else:queue()
