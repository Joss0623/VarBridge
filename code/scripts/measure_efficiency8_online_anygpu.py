"""Isolated online inference audit: no training and no adopted-source edits."""
import os,sys,time,csv,re,math,argparse,hashlib,subprocess,fcntl,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'test_results/efficiency8_4090'
OUT=SOURCE/'online_remeasure'
GPU=int(os.environ['EFFICIENCY_GPU'])
OUT.mkdir(exist_ok=True)
PY='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python'
MODELS=['DLinear','PatchTST','iTransformer','Crossformer','TimeMixer','TimesNet','TimeLLM','VarBridge']
def gpu():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 g=next([v.strip() for v in l.split(',')] for l in raw.splitlines() if l.split(',')[0].strip()==str(GPU))
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 return g,{int(l.split(',')[1]) for l in apps.splitlines() if l.split(',')[0].strip()==g[1]}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writecsv(path,rows):
 assert rows
 with path.with_suffix('.tmp').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 path.with_suffix('.tmp').replace(path)
def audit_training():
 rows=list(csv.DictReader((SOURCE/'all_training_sources.csv').open()))
 assert len(rows)==8 and {r['model'] for r in rows}==set(MODELS)
 for r in rows:
  assert r['dataset']=='CHO' and int(r['epochs'])==100 and int(r['seed'])==2021 and int(r['horizon'])==24 and int(r['global_batch'])==126 and int(r['test_windows'])==381
  assert all(math.isfinite(float(r[k])) for k in ['mse','mae'])
  p=Path(r['checkpoint']);assert p.exists()
  if r.get('checkpoint_sha256'):assert digest(p)==r['checkpoint_sha256']
  log=Path(r['source_log']).read_text()
  if r['model']=='VarBridge':
   assert 'completed_epochs=100' in log and 'CHO_BOUND_JOB_COMPLETE Fusion correct 24 2021' in log
   assert sorted(set(map(int,re.findall(r'Epoch:\s*(\d+)',log))))==list(range(1,101))
   assert set(map(int,re.findall(r'CHO_DDP_RANK (\d+)',log)))==set(range(6))
  else:
   assert re.findall(r'^EFF100_EPOCH (\d+) ',log,re.M)==[str(i) for i in range(1,101)]
   assert 'EFFICIENCY100_JOB_COMPLETE '+r['model'] in log
  print('TRAINING_RECORD_AUDIT_PASS',r['model'],digest(p),flush=True)
 return rows
def worker(model):
 import numpy as np,pandas as pd,torch
 from cho_data import CHOWindows,DATA,FEATURES
 from efficiency8_common import build,enforce_efficiency_fp32
 assert os.environ['CUDA_VISIBLE_DEVICES']==str(GPU)
 torch.set_num_threads(2);torch.set_num_interop_threads(2)
 random.seed(2021);np.random.seed(2021);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 row=next(r for r in csv.DictReader((SOURCE/'all_training_sources.csv').open()) if r['model']==model)
 ck=Path(row['checkpoint']);ck_hash=digest(ck)
 net=build('CHO',model).float()
 if model=='TimeLLM':enforce_efficiency_fp32(net)
 state=torch.load(ck,map_location='cpu');net.load_state_dict(state,strict=True)
 del state
 trainable_count=sum(p.numel() for p in net.parameters() if p.requires_grad)
 net=net.to('cuda:0').eval().requires_grad_(False)
 assert all(p.dtype==torch.float32 for p in net.parameters())
 d=CHOWindows('test',24,'roberta' if model=='VarBridge' else None);h48=CHOWindows('test',48)
 assert len(d)==381 and len(h48)==309
 lookup={s:i for i,s in enumerate(d.starts)};eligible=[lookup[s] for s in h48.starts]
 selected=[eligible[i] for i in np.linspace(0,len(eligible)-1,32,dtype=int)]
 assert len(set(selected))==32
 raw=pd.read_csv(DATA/'CHO.csv');manifest=pd.read_csv(DATA/'batch_manifest.csv')
 modes={int(r.batch_id):r.mode for r in manifest.itertuples()}
 rawv=raw[FEATURES].to_numpy(float)
 items=[dict(raw_window=rawv[s:s+48],age=[float(raw.hh.iloc[s])],modes=[modes[d.ids[i]]]) for i,s in enumerate(d.starts)]
 encoder=None;static=None;max_cache_error=0.
 if model=='VarBridge':
  from efficiency8_text import TextEncoder,texts_for_window
  encoder=TextEncoder('cuda:0')
  mechanisms=torch.load(DATA/'CHO_bound_mechanisms.pt',map_location='cpu')['correct']['embeddings'].float()
  static=d.cache['knowledge_embeddings'].float().clone();static[:,0]=mechanisms
 def normalized(i):
  return torch.from_numpy(((items[i]['raw_window']-d.mean)/d.std).astype('float32')).unsqueeze(0)
 def mark_for(window):
  bank=static[:,0].unsqueeze(0).expand(5,-1,-1)
  return torch.cat([window[:,None,:],bank,static[:,1:,:]],dim=1).unsqueeze(0)
 @torch.no_grad()
 def forward(x,mark=None):
  x=x.to('cuda:0')
  if model=='VarBridge':
   mark=mark.to('cuda:0');z=net(x,mark,None,mark)[:,-24:]
  else:
   dec=torch.cat([x[:,-24:],torch.zeros(1,24,5,device='cuda:0')],1)
   z=net(x,torch.zeros(1,48,4,device='cuda:0'),dec,torch.zeros(1,48,4,device='cuda:0'))[:,-24:]
  return z.float().cpu()
 @torch.no_grad()
 def online(i):
  x=normalized(i)
  if encoder is None:return forward(x)
  emb=encoder.encode(texts_for_window(items[i],x))
  return forward(x,mark_for(emb))
 # All 381 windows are evaluated; no subsampling of accuracy.
 se=ae=0.;max_pred_error=0.
 for i in range(len(d)):
  x,y,mark,_=d[i];assert torch.equal(normalized(i)[0],x)
  if encoder is not None:
   cache_window=d.cache['window_embeddings'][d.lookup[d.starts[i]]].float()
   mark=mark_for(cache_window)
   ref=forward(x.unsqueeze(0),mark)
   emb=encoder.encode(texts_for_window(items[i],x))
   err=float((emb-cache_window).abs().max());max_cache_error=max(max_cache_error,err)
   assert torch.allclose(emb,cache_window,atol=3e-5,rtol=3e-5),('text cache mismatch',i,err)
   pred=forward(x.unsqueeze(0),mark_for(emb));max_pred_error=max(max_pred_error,float((pred-ref).abs().max()))
   assert torch.allclose(pred,ref,atol=3e-5,rtol=3e-5),('online prediction mismatch',i,max_pred_error)
  else:pred=forward(x.unsqueeze(0))
  delta=(pred-y.unsqueeze(0)).double();assert torch.isfinite(delta).all()
  se+=float(delta.square().sum());ae+=float(delta.abs().sum())
  if i%100==0:print('FULL_TEST_AUDIT',model,i,flush=True)
 mse=se/(381*24*5);mae=ae/(381*24*5)
 # TimesNet selects FFT periods from the batch mean. A batch1 forecast can
 # differ from the original batch4 evaluation with the same unchanged state.
 # Reproduce batch4 separately to verify provenance, retain batch1 accuracy
 # for the online batch1 comparison; never alter model period selection.
 if model in ['TimesNet','TimeLLM']:
  if model=='TimesNet':
   source=Path(ROOT.parent/'Time-Series-Library/models/TimesNet.py').read_text()
   assert 'frequency_list = abs(xf).mean(0).mean(-1)' in source
  else:
   source=Path(ROOT.parent/'Time-LLM/models/TimeLLM.py').read_text()
   assert 'padding=True' in source and 'self.llm_model(inputs_embeds=llama_enc_out)' in source
   # Native dynamic prompt padding changes patch positions with batch peers.
   # Preserve the original tokenizer/LLM code and audit original batch4 separately.
  se4=ae4=0.
  with torch.no_grad():
   for off in range(0,len(d),4):
    batch=[d[i] for i in range(off,min(off+4,len(d)))]
    xx=torch.stack([v[0] for v in batch]).to('cuda:0')
    yy=torch.stack([v[1] for v in batch])
    n=len(xx);dec=torch.cat([xx[:,-24:],torch.zeros(n,24,5,device='cuda:0')],1)
    z=net(xx,torch.zeros(n,48,4,device='cuda:0'),dec,torch.zeros(n,48,4,device='cuda:0'))[:,-24:].float().cpu()
    delta=(z-yy).double();assert torch.isfinite(delta).all()
    se4+=float(delta.square().sum());ae4+=float(delta.abs().sum())
  mse4=se4/(381*24*5);mae4=ae4/(381*24*5)
  assert abs(mse4-float(row['mse']))<5e-6 and abs(mae4-float(row['mae']))<5e-6,('original batch4 metric mismatch',mse4,mae4,row['mse'],row['mae'])
  print('BATCH_DEPENDENCE_AUDIT_PASS',model,'original_batch4',mse4,mae4,'online_batch1',mse,mae,flush=True)
 else:
  assert abs(mse-float(row['mse']))<5e-6 and abs(mae-float(row['mae']))<5e-6,('checkpoint metric mismatch',model,mse,mae,row['mse'],row['mae'])
 print('FULL_TEST_AUDIT_PASS',model,mse,mae,'max_cache_error',max_cache_error,'max_prediction_error',max_pred_error,flush=True)
 # Warmup and timed work use the identical raw-window to CPU-forecast path.
 for k in range(20):online(selected[k%32])
 torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 samples=[]
 for repeat in range(5):
  for j,i in enumerate(selected):
   torch.cuda.synchronize();start=time.perf_counter_ns()
   pred=online(i)
   torch.cuda.synchronize();ms=(time.perf_counter_ns()-start)/1e6
   assert math.isfinite(ms) and ms>0 and torch.isfinite(pred).all()
   samples.append(dict(model=model,repeat=repeat,window_index=i,start=d.starts[i],batch_id=d.ids[i],latency_ms=ms))
 forecaster=sum(p.numel() for p in net.parameters())
 text_params=sum(p.numel() for p in encoder.net.parameters()) if encoder is not None else 0
 internal_text=sum(p.numel() for p in net.llm_model.parameters()) if model=='TimeLLM' else 0
 values=np.array([s['latency_ms'] for s in samples])
 result=dict(model=model,dataset='CHO',lookback=48,horizon=24,seed=2021,epochs=100,test_windows=381,mse=mse,mae=mae,
  latency_mean_ms=float(values.mean()),latency_median_ms=float(np.median(values)),latency_p90_ms=float(np.percentile(values,90)),
  latency_p95_ms=float(np.percentile(values,95)),latency_sd_ms=float(values.std(ddof=1)),samples=160,warmups=20,
  matched_windows=32,repeats=5,batch_size=1,precision='float32',cpu_threads=2,physical_gpu=GPU,gpu=torch.cuda.get_device_name(0),
  resident_parameters=forecaster+text_params,forecaster_parameters=forecaster,external_text_parameters=text_params,internal_text_parameters=internal_text,
  trainable_at_training_parameters=trainable_count,
  peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
  max_text_cache_abs_error=max_cache_error,max_prediction_abs_error=max_pred_error,checkpoint=str(ck),checkpoint_sha256=ck_hash,
  timing_scope='CPU raw window normalization; H2D; native causal text construction/tokenization/encoding if used; forecast; D2H; excludes model load and fixed context encoding')
 # Trainable count captured before disabling gradients for evaluation.
 if row.get('trainable_parameters'):assert trainable_count==int(row['trainable_parameters'])
 assert digest(ck)==ck_hash
 writecsv(OUT/(model+'_online_samples.csv'),samples);writecsv(OUT/(model+'_online.csv'),[result])
 print('EFF8_ONLINE_COMPLETE',model,result,flush=True)
def queue():
 lock=(OUT/'.online.queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (OUT/'online_coordinator.pid').write_text(str(os.getpid()))
 while not (SOURCE/'ALL8_TRAINING_RECORDS_COMPLETE').exists():
  time.sleep(30)
 trainlock=(SOURCE/'.queue.lock').open('a');fcntl.flock(trainlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 rows=audit_training()
 tracked=[ROOT/'models/KIBR_PatchTST_BoundMechanism.py',ROOT/'models/PatchTST.py',ROOT/'scripts/cho_data.py',ROOT/'scripts/common_bound_supplement.py',ROOT/'scripts/train_cho_bound_mechanism_ddp.py',
  ROOT.parent/'Time-LLM/models/TimeLLM.py']+[ROOT.parent/'Time-Series-Library/models'/f'{m}.py' for m in MODELS if m not in ['VarBridge','TimeLLM']]
 hashes={str(p):digest(p) for p in tracked}
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(GPU),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
 for m in MODELS:
  final=OUT/(m+'_online.csv');log=OUT/(m+'_online.log')
  if final.exists() and log.exists() and 'EFF8_ONLINE_COMPLETE '+m in log.read_text():print('SKIP_EXISTING_ONLINE',m,flush=True);continue
  while True:
   g,pids=gpu()
   if not pids and int(g[2])>=20000 and int(g[3])<=5:break
   time.sleep(30)
  if log.exists():log.rename(str(log)+'.technical.'+str(time.time_ns()))
  with log.open('w') as f:
   p=subprocess.Popen([PY,'-u',str(Path(__file__).resolve()),'--model',m],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   (OUT/'online_worker.pid').write_text(str(p.pid));print('ONLINE_START',m,p.pid,flush=True)
   while p.poll() is None:
    g,pids=gpu()
    if pids-{p.pid}:
     p.terminate();p.wait()
     # Outputs of a contested run cannot count as completed.
     if final.exists():final.rename(str(final)+'.technical.'+str(time.time_ns()))
     raise RuntimeError(('foreign process on selected GPU; isolated timing invalidated',m,pids))
    time.sleep(1)
   assert p.returncode==0,('online worker failed',m,p.returncode)
  assert 'EFF8_ONLINE_COMPLETE '+m in log.read_text() and final.exists()
  print('ONLINE_COMPLETE',m,flush=True)
 results=[next(csv.DictReader((OUT/(m+'_online.csv')).open())) for m in MODELS]
 training_by_model={r['model']:r for r in rows}
 for result in results:
  original=training_by_model[result['model']]
  result['original_batch4_mse']=original['mse']
  result['original_batch4_mae']=original['mae']
  result['accuracy_batch_size']=1
  result['accuracy_note']=('Batch1 full test evaluation; original batch4 checkpoint metrics verified separately: '+('native batch-mean FFT top-k' if result['model']=='TimesNet' else 'native prompt padding/patch position')) if result['model'] in ['TimesNet','TimeLLM'] else 'Full test batch1 agrees with source checkpoint metrics'

 assert all(r['samples']=='160' and r['test_windows']=='381' for r in results)
 assert hashes=={p:digest(p) for p in hashes},'adopted source changed'
 writecsv(OUT/'online_inference_results.csv',results)
 with (OUT/'online_source_audit.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['source','sha256']);w.writerows(hashes.items())
 (OUT/'ALL8_ONLINE_INFERENCE_COMPLETE').write_text('8 complete online inference results, 160 timings each; 381 full test windows; training records verified; adopted sources unchanged.\n')
 print('ALL8_ONLINE_INFERENCE_COMPLETE',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS);args=ap.parse_args()
 if args.model:worker(args.model)
 else:queue()
