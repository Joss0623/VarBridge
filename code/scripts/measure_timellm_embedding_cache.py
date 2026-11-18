"""Cached-text inference scenario only: unchanged checkpoints, no training, no main edits."""
import os,sys,time,csv,math,random,inspect,textwrap,types,argparse,subprocess,fcntl,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'test_results/efficiency8_4090';OUT=SOURCE/'cached_text_remeasure/timellm_embedding';OUT.mkdir(exist_ok=True)
GPU=int(os.environ['EFFICIENCY_GPU']);PY='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python'
MODELS=['TimeLLM']
from measure_efficiency8_online_anygpu import audit_training,digest,gpu,writecsv
def worker(model):
 import numpy as np,pandas as pd,torch
 from cho_data import CHOWindows,DATA,FEATURES
 from efficiency8_common import build,enforce_efficiency_fp32
 assert os.environ['CUDA_VISIBLE_DEVICES']==str(GPU)
 torch.set_num_threads(2);torch.set_num_interop_threads(2)
 random.seed(2021);np.random.seed(2021);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 row=next(r for r in csv.DictReader((SOURCE/'all_training_sources.csv').open()) if r['model']==model)
 refrow=next(r for r in csv.DictReader((SOURCE/'online_remeasure/online_inference_results.csv').open()) if r['model']==model)
 ck=Path(row['checkpoint']);ckhash=digest(ck);assert ckhash==refrow['checkpoint_sha256']
 net=build('CHO',model).float()
 if model=='TimeLLM':enforce_efficiency_fp32(net)
 net.load_state_dict(torch.load(ck,map_location='cpu'),strict=True)
 trainable=sum(p.numel() for p in net.parameters() if p.requires_grad)
 net=net.to('cuda:0').eval().requires_grad_(False)
 assert all(p.dtype==torch.float32 for p in net.parameters())
 ds=CHOWindows('test',24,'roberta' if model=='VarBridge' else None);ds48=CHOWindows('test',48)
 lookup={s:i for i,s in enumerate(ds.starts)}
 eligible=[lookup[s] for s in ds48.starts];selected=[eligible[i] for i in np.linspace(0,len(eligible)-1,32,dtype=int)]
 assert len(ds)==381 and len(set(selected))==32
 raw=pd.read_csv(DATA/'CHO.csv');rawv=raw[FEATURES].to_numpy(float)
 rawwindows=[rawv[s:s+48] for s in ds.starts]
 def normalized(i):return torch.from_numpy(((rawwindows[i]-ds.mean)/ds.std).astype('float32')).unsqueeze(0)
 mechanisms=torch.load(DATA/'CHO_bound_mechanisms.pt',map_location='cpu')['correct']['embeddings'].float() if model=='VarBridge' else None
 @torch.no_grad()
 def reference(i):
  x=normalized(i).to('cuda:0')
  if model=='VarBridge':
   mark=ds[i][2].unsqueeze(0).clone();mark[:,:,1:6,:]=mechanisms[None,None,:,:];mark=mark.to('cuda:0')
   return net(x,mark,None,mark)[:,-24:].float().cpu()
  dec=torch.cat([x[:,-24:],torch.zeros(1,24,5,device='cuda:0')],1)
  return net(x,torch.zeros(1,48,4,device='cuda:0'),dec,torch.zeros(1,48,4,device='cuda:0'))[:,-24:].float().cpu()
 if model=='TimeLLM':
  # Only an instance adapter: keep all numerical patch/reprogramming/GPT2/head operations.
  # Replace text construction/prefix encoding by causal prefix KV from the same native prompt.
  original=inspect.getsource(type(net).forecast)
  assert original.count('x_enc.to(torch.bfloat16)')==1
  s=original.replace('x_enc.to(torch.bfloat16)','x_enc')
  a=s.index('        min_values =');b=s.index('        source_embeddings =',a)
  s=s[:a]+'        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()\n\n'+s[b:]
  s=s.replace('    def forecast(', '    def forecast_cached_text(')
  line='        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)\n'
  assert s.count(line)==1;s=s.replace(line,'        llama_enc_out = torch.cat([self._eff_text_prefix_embeddings, enc_out], dim=1)\n')
  old='self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state'
  assert s.count(old)==1
  # Retain the entire native GPT2 prefix-plus-patch forward for this smaller input-embedding cache.
  namespace={};exec(textwrap.dedent(s),sys.modules[type(net).__module__].__dict__,namespace)
  net.forecast_cached_text=types.MethodType(namespace['forecast_cached_text'],net)
  assert net.llm_model.config.use_cache
 def nbytes(context):
  if context is None:return 0
  if torch.is_tensor(context):return context.numel()*context.element_size()
  return sum(nbytes(x) for x in context)
 @torch.no_grad()
 def prepare(i):
  if model=='VarBridge':
   mark=ds[i][2].unsqueeze(0).clone();mark[:,:,1:6,:]=mechanisms[None,None,:,:]
   assert mark.device.type=='cpu' and mark.shape==(1,5,8,768)
   return mark,reference(i),0
  if model=='TimeLLM':
   captured={}
   def capture(module,args,kwargs,result):
    assert 'inputs_embeds' in kwargs and result.past_key_values is not None
    prefix_len=kwargs['inputs_embeds'].shape[1]-net.patch_nums
    assert prefix_len>0
    captured['context']=kwargs['inputs_embeds'][:,:prefix_len,:].detach().cpu().contiguous()
    captured['prefix_len']=prefix_len
    assert captured['context'].device.type=='cpu' and captured['context'].shape[0]==5 and captured['context'].shape[-1]==768
   hook=net.llm_model.register_forward_hook(capture,with_kwargs=True)
   try:pred=reference(i)
   finally:hook.remove()
   assert 'context' in captured
   return captured['context'],pred,captured['prefix_len']
  return None,reference(i),0
 @torch.no_grad()
 def cached(i,context):
  # All inputs, including cached context, start on CPU; transfers remain timed.
  x=normalized(i).to('cuda:0')
  if model=='VarBridge':
   mark=context.to('cuda:0');return net(x,mark,None,mark)[:,-24:].float().cpu()
  dec=torch.cat([x[:,-24:],torch.zeros(1,24,5,device='cuda:0')],1)
  xm=torch.zeros(1,48,4,device='cuda:0');ym=torch.zeros(1,48,4,device='cuda:0')
  if model=='TimeLLM':
   net._eff_text_prefix_embeddings=context.to('cuda:0')
   try:pred=net.forecast_cached_text(x,xm,dec,ym)[:,-24:].float().cpu()
   finally:del net._eff_text_prefix_embeddings
   return pred
  return net(x,xm,dec,ym)[:,-24:].float().cpu()
 contexts={};context_meta=[];max_error=0.;se=ae=0.;prep_seconds=0.
 for i in range(len(ds)):
  assert torch.equal(normalized(i)[0],ds[i][0])
  start=time.perf_counter();context,ref,prefix_len=prepare(i);prep_seconds+=time.perf_counter()-start
  pred=cached(i,context);error=float((pred-ref).abs().max());max_error=max(max_error,error)
  assert torch.allclose(pred,ref,atol=3e-5,rtol=3e-5),('cached vs uncached prediction mismatch',model,i,error)
  delta=(pred-ds[i][1].unsqueeze(0)).double();assert torch.isfinite(delta).all()
  se+=float(delta.square().sum());ae+=float(delta.abs().sum())
  if i in selected:
   contexts[i]=context
   context_meta.append(dict(model=model,window_index=i,start=ds.starts[i],cpu_cache_bytes=nbytes(context),text_prefix_tokens=prefix_len))
  if i%100==0:print('CACHED_PREDICTION_AUDIT_PROGRESS',model,i,max_error,flush=True)
 mse=se/(381*24*5);mae=ae/(381*24*5)
 assert abs(mse-float(refrow['mse']))<5e-6 and abs(mae-float(refrow['mae']))<5e-6,('full381 accuracy changed',model,mse,mae)
 print('CACHED_PREDICTION_IDENTITY_PASS',model,'full381',mse,mae,'max_abs_error',max_error,flush=True)
 for k in range(20):i=selected[k%32];cached(i,contexts[i])
 torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 samples=[]
 for rep in range(5):
  for i in selected:
   torch.cuda.synchronize();t=time.perf_counter_ns()
   pred=cached(i,contexts[i])
   torch.cuda.synchronize();ms=(time.perf_counter_ns()-t)/1e6
   assert math.isfinite(ms) and ms>0 and torch.isfinite(pred).all()
   samples.append(dict(model=model,repeat=rep,window_index=i,start=ds.starts[i],batch_id=ds.ids[i],latency_ms=ms))
 v=np.array([r['latency_ms'] for r in samples])
 forecaster=sum(p.numel() for p in net.parameters())
 kind='VarBridge CPU cached RoBERTa/context features' if model=='VarBridge' else 'TimeLLM CPU native prompt input embeddings; all GPT2 prefix and numerical-patch processing retained' if model=='TimeLLM' else 'No text branch; unchanged numerical inference'
 result=dict(model=model,scenario='text_already_cached_CPU',dataset='CHO',lookback=48,horizon=24,seed=2021,epochs=100,test_windows=381,mse=mse,mae=mae,
  latency_median_ms=float(np.median(v)),latency_mean_ms=float(v.mean()),latency_p90_ms=float(np.percentile(v,90)),latency_p95_ms=float(np.percentile(v,95)),latency_sd_ms=float(v.std(ddof=1)),
  samples=160,warmups=20,matched_windows=32,repeats=5,batch_size=1,precision='float32',cpu_threads=2,physical_gpu=GPU,gpu=torch.cuda.get_device_name(0),
  resident_parameters=forecaster,forecaster_parameters=forecaster,external_text_parameters=0,internal_text_parameters=sum(p.numel() for p in net.llm_model.parameters()) if model=='TimeLLM' else 0,
  trainable_at_training_parameters=trainable,cache_kind=kind,mean_CPU_cache_bytes=float(np.mean([r['cpu_cache_bytes'] for r in context_meta])),
  total_32window_CPU_cache_bytes=sum(r['cpu_cache_bytes'] for r in context_meta),peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
  max_prediction_abs_error=max_error,checkpoint=str(ck),checkpoint_sha256=ckhash,
  timing_scope='CPU raw-window normalization and cached context lookup; H2D inputs/context; numerical forecasting including GPT2 numerical-patch processing if needed; D2H forecast. Excludes text cache generation, one-time load, fixed-context preparation.',
  cache_limit='Valid only for exactly the same observed-window text; new/changed descriptions require cache regeneration.',
  untimed_full381_cache_plus_reference_audit_seconds=prep_seconds)
 assert digest(ck)==ckhash
 writecsv(OUT/(model+'_cached_samples.csv'),samples);writecsv(OUT/(model+'_cache_metadata.csv'),context_meta);writecsv(OUT/(model+'_cached.csv'),[result])
 print('EFF8_CACHED_COMPLETE',model,result,flush=True)
def queue():
 lock=(OUT/'.queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (OUT/'queue.pid').write_text(str(os.getpid()));audit_training()
 tracked=list(csv.DictReader((SOURCE/'online_remeasure/online_source_audit.csv').open()))
 before={r['source']:digest(r['source']) for r in tracked};assert all(before[r['source']]==r['sha256'] for r in tracked)
 checkpoint_hashes={r['checkpoint']:r['checkpoint_sha256'] for r in csv.DictReader((SOURCE/'online_remeasure/online_inference_results.csv').open())}
 assert all(digest(p)==h for p,h in checkpoint_hashes.items())
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(GPU),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
 for m in MODELS:
  final=OUT/(m+'_cached.csv');log=OUT/(m+'_cached.log')
  if final.exists() and log.exists() and 'EFF8_CACHED_COMPLETE '+m in log.read_text():print('SKIP_COMPLETED_CACHED',m,flush=True);continue
  while True:
   g,pids=gpu()
   if not pids and int(g[2])>=20000 and int(g[3])<=5:break
   time.sleep(15)
  if log.exists():log.rename(str(log)+'.technical.'+str(time.time_ns()))
  print('START_CACHED',m,flush=True)
  with log.open('w') as f:
   job=subprocess.Popen([PY,'-u',str(Path(__file__).resolve()),'--model',m],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   (OUT/'worker.pid').write_text(str(job.pid))
   while job.poll() is None:
    g,pids=gpu()
    if pids-{job.pid}:
     job.terminate();job.wait()
     if final.exists():final.rename(str(final)+'.technical.'+str(time.time_ns()))
     raise RuntimeError(('foreign GPU process; invalidated own contested cached result',m,pids))
    time.sleep(1)
   assert job.returncode==0,('cached worker failed',m,job.returncode)
  assert final.exists() and 'EFF8_CACHED_COMPLETE '+m in log.read_text()
  print('COMPLETE_CACHED',m,flush=True)
 assert before=={p:digest(p) for p in before}
 assert all(digest(p)==h for p,h in checkpoint_hashes.items())
 results=[next(csv.DictReader((OUT/(m+'_cached.csv')).open())) for m in MODELS]
 writecsv(OUT/'cached_inference_results.csv',results)
 writecsv(OUT/'source_audit.csv',[dict(source=p,sha256=h) for p,h in before.items()])
 (OUT/'TIMELLM_EMBEDDING_CACHE_COMPLETE').write_text('8model cached-text scenario complete; 381window identical forecasts;160timings/model sameGPU; main sources/checkpoints unchanged.\n')
 print('TIMELLM_EMBEDDING_CACHE_COMPLETE',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS);a=ap.parse_args()
 if a.model:worker(a.model)
 else:queue()
