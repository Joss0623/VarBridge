"""Isolated original-VarBridge text tuning; no adopted source modifications."""
import os,sys,json,time,math,random,hashlib,csv,fcntl,subprocess,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
R=ROOT/'test_results/varbridge_text_finetune';CK=ROOT/'checkpoints/varbridge_text_finetune'
R.mkdir(exist_ok=True);CK.mkdir(exist_ok=True);(R/'logs').mkdir(exist_ok=True)
import torch,numpy as np,pandas as pd
from transformers import AutoTokenizer,AutoModel
from torch.utils.checkpoint import checkpoint
from common_bound_ablations import Windows
from common_bound_supplement import make_config,seed_all
from models.KIBR_PatchTST_BoundMechanism import Model as Original
import preprocess_window_semantic as efp
from cho_data import NAMES as CN,MECHANISMS as CM
SOURCE='FacebookAI/roberta-base';TASKS=['CHO','EFP_long_sub2'];MODES=['frozen','layernorm','layernorm_position']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savej(p,x):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def csvout(p,rows):
 t=Path(str(p)+'.tmp')
 with t.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 t.replace(p)
def guard():
 for p,h in json.loads((R/'protected_hashes.json').read_text()).items():assert sha(p)==h,p
 assert sha(__file__)==json.loads((R/'implementation_hashes.json').read_text())[str(Path(__file__).resolve())]
def setup():
 torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;seed_all(2021)
class Texts:
 def __init__(self,task):
  self.task=task;self.raw=pd.read_csv(ROOT/'dataset'/task/(task+'.csv'));self.features=[x for x in self.raw if x not in ['date','batch_id','hh']];self.v=self.raw[self.features].to_numpy(float);self.hh=self.raw.hh.to_numpy(float)
  self.lo={int(b):int(np.flatnonzero(self.raw.batch_id.to_numpy()==b)[0]) for b in self.raw.batch_id.unique()}
  if task=='CHO':self.modes=pd.read_csv(ROOT/'dataset/CHO/batch_manifest.csv').set_index('batch_id')['mode'].to_dict()
 def window(self,s):
  bid=int(self.raw.batch_id.iloc[s]);win=self.v[s:s+48];out=[]
  for j,name in enumerate(self.features):
   v=win[:,j];delta=v[-1]-v[0];direction='upward' if delta>1e-8 else 'downward' if delta<-1e-8 else 'flat';recent=', '.join(f'{z:.3f}' for z in v[-12:])
   if self.task=='CHO':
    age=self.hh[s];a=f'Data description: CHO-K1 {self.modes[bid]} cultivation, variable {CN[j]}. Task: forecast the next 24 hourly steps from the previous 48 observations. Input window culture age: {age:.1f} to {age+47:.1f} hours. '
    scores=[]
    for lag in range(1,25):
     x=v[:-lag]-v[:-lag].mean();y=v[lag:]-v[lag:].mean();den=float(np.sqrt((x*x).sum()*(y*y).sum()));scores.append((lag,float((x*y).sum()/den) if den>1e-8 else 0.))
    scores.sort(key=lambda z:(-abs(z[1]),z[0]));lags=', '.join(f'{l}:{z:+.2f}' for l,z in scores[:12])
   else:
    a=f'Data description: Erythromycin fermentation, variable {efp.NAMES[name]} ({name}). Task: forecast the next 24 hourly steps from the previous 48 observations. Input window batch age: {self.hh[s]-self.hh[self.lo[bid]]:.1f} to {self.hh[s+47]-self.hh[self.lo[bid]]:.1f} hours. ';lags=efp.top_lags(v)
   out.append(a+f'Input statistics: minimum {v.min():.3f}; maximum {v.max():.3f}; median {np.median(v):.3f}; mean {v.mean():.3f}; first {v[0]:.3f}; last {v[-1]:.3f}; trend {direction}, change {delta:+.3f}. Top 12 autocorrelation lags (lag:correlation): {lags}. Last 12 observed values: {recent}.')
  return out
 def static(self):
  if self.task=='CHO':
   out=[]
   for name,m in zip(CN,CM):out.extend([f'Mechanism for {name}: {m}',f'Stage knowledge for {name}: Only observed culture age is used; no biological stage label is assigned.',f'Operation context for {name}: CHO-K1 cell cultivation under batch or fed-batch operation. Oxygen, pH, optical and dielectric signals characterize the culture; no future measurements are supplied.'])
   return out
  randoms=list(efp.RANDOM_SEMANTICS);random.Random(2021).shuffle(randoms);out=[]
  for variant in efp.VARIANTS:
   for j,name in enumerate(self.features):
    m=efp.MECHANISM_UNITS[name] if variant=='correct' else efp.MECHANISM_UNITS[self.features[(j+1)%5]] if variant=='shuffled' else 'No variable-specific mechanism is supplied.' if variant=='none' else 'This process variable changes over time during fermentation.' if variant=='generic' else randoms[j]
    out.extend([f'Mechanism for {efp.NAMES[name]}: {m}',f'Stage knowledge for {efp.NAMES[name]}: {efp.STAGE_KNOWLEDGE}',f'Operation context for {efp.NAMES[name]}: {efp.OPERATION_CONTEXT[self.task]}'])
  return out

def tokenize(tok,texts,task):
 z=tok(texts,padding=True,truncation=task=='CHO',max_length=256 if task=='CHO' else None,return_tensors='pt');assert z['input_ids'].shape[1]<=320
 return {k:v.int().cpu() for k,v in z.items()}
def startids(ds):return ds.base.starts if ds.dataset=='CHO' else ds.base.valid_indices
class Online(torch.nn.Module):
 def __init__(self,task,mode):
  super().__init__();seed_all(2021);self.forecast=Original(make_config(task)).float();self.encoder=AutoModel.from_pretrained(SOURCE,local_files_only=True,add_pooling_layer=False).float().eval().requires_grad_(False);self.task=task;self.mode=mode;self.tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True);self.static_tokens=tokenize(self.tok,Texts(task).static(),task)
  for name,p in self.encoder.named_parameters():p.requires_grad_(mode!='frozen' and ('LayerNorm' in name or (mode=='layernorm_position' and name=='embeddings.position_embeddings.weight')))
 def train(self,mode=True):
  super().train(mode);self.encoder.eval();self.forecast.patch_backbone.eval();return self
 def encode(self,z):
  device=next(self.encoder.parameters()).device;result=[]
  for off in range(0,len(z['input_ids']),8):
   q={k:v[off:off+8].long().to(device) for k,v in z.items()};length=int(q['attention_mask'].sum(1).max());q={k:v[:,:length] for k,v in q.items()}
   def fn(input_ids,attention_mask):
    h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
    return (h*attention_mask[:,:,None]).sum(1)/attention_mask.sum(1)[:,None]
   pooled=checkpoint(fn,q['input_ids'],q['attention_mask'],use_reentrant=False) if self.mode!='frozen' and torch.is_grad_enabled() else fn(q['input_ids'],q['attention_mask'])
   result.append(pooled)
  return torch.cat(result)
 def mark(self,z):
  w=self.encode(z).reshape(-1,5,768);k=self.encode(self.static_tokens)
  if self.task=='CHO':k=k.reshape(5,3,768)
  else:
   w=w.half().float();k=k.half().float().reshape(5,5,3,768);k=k-k.mean((0,1),keepdim=True);k=(k/k.norm(dim=-1,keepdim=True).clamp_min(1e-6)*768**.5).half().float()[0]
  b=len(w);bank=k[:,0][None,None].expand(b,5,5,768);context=k[:,1:][None].expand(b,-1,-1,-1)
  return torch.cat([w[:,:,None],bank,context],dim=2)
 def forward(self,x,z):
  m=self.mark(z);return self.forecast(x,m,None,m)

def data(task):return [Windows(task,s,24) for s in ['train','val','test']]
def numeric(ds,ids,device):
 q=[ds[i] for i in ids];return tuple(torch.stack([p[k] for p in q]).float().to(device) for k in range(3))
def prepare(task):
 path=R/(task+'_tokens.pt')
 if path.exists():return
 tr,va,te=data(task);texts=Texts(task);tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True);parts={}
 for split,ds in zip(['train','val','test'],[tr,va,te]):
  starts=list(map(int,startids(ds)));ids=torch.full((len(starts)*5,320),tok.pad_token_id,dtype=torch.int32);mask=torch.zeros_like(ids)
  for off in range(0,len(starts),128):
   strings=[p for s in starts[off:off+128] for p in texts.window(s)];z=tokenize(tok,strings,task);a=off*5;b=a+len(strings);ids[a:b,:z['input_ids'].shape[1]]=z['input_ids'];mask[a:b,:z['attention_mask'].shape[1]]=z['attention_mask']
  parts[split]={'input_ids':ids,'attention_mask':mask,'starts':starts};print('TOKENIZED',task,split,len(starts),flush=True)
 tmp=path.with_suffix('.tmp');torch.save(parts,tmp);tmp.replace(path)
def tokenbatch(part,ids):
 sel=torch.tensor([i*5+j for i in ids for j in range(5)]);return {k:part[k].index_select(0,sel) for k in ['input_ids','attention_mask']}

def preflight():
 setup();guard();report=[]
 for task in TASKS:
  ds=data(task);n=Online(task,'frozen').eval();texts=Texts(task);checks=[]
  for split,d in zip(['train','val','test'],ds):
   ids=[0,len(d)-1];x,y,cached=numeric(d,ids,'cpu');strings=[p for i in ids for p in texts.window(int(startids(d)[i]))];z=tokenize(n.tok,strings,task)
   with torch.no_grad():
    m=n.mark(z);a=n.forecast(x,m,None,m);b=n.forecast(x,cached,None,cached)
   checks.append({'split':split,'mark_max_abs':float((m-cached).abs().max()),'forecast_max_abs':float((a-b).abs().max())})
   assert torch.allclose(m,cached,atol=.035 if task!='CHO' else 3e-5,rtol=2e-4),(task,checks)
   assert torch.allclose(a,b,atol=1e-4,rtol=1e-4),(task,checks)
  for mode in MODES[1:]:
   n.mode=mode
   for name,p in n.encoder.named_parameters():p.requires_grad_('LayerNorm' in name or (mode=='layernorm_position' and name=='embeddings.position_embeddings.weight'))
   n.train();n.zero_grad(set_to_none=True);x,y,c=numeric(ds[0],[0],'cpu');z=tokenize(n.tok,texts.window(int(startids(ds[0])[0])),task);p=n(x,z);loss=torch.nn.functional.huber_loss(p,y);loss.backward();active=[(name,q) for name,q in n.encoder.named_parameters() if q.requires_grad]
   assert torch.isfinite(p).all() and all(q.grad is not None and torch.isfinite(q.grad).all() for _,q in active)
   assert sum(float(q.grad.abs().sum()) for _,q in active)>0
   assert all(q.grad is None for q in n.forecast.patch_backbone.parameters()) and not n.encoder.training
   report.append({'task':task,'mode':mode,'encoder_trainable':sum(q.numel() for _,q in active),'encoder_names':[name for name,_ in active],'identity':checks,'finite_forward_backward':True,'frozen_numeric_backbone':True})
  del n
 guard();savej(R/'PREFLIGHT_PASS.json',report);print('ALL_TEXT_FINETUNE_PREFLIGHT_COMPLETE',flush=True)

def train(task,mode):
 setup();guard();dsets=data(task);parts=torch.load(R/(task+'_tokens.pt'),map_location='cpu');n=Online(task,mode).cuda();seed_all(2021);opt=torch.optim.Adam([p for p in n.parameters() if p.requires_grad],lr=1e-4);hist=[];best=float('inf');bestep=0;cp=CK/(task+'_'+mode+'.pth')
 frozen_hash={k:sha_tensor(v) for k,v in n.forecast.patch_backbone.state_dict().items()};encoder_frozen={k:sha_tensor(v) for k,v in n.encoder.named_parameters() if not v.requires_grad}
 def evaluate(ds,split,save=False):
  n.eval();se=ae=count=0;rows=[]
  with torch.no_grad():
   for off in range(0,len(ds),21):
    ids=list(range(off,min(off+21,len(ds))));x,y,m=numeric(ds,ids,'cuda');p=n(x,tokenbatch(parts[split],ids));d=(p-y).double();assert torch.isfinite(d).all();se+=float(d.square().sum());ae+=float(d.abs().sum());count+=d.numel()
    if save:
     rows.extend({'index':i,'mse':float(d[j].square().mean()),'mae':float(d[j].abs().mean())} for j,i in enumerate(ids))
  if save:csvout(R/(task+'_'+mode+'_per_window.csv'),rows)
  return se/count,ae/count
 for ep in range(1,101):
  guard();n.train();start=time.time();lr=1e-4*(1/(1+math.exp(-.5*(ep-10)))-1/(1+math.exp(-.05*(ep-100))))
  for pg in opt.param_groups:pg['lr']=lr
  ids=torch.randperm(len(dsets[0]),generator=torch.Generator().manual_seed(2021+ep)).tolist();total=0
  for off in range(0,len(ids),126):
   glob=ids[off:off+126];opt.zero_grad(set_to_none=True)
   for k in range(0,len(glob),21):
    cur=glob[k:k+21];x,y,m=numeric(dsets[0],cur,'cuda');p=n(x,tokenbatch(parts['train'],cur));loss=torch.nn.functional.huber_loss(p,y);assert torch.isfinite(loss);(loss*len(cur)/len(glob)).backward();total+=float(loss.detach())*len(cur)
   assert all(torch.isfinite(p.grad).all() for p in n.parameters() if p.grad is not None);opt.step()
   if off==0:print('FIRST_BATCH',task,mode,ep,'seconds',time.time()-start,'peak_MiB',torch.cuda.max_memory_allocated()/2**20,flush=True)
  vm,va=evaluate(dsets[1],'val');row={'epoch':ep,'train_huber':total/len(ids),'validation_mse':vm,'validation_mae':va,'lr':lr,'seconds':time.time()-start};hist.append(row);csvout(R/(task+'_'+mode+'_learning.csv'),hist)
  if vm<best:best=vm;bestep=ep;torch.save(n.state_dict(),cp)
  print('EPOCH',task,mode,json.dumps(row),flush=True)
 assert all(sha_tensor(v)==frozen_hash[k] for k,v in n.forecast.patch_backbone.state_dict().items())
 assert all(sha_tensor(dict(n.encoder.named_parameters())[k])==h for k,h in encoder_frozen.items())
 n.load_state_dict(torch.load(cp,map_location='cpu'));mse,mae=evaluate(dsets[2],'test',True)
 # Rebuild final best-checkpoint text caches for all three splits; these belong only to this experiment.
 n.eval();rebuilt={}
 with torch.no_grad():
  for split,ds in zip(['train','val','test'],dsets):
   marks=[]
   for off in range(0,len(ds),21):marks.append(n.mark(tokenbatch(parts[split],list(range(off,min(off+21,len(ds)))))).cpu())
   rebuilt[split]={'starts':parts[split]['starts'],'marks':torch.cat(marks)}
 torch.save(rebuilt,R/(task+'_'+mode+'_final_text_cache.pt'))
 out={'task':task,'mode':mode,'epochs':100,'seed':2021,'lookback':48,'horizon':24,'global_batch':126,'microbatch':21,'mse':mse,'mae':mae,'test_windows':len(dsets[2]),'validation_mse':best,'best_epoch':bestep,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'encoder_trainable':sum(p.numel() for p in n.encoder.parameters() if p.requires_grad),'implementation_sha256':sha(__file__)}
 guard();savej(R/(task+'_'+mode+'_COMPLETE.json'),out);print('TEXT_FINETUNE_COMPLETE',json.dumps(out),flush=True)
def sha_tensor(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def gpu_info():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True);apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],universal_newlines=True);busy={x.split(',')[0].strip() for x in apps.splitlines() if ',' in x}
 return [(int(i),int(f),int(u)) for i,uuid,f,u in [x.split(',') for x in rows.splitlines()] if uuid.strip() not in busy]
def queue():
 lock=(R/'queue.lock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 (R/'queue.pid').write_text(str(os.getpid()));guard()
 if not (R/'PREFLIGHT_PASS.json').exists():
  with (R/'logs/preflight.log').open('a') as f:subprocess.run([sys.executable,__file__,'preflight'],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
 for task in TASKS:prepare(task)
 # Frozen controls reuse only exact matched fresh-refit 100-epoch originals, after online/cache identity preflight.
 for task in TASKS:
  source=ROOT/'test_results/varbridge_relation_graph'/(task+'_original_COMPLETE')
  if source.exists():
   q=json.loads(source.read_text());assert q['epochs']==100 and q['seed']==2021 and q['global_batch']==126 and q['horizon']==24 and q['test_windows']==(381 if task=='CHO' else 4431)
   savej(R/(task+'_frozen_COMPLETE.json'),{'mode':'frozen','reused_matched_control':True,'source_result':str(source),'source_result_sha256':sha(source),'online_cache_identity_preflight':str(R/'PREFLIGHT_PASS.json'),'result':q})
 jobs=[(t,m) for t in TASKS for m in MODES if not (R/(t+'_'+m+'_COMPLETE.json')).exists()];active={};attempts={}
 while jobs or active:
  for g,(p,t,m,log) in list(active.items()):
   if p.poll() is not None:
    log.close();del active[g]
    if not (R/(t+'_'+m+'_COMPLETE.json')).exists():
     savej(R/(t+'_'+m+'_FAILURE.json'),{'exit':p.returncode,'time':time.time(),'log':log.name});print('FAILED',t,m,p.returncode,flush=True)
     # Leave technical history untouched, never launch a duplicate automatically after a fit failure.
  host=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1024
  free=[g for g,f,u in gpu_info() if f>=20000 and u<=5 and g not in active] if host>=50000 else []
  if jobs and free:
   g=free[0];stable=True
   for _ in range(3):
    time.sleep(3)
    if g not in [a for a,b,c in gpu_info() if b>=20000 and c<=5]:stable=False;break
   if stable:
    t,m=jobs.pop(0);env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false');log=(R/'logs'/(t+'_'+m+'.log')).open('a');p=subprocess.Popen([sys.executable,__file__,'train',t,m],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[g]=(p,t,m,log);print('LAUNCH',t,m,g,p.pid,flush=True)
  # During a live run detect foreign contexts; stop only our worker and preserve its partial history.
  apptext=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],universal_newlines=True);uuids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],universal_newlines=True);mapping={int(l.split(',')[0]):l.split(',')[1].strip() for l in uuids.splitlines()}
  for g,(p,t,m,log) in active.items():
   other=[int(l.split(',')[1]) for l in apptext.splitlines() if l.split(',')[0].strip()==mapping[g] and int(l.split(',')[1])!=p.pid]
   if other and p.poll() is None:print('GPU_CONTENTION_STOP_OWN_WORKER',g,t,m,other,flush=True);os.killpg(p.pid,signal.SIGTERM)
  time.sleep(15)
 complete=all((R/(t+'_'+m+'_COMPLETE.json')).exists() for t in TASKS for m in MODES)
 print('ALL_TEXT_FINETUNE_COMPLETE' if complete else 'TEXT_FINETUNE_REQUIRES_TECHNICAL_RECOVERY',flush=True)
if __name__=='__main__':
 setup()
 if sys.argv[1]=='preflight':preflight()
 elif sys.argv[1]=='queue':queue()
 elif sys.argv[1]=='train':train(sys.argv[2],sys.argv[3])
