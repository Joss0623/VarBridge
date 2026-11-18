"""Lookback sensitivity: identical original split membership and shared forecast origins.
Isolated VarBridge backbone100 + adaptation100; other predictors10 epochs.
"""
import os,sys,csv,json,time,math,random,hashlib,argparse,subprocess,fcntl,signal
from pathlib import Path
ROOT=Path('/home/home_new/qsmx/pycodes/GreyFerm');sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np,pandas as pd,torch
from varbridge_lhc_gemp import OUT,guard,writecsv,sha
from common_bound_supplement import make_config,seed_all
from analyze_varbridge_timebase import raw_task
from common_bound_ablations import Windows
B=OUT/'lookback';B.mkdir(exist_ok=True)
LENGTHS=[24,48,72,96];H=24
class Data:
 def __init__(self,task,L,split):
  self.task,self.L=task,L
  raw,features,ids,mean,std,manifest=raw_task(task);self.raw=raw;self.features=features;self.mean,self.std=mean,std
  self.values=torch.from_numpy(((raw[features].to_numpy(float)-mean)/std).astype('float32'))
  self.origins=[];self.bids=[]
  for bid in ids[split]:
   ix=np.flatnonzero(raw.batch_id.to_numpy()==bid);assert np.all(np.diff(ix)==1)
   for origin in range(int(ix[0])+max(LENGTHS),int(ix[-1])+1-H+1):
    self.origins.append(origin);self.bids.append(int(bid))
  assert self.origins
  self.mark=Windows(task,'train',24)[0][2].clone()
  self.modes={int(q.batch_id):str(q.mode) for q in manifest.itertuples()} if task=='CHO' else {}
  self.emb=None
  path=B/(task+'_L'+str(L)+'_observations.pt')
  if path.exists():
   d=torch.load(path,map_location='cpu');self.lookup={int(o):i for i,o in enumerate(d['origins'])};self.emb=d['embeddings'];assert set(self.origins)<=set(self.lookup)
 def __len__(self):return len(self.origins)
 def item(self,i):
  o=self.origins[i];x=self.values[o-self.L:o];y=self.values[o:o+H];m=self.mark.clone()
  if self.emb is not None:m[:,0]=self.emb[self.lookup[o]]
  return x,y,m
def prepare(task,L,device):
 from transformers import AutoTokenizer,AutoModel
 from preprocess_window_semantic import NAMES,top_lags
 from cho_data import NAMES as CN
 path=B/(task+'_L'+str(L)+'_observations.pt')
 if path.exists():return
 ds=[Data(task,L,s) for s in ['train','val','test']];base=ds[0];origins=sorted({o for d in ds for o in d.origins});raw=base.raw
 tok=AutoTokenizer.from_pretrained('FacebookAI/roberta-base',local_files_only=True);enc=AutoModel.from_pretrained('FacebookAI/roberta-base',local_files_only=True).to(device).eval().requires_grad_(False)
 embeds=[];texts=[]
 def flush():
  nonlocal texts
  if not texts:return
  p=tok(texts,padding=True,truncation=False,return_tensors='pt').to(device)
  assert p.input_ids.shape[1]<=320,('overlong observation text',p.input_ids.shape)
  with torch.no_grad():
   hidden=enc(**p).last_hidden_state;mask=p.attention_mask
   pooled=(hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
  embeds.append(pooled.half().cpu() if task!='CHO' else pooled.cpu());texts=[]
 for o in origins:
  window=raw.iloc[o-L:o];age=float(window.hh.iloc[0]);end=float(window.hh.iloc[-1]);bid=int(window.batch_id.iloc[0]);mode=base.modes.get(bid,'')
  for j,f in enumerate(base.features):
   v=window[f].to_numpy(float);delta=v[-1]-v[0];direction='upward' if delta>1e-8 else 'downward' if delta< -1e-8 else 'flat';recent=', '.join(f'{a:.3f}' for a in v[-12:])
   intro=(f'Data description: CHO-K1 {mode} cultivation, variable {CN[j]}. Task: forecast the next 24 hourly steps from the previous {L} observations. Input window culture age: {age:.1f} to {end:.1f} hours. ' if task=='CHO' else f'Data description: Erythromycin fermentation, variable {NAMES[f]} ({f}). Task: forecast the next 24 hourly steps from the previous {L} observations. Input window batch age: {age:.1f} to {end:.1f} hours. ')
   texts.append(intro+f'Input statistics: minimum {v.min():.3f}; maximum {v.max():.3f}; median {np.median(v):.3f}; mean {v.mean():.3f}; first {v[0]:.3f}; last {v[-1]:.3f}; trend {direction}, change {delta:+.3f}. Top 12 autocorrelation lags (lag:correlation): {top_lags(v)}. Last 12 observed values: {recent}.')
   if len(texts)==40:flush()
 flush();emb=torch.cat(embeds).reshape(len(origins),5,768).float();assert torch.isfinite(emb).all()
 tmp=Path(str(path)+'.tmp');torch.save(dict(origins=origins,embeddings=emb,lookback=L,horizon=24,features=base.features,recipe='Original frozen RoBERTa masked mean, causal window summary, original EFP half-storage/CHO FP32; no truncation; general static/context mechanism slots unchanged',original_data_sha256=sha(ROOT/'dataset'/task/(task+'.csv'))),tmp);tmp.replace(path)
 writecsv(B/(task+'_L'+str(L)+'_split_manifest.csv'),[dict(dataset=task,lookback=L,split=s,windows=len(d),batch_ids=';'.join(map(str,sorted(set(d.bids)))),shared_origin_sha256=hashlib.sha256(np.asarray(d.origins,dtype='int64').tobytes()).hexdigest()) for d,s in zip(ds,['train','val','test'])])
 del enc
 print('LOOKBACK_TEXT_READY',task,L,len(origins),flush=True)
def official_baseline(name,cfg):
 import importlib
 original_path=list(sys.path)
 saved={k:v for k,v in sys.modules.items() if k in ('models','layers') or k.startswith(('models.','layers.'))}
 for k in saved:del sys.modules[k]
 try:
  sys.path.insert(0,str(ROOT.parent/'Time-Series-Library'))
  model=importlib.import_module('models.'+name).Model(cfg)
 finally:
  for k in list(sys.modules):
   if k in ('models','layers') or k.startswith(('models.','layers.')):del sys.modules[k]
  sys.modules.update(saved);sys.path[:]=original_path
 return model
def run(task,L):
 torch.set_num_threads(2);guard();device='cuda:0';prepare(task,L,device)
 ds={s:Data(task,L,s) for s in ['train','val','test']};assert all(d.emb is not None for d in ds.values())
 for name in ['VarBridgeBackbone','VarBridge','DLinear','PatchTST','iTransformer']:
  stem=f'{task}_L{L}_{name}_seed2021';marker=B/(stem+'_COMPLETE')
  if marker.exists():continue
  seed_all(2021);cfg=make_config(task);cfg.seq_len=L;cfg.pred_len=cfg.token_len=24
  target=B/(stem+'.pth')
  if name=='VarBridge':
   cfg.numeric_ckpt=str(B/f'{task}_L{L}_VarBridgeBackbone_seed2021.pth')
   from models.KIBR_PatchTST_BoundMechanism import Model
   net=Model(cfg);epochs=100
  else:
   cfg.mix_embeds=False
   import importlib
   net=(importlib.import_module('models.PatchTST').Model(cfg) if name=='VarBridgeBackbone' else official_baseline(name,cfg));epochs=100 if name=='VarBridgeBackbone' else 10
  net=net.float().to(device);opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=1e-4);criterion=torch.nn.HuberLoss(delta=1.)
  def batch(split,ix):
   q=[ds[split].item(i) for i in ix];return tuple(torch.stack([v[k] for v in q]).to(device) for k in range(3))
  def forward(x,m):
   return net(x,m if name=='VarBridge' else None,None,m if name=='VarBridge' else None)[:,-24:]
  def evaluate(split):
   net.eval();se=ae=cnt=0
   with torch.no_grad():
    for a in range(0,len(ds[split]),64):
     x,y,m=batch(split,range(a,min(a+64,len(ds[split]))));p=forward(x,m);assert p.shape==y.shape and torch.isfinite(p).all();e=(p-y).double();se+=float(e.square().sum());ae+=float(e.abs().sum());cnt+=e.numel()
   return se/cnt,ae/cnt
  # Meaningful preflight before any optimization; original source remains untouched.
  x,y,m=batch('train',range(min(2,len(ds['train']))));p=forward(x,m);assert p.shape==y.shape and torch.isfinite(p).all();criterion(p,y).backward();opt.zero_grad(set_to_none=True)
  if name=='VarBridge':assert all(p.grad is None for p in net.patch_backbone.parameters())
  best=float('inf');bestep=0;history=[];baseline=name not in ['VarBridge','VarBridgeBackbone']
  for ep in range(1,epochs+1):
   guard();net.train();start=time.time()
   lr=(1e-4*.5*(1+math.cos(math.pi*(ep-1)/10)) if baseline else 1e-4/(1+math.exp(-.5*(ep-10)))-1e-4/(1+math.exp(-.05*(ep-100))))
   for g in opt.param_groups:g['lr']=lr
   order=torch.randperm(len(ds['train']),generator=torch.Generator().manual_seed(2021+ep)).tolist();total=0.
   for a in range(0,len(order),126):
    ix=order[a:a+126];opt.zero_grad(set_to_none=True)
    for k in range(0,len(ix),21):
     cur=ix[k:k+21];x,y,m=batch('train',cur);p=forward(x,m);loss=criterion(p,y);assert torch.isfinite(loss);(loss*len(cur)/len(ix)).backward();total+=float(loss.detach())*len(cur)
    opt.step()
   vm,va=evaluate('val')
   if vm<best:best=vm;bestep=ep;torch.save(net.state_dict(),target)
   row=dict(epoch=ep,training_huber=total/len(ds['train']),validation_mse=vm,validation_mae=va,lr=lr,seconds=time.time()-start)
   history.append(row);writecsv(B/(stem+'_learning.csv'),history);print('LOOKBACK_EPOCH',stem,json.dumps(row),flush=True)
  net.load_state_dict(torch.load(target,map_location='cpu'),strict=True);mse,mae=evaluate('test')
  result=dict(dataset=task,lookback=L,horizon=24,model=name,seed=2021,epochs=epochs,mse=mse,mae=mae,train_windows=len(ds['train']),validation_windows=len(ds['val']),test_windows=len(ds['test']),best_epoch=bestep,validation_mse=best,global_batch=126,microbatch=21,checkpoint=str(target),checkpoint_sha256=sha(target),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],resource_mode='memory_gated_shared_allowed_training_times_not_controlled')
  writecsv(B/(stem+'_result.csv'),[result]);marker.write_text(json.dumps(result,indent=2));print('LOOKBACK_FIT_COMPLETE',json.dumps(result),flush=True)
  del net,opt;torch.cuda.empty_cache()
def _plot():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rows=[]
 for p in B.glob('*_result.csv'):rows.extend(list(csv.DictReader(p.open())))
 if not rows:return
 writecsv(B/'results.csv',rows)
 r=pd.DataFrame(rows)
 for task,q in r.groupby('dataset'):
  fig,axs=plt.subplots(1,2,figsize=(9.7,3.4),layout='constrained')
  for name,color in [('VarBridge','#d45147'),('DLinear','#5785a6'),('PatchTST','#33988d'),('iTransformer','#947bb5')]:
   z=q[q.model==name].sort_values('lookback',key=lambda s:s.astype(int))
   for ax,metric in zip(axs,['mse','mae']):
    ax.plot(z.lookback.astype(int),z[metric].astype(float),marker='o',color=color,label=name);ax.set(xlabel='Lookback (hourly observations)',ylabel=metric.upper());ax.grid(alpha=.15)
  axs[0].legend(ncol=2,fontsize=9);fig.savefig(OUT/'figures'/('lookback_sensitivity_'+task+'.png'),dpi=360,bbox_inches='tight');plt.close(fig)
def plot():
 lock=(B/'plot.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
 try:_plot()
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
def pipeline():
 # Run only after context controls and matched one-GPU references drain.
 while not all((OUT/(t+'_own_target_COMPLETE')).exists() for t in ['EFP_long_sub2','CHO']):time.sleep(30)
 lock=(OUT/'.queue.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX)
 gl=(B/'queue.lock').open('w');fcntl.flock(gl,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def free():
  rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True).splitlines()
  busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],universal_newlines=True)
  return [(int(a),b.strip()) for a,b,c,d in [x.split(',') for x in rows] if int(a) in [0,1,3] and int(c)>=14000]
 while True:
  one=free();time.sleep(3);two=free();time.sleep(3);three=free();avail=[p for p in one if p in two and p in three]
  if avail:break
  time.sleep(30)
 avail=avail[:3];jobs=[(t,L) for L in LENGTHS for t in ['EFP_long_sub2','CHO']]
 print('LOOKBACK_GPUS',avail,flush=True)
 import concurrent.futures
 def slot(gpu,uuid,work):
  import shutil
  models=['VarBridgeBackbone','VarBridge','DLinear','PatchTST','iTransformer']
  for task,L in work:
   failures=0
   while not all((B/(f'{task}_L{L}_{m}_seed2021_COMPLETE')).exists() for m in models):
    while (gpu,uuid) not in free():time.sleep(30)
    # Retain failed-stage artifacts before retry; completed stages are immutable.
    failed=[m for m in models if not (B/f'{task}_L{L}_{m}_seed2021_COMPLETE').exists()]
    leftovers=[p for m in failed for p in B.glob(f'{task}_L{L}_{m}_seed2021*') if p.is_file()]
    if leftovers:
     archive=B/'technical_history'/f'{task}_L{L}_{time.time_ns()}';archive.mkdir(parents=True)
     for p in leftovers:shutil.move(str(p),str(archive/p.name))
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
    log=B/(f'{task}_L{L}.log')
    pressure=False
    with log.open('a') as f:
     p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--dataset',task,'--lookback',str(L)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     print('LOOKBACK_START',task,L,gpu,p.pid,flush=True)
     while p.poll() is None:
      time.sleep(20)
      if any(int(v.split(',')[2])<6000 for v in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],universal_newlines=True).splitlines() if int(v.split(',')[0])==gpu):
       pressure=True
       os.killpg(p.pid,signal.SIGTERM)
       try:p.wait(timeout=30)
       except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
       break
     code=p.wait()
    if code!=0:
     failures+=1
     print('LOOKBACK_TECHNICAL_RETRY',task,L,code,'memory_pressure',pressure,'attempt',failures,flush=True)
     if failures>=3 and not pressure:raise RuntimeError('Repeated technical failure; inspect '+str(log))
     time.sleep(30)
    else:plot()
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(avail)) as ex:
  for f in [ex.submit(slot,g,u,jobs[i::len(avail)]) for i,(g,u) in enumerate(avail)]:f.result()
 plot();guard();(B/'ALL_LOOKBACK_COMPLETE').write_text('40 unique fits: sixteen VarBridge stage fits100epochs, twenty-four baselines10epochs; shared origins and source guard.\n')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['run','pipeline','plot']);ap.add_argument('--dataset');ap.add_argument('--lookback',type=int,choices=LENGTHS);a=ap.parse_args()
 if a.mode=='run':
  assert (B/'CPU_PREFLIGHT_COMPLETE').exists()
  run(a.dataset,a.lookback)
 elif a.mode=='pipeline':pipeline()
 else:plot()

