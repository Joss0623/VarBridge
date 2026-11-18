"""Staged profiling, shared-static VJP, compact exact-recovery global checkpoints."""
import os,sys,time,json,math,copy,random,hashlib,csv,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import torch,numpy as np
import run_varbridge_text_finetune as e
import run_varbridge_text_finetune_fast as v1
R=e.R;CK=e.CK;D=R/'acceleration_diagnostics';CHUNK=210
class Online(v1.FastOnline):
 def __init__(self,task,mode):
  super().__init__(task,mode);self.active_static=None;self.active_window=None;self.fast_static=False;self.use_layer_checkpoint=False
  self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  size=self.static_tokens['input_ids'].shape[1];both=torch.cat([self.static_tokens['input_ids'],self.static_tokens['attention_mask']],1);unique,inverse=torch.unique(both,dim=0,return_inverse=True);self.unique_static={'input_ids':unique[:,:size].contiguous(),'attention_mask':unique[:,size:].contiguous()};self.static_inverse=inverse
 def train(self,mode=True):
  super().train(mode)
  # HF's layer checkpointing needs this container flag. All dropout children remain eval.
  self.encoder.encoder.training=bool(mode and self.use_layer_checkpoint)
  return self
 def encode(self,z):
  if not self.use_layer_checkpoint:return v1.FastOnline.encode(self,z)
  device=next(self.encoder.parameters()).device;chunks=[]
  for off in range(0,len(z['input_ids']),CHUNK):
   q={k:v[off:off+CHUNK].to(device,dtype=torch.long) for k,v in z.items()};length=int(q['attention_mask'].sum(1).max());q={k:v[:,:length] for k,v in q.items()};h=self.encoder(**q).last_hidden_state;mask=q['attention_mask'];chunks.append((h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None])
  return torch.cat(chunks)
 def window_bank(self,z):
  w=self.encode(z).reshape(-1,5,768);return w.half().float() if self.task!='CHO' else w
 def static_bank(self):
  k=self.encode(self.unique_static).index_select(0,self.static_inverse.to(next(self.parameters()).device))
  if self.task=='CHO':return k.reshape(5,3,768)
  k=k.half().float().reshape(5,5,3,768);k=k-k.mean((0,1),keepdim=True);return (k/k.norm(dim=-1,keepdim=True).clamp_min(1e-6)*768**.5).half().float()[0]
 def mark(self,z):
  if not self.fast_static:return e.Online.mark(self,z)
  w=self.active_window if self.active_window is not None else self.window_bank(z);k=self.active_static if self.active_static is not None else self.static_bank();b=len(w);return torch.cat([w[:,:,None],k[:,0][None,None].expand(b,5,5,768),k[:,1:][None].expand(b,-1,-1,-1)],2)

def guard():
 v1.guard()
 for p,h in json.loads((D/'v3_source_hash.json').read_text()).items():assert e.sha(p)==h,p

def one_step(n,opt,ds,part,ids,fast,profile=True):
 n.fast_static=fast;n.use_layer_checkpoint=fast;n.active_static=None;n.active_window=None;n.train();opt.zero_grad(set_to_none=True);phase={k:0. for k in ['data','static_forward','window_forward','forward','backward','encoder_backward','finite_check','optimizer']};gpu=[]
 def measure(k,fn):
  if profile:a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
  s=time.perf_counter();out=fn();phase[k]+=time.perf_counter()-s
  if profile:b.record();gpu.append((k,a,b))
  return out
 torch.cuda.synchronize();start=time.perf_counter();losses=[];predictions=[]
 if fast:
  all_tokens=measure('data',lambda:e.tokenbatch(part,ids));bank=measure('static_forward',n.static_bank);leaf=bank.detach().requires_grad_(True);n.active_static=leaf;observed=measure('window_forward',lambda:n.window_bank(all_tokens));window_leaf=observed.detach().requires_grad_(True)
 for off in range(0,len(ids),21):
  cur=ids[off:off+21]
  def fetch():
   x,y,_=v1.numeric(ds,cur,'cuda');return x,y,None if fast else e.tokenbatch(part,cur)
  x,y,z=measure('data',fetch)
  if fast:n.active_window=window_leaf[off:off+len(cur)]
  p=measure('forward',lambda:n(x,z));loss=torch.nn.functional.huber_loss(p,y);weighted=loss*(len(cur)/len(ids));losses.append(loss.detach().double()*len(cur))
  if not fast:assert torch.isfinite(loss);scalar=float(loss.detach())
  measure('backward',weighted.backward)
  if profile:predictions.append(p.detach().cpu())
 if fast:
  assert leaf.grad is not None and window_leaf.grad is not None;measure('encoder_backward',lambda:torch.autograd.backward([bank,observed],[leaf.grad,window_leaf.grad]));n.active_static=None;n.active_window=None
 def finite():
  gs=[q.grad for q in n.parameters() if q.grad is not None]
  if fast:assert torch.isfinite(torch.cat([q.reshape(-1) for q in gs])).all() and torch.isfinite(torch.stack(losses)).all()
  else:assert all(torch.isfinite(q).all() for q in gs)
 measure('finite_check',finite);measure('optimizer',opt.step);total=float(torch.stack(losses).sum());torch.cuda.synchronize();elapsed=time.perf_counter()-start;gpu_ms={k:0. for k in phase}
 if profile:
  for k,a,b in gpu:gpu_ms[k]+=a.elapsed_time(b)
 return {'seconds':elapsed,'cpu_phase_seconds':phase,'cuda_phase_ms':gpu_ms,'huber_sum':total,'prediction':torch.cat(predictions) if predictions else None}

def benchmark():
 e.setup();guard();torch.cuda.set_per_process_memory_fraction(.64);ds=e.Windows('EFP_long_sub2','train',24);part=torch.load(R/'EFP_long_sub2_tokens.pt',map_location='cpu')['train'];ids=torch.randperm(len(ds),generator=torch.Generator().manual_seed(2023)).tolist()[:126];rows=[]
 for mode in ['layernorm','layernorm_position']:
  cp=CK/('EFP_long_sub2_'+mode+'_latest_resume.pth');state=torch.load(cp,map_location='cpu');assert state['epoch']>=1 and len(state['history'])==state['epoch'] and state['optimizer']['state'];n=Online('EFP_long_sub2',mode).cuda();n.load_state_dict(state['model']);opt=torch.optim.Adam([p for p in n.parameters() if p.requires_grad],lr=1e-4);mutable={k:v.detach().cpu().clone() for k,v in n.named_parameters() if v.requires_grad}
  def reset():
   with torch.no_grad():
    for k,v in n.named_parameters():
     if k in mutable:v.copy_(mutable[k])
   opt.load_state_dict(copy.deepcopy(state['optimizer']));torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);n.zero_grad(set_to_none=True)
  # Warm both paths without retaining any resulting update.
  reset();one_step(n,opt,ds,part,ids[:21],False,False);reset();one_step(n,opt,ds,part,ids[:21],True,False)
  reset();a=one_step(n,opt,ds,part,ids,False);ga={k:p.grad.detach().cpu().clone() for k,p in n.named_parameters() if p.grad is not None};wa={k:p.detach().cpu().clone() for k,p in n.named_parameters() if p.requires_grad}
  reset();b=one_step(n,opt,ds,part,ids,True);gb={k:p.grad.detach().cpu().clone() for k,p in n.named_parameters() if p.grad is not None};wb={k:p.detach().cpu().clone() for k,p in n.named_parameters() if p.requires_grad};g=torch.cat([ga[k].flatten().double() for k in ga]);h=torch.cat([gb[k].flatten().double() for k in ga]);relative=float((g-h).norm()/g.norm().clamp_min(1e-20));err=float((a.pop('prediction')-b.pop('prediction')).abs().max());weightmax=max(float((wa[k]-wb[k]).abs().max()) for k in wa);row={'mode':mode,'baseline':a,'optimized':b,'speedup':a['seconds']/b['seconds'],'prediction_max_abs':err,'gradient_relative_l2':relative,'post_Adam_parameter_max_abs':weightmax,'static_texts_before':len(n.static_tokens['input_ids']),'static_texts_after_dedup':len(n.unique_static['input_ids']),'checkpoint_epoch':state['epoch']};row['equivalence_pass']=err<1e-4 and relative<.005 and weightmax<1e-4;assert row['equivalence_pass'];rows.append(row);e.savej(D/'v3_staged_benchmark.json',rows);print('STAGED_BENCHMARK',json.dumps(row),flush=True);del n,opt,state;torch.cuda.empty_cache()
 e.savej(D/'V3_PREFLIGHT_PASS.json',{'rows':rows,'full_optimizer_step_compared':True,'same_rng_and_optimizer_state':True,'all_pass':True,'source_sha256':e.sha(__file__)});print('V3_STAGED_PREFLIGHT_COMPLETE',flush=True)

def train(task,mode):
 e.setup();guard();torch.cuda.set_per_process_memory_fraction(.64);assert task=='EFP_long_sub2';assert (D/'V3_PREFLIGHT_PASS.json').exists();datasets=e.data(task);parts=torch.load(R/(task+'_tokens.pt'),map_location='cpu');n=Online(task,mode).cuda();opt=torch.optim.Adam([q for q in n.parameters() if q.requires_grad],lr=1e-4);mutable_ptrs={p.data_ptr() for p in n.parameters() if p.requires_grad};mutable_names={k for k,v in n.state_dict().items() if v.data_ptr() in mutable_ptrs};frozen_path=CK/(task+'_'+mode+'_frozen_components_v3.pth');latest=CK/(task+'_'+mode+'_latest_resume_v3.pth');cp=CK/(task+'_'+mode+'.pth');start_ep=1;offset=0;total_start=0.;hist=[];best=float('inf');bestep=0
 if latest.exists():
  z=torch.load(latest,map_location='cpu');assert z['source_sha256']==e.sha(__file__) and e.sha(frozen_path)==z['frozen_sha256'];fixed=torch.load(frozen_path,map_location='cpu');combined=dict(fixed,**z['model_mutable']);assert set(combined)==set(n.state_dict());n.load_state_dict(combined);opt.load_state_dict(z['optimizer']);hist=z['history'];best=z['best'];bestep=z['best_epoch'];start_ep=z['epoch']+1 if z['epoch_complete'] else z['epoch'];offset=0 if z['epoch_complete'] else z['next_offset'];total_start=0. if z['epoch_complete'] else z['running_huber_total'];del combined,fixed
 else:
  z=torch.load(CK/(task+'_'+mode+'_latest_resume.pth'),map_location='cpu');assert z['task']==task and z['mode']==mode and z['fast_source_sha256']==e.sha(v1.__file__);n.load_state_dict(z['model']);opt.load_state_dict(z['optimizer']);hist=z['history'];best=z['best'];bestep=z['best_epoch'];start_ep=z['epoch']+1;tmp=frozen_path.with_suffix('.tmp');torch.save({k:v.cpu() for k,v in n.state_dict().items() if k not in mutable_names},tmp);tmp.replace(frozen_path)
 torch.set_rng_state(z['torch_rng']);torch.cuda.set_rng_state_all(z['cuda_rng']);random.setstate(z['python_rng']);np.random.set_state(z['numpy_rng']);del z;frozen_sha=e.sha(frozen_path);frozen_numeric={k:e.sha_tensor(v) for k,v in n.forecast.patch_backbone.state_dict().items()} if hasattr(e,'sha_tensor') else {k:v1.sha_tensor(v) for k,v in n.forecast.patch_backbone.state_dict().items()};print('V3_RESUME',mode,'epoch',start_ep,'offset',offset,'completed_epochs',len(hist),flush=True)
 def snapshot(ep,done,next_offset,total):
  q={'epoch':ep,'epoch_complete':done,'next_offset':next_offset,'running_huber_total':total,'model_mutable':{k:v for k,v in n.state_dict().items() if k in mutable_names},'frozen_sha256':frozen_sha,'frozen_path':str(frozen_path),'optimizer':opt.state_dict(),'history':hist,'best':best,'best_epoch':bestep,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'source_sha256':e.sha(__file__)};tmp=latest.with_suffix('.tmp');torch.save(q,tmp);tmp.replace(latest)
 def evaluate(ds,split,save=False):
  n.eval();se=ae=count=0;rows=[]
  with torch.no_grad():
   n.active_static=n.static_bank()
   for off in range(0,len(ds),21):
    ids=list(range(off,min(off+21,len(ds))));x,y,_=v1.numeric(ds,ids,'cuda');p=n(x,e.tokenbatch(parts[split],ids));d=(p-y).double();assert torch.isfinite(d).all();se+=float(d.square().sum());ae+=float(d.abs().sum());count+=d.numel()
    if save:rows.extend({'index':i,'mse':float(d[j].square().mean()),'mae':float(d[j].abs().mean())} for j,i in enumerate(ids))
   n.active_static=None
  if save:e.csvout(R/(task+'_'+mode+'_per_window.csv'),rows)
  return se/count,ae/count
 for ep in range(start_ep,101):
  guard();n.train();started=time.perf_counter();lr=1e-4*(1/(1+math.exp(-.5*(ep-10)))-1/(1+math.exp(-.05*(ep-100))))
  for pg in opt.param_groups:pg['lr']=lr
  ids=torch.randperm(len(datasets[0]),generator=torch.Generator().manual_seed(2021+ep)).tolist();total=total_start if ep==start_ep else 0.;start_off=offset if ep==start_ep else 0
  for off in range(start_off,len(ids),126):
   cur=ids[off:off+126];q=one_step(n,opt,datasets[0],parts['train'],cur,True,False);total+=q['huber_sum'];heartbeat={'epoch':ep,'global_batch_index':off//126+1,'global_batches_this_epoch':math.ceil(len(ids)/126),'samples_seen':off+len(cur),'batch_seconds':q['seconds'],'cpu_phase_seconds':q['cpu_phase_seconds'],'loss_huber':q['huber_sum']/len(cur),'checkpoint':str(latest),'time':time.time()};e.savej(R/(task+'_'+mode+'_batch_progress.json'),heartbeat)
   if off//126%10==0:print('BATCH',mode,json.dumps(heartbeat),flush=True)
   if (off//126+1)%10==0:snapshot(ep,False,off+len(cur),total)
  vm,va=evaluate(datasets[1],'val');row={'epoch':ep,'train_huber':total/len(ids),'validation_mse':vm,'validation_mae':va,'lr':lr,'seconds':time.perf_counter()-started};hist.append(row)
  if vm<best:best=vm;bestep=ep;tmp=cp.with_suffix('.tmp');torch.save(n.state_dict(),tmp);tmp.replace(cp)
  snapshot(ep,True,0,0.);e.csvout(R/(task+'_'+mode+'_learning.csv'),hist);print('EPOCH',task,mode,json.dumps(row),flush=True)
 assert all(v1.sha_tensor(v)==frozen_numeric[k] for k,v in n.forecast.patch_backbone.state_dict().items());n.load_state_dict(torch.load(cp,map_location='cpu'));mse,mae=evaluate(datasets[2],'test',True);n.eval();rebuilt={}
 with torch.no_grad():
  n.active_static=n.static_bank()
  for split,ds in zip(['train','val','test'],datasets):
   marks=[]
   for off in range(0,len(ds),21):marks.append(n.mark(e.tokenbatch(parts[split],list(range(off,min(off+21,len(ds)))))).cpu())
   rebuilt[split]={'starts':parts[split]['starts'],'marks':torch.cat(marks)}
  n.active_static=None
 torch.save(rebuilt,R/(task+'_'+mode+'_final_text_cache.pt'));out={'task':task,'mode':mode,'epochs':100,'seed':2021,'lookback':48,'horizon':24,'global_batch':126,'microbatch':21,'precision':'FP32','mse':mse,'mae':mae,'test_windows':len(datasets[2]),'validation_mse':best,'best_epoch':bestep,'checkpoint':str(cp),'checkpoint_sha256':e.sha(cp),'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'implementation_sha256':e.sha(__file__),'encoder_trainable':sum(p.numel() for p in n.encoder.parameters() if p.requires_grad),'v2_preflight':str(D/'V3_PREFLIGHT_PASS.json')};guard();e.savej(R/(task+'_'+mode+'_COMPLETE.json'),out);print('TEXT_FINETUNE_COMPLETE',json.dumps(out),flush=True)
if __name__=='__main__':
 e.setup()
 if sys.argv[-1]=='profile':benchmark()
 else:train(sys.argv[2],sys.argv[3])
