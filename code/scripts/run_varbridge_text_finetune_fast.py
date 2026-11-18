"""FP32-equivalent text batching, direct numerical gathers, full-state recovery."""
import os,sys,time,json,math,hashlib,random,types,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import torch,numpy as np
from torch.utils.checkpoint import checkpoint
import run_varbridge_text_finetune as e
R=e.R;CK=e.CK;D=R/'acceleration_diagnostics';D.mkdir(exist_ok=True);CHUNK=64
class FastOnline(e.Online):
 def encode(self,z):
  device=next(self.encoder.parameters()).device;result=[]
  for off in range(0,len(z['input_ids']),CHUNK):
   q={k:v[off:off+CHUNK].to(device,dtype=torch.long) for k,v in z.items()};length=int(q['attention_mask'].sum(1).max());q={k:v[:,:length] for k,v in q.items()}
   def fn(input_ids,attention_mask):
    h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state;return (h*attention_mask[:,:,None]).sum(1)/attention_mask.sum(1)[:,None]
   value=checkpoint(fn,q['input_ids'],q['attention_mask'],use_reentrant=False) if self.mode!='frozen' and torch.is_grad_enabled() else fn(q['input_ids'],q['attention_mask']);result.append(value)
  return torch.cat(result)
def numeric(ds,ids,device):
 starts=torch.tensor([int(e.startids(ds)[i]) for i in ids]);values=ds.base.data if ds.dataset!='CHO' else ds.base.values;x=values[starts[:,None]+torch.arange(48)[None]];y=values[starts[:,None]+48+torch.arange(24)[None]];return x.to(device),y.to(device),None
def sha_tensor(q):return hashlib.sha256(q.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def guard():
 e.guard()
 for p,h in json.loads((D/'fast_source_hash.json').read_text()).items():assert e.sha(p)==h,p

def benchmark():
 e.setup();guard();torch.cuda.set_per_process_memory_fraction(.60);ds=e.Windows('EFP_long_sub2','train',24);part=torch.load(R/'EFP_long_sub2_tokens.pt',map_location='cpu')['train'];n=e.Online('EFP_long_sub2','layernorm').cuda();n.load_state_dict(torch.load(CK/'EFP_long_sub2_layernorm.pth',map_location='cpu'));n.train();orig=n.encode;ids=torch.randperm(len(ds),generator=torch.Generator().manual_seed(2023)).tolist()[:126]
 for cur in [ids[:21],ids[-21:]]:
  x,y,_=e.numeric(ds,cur,'cpu');a,b,_=numeric(ds,cur,'cpu');assert torch.equal(x,a) and torch.equal(y,b)
 def run(fast,global_batch=True):
  n.encode=types.MethodType(FastOnline.encode,n) if fast else orig;n.zero_grad(set_to_none=True);e.seed_all(777);pred=[];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter();cohort=ids if global_batch else ids[:21]
  for off in range(0,len(cohort),21):
   cur=cohort[off:off+21];x,y,_=(numeric if fast else e.numeric)(ds,cur,'cuda');z=e.tokenbatch(part,cur);p=n(x,z);loss=torch.nn.functional.huber_loss(p,y)*(len(cur)/len(cohort));loss.backward();pred.append(p.detach().cpu())
  torch.cuda.synchronize();elapsed=time.perf_counter()-started;g={k:q.grad.detach().cpu().clone() for k,q in n.named_parameters() if q.grad is not None};return torch.cat(pred),g,elapsed,torch.cuda.max_memory_allocated()/2**20
 # Warm both execution paths before any timed comparison.
 run(False,False);run(True,False);rows=[]
 for repeat in range(2):
  a,ga,t,mem=run(False);b,gb,u,peak=run(True);x=torch.cat([ga[k].flatten().double() for k in ga]);y=torch.cat([gb[k].flatten().double() for k in ga]);rel=float((x-y).norm()/x.norm().clamp_min(1e-20));cos=float(torch.nn.functional.cosine_similarity(x,y,dim=0));err=float((a-b).abs().max());row={'repeat':repeat,'baseline_global126_seconds':t,'accelerated_global126_seconds':u,'speedup':t/u,'prediction_max_abs':err,'gradient_relative_l2':rel,'gradient_cosine':cos,'peak_MiB':peak};row['equivalence_pass']=err<1e-4 and rel<.005 and cos>.9999;assert row['equivalence_pass'];rows.append(row);e.savej(D/'global_batch_benchmark.json',rows);print('GLOBAL_BATCH_BENCHMARK',json.dumps(row),flush=True)
 # Independently check the position-embedding gradient path using the same chosen chunk64.
 for name,p in n.encoder.named_parameters():p.requires_grad_('LayerNorm' in name or name=='embeddings.position_embeddings.weight')
 n.mode='layernorm_position';a,ga,t,mem=run(False,False);b,gb,u,peak=run(True,False);keys=[k for k in ga if k.startswith('encoder.')];x=torch.cat([ga[k].flatten().double() for k in keys]);y=torch.cat([gb[k].flatten().double() for k in keys]);rel=float((x-y).norm()/x.norm().clamp_min(1e-20));err=float((a-b).abs().max());assert err<1e-4 and rel<.005;e.savej(D/'ACCELERATION_PREFLIGHT_PASS.json',{'global126':rows,'position_gradient_relative_l2':rel,'position_prediction_max_abs':err,'numeric_inputs_bitwise_identical':True,'FP32':True,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'chunk':CHUNK,'microbatch':21,'global_batch':126,'source_sha256':e.sha(__file__)});print('ALL_ACCELERATION_PREFLIGHT_COMPLETE',flush=True)

def train(task,mode):
 e.setup();guard();assert task=='EFP_long_sub2' and mode in ['layernorm','layernorm_position'];assert (D/'ACCELERATION_PREFLIGHT_PASS.json').exists();datasets=e.data(task);parts=torch.load(R/(task+'_tokens.pt'),map_location='cpu');n=FastOnline(task,mode).cuda();e.seed_all(2021);opt=torch.optim.Adam([q for q in n.parameters() if q.requires_grad],lr=1e-4);best=float('inf');bestep=0;hist=[];start_epoch=1;cp=CK/(task+'_'+mode+'.pth');latest=CK/(task+'_'+mode+'_latest_resume.pth');frozen={k:sha_tensor(v) for k,v in n.forecast.patch_backbone.state_dict().items()};fixed={k:sha_tensor(v) for k,v in n.encoder.named_parameters() if not v.requires_grad}
 if latest.exists():
  z=torch.load(latest,map_location='cpu');assert z['task']==task and z['mode']==mode and z['fast_source_sha256']==e.sha(__file__);n.load_state_dict(z['model']);opt.load_state_dict(z['optimizer']);hist=z['history'];best=z['best'];bestep=z['best_epoch'];start_epoch=z['epoch']+1;torch.set_rng_state(z['torch_rng']);torch.cuda.set_rng_state_all(z['cuda_rng']);random.setstate(z['python_rng']);np.random.set_state(z['numpy_rng'])
 def evaluate(ds,split,save=False):
  n.eval();se=ae=count=0;rows=[]
  with torch.no_grad():
   for off in range(0,len(ds),21):
    ids=list(range(off,min(off+21,len(ds))));x,y,_=numeric(ds,ids,'cuda');p=n(x,e.tokenbatch(parts[split],ids));d=(p-y).double();assert torch.isfinite(d).all();se+=float(d.square().sum());ae+=float(d.abs().sum());count+=d.numel()
    if save:rows.extend({'index':i,'mse':float(d[j].square().mean()),'mae':float(d[j].abs().mean())} for j,i in enumerate(ids))
  if save:e.csvout(R/(task+'_'+mode+'_per_window.csv'),rows)
  return se/count,ae/count
 print('FAST_TRAIN_START',task,mode,'start_epoch',start_epoch,'seed2021_global126_micro21_FP32',flush=True)
 replay=json.loads((D/'replay_reference.json').read_text())[mode]
 for ep in range(start_epoch,101):
  guard();n.train();started=time.time();lr=1e-4*(1/(1+math.exp(-.5*(ep-10)))-1/(1+math.exp(-.05*(ep-100))))
  for pg in opt.param_groups:pg['lr']=lr
  ids=torch.randperm(len(datasets[0]),generator=torch.Generator().manual_seed(2021+ep)).tolist();total=0
  for off in range(0,len(ids),126):
   glob=ids[off:off+126];opt.zero_grad(set_to_none=True)
   for k in range(0,len(glob),21):
    cur=glob[k:k+21];x,y,_=numeric(datasets[0],cur,'cuda');p=n(x,e.tokenbatch(parts['train'],cur));loss=torch.nn.functional.huber_loss(p,y);assert torch.isfinite(loss);(loss*len(cur)/len(glob)).backward();total+=float(loss.detach())*len(cur)
   assert all(torch.isfinite(q.grad).all() for q in n.parameters() if q.grad is not None);opt.step()
   if off==0:print('FIRST_GLOBAL_BATCH',ep,time.time()-started,'peak_MiB',torch.cuda.max_memory_allocated()/2**20,flush=True)
  vm,va=evaluate(datasets[1],'val');row={'epoch':ep,'train_huber':total/len(ids),'validation_mse':vm,'validation_mae':va,'lr':lr,'seconds':time.time()-started};hist.append(row)
  if vm<best:best=vm;bestep=ep;tmp=cp.with_suffix('.tmp');torch.save(n.state_dict(),tmp);tmp.replace(cp)
  if ep==replay['best_epoch']:
   old=torch.load(replay['checkpoint'],map_location='cpu');deltas={k:float((v.detach().cpu()-old[k]).abs().max()) for k,v in n.state_dict().items() if v.dtype.is_floating_point};worst=max(deltas.values());assert worst<1e-3,'Replay diverges beyond predefined implementation tolerance';e.savej(D/(mode+'_replay_audit.json'),{'epoch':ep,'old_completed_epochs':replay['completed_epochs'],'old_best_epoch':replay['best_epoch'],'max_weight_absolute_difference':worst,'old_checkpoint':replay['checkpoint'],'old_optimizer_state_available':False,'new_optimizer_reconstructed_by_seed_and_data_order_replay':True});print('REPLAY_AUDIT',mode,ep,worst,flush=True)
  state={'task':task,'mode':mode,'epoch':ep,'model':n.state_dict(),'optimizer':opt.state_dict(),'history':hist,'best':best,'best_epoch':bestep,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'fast_source_sha256':e.sha(__file__)};tmp=latest.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(latest);e.csvout(R/(task+'_'+mode+'_learning.csv'),hist);print('EPOCH',task,mode,json.dumps(row),flush=True)
 assert all(sha_tensor(v)==frozen[k] for k,v in n.forecast.patch_backbone.state_dict().items());assert all(sha_tensor(dict(n.encoder.named_parameters())[k])==h for k,h in fixed.items());n.load_state_dict(torch.load(cp,map_location='cpu'));mse,mae=evaluate(datasets[2],'test',True);n.eval();rebuilt={}
 with torch.no_grad():
  for split,ds in zip(['train','val','test'],datasets):
   marks=[]
   for off in range(0,len(ds),21):marks.append(n.mark(e.tokenbatch(parts[split],list(range(off,min(off+21,len(ds)))))).cpu())
   rebuilt[split]={'starts':parts[split]['starts'],'marks':torch.cat(marks)}
 torch.save(rebuilt,R/(task+'_'+mode+'_final_text_cache.pt'));out={'task':task,'mode':mode,'epochs':100,'seed':2021,'lookback':48,'horizon':24,'global_batch':126,'microbatch':21,'precision':'FP32','text_chunk':CHUNK,'mse':mse,'mae':mae,'test_windows':len(datasets[2]),'validation_mse':best,'best_epoch':bestep,'checkpoint':str(cp),'checkpoint_sha256':e.sha(cp),'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'encoder_trainable':sum(q.numel() for q in n.encoder.parameters() if q.requires_grad),'implementation_sha256':e.sha(__file__),'original_implementation_sha256':e.sha(e.__file__),'acceleration_preflight':str(D/'ACCELERATION_PREFLIGHT_PASS.json')};guard();e.savej(R/(task+'_'+mode+'_COMPLETE.json'),out);print('TEXT_FINETUNE_COMPLETE',json.dumps(out),flush=True)
if __name__=='__main__':
 e.setup()
 if sys.argv[-1]=='profile':benchmark()
 else:train(sys.argv[2],sys.argv[3])
