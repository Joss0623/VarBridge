"""Isolated four-condition semantic-draft experiment, seed2021/100 epochs."""
import os,sys,time,json,math,csv,hashlib,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
R=ROOT/'test_results/varbridge_semantic_refinement';CK=ROOT/'checkpoints/varbridge_semantic_refinement';R.mkdir(exist_ok=True);CK.mkdir(exist_ok=True);(R/'logs').mkdir(exist_ok=True)
import torch
from common_bound_supplement import make_config,seed_all
from common_bound_ablations import Windows
from models.VarBridge_SemanticRefinement import Model
TASKS=['EFP_long_sub2','CHO'];MODES=['original','llm_static','llm_repair','neutral_repair']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,z):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(z,indent=2));t.replace(p)
def csvout(p,rows):
 t=Path(str(p)+'.tmp')
 with t.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 t.replace(p)
def guard():
 for name in ['protected_hashes.json','implementation_hashes.json']:
  for p,h in json.loads((R/name).read_text()).items():assert sha(p)==h,p

def setup():
 torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;seed_all(2021)
def make(task,mode):
 seed_all(2021);cache=torch.load(R/(task+'_semantics.pt'),map_location='cpu') if mode!='original' else None
 return Model(make_config(task),mode,None if cache is None else cache['neutral' if mode=='neutral_repair' else 'llm']).float()
def data(task):return [Windows(task,s,24) for s in ['train','val','test']]
def batch(ds,ids,device):
 q=[ds[i] for i in ids];return tuple(torch.stack([p[k] for p in q]).float().to(device) for k in range(3))
def preflight():
 setup();guard();report=[]
 from models.KIBR_PatchTST_BoundMechanism import Model as Original
 for task in TASKS:
  tr,va,te=data(task);assert len(te)==(381 if task=='CHO' else 4431);a=make(task,'original').eval();seed_all(2021);b=Original(make_config(task)).float().eval()
  for split,ds in zip(['train','val','test'],[tr,va,te]):
   x,y,m=batch(ds,[0,len(ds)-1],'cpu')
   with torch.no_grad():assert torch.equal(a(x,m),b(x,m,None,m))
  shared={k:v.clone() for k,v in a.model.state_dict().items()};del a,b
  repair_initial=None
  for mode in MODES:
   n=make(task,mode);assert all(torch.equal(n.model.state_dict()[k],v) for k,v in shared.items());n.train();x,y,m=batch(tr,[0,1],'cpu');p=n(x,m);assert p.shape==y.shape and torch.isfinite(p).all();loss=torch.nn.functional.huber_loss(p,y);loss.backward();assert all(torch.isfinite(q.grad).all() for q in n.parameters() if q.grad is not None);assert all(q.grad is None for q in n.model.patch_backbone.parameters()) and not n.model.patch_backbone.training
   if 'repair' in mode:
    params=dict(n.repair.named_parameters());assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in params.values());assert all(params[k].grad.abs().sum()>0 for k in ['delta.2.weight','gate.2.weight','attention.in_proj_weight'])
    init={k:v.detach().clone() for k,v in n.repair.state_dict().items()}
    if mode=='llm_repair':repair_initial=init
    else:assert all(torch.equal(v,repair_initial[k]) for k,v in init.items()),'Capacity-matched controls must have identical initialization'
    n.eval()
    with torch.no_grad():
     n(x,m);before=n.repair.last_refined.clone();z=x.clone();z[:,:,0]+=torch.sin(torch.arange(48).float())[None];n(z,m);after=n.repair.last_refined
    assert not torch.allclose(before,after,atol=1e-7),'Observation-conditioned correction is inactive'
   report.append({'task':task,'mode':mode,'train_windows':len(tr),'validation_windows':len(va),'test_windows':len(te),'parameters':sum(q.numel() for q in n.parameters()),'trainable_parameters':sum(q.numel() for q in n.parameters() if q.requires_grad),'finite_forward_backward':True,'shared_original_initialization':True,'frozen_numeric_backbone':True});del n
 guard();save(R/'PREFLIGHT_PASS.json',report);print('ALL_SEMANTIC_REFINEMENT_PREFLIGHT_COMPLETE',flush=True)
def train(task,mode):
 setup();guard();assert (R/'PREFLIGHT_PASS.json').exists();assert mode!='original';tr,va,te=data(task);n=make(task,mode).cuda();seed_all(2021);opt=torch.optim.Adam([p for p in n.parameters() if p.requires_grad],lr=1e-4);hist=[];best=float('inf');bestep=0;cp=CK/(task+'_'+mode+'_seed2021.pth');latest=CK/(task+'_'+mode+'_latest_resume.pth');backbone={k:v.detach().cpu().clone() for k,v in n.model.patch_backbone.state_dict().items()};resume=False;start_epoch=1
 if latest.exists():
  state=torch.load(latest,map_location='cpu');assert state['task']==task and state['mode']==mode and state['implementation_hashes']==json.loads((R/'implementation_hashes.json').read_text());n.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);hist=state['history'];best=state['best'];bestep=state['best_epoch'];start_epoch=state['epoch']+1;torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);random.setstate(state['python_rng']);import numpy as np;np.random.set_state(state['numpy_rng']);resume=True
 def evaluate(ds,save_rows=False):
  n.eval();se=ae=count=0;rows=[];diagnostics=[]
  with torch.no_grad():
   for off in range(0,len(ds),64):
    ids=list(range(off,min(off+64,len(ds))));x,y,m=batch(ds,ids,'cuda');p=n(x,m);d=(p-y).double();assert torch.isfinite(d).all();se+=float(d.square().sum());ae+=float(d.abs().sum());count+=d.numel()
    if save_rows:
     rows.extend({'index':i,'mse':float(d[j].square().mean()),'mae':float(d[j].abs().mean())} for j,i in enumerate(ids))
     if hasattr(n,'repair'):
      diagnostics.extend({'index':i,'mean_gate':float(n.repair.last_gate[j].mean()),'relative_correction':float(n.repair.last_relative_correction[j].mean())} for j,i in enumerate(ids))
  if save_rows:
   csvout(R/(task+'_'+mode+'_per_window.csv'),rows)
   if diagnostics:csvout(R/(task+'_'+mode+'_test_semantic_diagnostics.csv'),diagnostics)
  return se/count,ae/count
 print('TRAIN_START',task,mode,'resume',resume,'start_epoch',start_epoch,flush=True)
 for ep in range(start_epoch,101):
  guard();n.train();started=time.time();lr=1e-4*(1/(1+math.exp(-.5*(ep-10)))-1/(1+math.exp(-.05*(ep-100))))
  for pg in opt.param_groups:pg['lr']=lr
  ids=torch.randperm(len(tr),generator=torch.Generator().manual_seed(2021+ep)).tolist();total=0
  for off in range(0,len(ids),126):
   glob=ids[off:off+126];opt.zero_grad(set_to_none=True)
   for k in range(0,len(glob),21):
    cur=glob[k:k+21];x,y,m=batch(tr,cur,'cuda');p=n(x,m);loss=torch.nn.functional.huber_loss(p,y,delta=1.);assert torch.isfinite(loss);(loss*len(cur)/len(glob)).backward();total+=float(loss.detach())*len(cur)
   assert all(torch.isfinite(q.grad).all() for q in n.parameters() if q.grad is not None);opt.step()
  vm,am=evaluate(va);row={'epoch':ep,'train_huber':total/len(tr),'validation_mse':vm,'validation_mae':am,'lr':lr,'seconds':time.time()-started};hist.append(row)
  if vm<best:best=vm;bestep=ep;tmp=cp.with_suffix('.tmp');torch.save(n.state_dict(),tmp);tmp.replace(cp)
  # Epoch-boundary complete recovery state, including optimizer and all RNG states.
  import numpy as np
  state={'task':task,'mode':mode,'epoch':ep,'model':n.state_dict(),'optimizer':opt.state_dict(),'history':hist,'best':best,'best_epoch':bestep,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'implementation_hashes':json.loads((R/'implementation_hashes.json').read_text())};tmp=latest.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(latest);csvout(R/(task+'_'+mode+'_learning.csv'),hist);print('EPOCH',task,mode,json.dumps(row),flush=True)
 assert all(torch.equal(v.cpu(),backbone[k]) for k,v in n.model.patch_backbone.state_dict().items());n.load_state_dict(torch.load(cp,map_location='cpu'));mse,mae=evaluate(te,True);out={'task':task,'mode':mode,'seed':2021,'epochs':100,'lookback':48,'horizon':24,'global_batch':126,'microbatch':21,'precision':'FP32','mse':mse,'mae':mae,'validation_mse':best,'best_epoch':bestep,'test_windows':len(te),'parameters':sum(p.numel() for p in n.parameters()),'trainable_parameters':sum(p.numel() for p in n.parameters() if p.requires_grad),'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'host':os.uname().nodename,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'implementation_hashes':json.loads((R/'implementation_hashes.json').read_text())};guard();save(R/(task+'_'+mode+'_COMPLETE.json'),out);print('SEMANTIC_REFINEMENT_COMPLETE',json.dumps(out),flush=True)
if __name__=='__main__':
 setup()
 if sys.argv[1]=='preflight':preflight()
 elif sys.argv[1]=='train':train(sys.argv[2],sys.argv[3])
