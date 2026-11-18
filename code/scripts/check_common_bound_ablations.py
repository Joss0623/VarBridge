"""Check every variant and prove isolated full-model identity before scheduling fits."""
import os,sys,textwrap,importlib
from types import SimpleNamespace
import torch
import torch.distributed as dist
from common_bound_ablations import ROOT,VARIANTS,Windows,configure
from train_common_bound_ablation_ddp import defaults
def main():
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==6 and os.environ['CUDA_VISIBLE_DEVICES']=='2,3,4,5,6,7'
 torch.cuda.set_device(local);dist.init_process_group('nccl');torch.set_num_threads(2);sys.path.insert(0,str(ROOT))
 src=(ROOT/'scripts/train_common_bound_ablation_ddp.py').read_text();block=src[src.index(' cfg=defaults'):src.index(' config=SimpleNamespace')]
 Original=importlib.import_module('models.KIBR_PatchTST_BoundMechanism').Model
 Ablation=importlib.import_module('models.KIBR_PatchTST_BoundAblation').Model
 for dataset in ['CHO','EFP_long_sub1','EFP_long_sub2','EFP_long_sub3']:
  for h in [6,12,24]:
   def config(v):
    env=dict(ROOT=ROOT,repo=ROOT,defaults=defaults,args=SimpleNamespace(dataset=dataset,horizon=h,variant=v),fusion=True,tl=False,configure=configure)
    exec(textwrap.dedent(block),env);return SimpleNamespace(**env['cfg'])
   ds=Windows(dataset,'val',h,variant='full');x,y,m,_=ds[rank];x=x[None].cuda();y=y[None].cuda();m=m[None].cuda()
   torch.manual_seed(2021);ref=Original(config('full')).cuda().eval()
   torch.manual_seed(2021);net=Ablation(config('full')).cuda().eval()
   assert ref.state_dict().keys()==net.state_dict().keys()
   for key,value in ref.state_dict().items():assert torch.equal(value,net.state_dict()[key]),key
   with torch.no_grad():
    a=ref(x,m,None,m);b=net(x,m,None,m);assert torch.equal(a,b),(dataset,h,float((a-b).abs().max()))
   del ref,net,a,b
   for v in VARIANTS[rank::world]:
    cfg=config(v);torch.manual_seed(2021);net=Ablation(cfg).cuda().eval()
    control=Windows(dataset,'val',h,variant=v);xx,yy,mm,_=control[rank];xx=xx[None].cuda();yy=yy[None].cuda();mm=mm[None].cuda()
    pred=net(xx,mm,None,mm);assert pred.shape==yy.shape and torch.isfinite(pred).all()
    loss=(pred-yy).square().mean();loss.backward()
    grads=[p.grad for p in net.parameters() if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads)
    if v in ['learned_condition','no_knowledge']:
     with torch.no_grad():
      reference=net(xx,mm,None,mm);changed=net(xx,mm+torch.randn_like(mm),None,mm);assert torch.equal(reference,changed),v
     markcheck=mm.detach().requires_grad_();check=net(xx,markcheck,None,markcheck);textgrad=torch.autograd.grad(check.sum(),markcheck,allow_unused=True)[0]
     assert textgrad is None or int(torch.count_nonzero(textgrad))==0,v
    if v=='no_dynamic_stats':assert not hasattr(net,'current_stats_encoder')
    if v=='separate_numeric_path':assert net.separate_numeric_path
    if v=='no_forecast_conditioning':assert not hasattr(net,'semantic_attention')
    print('COMMON_ABLATION_PREFLIGHT_RANK',rank,'dataset',dataset,'h',h,'variant',v,'windows',len(control),'PASS',flush=True)
    del net,control,xx,yy,mm,pred,loss,grads;torch.cuda.empty_cache()
   del ds,x,y,m
 dist.barrier()
 if rank==0:print('ALL_COMMON_BOUND_ABLATION_PREFLIGHT_COMPLETE',flush=True)
 dist.destroy_process_group()
if __name__=='__main__':main()
