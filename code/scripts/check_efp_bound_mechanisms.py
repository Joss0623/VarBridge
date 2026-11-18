"""Structural checks before expensive mechanism-control training."""
import os,sys,textwrap,importlib
from types import SimpleNamespace
import torch
import torch.distributed as dist
from efp_bound_data import ROOT,EFPBoundWindows
from train_efp_bound_mechanism_ddp import defaults

def main():
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==6 and os.environ['CUDA_VISIBLE_DEVICES']=='2,3,4,5,6,7'
 torch.cuda.set_device(local);dist.init_process_group('nccl');torch.set_num_threads(2);torch.manual_seed(2021)
 sys.path.insert(0,str(ROOT));src=(ROOT/'scripts/train_efp_bound_mechanism_ddp.py').read_text();block=src[src.index(' cfg=defaults'):src.index(' config=SimpleNamespace')]
 for dataset in ['EFP_long_sub1','EFP_long_sub2','EFP_long_sub3']:
  for h in [6,12,24]:
   env=dict(ROOT=ROOT,repo=ROOT,defaults=defaults,args=SimpleNamespace(horizon=h,dataset=dataset),fusion=True,tl=False);exec(textwrap.dedent(block),env)
   net=importlib.import_module('models.KIBR_PatchTST_BoundMechanism').Model(SimpleNamespace(**env['cfg'])).cuda().eval()
   ds=EFPBoundWindows(dataset,'val',h);x,y,mark,_=ds[rank];x=x[None].cuda();y=y[None].cuda();mark=mark[None].cuda();captured={}
   def semhook(module,inputs,output):captured['memory']=inputs[1].detach().clone()
   def prompt_hook(module,inputs,output):captured['general']=inputs[0].detach().clone()
   handles=[net.semantic_attention.register_forward_hook(semhook),net.Prompt_encoder.register_forward_hook(prompt_hook)]
   a=mark.clone().requires_grad_();pred=net(x,a,None,a);memory=captured['memory'];general=captured['general'];g=torch.autograd.grad((pred-y).square().mean(),a)[0]
   assert general.shape[1]==3 and memory.shape[1]==1 and pred.shape==y.shape and torch.isfinite(pred).all()
   for target in range(5):
    for mechanism in range(5):
     norm=float(g[0,target,1+mechanism].norm());assert (norm>0 if target==mechanism else norm==0),(target,mechanism,norm)
   with torch.no_grad():
    net(x+.25,mark,None,mark);assert torch.equal(memory,captured['memory']) and torch.equal(general,captured['general'])
    one=mark.clone();one[:,:,1]=mark[:,:,2];changed=net(x,one,None,one)
    assert float((changed[:,:,1:]-pred.detach()[:,:,1:]).abs().max())<1e-7
    assert float((changed[:,:,0]-pred.detach()[:,:,0]).abs().max())>0
   for handle in handles:handle.remove()
   print('EFP_BOUND_PREFLIGHT_RANK',rank,'dataset',dataset,'h',h,'features',ds.feature_names,'val_windows',len(ds),flush=True)
   del net,ds,x,y,mark,a,pred,g;torch.cuda.empty_cache()
 dist.barrier()
 if rank==0:print('ALL_EFP_BOUND_PREFLIGHT_COMPLETE',flush=True)
 dist.destroy_process_group()
if __name__=='__main__':main()
