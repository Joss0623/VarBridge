"""Structural checks before expensive mechanism-control training."""
import os,sys,textwrap,importlib
from types import SimpleNamespace
import torch
import torch.distributed as dist
from cho_data import ROOT,CHOWindows
from train_cho_ddp import defaults

def main():
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==6 and os.environ['CUDA_VISIBLE_DEVICES']=='2,3,4,5,6,7'
 torch.cuda.set_device(local);dist.init_process_group('nccl');torch.set_num_threads(2);torch.manual_seed(2021)
 sys.path.insert(0,str(ROOT));src=(ROOT/'scripts/train_cho_ddp.py').read_text();block=src[src.index(' cfg=defaults'):src.index(' config=SimpleNamespace')]
 env=dict(ROOT=ROOT,repo=ROOT,defaults=defaults,args=SimpleNamespace(horizon=6),fusion=True,tl=False);exec(textwrap.dedent(block),env)
 net=importlib.import_module('models.KIBR_PatchTST_BoundMechanism').Model(SimpleNamespace(**env['cfg'])).cuda().eval()
 ds=CHOWindows('val',6,'roberta');x,y,mark,_=ds[rank];x=x[None].cuda();y=y[None].cuda();mark=mark[None].cuda();cache=torch.load(ROOT/'dataset/CHO/CHO_bound_mechanisms.pt')
 mark[:,:,1:6]=cache['correct']['embeddings'].cuda()[None,None]
 captured={}
 def semhook(module,inputs,output):captured['memory']=inputs[1].detach().clone()
 def prompt_hook(module,inputs,output):captured['general']=inputs[0].detach().clone()
 handles=[net.semantic_attention.register_forward_hook(semhook),net.Prompt_encoder.register_forward_hook(prompt_hook)]
 a=mark.clone().requires_grad_();pred=net(x,a,None,a);memory=captured['memory'];general=captured['general'];g=torch.autograd.grad((pred-y).square().mean(),a)[0]
 assert general.shape[1]==3 and memory.shape[1]==1
 for target in range(5):
  for mechanism in range(5):
   norm=float(g[0,target,1+mechanism].norm())
   assert (norm>0 if target==mechanism else norm==0),(target,mechanism,norm)
 with torch.no_grad():
  net(x+0.25,mark,None,mark);assert torch.equal(memory,captured['memory']) and torch.equal(general,captured['general'])
  one=mark.clone();one[:,:,1]=cache['contradictory']['embeddings'][0].cuda();changed=net(x,one,None,one)
  assert float((changed[:,:,1:]-pred.detach()[:,:,1:]).abs().max())<1e-7
  assert float((changed[:,:,0]-pred.detach()[:,:,0]).abs().max())>0
  for variant in ['correct','contradictory','unrelated']:
   z=mark.clone();z[:,:,1:6]=cache[variant]['embeddings'].cuda()[None,None];q=net(x,z,None,z)
   assert torch.isfinite(q).all();print('BOUND_PREFLIGHT',rank,variant,'max_change',float((q-pred.detach()).abs().max()),flush=True)
 for handle in handles:handle.remove()
 dist.barrier()
 if rank==0:print('ALL_CHO_BOUND_PREFLIGHT_COMPLETE',flush=True)
 dist.destroy_process_group()
if __name__=='__main__':main()
