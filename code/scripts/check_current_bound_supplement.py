"""Exact default identity and finite gradients for every isolated setting."""
import argparse,os,sys,gc
from pathlib import Path
import torch
from common_bound_supplement import *
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--distributed',action='store_true');args=ap.parse_args()
 torch.set_num_threads(2);rank=int(os.environ.get('RANK','0'))
 if args.distributed:
  import torch.distributed as dist
  assert int(os.environ['WORLD_SIZE'])==6 and os.environ['CUDA_VISIBLE_DEVICES']=='2,3,4,5,6,7'
  local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);dist.init_process_group('nccl');device=torch.device('cuda',local)
  print('SUPPLEMENT_PREFLIGHT_RANK',rank,flush=True)
 else:device=torch.device('cpu')
 for ds in DATASETS:
  cfg=make_config(ds);seed_all();original=AdoptedModel(cfg).to(device).eval();seed_all();new=SupplementModel(cfg).to(device).eval()
  assert tensor_hash(original)==tensor_hash(new)
  ck=torch.load(main_checkpoint(ds),map_location='cpu');original.load_state_dict(ck,strict=True);new.load_state_dict(ck,strict=True);del ck
  g=torch.Generator().manual_seed(1987);x=torch.randn(2,48,5,generator=g).to(device);mark=torch.randn(2,5,8,768,generator=g).to(device)
  with torch.no_grad():a=original(x,mark,None,mark);b=new(x,mark,None,mark)
  assert torch.equal(a,b),(ds,float((a-b).abs().max()))
  frozen=tensor_hash(original.patch_backbone)
  del original,new;gc.collect()
  for name,changes in TASKS.items():
   seed_all();net=SupplementModel(make_config(ds,**changes)).to(device).train()
   assert tensor_hash(net.patch_backbone)==frozen
   y=net(x,mark,None,mark);assert y.shape==(2,24,5) and torch.isfinite(y).all()
   y.square().mean().backward();grads=[p.grad for p in net.parameters() if p.requires_grad and p.grad is not None]
   assert grads and all(torch.isfinite(q).all() for q in grads)
   assert all(p.grad is None for p in net.patch_backbone.parameters())
   assert tensor_hash(net.patch_backbone)==frozen
   if rank==0:print('SUPPLEMENT_FINITE',ds,name,'trainable',sum(p.numel() for p in net.parameters() if p.requires_grad),flush=True)
   del net;gc.collect()
  if rank==0:print('SUPPLEMENT_DEFAULT_EXACT',ds,'state_and_outputs_equal; backbone_fixed',flush=True)
 if args.distributed:
  dist.barrier();dist.destroy_process_group()
 print('ALL_CURRENT_SUPPLEMENT_'+('DDP6' if args.distributed else 'CPU')+'_PREFLIGHT_COMPLETE',flush=True)
if __name__=='__main__':main()
