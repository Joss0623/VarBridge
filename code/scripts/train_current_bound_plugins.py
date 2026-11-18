"""Isolated exact-global-batch DDP training for interchangeable encoders."""
import argparse,os,sys,math,re,hashlib
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Subset
from current_bound_plugins import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=DATASETS,required=True);ap.add_argument('--numeric',choices=NUMERICS,required=True);ap.add_argument('--text',choices=TEXTS,default='roberta');ap.add_argument('--phase',choices=['preflight','train','reuse','reference'],required=True);ap.add_argument('--stage',choices=['adapt','pretrain'],default='adapt');args=ap.parse_args()
 world=int(os.environ['WORLD_SIZE']);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);assert world==2
 torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl');torch.set_num_threads(2);seed_all(2021);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 sources=source_hashes();print('PLUGIN_DDP_RANK',rank,'WORLD',world,'PHYSICAL',os.environ['CUDA_VISIBLE_DEVICES'],flush=True)
 cfg=config(args.dataset,args.numeric)
 if args.stage=='pretrain' or args.phase=='reference':
  cfg.mix_embeds=False;module=__import__('models.'+('TimesNet' if args.numeric=='timesnet' else 'PatchTST'),fromlist=['Model']);model=module.Model(cfg)
  out=numerical_checkpoint(args.dataset,args.numeric)
  if args.phase=='reference':model.load_state_dict(torch.load(out,map_location='cpu'),strict=True)
 else:
  assert Path(cfg.numeric_ckpt).exists(),cfg.numeric_ckpt
  model=Model(cfg);out=checkpoint(args.dataset,args.numeric,args.text)
  if args.phase=='reuse':
   assert args.numeric=='patchtst' and args.text=='roberta';out=main_checkpoint(args.dataset);model.load_state_dict(torch.load(out,map_location='cpu'),strict=True)
 model=model.float().to(device)
 train=Windows(args.dataset,'train',args.text if args.stage=='adapt' and args.phase!='reference' else 'roberta');val=Windows(args.dataset,'val',args.text if args.stage=='adapt' and args.phase!='reference' else 'roberta')
 def forward(net,x,mark):
  return net(x,None,None,None)[:,-24:,:].float() if args.stage=='pretrain' or args.phase=='reference' else net(x,mark,None,mark)[:,-24:,:].float()
 def evaluate(ds):
  model.eval();sums=torch.zeros(3,dtype=torch.float64,device=device);count=torch.zeros(1,dtype=torch.int64,device=device)
  with torch.no_grad():
   for x,y,mark,_ in DataLoader(Subset(ds,list(range(rank,len(ds),world))),batch_size=32,shuffle=False,num_workers=0):
    x,y,mark=x.to(device),y.to(device),mark.to(device);pred=forward(model,x,mark);assert pred.shape==y.shape and torch.isfinite(pred).all()
    d=pred-y;sums+=torch.stack([d.square().sum().double(),d.abs().sum().double(),torch.tensor(d.numel(),dtype=torch.float64,device=device)]);count+=len(x)
  dist.all_reduce(sums);dist.all_reduce(count);assert int(count)==len(ds)
  return float(sums[0]/sums[2]),float(sums[1]/sums[2])
 if args.phase=='reuse':
  # Exact default-source and observed-text identity before checkpoint evaluation.
  reference=Model(original_config(args.dataset));reference.load_state_dict(torch.load(main_checkpoint(args.dataset),map_location='cpu'),strict=True);reference=reference.float().to(device).eval();model.eval()
  assert model.state_dict().keys()==reference.state_dict().keys()
  assert all(torch.equal(v,reference.state_dict()[k]) for k,v in model.state_dict().items())
  original=CurrentWindows(args.dataset,'val',24,'roberta','full')
  for index in [0,len(val)//2,len(val)-1]:
   a=val[index];b=original[index];assert all(torch.equal(a[k],b[k]) for k in [0,1,2])
  x,y,mark,_=val[rank%len(val)];x=x[None].to(device);mark=mark[None].to(device)
  with torch.no_grad():assert torch.equal(forward(model,x,mark),reference(x,mark,None,mark)[:,-24:,:])
  del reference,original
  test=Windows(args.dataset,'test','roberta');mse,mae=evaluate(test)
  text=main_log(args.dataset).read_text();lines=[l for l in text.splitlines() if l.startswith('EFP_BOUND_RESULT ' if args.dataset!='CHO' else 'CHO_RESULT ') and ' h=24 ' in l]
  assert len(lines)==1,lines;fields=dict(re.findall(r'(\w+)=([^\s]+)',lines[0]));assert abs(mse-float(fields['mse']))<2e-6 and abs(mae-float(fields['mae']))<2e-6
  if rank==0:print(f'PLUGIN_RESULT dataset={args.dataset} numeric=patchtst text=roberta mse={mse:.10f} mae={mae:.10f} windows={len(test)} epochs=100 reused=1 checkpoint={out}',flush=True);print('PLUGIN_DEFAULT_IDENTITY_COMPLETE',args.dataset,flush=True)
 elif args.phase=='reference':
  test=Windows(args.dataset,'test','roberta');mse,mae=evaluate(test)
  if rank==0:print(f'PLUGIN_REFERENCE dataset={args.dataset} numeric={args.numeric} mse={mse:.10f} mae={mae:.10f} windows={len(test)} checkpoint={out}',flush=True)
 else:
  ddp=DDP(model,device_ids=[local],find_unused_parameters=True,broadcast_buffers=False)
  if args.phase=='preflight':
   model.train();batch=[train[i] for i in [rank,rank+world]];x=torch.stack([v[0] for v in batch]).to(device);y=torch.stack([v[1] for v in batch]).to(device);mark=torch.stack([v[2] for v in batch]).to(device)
   loss=torch.nn.functional.huber_loss(forward(ddp,x,mark),y,delta=1.);assert torch.isfinite(loss);loss.backward();assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
   assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
   if args.stage=='adapt':assert not model.patch_backbone.training and all(p.grad is None for p in model.patch_backbone.parameters())
   if rank==0:print('PLUGIN_PREFLIGHT_COMPLETE',args.dataset,args.numeric,args.text,args.stage,'train',len(train),'val',len(val),flush=True)
  else:
   assert not out.exists(),('Never overwrite a completed/new partial fit',out)
   frozen_before={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith('patch_backbone.')} if args.stage=='adapt' else {}
   optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4);best=float('inf');best_epoch=0
   if rank==0:print('PLUGIN_PROTOCOL',vars(args),'epochs100 patience100 batch126 seed2021 Adam Huber1 LR1e-4 dual-sigmoid validationMSE FP32 world2',flush=True)
   for epoch in range(1,101):
    import time
    model.train();started=time.time();order=torch.randperm(len(train),generator=torch.Generator().manual_seed(2021+epoch)).tolist();total=torch.zeros(2,dtype=torch.float64,device=device)
    lr=1e-4/(1+math.exp(-.5*(epoch-10)))-1e-4/(1+math.exp(-.05*(epoch-100)))
    for group in optimizer.param_groups:group['lr']=lr
    for off in range(0,len(order),126):
     ids=order[off:off+126];mine=ids[rank::world];n=len(mine);batch=[train[i] for i in (mine or ids[:1])];x=torch.stack([v[0] for v in batch]).to(device);y=torch.stack([v[1] for v in batch]).to(device);mark=torch.stack([v[2] for v in batch]).to(device)
     optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.huber_loss(forward(ddp,x,mark),y,delta=1.);assert torch.isfinite(loss);(loss*(world*n/len(ids))).backward();optimizer.step();total+=torch.tensor([float(loss.detach())*n,n],dtype=torch.float64,device=device)
    for buffer in model.buffers():dist.broadcast(buffer,src=0)
    dist.all_reduce(total);vm,va=evaluate(val)
    if vm<best:
     best=vm;best_epoch=epoch
     if rank==0:tmp=out.with_suffix('.tmp');torch.save(model.state_dict(),tmp);tmp.replace(out)
    dist.barrier()
    if rank==0:print(f'Epoch: {epoch} Train Loss: {float(total[0]/total[1]):.9f} Vali Loss: {vm:.9f} Vali MAE: {va:.9f} lr:{lr:.10f} seconds:{time.time()-started:.2f}',flush=True)
   if args.stage=='adapt':assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in frozen_before.items())
   model.load_state_dict(torch.load(out,map_location='cpu'),strict=True);vm,va=evaluate(val)
   if rank==0:print(f'PLUGIN_VALIDATION dataset={args.dataset} numeric={args.numeric} text={args.text} mse={vm:.10f} mae={va:.10f} best_epoch={best_epoch} windows={len(val)}',flush=True)
   if args.stage=='adapt':
    test=Windows(args.dataset,'test',args.text);mse,mae=evaluate(test)
    if rank==0:print(f'PLUGIN_RESULT dataset={args.dataset} numeric={args.numeric} text={args.text} mse={mse:.10f} mae={mae:.10f} windows={len(test)} epochs=100 reused=0 best_epoch={best_epoch} checkpoint={out}',flush=True)
   elif rank==0:print('PLUGIN_BACKBONE_READY',args.dataset,args.numeric,'epochs100',out,flush=True)
 assert sources==source_hashes(),'Adopted source changed during experiment'
 dist.barrier()
 if rank==0:print('PLUGIN_JOB_COMPLETE',args.phase,args.stage,args.dataset,args.numeric,args.text,flush=True)
 dist.destroy_process_group()
if __name__=='__main__':main()
