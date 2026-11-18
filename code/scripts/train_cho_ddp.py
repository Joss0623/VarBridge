"""Unified six-rank CHO training; no test-based selection or budget changes."""
import argparse,ast,importlib,os,random,sys,time,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Subset
from cho_data import ROOT,DATA,FEATURES,CHOWindows

def defaults(path):
 out={}
 for node in ast.walk(ast.parse(path.read_text())):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
   try:
    name=ast.literal_eval(node.args[0]);d=next(x.value for x in node.keywords if x.arg=='default');value=ast.literal_eval(d)
    if name.startswith('--'):out[name[2:].replace('-','_')]=value
   except (ValueError,StopIteration,TypeError,IndexError):pass
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--horizon',type=int,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--variant',default='full',choices=['full','no_mechanism','shuffled_mechanism']);ap.add_argument('--epochs',type=int);args=ap.parse_args()
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK']);assert world==6 and os.environ['CUDA_VISIBLE_DEVICES']=='2,3,4,5,6,7'
 torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl');torch.set_num_threads(4)
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 print(f'CHO_DDP_RANK {rank} LOCAL {local} WORLD {world} SEED {args.seed} MODEL {args.model}',flush=True)
 fusion=args.model=='Fusion';backbone=args.model=='Backbone';multi=args.model in ['TimeCMA','T3Time'];tl=args.model=='TimeLLM';xp=args.model=='xPatch'
 repo=ROOT if fusion or backbone else ROOT.parent/('Time-Series-Library' if not multi and not tl and not xp else 'Time-LLM' if tl else args.model)
 cfg=defaults(repo/'run.py') if (repo/'run.py').exists() else {}
 cfg.update(task_name='long_term_forecast',seq_len=48,label_len=24,pred_len=args.horizon,token_len=args.horizon,enc_in=5,dec_in=5,c_out=5,d_model=128,d_ff=256,n_heads=4,e_layers=2,d_layers=1,dropout=.1,patch_len=24,stride=12,padding_patch='end',embed='timeF',freq='h',factor=1,activation='gelu',output_attention=False,moving_avg=25,distil=True,mix_embeds=fusion,ma_type='ema',alpha=.3,beta=.3,kernel_size=25,down_sampling_layers=2,down_sampling_window=2,down_sampling_method='avg',channel_independence=1,decomp_method='moving_avg',use_norm=1,top_k=5,num_kernels=6,revin=1,individual=False)
 if tl:cfg.update(d_model=16,d_ff=32,patch_len=16,stride=8,llm_model='GPT2',llm_dim=768,llm_layers=6,prompt_domain=1,content='CHO-K1 cultivation with dissolved carbon dioxide, optical transmission, permittivity, pH and dissolved oxygen measurements. Optical and dielectric measurements are process signals, not direct cell-count labels.')
 if fusion:cfg.update(numeric_encoder='patchtst',numeric_ckpt=str(ROOT/'checkpoints/cho/Backbone_h24_seed2021_full.pth'),patchtst_ckpt='',text_embed_dim=768,channel_interaction=True,early_channel_mode='patch',forecast_conditioned=True,tri_mmoe=True,dynamic_window_stats=True,separate_numeric_path=False,embedding_fusion='output',prompt_control='text',initial_cross_target='seasonal',disable_prompt_self_attn=False,use_cross_attn=True,use_dynamic_gate=True,coarse_factor=0)
 config=SimpleNamespace(**cfg);sys.path.insert(0,str(repo))
 if multi:
  module=importlib.import_module('models.'+args.model);Net=module.Dual if args.model=='TimeCMA' else module.TriModal
  model=Net(device=str(device),channel=32,num_nodes=5,seq_len=48,pred_len=args.horizon,dropout_n=.2,d_llm=768,e_layer=1,d_layer=1,head=8)
 else:
  name='KIBR_PatchTST' if fusion else 'PatchTST' if backbone else args.model
  model=importlib.import_module('models.'+name).Model(config)
 model=model.float().to(device)
 ddp=DDP(model,device_ids=[local],find_unused_parameters=True,broadcast_buffers=False)
 encoder='roberta' if fusion else 'gpt2' if multi else None
 train=CHOWindows('train',args.horizon,encoder);val=CHOWindows('val',args.horizon,encoder)
 budget=100 if fusion or backbone else 10;patience=100 if fusion or backbone else 3
 epochs=args.epochs or budget;assert epochs<=budget
 global_batch=126 if fusion or backbone or args.model=='Crossformer' else 32
 optimizer=(torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-3) if multi else torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4))
 huber=fusion or backbone or args.model=='Crossformer';criterion=torch.nn.HuberLoss(delta=1.) if huber else torch.nn.MSELoss()
 out=ROOT/'checkpoints/cho'/f'{args.model}_h{args.horizon}_seed{args.seed}_{args.variant}.pth';out.parent.mkdir(exist_ok=True,parents=True)
 if rank==0:print('CHO_PROTOCOL',dict(model=args.model,variant=args.variant,train_windows=len(train),val_windows=len(val),max_epochs=budget,patience=patience,global_batch=global_batch,loss='Huber1' if huber else 'MSE',checkpoint='validation_MSE',fixed_backbone_seed=2021 if fusion else None),flush=True)
 def forward(net,x,mark,h):
  if fusion:
   if args.variant!='full':
    mark=mark.clone();knowledge=mark[:,:,1:6,:].clone()
    if args.variant=='shuffled_mechanism':mark[:,:,1:6,:]=train.cache['shuffled_knowledge_embeddings'][:,0,:].to(device)[None,None,:,:]
    else:mark[:,:,1:6,:]=train.cache['no_mechanism_embedding'].to(device)[None,None,None,:]
   if h==args.horizon:return net(x,mark,None,mark)[:,-h:,:]
   # Same H24 checkpoint and initial observed text as EFP; dynamic numerical statistics update from predictions.
   hist=x;pred=[]
   for _ in range(math.ceil(h/args.horizon)):
    y=net(hist[:,-48:,:],mark,None,mark)[:,-args.horizon:,:];pred.append(y);hist=torch.cat([hist,y],1)
   return torch.cat(pred,1)[:,:h,:]
  if multi:
   emb=mark[:,:,0,:].transpose(1,2).unsqueeze(-1);return net(x,torch.zeros(x.shape[0],48,4,device=device),emb)
  if xp:return net(x)
  xm=torch.zeros(x.shape[0],48,4,device=device);ym=torch.zeros(x.shape[0],24+h,4,device=device);dec=torch.cat([x[:,-24:,:],torch.zeros(x.shape[0],h,5,device=device)],1)
  with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=tl):
   pred=net(x,xm,dec,ym)[:,-h:,:]
  return pred.float()
 def evaluate(ds,h,report=False):
  model.eval();sums=torch.zeros(3,dtype=torch.float64,device=device);windows=torch.zeros(1,dtype=torch.int64,device=device)
  per_batch={}
  loader=DataLoader(Subset(ds,list(range(rank,len(ds),world))),batch_size=4 if tl else 32,shuffle=False,num_workers=0)
  with torch.no_grad():
   for x,y,mark,bids in loader:
    x,y,mark=x.to(device),y.to(device),mark.to(device)
    pred=forward(model,x,mark,h);assert pred.shape==y.shape,(pred.shape,y.shape)
    delta=pred-y;assert torch.isfinite(delta).all();sums+=torch.stack([delta.square().sum().double(),delta.abs().sum().double(),torch.tensor(delta.numel(),device=device,dtype=torch.float64)]);windows+=len(x)
    if report:
     se=delta.square().sum((1,2)).cpu().tolist();ae=delta.abs().sum((1,2)).cpu().tolist()
     for bid,a,b in zip(bids.tolist(),se,ae):
      q=per_batch.setdefault(bid,[0.,0.,0]);q[0]+=a;q[1]+=b;q[2]+=h*5
  dist.all_reduce(sums);dist.all_reduce(windows);assert int(windows)==len(ds)
  if report:
   all_parts=[None]*world;dist.all_gather_object(all_parts,per_batch)
   if rank==0:
    merged={}
    for part in all_parts:
     for bid,v in part.items():
      q=merged.setdefault(bid,[0.,0.,0]);merged[bid]=[a+b for a,b in zip(q,v)]
    for bid,(se,ae,n) in sorted(merged.items()):print(f'CHO_BATCH_RESULT h={h} batch={bid} mse={se/n:.10f} mae={ae/n:.10f} elements={n}',flush=True)
  return (sums[:2]/sums[2]).tolist()
 best=float('inf');stale=0;best_epoch=0
 for epoch in range(1,epochs+1):
  model.train();g=torch.Generator().manual_seed(args.seed+epoch);order=torch.randperm(len(train),generator=g).tolist();total=0.;cnt=0;start=time.time()
  if huber:
   lr=1e-4/(1+math.exp(-.5*(epoch-10)))-1e-4/(1+math.exp(-.05*(epoch-100)))
  elif multi:lr=1e-4
  elif tl:lr=1e-4*.5*(1+math.cos(math.pi*(epoch-1)/10))
  else:lr=1e-4*(.5**max(epoch-2,0))
  for group in optimizer.param_groups:group['lr']=lr
  for off in range(0,len(order),global_batch):
   ids=order[off:off+global_batch];mine=ids[rank::world];n=len(mine)
   # Empty rank participates with zero-weight dummy; no duplicated sample contributes to the gradient.
   batch=[train[i] for i in (mine or ids[:1])];x=torch.stack([v[0] for v in batch]).to(device);y=torch.stack([v[1] for v in batch]).to(device);mark=torch.stack([v[2] for v in batch]).to(device)
   optimizer.zero_grad(set_to_none=True);pred=forward(ddp,x,mark,args.horizon);loss=criterion(pred,y);assert torch.isfinite(loss)
   weighted=loss*(world*n/len(ids));weighted.backward()
   if multi:torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
   optimizer.step();total+=float(loss.detach())*n;cnt+=n
  # Synchronize model buffers before checkpoint and exact distributed validation.
  for buffer in model.buffers():dist.broadcast(buffer,src=0)
  stats=torch.tensor([total,cnt],dtype=torch.float64,device=device);dist.all_reduce(stats)
  vm,va=evaluate(val,args.horizon)
  if rank==0:print(f'Epoch: {epoch} Train Loss: {stats[0]/stats[1]:.9f} Vali Loss: {vm:.9f} Vali MAE: {va:.9f} lr:{lr:.10f} seconds:{time.time()-start:.2f}',flush=True)
  if vm<best:
   best=vm;stale=0;best_epoch=epoch
   if rank==0:torch.save(model.state_dict(),out)
  else:stale+=1
  dist.barrier()
  if stale>=patience:break
 model.load_state_dict(torch.load(out,map_location='cpu'),strict=True);dist.barrier()
 if not backbone:
  hs=[args.horizon,48] if fusion and args.horizon==24 else [args.horizon]
  for h in hs:
   ds=CHOWindows('test',h,encoder);mse,mae=evaluate(ds,h,True)
   if rank==0:print(f'CHO_RESULT model={args.model} variant={args.variant} seed={args.seed} h={h} mse={mse:.10f} mae={mae:.10f} test_windows={len(ds)} best_epoch={best_epoch} completed_epochs={epoch} protocol={"recursive" if fusion and h==48 else "direct"}',flush=True)
 if rank==0:print('CHO_JOB_COMPLETE',args.model,args.variant,args.horizon,args.seed,flush=True)
 dist.destroy_process_group()
if __name__=='__main__':main()
