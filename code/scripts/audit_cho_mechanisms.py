"""Fixed-checkpoint validation intervention audit; no training or test-based selection."""
import os,sys,json,importlib,textwrap
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset
from cho_data import ROOT,CHOWindows
from train_cho_ddp import defaults

def emit(kind,obj): print(kind,json.dumps(obj),flush=True)
def main():
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
 assert world==6 and os.environ['CUDA_VISIBLE_DEVICES']=='2,3,4,5,6,7'
 torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl');torch.set_num_threads(2)
 print('AUDIT_RANK',rank,'PHYSICAL_GPU',local+2,flush=True)
 sys.path.insert(0,str(ROOT))
 source=(ROOT/'scripts/train_cho_ddp.py').read_text();cfg_source=source[source.index(' cfg=defaults'):source.index(' config=SimpleNamespace')]
 modes=['full','no_mechanism','wrong_mechanism','zero_mechanism','permute_mechanism','zero_observed_text']
 if os.environ.get('AUDIT_STATS')=='1':modes=['full','zero_dynamic_stats','zero_all_cached_text','zero_all_text_and_stats']
 for h in [6,12,24]:
  ds=CHOWindows('val',h,'roberta');cache=ds.cache
  if rank==0:
   original=cache['knowledge_embeddings'][:,0,:];wrong=cache['shuffled_knowledge_embeddings'][:,0,:];null=cache['no_mechanism_embedding'].expand_as(original)
   emit('CACHE',dict(h=h,wrong_cosine=torch.nn.functional.cosine_similarity(original,wrong).tolist(),null_cosine=torch.nn.functional.cosine_similarity(original,null).tolist(),wrong_relative_norm=float((wrong-original).norm()/original.norm()),null_relative_norm=float((null-original).norm()/original.norm())))
  def intervene(mark,mode):
   z=mark.clone()
   if mode in ['zero_all_cached_text','zero_all_text_and_stats']:return z.zero_()
   if mode=='no_mechanism':z[:,:,1:6]=cache['no_mechanism_embedding'].to(device)[None,None,None,:]
   elif mode=='wrong_mechanism':z[:,:,1:6]=cache['shuffled_knowledge_embeddings'][:,0,:].to(device)[None,None,:,:]
   elif mode=='zero_mechanism':z[:,:,1:6]=0
   elif mode=='permute_mechanism':z[:,:,1:6]=z[:,:,1:6].flip(2)
   elif mode=='zero_observed_text':z[:,:,0]=0
   return z
  for seed in [2021,2022,2023]:
   args=SimpleNamespace(horizon=h);scope=dict(ROOT=ROOT,defaults=defaults,repo=ROOT,args=args,fusion=True,tl=False)
   exec(textwrap.dedent(cfg_source),scope);cfg=scope['cfg'];model=importlib.import_module('models.KIBR_PatchTST').Model(SimpleNamespace(**cfg)).to(device).eval()
   ckpt=ROOT/f'checkpoints/cho/Fusion_h{h}_seed{seed}_full.pth';model.load_state_dict(torch.load(ckpt,map_location=device),strict=True)
   sums=torch.zeros(len(modes),7,dtype=torch.float64,device=device)
   loader=DataLoader(Subset(ds,list(range(rank,len(ds),world))),batch_size=16,shuffle=False)
   with torch.no_grad():
    for x,y,mark,_ in loader:
     x,y,mark=x.to(device),y.to(device),mark.to(device);base=model(x,mark,None,mark)
     for j,mode in enumerate(modes):
      z=intervene(mark,mode)
      handle=model.current_stats_encoder.register_forward_hook(lambda module,inputs,output: torch.zeros_like(output)-model.current_stats_role) if mode in ['zero_dynamic_stats','zero_all_text_and_stats'] else None
      pred=base if j==0 else model(x,z,None,z)
      if handle:handle.remove()
      err=pred-y;delta=pred-base
      sums[j]+=torch.tensor([err.square().sum(),err.abs().sum(),delta.square().sum(),delta.abs().sum(),base.square().sum(),err.numel(),len(x)],dtype=torch.float64,device=device)
   dist.all_reduce(sums)
   if rank==0:
    for mode,s in zip(modes,sums.cpu().tolist()):
     assert int(s[6])==len(ds)
     emit('INTERVENTION',dict(h=h,seed=seed,mode=mode,windows=int(s[6]),validation_mse=s[0]/s[5],validation_mae=s[1]/s[5],prediction_delta_rms=(s[2]/s[5])**.5,prediction_delta_mae=s[3]/s[5],relative_prediction_delta_rms=(s[2]/s[4])**.5))
    # Balanced sample spans all validation segments. Inspect input gradients and internal pathways.
    ids=torch.linspace(0,len(ds)-1,16).long().tolist();batch=[ds[i] for i in ids];x=torch.stack([v[0] for v in batch]).to(device);y=torch.stack([v[1] for v in batch]).to(device);mark=torch.stack([v[2] for v in batch]).to(device).requires_grad_()
    captured={};handles=[]
    def hook(name):
     def f(module,inputs,output):
      captured[name]=(inputs[0].detach(),(output[0] if isinstance(output,tuple) else output).detach())
      if name=='semantic_attention' or name=='cross_attention':
       attn=module if name=='semantic_attention' else module.cross_attn
       with torch.no_grad():
        _,w=attn.forward(inputs[0],inputs[1],inputs[1],need_weights=True)
       captured[name+'_weights']=w.detach().mean((0,1)).cpu().tolist()
     return f
    for name in ['LLMtodim','Prompt_encoder','cross_attention','semantic_attention','numeric_router','semantic_router','tri_router','semantic_delta']:
     if hasattr(model,name):handles.append(getattr(model,name).register_forward_hook(hook(name)))
    pred=model(x,mark,None,mark);loss=(pred-y).square().mean();g=torch.autograd.grad(loss,mark)[0]
    info=dict(h=h,seed=seed,add_scale=float(model.add_scale.detach()),gradient_norms={name:float(g[:,:,sl,:].norm()) for name,sl in [('observed',slice(0,1)),('mechanism',slice(1,6)),('stage_operation',slice(6,8))]},attention={k:v for k,v in captured.items() if k.endswith('_weights')},routers={k:torch.softmax(v[1],-1).mean((0,1)).cpu().tolist() for k,v in captured.items() if k.endswith('_router')})
    emit('PATHWAY',info)
    for handle in handles:handle.remove()
   dist.barrier();del model
 if rank==0:print('ALL_CHO_MECHANISM_AUDITS_COMPLETE',flush=True)
 dist.destroy_process_group()
if __name__=='__main__':main()
