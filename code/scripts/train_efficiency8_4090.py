"""Separate matched100-epoch CHO-H24 efficiency study; adopted models unchanged."""
import os,sys,time,math,random,csv,argparse,gc
from pathlib import Path
import numpy as np
import torch
from efficiency8_common import build
from cho_data import CHOWindows
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'test_results/efficiency8_4090';MODELS=['DLinear','TimesNet','PatchTST','Crossformer','iTransformer','TimeMixer']
def enforce_efficiency_fp32(net):
 # Instance-only removal of a hardcoded BF16 cast for this FP32 comparison.
 # Original baseline files, architecture and parameters are unchanged.
 import inspect,textwrap,types
 source=inspect.getsource(type(net).forecast)
 assert source.count('x_enc.to(torch.bfloat16)')==1
 namespace={};exec(textwrap.dedent(source.replace('x_enc.to(torch.bfloat16)','x_enc')),sys.modules[type(net).__module__].__dict__,namespace)
 net.forecast=types.MethodType(namespace['forecast'],net)
 return net
def seed():
 random.seed(2021);np.random.seed(2021);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True);ap.add_argument('--preflight-only',action='store_true');args=ap.parse_args();m=args.model
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='6';torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;seed();device='cuda:0'
 OUT.mkdir(exist_ok=True,parents=True);net=build('CHO',m).float().to(device)
 if m=='TimeLLM':enforce_efficiency_fp32(net)
 ds={s:CHOWindows(s,24,'roberta' if m=='VarBridge' else None) for s in ['train','val','test']}
 data={}
 for split,d in ds.items():
  values=[d[i] for i in range(len(d))];data[split]=(torch.stack([v[0] for v in values]),torch.stack([v[1] for v in values]),torch.stack([v[2] for v in values]) if m=='VarBridge' else None)
 def forward(x,mark):
  if m=='VarBridge':return net(x,mark,None,mark)[:,-24:]
  dec=torch.cat([x[:,-24:],torch.zeros(len(x),24,5,device=device)],1)
  return net(x,torch.zeros(len(x),48,4,device=device),dec,torch.zeros(len(x),48,4,device=device))[:,-24:]
 # Finite gradient preflight uses a forked RNG, preserving the training RNG stream.
 with torch.random.fork_rng(devices=[0]):
  x,y,mark=data['train'];pred=forward(x[:4].to(device),None if mark is None else mark[:4].to(device));loss=torch.nn.functional.huber_loss(pred,y[:4].to(device),delta=1.);assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
 net.zero_grad(set_to_none=True);seed()
 if args.preflight_only:print('EFF8_PREFLIGHT_COMPLETE',m,tuple(pred.shape),flush=True);return
 opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=1e-4)
 ck=ROOT/'checkpoints/efficiency8_4090'/f'{m}_CHO_h24_seed2021_100epochs.pth';ck.parent.mkdir(exist_ok=True,parents=True)
 @torch.no_grad()
 def evaluate(split):
  net.eval();x,y,marks=data[split];se=ae=0.;n=0
  for k in range(0,len(x),4):
   pred=forward(x[k:k+4].to(device),None if marks is None else marks[k:k+4].to(device));delta=(pred-y[k:k+4].to(device)).double();assert torch.isfinite(delta).all();se+=float(delta.square().sum());ae+=float(delta.abs().sum());n+=delta.numel()
  return se/n,ae/n
 total=0.;best=float('inf');best_epoch=0;x,y,marks=data['train'];torch.cuda.reset_peak_memory_stats()
 print('EFF100_PROTOCOL',dict(model=m,dataset='CHO',horizon=24,seed=2021,epochs=100,physical_gpu=6,world_size=1,global_batch=126,microbatch=4,precision='float32',optimizer='Adam',loss='Huber1',lr='1e-4 dual sigmoid',checkpoint='validation MSE',resident_parameters=sum(p.numel() for p in net.parameters()),trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad)),flush=True)
 for epoch in range(1,101):
  net.train();g=torch.Generator().manual_seed(2021+epoch);order=torch.randperm(len(x),generator=g);lr=1e-4/(1+math.exp(-.5*(epoch-10)))-1e-4/(1+math.exp(-.05*(epoch-100)))
  for group in opt.param_groups:group['lr']=lr
  torch.cuda.synchronize();start=time.perf_counter()
  for off in range(0,len(order),126):
   ids=order[off:off+126];opt.zero_grad(set_to_none=True)
   for j in range(0,len(ids),4):
    ix=ids[j:j+4];xx=x[ix].to(device);yy=y[ix].to(device);mm=None if marks is None else marks[ix].to(device);pred=forward(xx,mm);loss=torch.nn.functional.huber_loss(pred,yy,delta=1.);assert torch.isfinite(loss);(loss*(len(ix)/len(ids))).backward()
   opt.step()
  torch.cuda.synchronize();seconds=time.perf_counter()-start;total+=seconds;vm,va=evaluate('val')
  if vm<best:best=vm;best_epoch=epoch;torch.save(net.state_dict(),ck)
  print('EFF100_EPOCH',epoch,'training_seconds',round(seconds,6),'val_mse',vm,'val_mae',va,flush=True)
 net.load_state_dict(torch.load(ck,map_location='cpu'),strict=True);mse,mae=evaluate('test');vm,va=evaluate('val')
 row=dict(model=m,dataset='CHO',horizon=24,seed=2021,epochs=100,global_batch=126,microbatch=4,world_size=1,gpu=torch.cuda.get_device_name(0),precision='float32',training100_seconds=total,mse=mse,mae=mae,val_mse=vm,val_mae=va,best_epoch=best_epoch,resident_parameters=sum(p.numel() for p in net.parameters()),trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),peak_memory_MiB=torch.cuda.max_memory_allocated()/2**20,train_windows=len(ds['train']),val_windows=len(ds['val']),test_windows=len(ds['test']),checkpoint=str(ck))
 with (OUT/(m+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
 print('EFFICIENCY100_JOB_COMPLETE',m,flush=True)
if __name__=='__main__':main()
