"""One model per process prevents sibling-repository import collisions."""
import os,sys,ast,time,csv,argparse,importlib,math,gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'test_results/current_bound_supplement'
MODELS=['VarBridge','PatchTST','Crossformer','DLinear','T3Time','TimeLLM']
def defaults(path):
 cfg={}
 if not path.exists():return cfg
 for node in ast.walk(ast.parse(path.read_text())):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
   try:
    name=ast.literal_eval(node.args[0]);value=ast.literal_eval(next(k.value for k in node.keywords if k.arg=='default'))
    if name.startswith('--'):cfg[name[2:].replace('-','_')]=value
   except (ValueError,StopIteration,TypeError,IndexError):pass
 return cfg
def checkpoint(dataset,model):
 if model=='VarBridge':
  return ROOT/('checkpoints/cho_bound_mechanism' if dataset=='CHO' else 'checkpoints/efp_bound_mechanism')/('Fusion_h24_seed2021_correct.pth' if dataset=='CHO' else dataset+'_h24_seed2021_correct.pth')
 if dataset=='CHO':return ROOT/'checkpoints/cho'/f'{model}_h24_seed2021_full.pth'
 if model=='T3Time':return ROOT/'checkpoints/efp_multimodal'/f'{model}_{dataset}_h24_seed2021.pt'
 if model=='TimeLLM':pattern=f'long_term_forecast_{dataset}_h24_TimeLLM_*/checkpoint';paths=list((ROOT/'checkpoints/efp_multimodal/TimeLLM').glob(pattern))
 else:
  pattern=(f'long_term_forecast_{dataset}_crossformer_3seed_h24_seed2021_Crossformer_*/checkpoint.pth' if model=='Crossformer' else f'long_term_forecast_{dataset}_h24_{model}_*/checkpoint.pth');paths=list((ROOT.parent/'Time-Series-Library/checkpoints').glob(pattern))
 assert len(paths)==1,paths
 return paths[0]
def build(dataset,model):
 if model=='VarBridge':
  sys.path.insert(0,str(ROOT/'scripts'));from common_bound_supplement import make_config,SupplementModel
  return SupplementModel(make_config(dataset))
 repo=ROOT.parent/('T3Time' if model=='T3Time' else 'Time-LLM' if model=='TimeLLM' else 'Time-Series-Library');sys.path.insert(0,str(repo));cfg=defaults(repo/('run_main.py' if model=='TimeLLM' else 'run.py'))
 cfg.update(model=model,task_name='long_term_forecast',data='efp_batch',seq_len=48,label_len=24,pred_len=24,token_len=24,enc_in=5,dec_in=5,c_out=5,d_model=128,d_ff=256,n_heads=4,e_layers=2,d_layers=1,dropout=.1,embed='timeF',freq='h',factor=1,activation='gelu',output_attention=False,down_sampling_layers=2,down_sampling_window=2,down_sampling_method='avg')
 if dataset=='CHO':cfg.update(patch_len=24,stride=12,padding_patch='end',moving_avg=25,distil=True,mix_embeds=False,ma_type='ema',alpha=.3,beta=.3,kernel_size=25,channel_independence=1,decomp_method='moving_avg',use_norm=1,top_k=5,num_kernels=6,revin=1,individual=False)
 if model=='TimeLLM':
  cfg.update(d_model=16,d_ff=32,patch_len=16,stride=8,llm_model='GPT2',llm_dim=768,llm_layers=6,prompt_domain=1)
  cfg['content']='CHO-K1 cultivation with dissolved carbon dioxide, optical transmission, permittivity, pH and dissolved oxygen measurements. Optical and dielectric measurements are process signals, not direct cell-count labels.' if dataset=='CHO' else (repo/'dataset/prompt_bank'/f'{dataset}.txt').read_text()
 module=importlib.import_module('models.'+model)
 if model=='T3Time':return module.TriModal(device='cuda:0',channel=32,num_nodes=5,seq_len=48,pred_len=24,dropout_n=.2,d_llm=768,e_layer=1,d_layer=1,head=8)
 return module.Model(SimpleNamespace(**cfg))
def inputs(dataset,data,model):
 if model!='T3Time':return None
 path=ROOT/'dataset'/dataset/(f'CHO__gpt2_h24.pt' if dataset=='CHO' else dataset+'__gpt2_window_semantic.pt');c=torch.load(path,map_location='cpu');lookup={int(s):i for i,s in enumerate(c['starts'])};return c['window_embeddings'][[lookup[int(s)] for s in data['starts']]].float()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--model',choices=MODELS,required=True);ap.add_argument('--predictions-only',action='store_true');args=ap.parse_args();ds=args.dataset;m=args.model
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2','Benchmark uses physical GPU2 only'
 torch.set_num_threads(2);torch.manual_seed(2021);np.random.seed(2021);torch.cuda.manual_seed_all(2021);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 data=torch.load(OUT/ds/'samples.pt',map_location='cpu');net=build(ds,m).float().cuda().eval();path=checkpoint(ds,m);state=torch.load(path,map_location='cpu');state={(k[7:] if k.startswith('module.') else k):v for k,v in state.items()};net.load_state_dict(state,strict=True);del state
 emb=inputs(ds,data,m)
 def forward(x,mark,ee=None):
  if m=='VarBridge':return net(x,mark,None,mark)[:,-24:]
  if m=='T3Time':return net(x,torch.zeros(len(x),48,4,device=x.device),ee.transpose(1,2).unsqueeze(-1))
  dec=torch.cat([x[:,-24:],torch.zeros(len(x),24,5,device=x.device)],1)
  return net(x,torch.zeros(len(x),48,4,device=x.device),dec,torch.zeros(len(x),48,4,device=x.device))[:,-24:]
 pred=[]
 with torch.no_grad():
  for k in range(0,len(data['x']),32):
   y=forward(data['x'][k:k+32].cuda(),data['mark'][k:k+32].cuda(),None if emb is None else emb[k:k+32].cuda());assert torch.isfinite(y).all();pred.append(y.cpu())
 pred=torch.cat(pred);assert pred.shape==data['y'][:,:24].shape
 torch.save(dict(prediction=pred,starts=data['starts'],checkpoint=str(path),precision='float32'),OUT/ds/(m+'_benchmark_predictions.pt'))
 if args.predictions_only:
  print('CURRENT_SUPPLEMENT_PREDICTIONS_ONLY_COMPLETE',ds,m,flush=True);return
 print('CURRENT_SUPPLEMENT_PREDICTIONS_ONLY_COMPLETE',ds,m,flush=True)
if __name__=='__main__':main()
