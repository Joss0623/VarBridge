"""Resume unchanged V3 on4090; path-aware source guards and hardware audit only."""
import sys,os,json,time,hashlib,copy,random,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import run_varbridge_text_finetune_fast_v3 as v
import torch,numpy as np
D=v.D;OLD='/home/qsmx/VarBridge_supplement/pycodes/GreyFerm'
def guard():
 for name in ['migration_protected_hashes.json','migration_implementation_hashes.json','fast_source_hash.json','v3_source_hash.json']:
  for p,h in json.loads((D/name).read_text()).items():
   q=Path(p.replace(OLD,str(ROOT)));assert v.e.sha(q)==h,str(q)
def freegpu():
 z=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 assert all(line.split(',')[0].strip()!=uuid or int(line.split(',')[1])==os.getpid() for line in z.splitlines()),'GPU1 occupied'
 return uuid
v.guard=guard;v.v1.guard=guard;v.e.guard=guard
if sys.argv[1]=='benchmark':
 v.e.setup();guard();uuid=freegpu();torch.cuda.set_per_process_memory_fraction(.64)
 z=torch.load(v.CK/'EFP_long_sub2_layernorm_position_latest_resume_v3.pth',map_location='cpu');fixed=torch.load(v.CK/'EFP_long_sub2_layernorm_position_frozen_components_v3.pth',map_location='cpu');assert v.e.sha(v.CK/'EFP_long_sub2_layernorm_position_frozen_components_v3.pth')==z['frozen_sha256']
 n=v.Online('EFP_long_sub2','layernorm_position').cuda();n.load_state_dict(dict(fixed,**z['model_mutable']));del fixed
 opt=torch.optim.Adam([p for p in n.parameters() if p.requires_grad],lr=1e-4);ds=v.e.Windows('EFP_long_sub2','train',24);part=torch.load(v.R/'EFP_long_sub2_tokens.pt',map_location='cpu')['train'];ids=torch.randperm(len(ds),generator=torch.Generator().manual_seed(2021+z['epoch'])).tolist()[z['next_offset']:z['next_offset']+126]
 if len(ids)!=126:ids=list(range(126))
 def reset():
  with torch.no_grad():
   for k,p in n.state_dict().items():
    if k in z['model_mutable']:p.copy_(z['model_mutable'][k])
  opt.load_state_dict(copy.deepcopy(z['optimizer']));torch.set_rng_state(z['torch_rng']);torch.cuda.set_rng_state_all(z['cuda_rng']);random.setstate(z['python_rng']);np.random.set_state(z['numpy_rng'])
 reset();a=v.one_step(n,opt,ds,part,ids,False,True);wa={k:p.detach().cpu().clone() for k,p in n.named_parameters() if p.requires_grad}
 reset();b=v.one_step(n,opt,ds,part,ids,True,True);wb={k:p.detach().cpu().clone() for k,p in n.named_parameters() if p.requires_grad}
 err=float((a['prediction']-b['prediction']).abs().max());weight=max(float((wa[k]-wb[k]).abs().max()) for k in wa);assert err<1e-4 and weight<1e-4,(err,weight)
 times=[]
 for _ in range(3):
  freegpu();reset();q=v.one_step(n,opt,ds,part,ids,True,False);times.append(q['seconds'])
 out={'uuid':uuid,'physical_gpu':1,'FP32':True,'global_batch':126,'source_unchanged':True,'prediction_max_abs':err,'post_Adam_parameter_max_abs':weight,'batch_seconds':times,'source_checkpoint_epoch':z['epoch'],'next_offset':z['next_offset'],'checkpoint_sha256':v.e.sha(v.CK/'EFP_long_sub2_layernorm_position_latest_resume_v3.pth')};v.e.savej(D/'4090_migration_position_benchmark.json',out);print(json.dumps(out),flush=True)
else:
 guard();freegpu();assert (D/'4090_migration_position_benchmark.json').exists();v.train('EFP_long_sub2','layernorm_position')
