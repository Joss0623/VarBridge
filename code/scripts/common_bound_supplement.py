"""Isolated model-parameter sensitivity; adopted sources and frozen backbone stay fixed."""
import ast,copy,sys,random,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.KIBR_PatchTST_BoundMechanism import Model as AdoptedModel
DATASETS=['EFP_long_sub2','CHO']
GRIDS={'alpha':[.1,.2,.3,.5,.7,.9],'fusion_heads':[1,2,4,8],'expert_width':[32,64,128,256]}
DEFAULTS={'alpha':.3,'fusion_heads':4,'expert_width':128}
def tag(parameter,value):return parameter+'_'+str(value).replace('.','p')
TASKS={tag(k,v):{k:v} for k,vs in GRIDS.items() for v in vs if v!=DEFAULTS[k]}
def make_config(dataset,**overrides):
 cfg={}
 for node in ast.walk(ast.parse((ROOT/'run.py').read_text())):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
   try:
    name=ast.literal_eval(node.args[0]);d=next(x.value for x in node.keywords if x.arg=='default');value=ast.literal_eval(d)
    if name.startswith('--'):cfg[name[2:].replace('-','_')]=value
   except (ValueError,StopIteration,TypeError,IndexError):pass
 cfg.update(task_name='long_term_forecast',seq_len=48,label_len=24,pred_len=24,token_len=24,enc_in=5,dec_in=5,c_out=5,d_model=128,d_ff=256,n_heads=4,e_layers=2,d_layers=1,dropout=.1,patch_len=24,stride=12,padding_patch='end',embed='timeF',freq='h',factor=1,activation='gelu',output_attention=False,moving_avg=25,distil=True,mix_embeds=True,ma_type='ema',alpha=.3,beta=.3,kernel_size=25,down_sampling_layers=2,down_sampling_window=2,down_sampling_method='avg',channel_independence=1,decomp_method='moving_avg',use_norm=1,top_k=5,num_kernels=6,revin=1,individual=False)
 ckpts=[ROOT/'checkpoints/cho/Backbone_h24_seed2021_full.pth'] if dataset=='CHO' else sorted((ROOT/'checkpoints').glob('long_term_forecast_'+dataset+'_noprompt_seed2021_PatchTST_*/checkpoint.pth'))
 assert len(ckpts)==1 and ckpts[0].exists(),ckpts
 cfg.update(numeric_encoder='patchtst',numeric_ckpt=str(ckpts[0]),patchtst_ckpt='',text_embed_dim=768,channel_interaction=True,early_channel_mode='patch',forecast_conditioned=True,tri_mmoe=True,dynamic_window_stats=True,separate_numeric_path=False,embedding_fusion='output',prompt_control='text',initial_cross_target='seasonal',disable_prompt_self_attn=False,use_cross_attn=True,use_dynamic_gate=True,coarse_factor=0,**{k:v for k,v in DEFAULTS.items() if k!='alpha'})
 assert set(overrides)<=set(DEFAULTS),overrides
 cfg.update(overrides)
 return SimpleNamespace(**cfg)
class SupplementModel(AdoptedModel):
 def __init__(self,config):
  # Always construct the frozen PatchTST using its original four attention heads.
  assert config.n_heads==4 and config.d_model==128
  super().__init__(config)
  heads=config.fusion_heads;width=config.expert_width
  assert heads in GRIDS['fusion_heads'] and width in GRIDS['expert_width']
  for name,module in self.named_modules():
   if isinstance(module,torch.nn.MultiheadAttention) and not name.startswith('patch_backbone.'):
    assert module.embed_dim%heads==0
    module.num_heads=heads;module.head_dim=module.embed_dim//heads
  if width!=128:
   for bank_name in ['numeric_experts','semantic_experts']:
    bank=getattr(self,bank_name)
    for i,old in enumerate(bank):
     net=torch.nn.Sequential(torch.nn.Linear(old[0].in_features,width),torch.nn.GELU(),torch.nn.Linear(width,1))
     torch.nn.init.normal_(net[-1].weight,std=.01);torch.nn.init.zeros_(net[-1].bias);bank[i]=net
  for name,module in self.patch_backbone.named_modules():
   if hasattr(module,'n_heads'):assert module.n_heads==4,(name,module.n_heads)
def main_checkpoint(dataset,seed=2021):
 p=ROOT/('checkpoints/cho_bound_mechanism' if dataset=='CHO' else 'checkpoints/efp_bound_mechanism')/((f'Fusion_h24_seed{seed}_correct.pth') if dataset=='CHO' else f'{dataset}_h24_seed{seed}_correct.pth')
 assert p.exists(),p
 return p
def main_log(dataset,seed=2021):
 return ROOT/('test_results/cho_bound_mechanism' if dataset=='CHO' else 'test_results/efp_bound_mechanism')/((f'Fusion_h24_seed{seed}_correct.log') if dataset=='CHO' else f'{dataset}_h24_seed{seed}_correct.log')
def seed_all(seed=2021):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def tensor_hash(net):
 h=hashlib.sha256()
 for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
