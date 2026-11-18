from pathlib import Path
import ast,sys,importlib,torch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
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

def enforce_efficiency_fp32(net):
 # Instance-only removal of a hardcoded BF16 cast for this FP32 comparison.
 # Original baseline files, architecture and parameters are unchanged.
 import inspect,textwrap,types
 source=inspect.getsource(type(net).forecast)
 assert source.count('x_enc.to(torch.bfloat16)')==1
 namespace={};exec(textwrap.dedent(source.replace('x_enc.to(torch.bfloat16)','x_enc')),sys.modules[type(net).__module__].__dict__,namespace)
 net.forecast=types.MethodType(namespace['forecast'],net)
 return net
