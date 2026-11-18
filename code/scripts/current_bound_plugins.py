"""Current VarBridge encoder compatibility experiment. No adopted source mutations."""
from pathlib import Path
import sys,copy,hashlib,random
import numpy as np,torch
from torch.utils.data import Dataset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from common_bound_supplement import make_config as original_config,main_checkpoint,main_log,seed_all
from models.KIBR_PatchTST_BoundMechanism import Model
from common_bound_ablations import Windows as CurrentWindows
OUT=ROOT/'test_results/current_bound_plugins';CK=ROOT/'checkpoints/current_bound_plugins';CACHE=OUT/'cache'
DATASETS=['EFP_long_sub2','CHO'];NUMERICS=['patchtst','timesnet'];TEXTS=['roberta','bert','gpt2']
SOURCES=['models/KIBR_PatchTST_BoundMechanism.py','models/PatchTST.py','models/TimesNet.py','data_provider/data_loader.py','scripts/cho_data.py','scripts/common_bound_supplement.py','scripts/train_cho_bound_mechanism_ddp.py','scripts/train_efp_bound_mechanism_ddp.py']
def source_hashes():return {s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES}
def numerical_checkpoint(dataset,numeric):
 if numeric=='patchtst':return Path(original_config(dataset).numeric_ckpt)
 if dataset=='CHO':return CK/'CHO_timesnet_backbone_h24_seed2021.pth'
 p=sorted((ROOT/'checkpoints').glob('long_term_forecast_EFP_long_sub2_noprompt_seed2021_TimesNet_*/checkpoint.pth'));assert len(p)==1,p;return p[0]
def config(dataset,numeric):
 cfg=original_config(dataset);cfg.numeric_encoder=numeric;cfg.numeric_ckpt=str(numerical_checkpoint(dataset,numeric));return cfg
def checkpoint(dataset,numeric,text):return CK/f'{dataset}_{numeric}_{text}_h24_seed2021.pth'
class Windows(Dataset):
 def __init__(self,dataset,split,text='roberta'):
  self.dataset=dataset;self.text=text
  if text=='roberta':
   self.base=CurrentWindows(dataset,split,24,'roberta','full');self.feature_names=self.base.feature_names;self.current=True
  elif dataset=='CHO':
   from cho_data import CHOWindows,FEATURES
   self.base=CHOWindows(split,24,None);self.base.cache=torch.load(CACHE/f'CHO_{text}_h24.pt',map_location='cpu');self.base.lookup={int(x):i for i,x in enumerate(self.base.cache['starts'])};self.feature_names=FEATURES;self.current=False
   assert self.base.cache['features']==FEATURES and set(self.base.starts)<=set(self.base.lookup)
  else:
   from data_provider.data_loader import Dataset_BatchMTSF
   self.base=Dataset_BatchMTSF(str(ROOT/'dataset'/dataset),flag=split,size=[48,24,24],data_path=dataset+'.csv',prompt_encoder=text+'_window_horizon_correct');self.feature_names=self.base.feature_names;self.current=False
   assert set(self.base.valid_indices)<=set(self.base.window_lookup)
  assert len(self)>0 and len(self.feature_names)==5
 def __len__(self):return len(self.base)
 def __getitem__(self,i):
  if self.current:return self.base[i]
  x,y,mark,bid=self.base[i]
  if self.dataset!='CHO':y=y[-24:];bid=torch.tensor(0)
  assert x.shape==(48,5) and y.shape==(24,5) and mark.shape==(5,8,768)
  assert torch.isfinite(mark).all()
  return x,y,mark,bid
