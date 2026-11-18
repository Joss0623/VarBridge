"""Original EFP partitions/scaling/caches with exact windows for the common bound model."""
from pathlib import Path
import sys,torch
from torch.utils.data import Dataset
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_provider.data_loader import Dataset_BatchMTSF
class EFPBoundWindows(Dataset):
 def __init__(self,dataset,flag,horizon,encoder='roberta'):
  assert dataset in ['EFP_long_sub1','EFP_long_sub2','EFP_long_sub3'] and encoder=='roberta'
  self.horizon=horizon
  self.base=Dataset_BatchMTSF(str(ROOT/'dataset'/dataset),flag=flag,size=[48,24,horizon],data_path=dataset+'.csv',prompt_encoder='roberta_window_horizon_correct')
  assert len(self.base.feature_names)==5 and self.base.knowledge_embeddings.shape==(5,3,768)
  missing=set(self.base.valid_indices)-set(self.base.window_lookup)
  assert not missing,('incomplete observed cache',dataset,flag,horizon,len(missing))
  assert torch.isfinite(self.base.window_embeddings).all() and torch.isfinite(self.base.knowledge_embeddings).all()
  self.feature_names=self.base.feature_names
 def __len__(self):return len(self.base)
 def __getitem__(self,i):
  x,y,mark,_=self.base[i]
  assert x.shape==(48,5) and mark.shape==(5,8,768)
  return x,y[-self.horizon:],mark,torch.tensor(0)
