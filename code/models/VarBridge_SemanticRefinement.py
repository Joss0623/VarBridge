"""Observation-conditioned residual semantic correction, without any relation graph."""
import torch
from torch import nn
from models.KIBR_PatchTST_BoundMechanism import Model as Original
class SemanticRepair(nn.Module):
 def __init__(self,d=128):
  super().__init__();self.semantic=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,d),nn.GELU());self.numeric=nn.Sequential(nn.Linear(d+5,d),nn.GELU(),nn.LayerNorm(d));self.attention=nn.MultiheadAttention(d,4,dropout=.1,batch_first=True)
  self.delta=nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.Linear(d,768));self.gate=nn.Sequential(nn.Linear(3*d,32),nn.GELU(),nn.Linear(32,1));nn.init.normal_(self.delta[-1].weight,std=.001);nn.init.zeros_(self.delta[-1].bias);nn.init.constant_(self.gate[-1].bias,-1.)
 def forward(self,base,h,x):
  z=x.transpose(1,2);stats=torch.stack([z.mean(-1),z.std(-1,unbiased=False),z[...,0],z[...,-1],z[...,-1]-z[...,0]],-1);n=self.numeric(torch.cat([h,stats],-1));s=self.semantic(base);context,_=self.attention(s+n,n,n,need_weights=False);joint=torch.cat([s,n,context],-1);delta=.5*base.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)*torch.tanh(self.delta(joint));gate=torch.sigmoid(self.gate(joint));out=base+gate*delta
  self.last_base=base.detach();self.last_refined=out.detach();self.last_gate=gate.detach();self.last_relative_correction=((out-base).norm(dim=-1)/base.norm(dim=-1).clamp_min(1e-6)).detach();return out
class Model(nn.Module):
 def __init__(self,config,mode,semantic=None):
  super().__init__();self.model=Original(config).float();self.mode=mode
  if mode!='original':
   assert semantic.shape==(5,768);self.register_buffer('semantic_base',semantic.float().clone())
  if mode in ['llm_repair','neutral_repair']:self.repair=SemanticRepair(config.d_model)
 def train(self,mode=True):
  super().train(mode);self.model.patch_backbone.eval();return self
 def numerical_state(self,x):
  # The same frozen numerical encoder used by original VarBridge, observed inputs only.
  with torch.no_grad():
   v=x-x.mean(1,keepdim=True);v=v/torch.sqrt(v.var(1,keepdim=True,unbiased=False)+1e-5);p,_=self.model.patch_backbone.patch_embedding(v.permute(0,2,1));h,_=self.model.patch_backbone.encoder(p);return h.mean(1).reshape(len(x),5,-1)
 def forward(self,x,m):
  if self.mode=='original':return self.model(x,m,None,m)
  base=self.semantic_base[None].expand(len(x),-1,-1)
  if self.mode in ['llm_repair','neutral_repair']:base=self.repair(base,self.numerical_state(x),x)
  mark=m.clone();mark[:,:,1:6,:]=base[:,None].expand(-1,5,-1,-1)
  # Only the bound-mechanism bank changes. Observed/stage/operation slots and all original fusion/prediction paths are retained.
  return self.model(x,mark,None,mark)
