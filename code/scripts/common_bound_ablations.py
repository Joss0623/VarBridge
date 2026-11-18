"""Fixed common-model ablations; never mutate the adopted main-model implementation."""
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
VARIANTS=['shuffled_mechanism','no_mechanism','generic_description','random_semantic','random_init_text','learned_condition','swap_to_trend','cross_both','no_prompt_self_attention','no_knowledge','no_channel','no_early_channel','early_raw','no_dynamic_stats','separate_numeric_path','no_dynamic_gate','no_initial_cross_attention','sma_decomposition','dema_decomposition','no_forecast_conditioning','additive_fusion','dual_moe','coarse_scale_2','coarse_scale_4']
DEFINITIONS={
'shuffled_mechanism':'Only own-target mechanism content reassigned cyclically; target binding retained.',
'no_mechanism':'Own mechanism replaced with fixed no-mechanism placeholder; general context retained.',
'generic_description':'Own mechanism replaced with generic process description; general context retained.',
'random_semantic':'Own mechanism replaced with fixed unrelated-domain description; general context retained.',
'random_init_text':'All observed/general/mechanism embeddings from frozen randomly initialized RoBERTa(seed2021).',
'learned_condition':'Replace all text with trainable target-specific three general tokens and one mechanism memory; no text input.',
'swap_to_trend':'Initial general-context cross-attention conditions trend instead of seasonal features.',
'cross_both':'Initial general-context cross-attention conditions both trend and seasonal features.',
'no_prompt_self_attention':'Remove residual self-attention over three general-context tokens.',
'no_knowledge':'Remove all text conditioning and forecast-conditioned semantic/expert routers; retain frozen base and numerical residual decoder. Joint path removal.',
'no_channel':'Remove both early patch and late channel attention.',
'no_early_channel':'Remove early patch channel attention only.',
'early_raw':'Move early channel attention from patches to raw normalized channels.',
'no_dynamic_stats':'Remove explicitly recomputed numerical statistics; keep initial observed-window text.',
'separate_numeric_path':'Restore independent unconditioned numerical adaptation path (inverse of historical no_separate_numeric).',
'no_dynamic_gate':'Replace trend/seasonal sigmoid gate with equal mean.',
'no_initial_cross_attention':'Replace initial general-context cross-attention with additive general-context mean.',
'sma_decomposition':'Replace EMA with SMA(kernel25).',
'dema_decomposition':'Replace EMA with DEMA(alpha.3,beta.3).',
'no_forecast_conditioning':'Remove target/horizon semantic retrieval and multi-expert routing; retain general-text seasonal conditioning, statistics and base-plus-residual. Joint path removal.',
'additive_fusion':'Replace multi-expert softmax forecast fusion with gated additive semantic residual; retain numerical statistics.',
'dual_moe':'Alternative direct numerical/semantic dual-expert predictor without frozen base forecast candidate; joint forecast/fusion change.',
'coarse_scale_2':'Add pooled coarse-scale trend and seasonal paths, factor2.',
'coarse_scale_4':'Add pooled coarse-scale trend and seasonal paths, factor4.'}
def configure(cfg,v):
 assert v in VARIANTS or v=='full'
 changes={
 'learned_condition':dict(prompt_control='learned'),
 'swap_to_trend':dict(initial_cross_target='trend'),
 'cross_both':dict(initial_cross_target='both'),
 'no_prompt_self_attention':dict(disable_prompt_self_attn=True),
 'no_knowledge':dict(mix_embeds=False,forecast_conditioned=False,tri_mmoe=False,dynamic_window_stats=False),
 'no_channel':dict(channel_interaction=False,early_channel_mode='none'),
 'no_early_channel':dict(early_channel_mode='none'),
 'early_raw':dict(early_channel_mode='raw'),
 'no_dynamic_stats':dict(dynamic_window_stats=False),
 'separate_numeric_path':dict(separate_numeric_path=True),
 'no_dynamic_gate':dict(use_dynamic_gate=False),
 'no_initial_cross_attention':dict(use_cross_attn=False),
 'sma_decomposition':dict(ma_type='sma'),
 'dema_decomposition':dict(ma_type='dema'),
 'no_forecast_conditioning':dict(forecast_conditioned=False,tri_mmoe=False),
 'additive_fusion':dict(tri_mmoe=False),
 'dual_moe':dict(tri_mmoe=False,direct_forecast=True,dual_moe=True),
 'coarse_scale_2':dict(coarse_factor=2),
 'coarse_scale_4':dict(coarse_factor=4)}
 cfg.update(changes.get(v,{}));return cfg

class Windows(torch.utils.data.Dataset):
 def __init__(self,dataset,split,horizon,encoder='roberta',variant='full'):
  self.variant=variant;self.dataset=dataset;self.horizon=horizon
  encoder='robertarandom' if variant=='random_init_text' else 'roberta'
  if dataset=='CHO':
   from cho_data import CHOWindows
   self.base=CHOWindows(split,horizon,encoder);self.feature_names=self.base.cache['features']
   if variant=='random_init_text':
    self.knowledge=None
   else:
    banks=torch.load(ROOT/'dataset/CHO/CHO_bound_mechanisms.pt',map_location='cpu')
    self.knowledge=banks['correct']['embeddings']
    cache=self.base.cache
    if variant=='shuffled_mechanism':self.knowledge=cache['shuffled_knowledge_embeddings'][:,0,:]
    if variant=='random_semantic':self.knowledge=banks['unrelated']['embeddings']
    if variant in ('no_mechanism','generic_description'):
     controls=torch.load(ROOT/'dataset/CHO/CHO_ablation_mechanisms.pt',map_location='cpu')
     self.knowledge=controls[variant]['embeddings']
   assert set(self.base.starts)<=set(self.base.lookup),'incomplete CHO cache'
  else:
   from data_provider.data_loader import Dataset_BatchMTSF
   # Keep correct general context and swap only mechanism bank entries for content controls.
   self.base=Dataset_BatchMTSF(str(ROOT/'dataset'/dataset),flag=split,size=[48,24,horizon],data_path=dataset+'.csv',prompt_encoder=encoder+'_window_horizon_correct')
   self.feature_names=self.base.feature_names
   assert set(self.base.valid_indices)<=set(self.base.window_lookup),'incomplete EFP cache'
   v={'shuffled_mechanism':'shuffled','no_mechanism':'none','generic_description':'generic','random_semantic':'random'}.get(variant)
   self.knowledge=None
   if v:
    h=24 if horizon==48 else horizon
    cache=torch.load(ROOT/'dataset'/dataset/(dataset+'__roberta_window_semantic'+('' if h==24 else '_h'+str(h))+'.pt'),map_location='cpu')
    assert cache['features']==self.feature_names
    self.knowledge=cache['knowledge_embeddings'][cache['variants'].index(v),:,0,:].float()
  assert len(self.feature_names)==5 and len(self)>0
 def __len__(self):return len(self.base)
 def __getitem__(self,i):
  if self.dataset=='CHO':x,y,mark,bid=self.base[i]
  else:
   x,y,mark,_=self.base[i];bid=torch.tensor(0);y=y[-self.horizon:]
  mark=mark.clone()
  if self.knowledge is not None:mark[:,1:6]=self.knowledge[None,:,:]
  assert mark.shape==(5,8,768) and x.shape==(48,5) and y.shape==(self.horizon,5)
  assert torch.isfinite(mark).all()
  return x,y,mark,bid
