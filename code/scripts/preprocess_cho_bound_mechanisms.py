"""Frozen RoBERTa mechanism controls. Intentionally false controls are not scientific claims."""
import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer,AutoModel
from cho_data import ROOT,DATA,FEATURES,NAMES,MECHANISMS
CONTRADICTORY=[
 'Dissolved carbon dioxide is a direct cell-count measurement; production, gas exchange and supplied gas composition have no influence on this signal.',
 'Optical transmission is a calibrated biomass concentration; light scattering and absorption in the culture have no influence on this signal.',
 'Permittivity is independent of polarization and viable cell volume; cell physiology and sensor conditions have no influence on this signal.',
 'Culture pH is independent of acid-base balance and process control; metabolism, gas exchange and base addition have no influence on its dynamics.',
 'Dissolved oxygen is independent of the balance between oxygen transfer and cellular demand; operating control has no influence on this signal.'
]
UNRELATED=[
 'The orbital period of a planet depends on its distance from its star and the gravitational mass; an orbit is not a direct measurement of stellar temperature.',
 'A compiler translates program source code into executable instructions; parsing and type checking affect compilation, not the physical size of the computer.',
 'Sedimentary rock layers record deposition and erosion; the order and thickness of layers depend on geological history and local environmental conditions.',
 'The focal length of a camera lens influences its field of view; image framing also depends on sensor dimensions and the distance to the photographed object.',
 'A musical instrument produces sound through vibrating components; pitch depends on vibration frequency, while loudness depends on amplitude and acoustic coupling.'
]
def main():
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==6 and os.environ['CUDA_VISIBLE_DEVICES']=='2,3,4,5,6,7'
 torch.cuda.set_device(local);dist.init_process_group('nccl');torch.set_num_threads(2)
 print('CHO_BOUND_TEXT_RANK',rank,'GPU',local+2,flush=True)
 source='FacebookAI/roberta-base';tok=AutoTokenizer.from_pretrained(source,local_files_only=True);net=AutoModel.from_pretrained(source,local_files_only=True).cuda().eval().requires_grad_(False)
 banks={'correct':MECHANISMS,'contradictory':CONTRADICTORY,'unrelated':UNRELATED}
 jobs=[(variant,j,f'Mechanism for {name}: {text}') for variant,texts in banks.items() for j,(name,text) in enumerate(zip(NAMES,texts))]
 local_results=[]
 for index in range(rank,len(jobs),world):
  variant,j,text=jobs[index];inp=tok([text],padding=True,truncation=True,max_length=256,return_tensors='pt').to(local)
  with torch.no_grad():
   hidden=net(**inp).last_hidden_state;mask=inp['attention_mask'];emb=(hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
  assert torch.isfinite(emb).all();local_results.append((variant,j,text,emb[0].cpu()))
 parts=[None]*world;dist.all_gather_object(parts,local_results)
 if rank==0:
  result={v:{'features':FEATURES,'texts':['']*5,'embeddings':torch.empty(5,768)} for v in banks};seen=[]
  for part in parts:
   for v,j,text,emb in part:result[v]['texts'][j]=text;result[v]['embeddings'][j]=emb;seen.append((v,j))
  assert len(set(seen))==15
  for v in banks:
   print('TEXTS',v,result[v]['texts'],flush=True)
   if v!='correct':print('COSINE',v,torch.nn.functional.cosine_similarity(result['correct']['embeddings'],result[v]['embeddings']).tolist(),flush=True)
  tmp=DATA/'CHO_bound_mechanisms.tmp';torch.save(result,tmp);tmp.replace(DATA/'CHO_bound_mechanisms.pt')
  print('ALL_CHO_BOUND_TEXT_COMPLETE',flush=True)
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
