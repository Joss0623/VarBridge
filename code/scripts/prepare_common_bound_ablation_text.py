"""Prepare only missing CHO ablation caches on six ranks, preserving original text/cache."""
from pathlib import Path
import os,torch
import torch.distributed as dist
from transformers import AutoModel,AutoTokenizer
from cho_data import ROOT,DATA,NAMES
def main():
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
 assert world==6 and os.environ['CUDA_VISIBLE_DEVICES']=='2,3,4,5,6,7'
 torch.cuda.set_device(local);dist.init_process_group('nccl')
 print('COMMON_ABLATION_TEXT_RANK',rank,'GPU',local+2,flush=True)
 out=DATA/'CHO_ablation_mechanisms.pt'
 if not out.exists():
  source='FacebookAI/roberta-base';tok=AutoTokenizer.from_pretrained(source,local_files_only=True);net=AutoModel.from_pretrained(source,local_files_only=True).cuda().eval().requires_grad_(False)
  variants={'no_mechanism':'No variable-specific mechanism is supplied.','generic_description':'CHO-K1 cultivation is monitored using process measurements under batch or fed-batch operation.'}
  rows=[]
  for k,(v,text) in enumerate((v,text) for v,text in variants.items() for name in NAMES):
   j=k%5
   if k%world!=rank:continue
   phrase=f'Mechanism for {NAMES[j]}: {text}'
   inp=tok([phrase],padding=True,truncation=True,max_length=256,return_tensors='pt').to(local)
   with torch.no_grad():hidden=net(**inp).last_hidden_state;mask=inp['attention_mask'];emb=(hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
   rows.append((v,j,phrase,emb[0].cpu()))
  parts=[None]*world;dist.all_gather_object(parts,rows)
  if rank==0:
   cache={v:{'texts':['']*5,'embeddings':torch.empty(5,768)} for v in variants};seen=[]
   for part in parts:
    for v,j,text,emb in part:cache[v]['texts'][j]=text;cache[v]['embeddings'][j]=emb;seen.append((v,j))
   assert len(set(seen))==10
   tmp=out.with_suffix('.tmp');torch.save(cache,tmp);tmp.replace(out)
  del net;torch.cuda.empty_cache()
 dist.barrier()
 # Reuse the exact existing CHO text construction, changing only frozen encoder initialization.
 src=(ROOT/'scripts/preprocess_cho_ddp.py').read_text()
 src=src.replace('from transformers import AutoModel,AutoTokenizer','from transformers import AutoModel,AutoTokenizer,AutoConfig')
 src=src.replace("for encoder,source in [('roberta','FacebookAI/roberta-base'),('gpt2','/home/home_new/qsmx/.cache/modelscope/hub/models/AI-ModelScope/gpt2')]:","for encoder,source in [('robertarandom','FacebookAI/roberta-base')]:")
 src=src.replace("model = AutoModel.from_pretrained(source).cuda().eval()", "model = AutoModel.from_pretrained(source).cuda().eval()") # Original contains local_files_only.
 src=src.replace("model=AutoModel.from_pretrained(source,local_files_only=True).cuda().eval();model.requires_grad_(False)", "torch.manual_seed(2021);model=AutoModel.from_config(AutoConfig.from_pretrained(source,local_files_only=True)).cuda().eval();model.requires_grad_(False)")
 src=src.replace("dist.init_process_group('nccl')", "None")
 assert "from_config" in src
 exec(compile(src,'CHO_random_text_recipe','exec'),{'__name__':'__main__'})
 print('ALL_COMMON_ABLATION_TEXT_COMPLETE',flush=True)
if __name__=='__main__':main()
