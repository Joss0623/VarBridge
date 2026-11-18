"""Generate fixed semantic drafts from definitions, then freeze RoBERTa embeddings."""
import os,sys,json,time,hashlib,fcntl,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
R=ROOT/'test_results/varbridge_semantic_refinement';R.mkdir(exist_ok=True)
def save(p,z):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(z,indent=2));t.replace(p)
def main():
 lock=(R/'preparation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (R/'PREPARATION_COMPLETE.json').exists():return
 account=ROOT/'test_results/varbridge_api_accounting';bl=(account/'budget.lock').open('a');fcntl.flock(bl,fcntl.LOCK_EX)
 c=json.loads((Path.home()/'.config/attentionpit/siliconflow.json').read_text());references=json.loads((ROOT/'test_results/varbridge_lhc_gemp/api/researcher_reference.json').read_text());receipt=R/'api_receipts.jsonl'
 for task,d in references.items():
  path=R/(task+'_semantic_draft.json')
  if path.exists():continue
  msg=[{'role':'system','content':'Write a conservative reusable semantic reference for bioprocess sensor forecasting. Ground every statement in the supplied definitions. Express uncertainty, not deterministic causal laws. Never invent kinetics, signs, biological stage labels, sensor calibration, units or numerical thresholds. No relation matrix or graph. Return JSON only.'},{'role':'user','content':json.dumps({'process':d['process'],'variables':[{'index':i,'name':n,'definition':t} for i,(n,t) in enumerate(zip(d['names'],d['definitions']))],'request':'Return {variables:[{index:0,measurement:"...",possible_explanations:"...",conditions:"...",limitations:"..."},...]} for all five indices in order. Each field <=26 English words. Measurement: what this variable represents. Possible_explanations: possible explanations of changes supported by supplied definitions, without assigning one cause. Conditions: context required to interpret it; say unavailable when not supplied. Limitations: what the measurement alone cannot establish. Do not use measurements, forecasting targets, dataset split or results. Keep genuine ambiguities. Do not generate a graph.'})}]
  payload={'model':'deepseek-ai/DeepSeek-V3.2','messages':msg,'temperature':.01,'max_tokens':2200,'enable_thinking':False,'response_format':{'type':'json_object'},'stream':False}
  # Reserve conservatively at30CNY/million tokens, input tokens bounded by UTF8 bytes plus margin.
  bound=(len(json.dumps(msg).encode())+2200+1024)*30/1e6;budget=json.loads((account/'cumulative_budget.json').read_text());assert budget['known_cumulative_upper_cny']+bound<=budget['cap_cny']
  pending=R/'api_pending.json';assert not pending.exists(),'Ambiguous prior call: inspect before retry';rec={'task':task,'payload':payload,'time':time.time(),'reserved_upper_cny':bound};save(pending,rec);budget['known_cumulative_upper_cny']+=bound;save(account/'cumulative_budget.json',budget)
  req=urllib.request.Request(c['base_url']+'/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+c['api_key'],'Content-Type':'application/json'})
  try:
   with urllib.request.urlopen(req,timeout=180) as q:out=json.load(q)
   rec.update(status='received',response=out);z=json.loads(out['choices'][0]['message']['content']);rows=z['variables'];assert [q['index'] for q in rows]==list(range(5))
   for row in rows:
    for key in ['measurement','possible_explanations','conditions','limitations']:assert isinstance(row[key],str) and 0<len(row[key])<=500
   save(path,{'task':task,'process':d['process'],'names':d['names'],'definitions':d['definitions'],'variables':rows,'model':payload['model'],'inputs':'Definitions only; no numerical observations, split, labels or results','api_seed':'not provided by service'})
  except Exception as exc:
   rec.update(status='received_invalid' if 'out' in locals() else 'ambiguous_failure',error_type=type(exc).__name__)
   with receipt.open('a') as f:f.write(json.dumps(rec)+'\n')
   raise
  with receipt.open('a') as f:f.write(json.dumps(rec)+'\n')
  pending.unlink();print('DRAFT_COMPLETE',task,flush=True)
 fcntl.flock(bl,fcntl.LOCK_UN)
 import torch
 from transformers import AutoTokenizer,AutoModel
 from run_varbridge_text_finetune import Texts
 torch.set_num_threads(2);tok=AutoTokenizer.from_pretrained('FacebookAI/roberta-base',local_files_only=True);net=AutoModel.from_pretrained('FacebookAI/roberta-base',local_files_only=True,add_pooling_layer=False).eval().requires_grad_(False)
 def encode(texts):
  z=tok(texts,padding=True,truncation=False,return_tensors='pt');assert z['input_ids'].shape[1]<=320
  with torch.no_grad():h=net(**z).last_hidden_state;return (h*z['attention_mask'][:,:,None]).sum(1)/z['attention_mask'].sum(1)[:,None]
 for task in references:
  z=json.loads((R/(task+'_semantic_draft.json')).read_text());texts=[]
  for name,row in zip(z['names'],z['variables']):texts.append(f'Mechanism for {name}: Measurement: {row["measurement"]} Possible explanations: {row["possible_explanations"]} Conditions: {row["conditions"]} Limitations: {row["limitations"]}')
  neutral='Mechanism for process variable: No variable-specific interpretation is supplied. Observed changes may have multiple explanations. Applicable conditions are not supplied. No variable-specific process knowledge is provided.'
  em=encode(texts+[neutral]*5)
  if task=='EFP_long_sub2':
   allstatic=encode(Texts(task).static()).half().float().reshape(5,5,3,768);center=allstatic.mean((0,1))[0];em=em.half().float()-center;em=(em/em.norm(dim=-1,keepdim=True).clamp_min(1e-6)*768**.5).half().float()
  assert torch.isfinite(em).all();tmp=R/(task+'_semantics.tmp');torch.save({'task':task,'llm':em[:5].contiguous(),'neutral':em[5:].contiguous(),'texts':texts,'neutral_text':neutral,'encoder':'FacebookAI/roberta-base','pooling':'masked mean; all supplied text fits <=320 tokens','source_draft_sha256':hashlib.sha256((R/(task+'_semantic_draft.json')).read_bytes()).hexdigest()},tmp);tmp.replace(R/(task+'_semantics.pt'));print('EMBEDDINGS_COMPLETE',task,flush=True)
 save(R/'PREPARATION_COMPLETE.json',{'tasks':list(references),'time':time.time(),'encoder_frozen':True,'public_definitions_only':True,'budget':json.loads((account/'cumulative_budget.json').read_text())})
if __name__=='__main__':main()
