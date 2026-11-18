"""Supplementary experiment protocol and isolated VarBridge context controls.
No adopted source edits. No change to current manuscript or formal result files.
"""
import os,sys,time,json,csv,math,hashlib,random,argparse,fcntl,subprocess
from pathlib import Path
ROOT=Path('/home/home_new/qsmx/pycodes/GreyFerm')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
OUT=ROOT/'test_results/varbridge_lhc_gemp'
CK=ROOT/'checkpoints/varbridge_lhc_gemp'
TASKS=['EFP_long_sub2','CHO']
VARIANTS=['own_target','contradictory','noisy_text','correlation_pool','group_1','group_2','group_3','group_4']
SOURCES=['models/KIBR_PatchTST_BoundMechanism.py','models/PatchTST.py','data_provider/data_loader.py','scripts/cho_data.py','scripts/common_bound_supplement.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writecsv(path,rows):
 if not rows:return
 tmp=Path(str(path)+'.tmp')
 with tmp.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 tmp.replace(path)
def guard():
 expected=json.loads((OUT/'adopted_source_hashes.json').read_text())
 assert all(sha(ROOT/p)==h for p,h in expected.items()),'Adopted source mutation'
def init():
 if (OUT/'figure_table_correspondence.csv').exists() and (OUT/'adopted_source_hashes.json').exists():
  guard();return
 OUT.mkdir(exist_ok=True);CK.mkdir(exist_ok=True)
 for d in ['figures','traces','logs']: (OUT/d).mkdir(exist_ok=True)
 if not (OUT/'adopted_source_hashes.json').exists():
  (OUT/'adopted_source_hashes.json').write_text(json.dumps({s:sha(ROOT/s) for s in SOURCES},indent=2))
 guard()
 mapping=[
 ('LHC','Table I','Training-only descriptive statistics','new CPU table','descriptive_statistics'),
 ('LHC','Table II','Baseline modality / channel-strategy inventory','reuse audited main protocol','baseline_inventory'),
 ('LHC','Table III','Variable meanings and own-target bindings; no invented expert groups','new source-derived table','target_bindings'),
 ('LHC','Table IV','Four-task four-horizon main forecasting; original three seeds','reuse 100/at-most-10 formal runs','main_results'),
 ('LHC','Table V','Three-seed means, sample SD and t intervals; no fixed-seed pseudo-replication','derive from existing distinct-seed runs','statistical_summary'),
 ('LHC','Table VI','LLM-generated descriptions: latency, phrasing stability, downstream prediction','waiting API location; expert agreement only with independent expert labels','llm_comparison'),
 ('LHC','Figure 1','Original VarBridge architecture','current PNG','architecture'),
 ('LHC','Figure 2','Training-value distribution gallery','new CPU PNG','distributions'),
 ('LHC','Figure 3','Erroneous-text training trajectories and context-weight changes','new isolated 100-epoch fits','context_learning'),
 ('LHC','Figure 4','Training-only numeric level/difference relations','reuse / restyle existing','correlation'),
 ('LHC','Figure 5','Ablation bars from current 100-epoch ablations','reuse audited original results','ablation_bars'),
 ('LHC','Figure 6','Predictions with scatter and marginal densities','reuse current matched-window outputs','joint_predictions'),
 ('LHC','Figure 7','Actual model parameters: EMA, fusion heads, expert width','reuse all 22 completed 100-epoch fits','model_sensitivity'),
 ('LHC','Figure 8','Lookback sensitivity with original process-disjoint splits','new variable-L cache/backbone/fusion fits; baselines 10 epochs','lookback'),
 ('LHC','Figure 9','Actual description prompt, returned text and target binding','waiting actual API response; no invented expert truth','text_workflow'),
 ('LHC','Figure 10','Cross-variable attention, general-context attention and forecast-step routing','reuse saved traces; not single-key semantic retrieval','attention'),
 ('LHC','Figure 11','Controlled inference cost and separately labelled 100/10 training budgets','reuse audited inference; do not mix old shared-rank training durations','efficiency'),
 ('LHC','Figure 12','Own-target versus numeric/group-pooled mechanism conditioning','new 100-epoch input controls; no hypergraph claim','binding_controls'),
 ('GEMP','Table I','Training-only descriptive statistics','same evidence as LHC Table I','descriptive_statistics'),
 ('GEMP','Table II','Baseline inventory','same as LHC Table II','baseline_inventory'),
 ('GEMP','Table III','Main results','same as LHC Table IV','main_results'),
 ('GEMP','Table IV','LLM source and prompt stability','same as LHC Table VI; actual available models only','llm_comparison'),
 ('GEMP','Figure 1','Full architecture and bioprocess scene','current first/third PNGs','architecture'),
 ('GEMP','Figure 2','Variable overlap / meanings and empirical distributions','new CPU gallery + current scenario PNG','distributions'),
 ('GEMP','Figure 3','Lagged predictive relations fitted within training batches','new diagnostic; no causal interpretation','lagged_relations'),
 ('GEMP','Figure 4','Training-only Pearson relations','same as LHC Figure 4','correlation'),
 ('GEMP','Figure 5','Core ablation results','same as LHC Figure 5','ablation_bars'),
 ('GEMP','Figure 6','Parameter/memory/prediction latency','existing same-card audited cached inference','efficiency'),
 ('GEMP','Figure 7','Scatter and marginal densities','same as LHC Figure 6','joint_predictions'),
 ('GEMP','Figure 8','Three routers along observed batch age and forecast step','new plots from original checkpoint traces','routing_trajectories'),
 ('GEMP','Figure 9','Erroneous-text learning and routing entropy','same new fits as LHC Figure 3; VarBridge has no trust lambda','context_learning'),
 ('GEMP','Figure 10','Missing-specificity context / global pooling training routing','new group_1 100-epoch trace; no CI/CD expert claim','context_learning'),
 ('GEMP','Figure 11','Corrupted description training routing','new noisy/contradictory 100-epoch traces','context_learning'),
 ('GEMP','Figure 12','Routing-by-epoch heatmaps: correct reference versus erroneous text','new snapshots; reused reference is final state only','context_learning'),
 ('GEMP','Figure 13','Lookback sensitivity','same as LHC Figure 8','lookback'),
 ('GEMP','Figure 14','Actual VarBridge parameter sensitivity, not absent Smax','same as LHC Figure 7','model_sensitivity'),
 ('GEMP','Figure 15','Reproducible text preparation / binding case','same as LHC Figure 9','text_workflow'),
 ('GEMP','Figure 16','Routing and sensor histories aligned on original timestamps','new trace-derived plots; no annotated biological phase truth','routing_sensor_alignment'),
 ]
 writecsv(OUT/'figure_table_correspondence.csv',[dict(paper=a,item=b,varbridge_counterpart=c,execution=d,family=e) for a,b,c,d,e in mapping])
 protocol="""# LHC-Net / GEMP experiment-system completion for original VarBridge
User authorization 2026-11-14. This is an isolated supplement; adopted model source, current paper and formal results are not overwritten.
## Budget
New VarBridge adaptation/control fits: seed2021, EXACT100 epochs, patience100, global126, FP32 Adam Huber delta1, existing dual-sigmoid LR, original frozen backbone and text caches. Other newly trained forecasting baselines: EXACT10 epochs. Existing formal three-seed main results remain unchanged and labelled original at-most10 baseline budget. Historical100-epoch efficiency comparison is retained as separate evidence, never relabelled10.
## Scope
Use EFP-sub2 and CHO H24 for new targeted fits; all four original tasks for descriptive statistics and existing-result summaries. EFP subviews are variable subsets, NOT known fermentation stages. Training/validation/test memberships remain frozen. No API receives sensor trajectories, validation or test targets; only public variable definitions, source descriptions and process context.
## Interpretation
Original VarBridge has no hypergraph incidence, GEMP prior dependence score/trust factor, or CI/CD pair of experts. Correspondence is by scientific question, never invented modules. Own-target binding alternatives are INPUT-CONTEXT controls, not adopted architecture changes. group_5 is exact own-target identity and reuses full results. Correct, shuffled/generic/unrelated/learned/random-encoder fitted controls already exist and are reused when compatible.
## New input controls
contradictory: intentionally false descriptions, same frozen encoder recipe.
noisy_text: deterministic seed2021 orthogonal random perturbations of standardized mechanism vectors with matched vector norm.
correlation_pool: row-normalized absolute TRAIN within-batch first-difference correlations pool static mechanism vectors.
group_1..4: KMeans(seed2021,n_init20) on TRAIN-only correlation profiles groups variables; each own-slot receives its group mean mechanism vector. Numerical histories still have all five sensors; no model module changed. group_5 is original identity reuse.
No selection by test metrics. Every condition is recorded, including poor scores.
## Diagnostics
Initial and every-epoch fixed TRAIN-window routing; best-checkpoint test only after100 epochs. Save mean routing, entropy, parameter snapshots and paired per-window errors. Fixed train probe first32 chronological windows. Batch-level bootstrap does not imply training-seed uncertainty. Across-seed statistics use existing actual seeds2021/2022/2023; never treat timing repetitions as seeds.
## LLM comparison
No expert labels are fabricated. QWK/NDCG/FMI require independent expert annotations; until supplied use format validity, five-call variability, prompt paraphrase stability, latency and downstream errors. Record actual API model identifier/revision, prompt, response, usage and temperature support. No secret values printed or put in results. API path pending.
## Hardware
Select only currently empty4090 GPUs with no compute processes, >=20000MiB free, util<=5 and host available>=50000MiB, three snapshots. Per-GPU flock, preflight before training, one worker per selected GPU. Re-check exclusivity; terminate only own task if foreign compute appears. No other-user signals or allocation changes. Unique task markers; no duplicate completed/live tasks. Recovery archives incomplete attempts, never replaces formal outputs.
## Files
All experimental code, data, checkpoints, raw traces and result tables stay on server. Only final manuscript PNGs may be synchronized locally. No local experimental JSON/CSV. Current paper unchanged until audited result integration.
"""
 (OUT/'protocol.md').write_text(protocol)
 print('CORRESPONDENCE_READY',len(mapping),'source figure/table entries',flush=True)
def bank(dataset,variant,device='cpu'):
 import numpy as np,torch
 from common_bound_ablations import Windows
 from analyze_varbridge_timebase import raw_task
 from sklearn.cluster import KMeans
 ds=Windows(dataset,'train',24)
 correct=ds[0][2][0,1:6].clone().float()
 raw,features,ids,mean,std,_=raw_task(dataset)
 xs=[]
 for b in ids['train']:
  z=(raw[raw.batch_id==b][features].to_numpy(float)-mean)/std;xs.append(np.diff(z,axis=0))
 corr=np.corrcoef(np.concatenate(xs),rowvar=False);assert np.isfinite(corr).all()
 matrix=np.eye(5);groups=list(range(5))
 if variant.startswith('group_'):
  k=int(variant[-1]);groups=KMeans(n_clusters=k,random_state=2021,n_init=20).fit_predict(corr).tolist() if k>1 else [0]*5
  matrix=np.array([[float(groups[i]==groups[j]) for j in range(5)] for i in range(5)]);matrix/=matrix.sum(1,keepdims=True)
 elif variant=='correlation_pool':
  matrix=abs(corr);matrix/=matrix.sum(1,keepdims=True)
 if variant.startswith('group_') or variant=='correlation_pool':
  result=torch.from_numpy(matrix).float()@correct
 elif variant=='own_target':
  result=correct.clone()
 elif variant=='noisy_text':
  g=torch.Generator().manual_seed(2021);noise=torch.randn(correct.shape,generator=g)
  noise=noise-(noise*correct).sum(-1,keepdim=True)*correct/correct.square().sum(-1,keepdim=True).clamp_min(1e-10)
  result=correct+noise/noise.norm(dim=-1,keepdim=True)*correct.norm(dim=-1,keepdim=True)
  result=result/result.norm(dim=-1,keepdim=True)*correct.norm(dim=-1,keepdim=True)
 elif variant=='contradictory':
  if dataset=='CHO':
   result=torch.load(ROOT/'dataset/CHO/CHO_bound_mechanisms.pt',map_location='cpu')['contradictory']['embeddings'].float()
  else:
   # Reconstruct ORIGINAL per-role centroid from the unchanged 5-variant text inventory.
   from analyze_current_bound_supplement import TextEncoder
   from preprocess_window_semantic import NAMES,MECHANISM_UNITS,RANDOM_SEMANTICS
   enc=TextEncoder(device)
   units=[MECHANISM_UNITS[f] for f in features]
   random_units=list(RANDOM_SEMANTICS);random.Random(2021).shuffle(random_units)
   shuffled=units[1:]+units[:1]
   alltexts=[]
   for v in ['correct','shuffled','none','generic','random']:
    for j,f in enumerate(features):
     text=units[j] if v=='correct' else shuffled[j] if v=='shuffled' else 'No variable-specific mechanism is supplied.' if v=='none' else 'This process variable changes over time during fermentation.' if v=='generic' else random_units[j]
     alltexts.append(f'Mechanism for {NAMES[f]}: {text}')
   emb=enc.encode(alltexts).half().float();center=emb.mean(0)
   rebuilt=emb[:5]-center;rebuilt=(rebuilt/rebuilt.norm(dim=-1,keepdim=True)*768**.5).half().float()
   assert torch.allclose(rebuilt,correct,atol=.005,rtol=.005),('original text recipe mismatch',float((rebuilt-correct).abs().max()))
   # Original name style, not later diagnostic name-with-code.
   wrong={'hx':'Product formation is independent of metabolic activity, substrate availability and oxygen availability.','jn':'Biomass growth consumes no substrate and cannot affect broth viscosity.','doz':'Dissolved oxygen does not depend on oxygen supply or biological consumption.','phzx':'Broth pH cannot influence enzyme activity or metabolic conversion.','fi':'Airflow does not contribute to oxygen supply and has no relationship to mixing or broth viscosity.'}
   texts=[f'Mechanism for {NAMES[f]}: {wrong[f]}' for f in features];e=enc.encode(texts).half().float()-center
   result=(e/e.norm(dim=-1,keepdim=True).clamp_min(1e-6)*768**.5).half().float()
   del enc
 else:raise ValueError(variant)
 assert result.shape==(5,768) and torch.isfinite(result).all()
 return result,dict(dataset=dataset,variant=variant,features=features,groups=groups,mixing_matrix=matrix.tolist(),train_only=True)
def prepare_banks(device):
 import torch
 for ds in TASKS:
  for v in VARIANTS:
   path=OUT/(ds+'_'+v+'_bank.pt')
   if path.exists():continue
   value,meta=bank(ds,v,device);torch.save(dict(embeddings=value,metadata=meta),path);print('BANK_READY',ds,v,flush=True)
def train(dataset,variant):
 import torch,numpy as np
 from common_bound_supplement import make_config,seed_all
 from common_bound_ablations import Windows
 from models.KIBR_PatchTST_BoundMechanism import Model
 from analyze_current_bound_supplement import predict
 assert variant in VARIANTS
 guard();seed_all(2021);torch.set_num_threads(2)
 device=torch.device('cuda:0')
 path=OUT/(dataset+'_'+variant+'_bank.pt');assert path.exists()
 replacement=torch.load(path,map_location='cpu')['embeddings']
 tr,va,te=[Windows(dataset,s,24) for s in ['train','val','test']]
 n=Model(make_config(dataset)).float().to(device);opt=torch.optim.Adam([p for p in n.parameters() if p.requires_grad],lr=1e-4)
 criterion=torch.nn.HuberLoss(delta=1.)
 out=CK/(dataset+'_'+variant+'_seed2021.pth')
 frozen={k:sha(ROOT/k) for k in SOURCES}
 probe=[tr[i] for i in range(min(32,len(tr)))]
 def batch(ds,ix):
  q=[ds[i] for i in ix];x=torch.stack([p[0] for p in q]);y=torch.stack([p[1] for p in q]);m=torch.stack([p[2] for p in q]);m[:,:,1:6]=replacement[None,None]
  return x.to(device),y.to(device),m.to(device)
 def evaluate(ds):
  n.eval();se=ae=count=0
  with torch.no_grad():
   for k in range(0,len(ds),64):
    x,y,m=batch(ds,range(k,min(k+64,len(ds))));p=n(x,m,None,m);d=(p-y).double()
    assert torch.isfinite(d).all();se+=float(d.square().sum());ae+=float(d.abs().sum());count+=d.numel()
  return se/count,ae/count
 history=[];best=float('inf');bestep=0
 def snapshot(epoch):
  x,y,m=batch(tr,range(min(32,len(tr))));n.eval()
  pp,t=predict(n,x.cpu(),m.cpu(),trace=True)
  weights={k:t[k] for k in ['numeric_router','semantic_router','tri_router']}
  torch.save(dict(epoch=epoch,probe='first32 train windows',weights=weights,prediction=pp,target=y.cpu()),OUT/'traces'/f'{dataset}_{variant}_epoch{epoch:03d}.pt')
 snapshot(0)
 for ep in range(1,101):
  guard();n.train();lr=1e-4/(1+math.exp(-.5*(ep-10)))-1e-4/(1+math.exp(-.05*(ep-100)))
  for pg in opt.param_groups:pg['lr']=lr
  order=torch.randperm(len(tr),generator=torch.Generator().manual_seed(2021+ep)).tolist();total=0;start=time.time()
  for off in range(0,len(order),126):
   ids=order[off:off+126];opt.zero_grad(set_to_none=True)
   for k in range(0,len(ids),21):
    cur=ids[k:k+21];x,y,m=batch(tr,cur);p=n(x,m,None,m);loss=criterion(p,y)
    assert torch.isfinite(loss);(loss*(len(cur)/len(ids))).backward();total+=float(loss.detach())*len(cur)
   opt.step()
  vm,am=evaluate(va);snapshot(ep)
  row=dict(dataset=dataset,variant=variant,seed=2021,epoch=ep,training_loss=total/len(tr),validation_mse=vm,validation_mae=am,lr=lr,seconds=time.time()-start)
  history.append(row);writecsv(OUT/(dataset+'_'+variant+'_learning.csv'),history)
  if vm<best:best=vm;bestep=ep;torch.save(n.state_dict(),out)
  print('EPOCH',json.dumps(row),flush=True)
 n.load_state_dict(torch.load(out,map_location='cpu'),strict=True);mse,mae=evaluate(te)
 result=dict(dataset=dataset,variant=variant,seed=2021,epochs=100,global_batch=126,microbatch=21,precision='FP32',mse=mse,mae=mae,test_windows=len(te),best_epoch=bestep,validation_mse=best,checkpoint=str(out),checkpoint_sha256=sha(out),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],resource_mode='memory_gated_shared_allowed_training_times_not_controlled')
 writecsv(OUT/(dataset+'_'+variant+'_result.csv'),[result])
 guard();assert all(sha(ROOT/k)==v for k,v in frozen.items())
 (OUT/(dataset+'_'+variant+'_COMPLETE')).write_text(json.dumps(result,indent=2))
 print('CURRENT_LHC_GEMP_FIT_COMPLETE',json.dumps(result),flush=True)
def preflight(device):
 import torch
 from common_bound_supplement import make_config,seed_all,main_checkpoint
 from common_bound_ablations import Windows
 from models.KIBR_PatchTST_BoundMechanism import Model
 torch.set_num_threads(2)
 for task in TASKS:
  ds=Windows(task,'train',24);x,y,m,_=ds[0];x=x[None].to(device);m=m[None].to(device)
  seed_all();a=Model(make_config(task)).to(device).eval();b=Model(make_config(task)).to(device).eval()
  state=torch.load(main_checkpoint(task),map_location='cpu');a.load_state_dict(state,strict=True);b.load_state_dict(state,strict=True)
  with torch.no_grad():assert torch.equal(a(x,m,None,m),b(x,m,None,m))
  del a,b,state
  for v in VARIANTS:
   replacement=torch.load(OUT/(task+'_'+v+'_bank.pt'),map_location='cpu')['embeddings']
   seed_all();n=Model(make_config(task)).to(device).train();mm=m.clone();mm[:,:,1:6]=replacement.to(device)[None,None];p=n(x,mm,None,mm)
   assert p.shape==(1,24,5) and torch.isfinite(p).all();p.square().mean().backward()
   assert all(torch.isfinite(q.grad).all() for q in n.parameters() if q.grad is not None)
   assert all(q.grad is None for q in n.patch_backbone.parameters());del n
  print('PREFLIGHT_PASS',task,flush=True)
 guard();(OUT/'PREFLIGHT_COMPLETE').write_text('State/output identity of adopted default; fourteen finite controls; frozen backbone.\n')
def pipeline(wait=False):
 import concurrent.futures,signal
 init();lock=open(OUT/'.queue.lock','w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|(0 if wait else fcntl.LOCK_NB))
 except BlockingIOError:print('QUEUE_ALREADY_LIVE',flush=True);return
 py='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python';script=str(Path(__file__).resolve())
 def gpu_info():
  lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True).splitlines()
  active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],universal_newlines=True)
  used={l.split(',')[0].strip() for l in active.splitlines() if ',' in l}
  return {int(a.strip()):dict(uuid=b.strip(),free=int(c),util=int(d),busy=b.strip() in used) for a,b,c,d in [l.split(',') for l in lines]}
 while True:
  candidates=None
  for _ in range(3):
   g=gpu_info();ids={i for i,state in g.items() if i in [0,1,3] and state['free']>=14000}
   candidates=ids if candidates is None else candidates&ids;time.sleep(3)
  available=sorted(candidates or [])[:3]
  if available:break
  print('WAIT_EXCLUSIVE_GPU',flush=True);time.sleep(30)
 host=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1024
 assert host>=50000
 print('SELECTED',available,'HOST',host,flush=True)
 def env(gpu):
  e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1');return e
 # Preparation and preflight use one selected GPU; no optimizer or adopted checkpoint writes.
 for mode in ['prepare','preflight']:
  if mode=='preflight' and (OUT/'PREFLIGHT_COMPLETE').exists():continue
  p=OUT/'logs'/(mode+'.log')
  with p.open('a') as f:subprocess.run([py,script,mode],cwd=ROOT,env=env(available[0]),stdout=f,stderr=subprocess.STDOUT,check=True)
 jobs=[(d,v) for v in VARIANTS for d in TASKS]
 def slot(gpu,work):
  gl=open(OUT/('.gpu'+str(gpu)+'.lock'),'w');fcntl.flock(gl,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for ds,v in work:
   if (OUT/(ds+'_'+v+'_COMPLETE')).exists():print('SKIP_COMPLETE',ds,v,flush=True);continue
   attempt=0
   while attempt<3:
    attempt+=1
    while True:
     state=gpu_info()[gpu]
     if state['free']>=14000:break
     time.sleep(30)
    log=OUT/'logs'/(ds+'_'+v+'.log')
    if log.exists():
     import shutil
     stamp=str(time.time_ns());hist=OUT/'technical_history'/(ds+'_'+v+'_'+stamp);hist.mkdir(parents=True,exist_ok=True)
     log.rename(hist/log.name)
     for old in [CK/(ds+'_'+v+'_seed2021.pth'),OUT/(ds+'_'+v+'_learning.csv')]+list((OUT/'traces').glob(ds+'_'+v+'_epoch*.pt')):
      if old.exists():shutil.move(str(old),str(hist/old.name))
    with log.open('w') as f:
     proc=subprocess.Popen([py,script,'train','--dataset',ds,'--variant',v],cwd=ROOT,env=env(gpu),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     print('START',ds,v,'GPU',gpu,'PID',proc.pid,flush=True)
     # Foreign compute process on selected UUID is an exclusivity failure.
     foreign=False
     while proc.poll() is None:
      time.sleep(20)
      rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],universal_newlines=True)
      uuid=gpu_info()[gpu]['uuid']
      pids=[int(l.split(',')[1]) for l in rows.splitlines() if l.startswith(uuid+',')]
      if gpu_info()[gpu]['free']<6000:
       foreign=True;os.killpg(proc.pid,signal.SIGTERM);print('MEMORY_PRESSURE_STOP_OWN',ds,v,gpu,flush=True);break
     code=proc.wait()
    if code==0 and (OUT/(ds+'_'+v+'_COMPLETE')).exists():break
    print('TECHNICAL_RETRY',ds,v,attempt,code,'foreign',foreign,flush=True)
    if foreign:attempt-=1
   else:raise RuntimeError('Three failed attempts '+ds+' '+v)
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(available)) as ex:
  for f in [ex.submit(slot,gpu,jobs[i::len(available)]) for i,gpu in enumerate(available)]:f.result()
 guard();(OUT/'ALL_CONTEXT_TRAINING_COMPLETE').write_text(str(len(TASKS)*len(VARIANTS))+' isolated controls,100 epochs,seed2021.\n')
 print('ALL_CONTEXT_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['init','prepare','preflight','train','pipeline']);ap.add_argument('--dataset');ap.add_argument('--variant');ap.add_argument('--wait',action='store_true');a=ap.parse_args()
 init()
 if a.mode=='prepare':prepare_banks('cuda:0')
 elif a.mode=='preflight':preflight('cuda:0')
 elif a.mode=='train':train(a.dataset,a.variant)
 elif a.mode=='pipeline':pipeline(a.wait)

