"""Audit effective30-epoch variants; full reference100 retained and explicitly labeled."""
from pathlib import Path
import json,re,math,hashlib,statistics,argparse
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'test_results/common_bound_ablations_30'
def guard():
 p=ROOT/'scripts/train_common_bound_ablation_30_single.py'
 if (R/'single_trainer_sha256.txt').exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==(R/'single_trainer_sha256.txt').read_text()
 for rel,h in json.loads((R/'protected_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
 assert hashlib.sha256((ROOT/'scripts/train_common_bound_ablation_30_ddp.py').read_bytes()).hexdigest()==(R/'trainer_sha256.txt').read_text()
def fields(s):return dict(q.split('=',1) for q in s.split() if '=' in q)
def record(j):
 p=R/(j['stem']+'_REUSED.json')
 if p.exists():
  z=json.loads(p.read_text());cp=ROOT/'checkpoints/common_bound_ablations_30'/(j['stem']+'.pth')
  assert hashlib.sha256(cp.read_bytes()).hexdigest()==z['checkpoint_sha256'];assert z['validation']['best_epoch']<=30
  return z
 p=R/(j['stem']+'.log')
 if not p.exists():return None
 text=p.read_text(errors='replace')
 if 'COMMON_ABLATION_JOB_COMPLETE' not in text:return None
 assert re.findall(r'^Epoch: (\d+) ',text,re.M)==[str(i) for i in range(1,31)],j['stem']
 ranks=set(re.findall(r'COMMON_ABLATION_DDP_RANK (\d+) ',text))
 if "'actual_ranks': 1" in text:
  assert ranks=={'0'} and "'virtual_shards': 6" in text
  audit=json.loads((R/(j['stem']+'_SINGLE_PREFLIGHT.json')).read_text());assert audit['exact_sample_weighting'] and audit['prediction_max_abs']<1e-4 and audit['gradient_relative_l2']<.005
 else:assert ranks==set('012345')
 assert "'max_epochs': 30" in text and "'global_batch': 126" in text
 tests=[]
 for line in re.findall(r'^COMMON_ABLATION_RESULT (.+)$',text,re.M):
  a=fields(line);h=int(a['h']);assert int(a['seed'])==2021 and a['dataset']==j['dataset'] and a['variant']==j['variant'];assert int(a['completed_epochs'])==30 and int(a['test_windows'])==j['expected_test_windows'][str(h)]
  assert all(math.isfinite(float(a[k])) for k in ['mse','mae']);assert 1<=int(a['best_epoch'])<=30
  tests.append(dict(dataset=j['dataset'],variant=j['variant'],seed=2021,h=h,mse=float(a['mse']),mae=float(a['mae']),windows=int(a['test_windows']),best_epoch=int(a['best_epoch']),protocol=a['protocol'],source=str(p)))
 assert len(tests)==(2 if j['horizon']==24 else 1)
 vv=re.findall(r'^COMMON_ABLATION_VALIDATION (.+)$',text,re.M);assert len(vv)==1;a=fields(vv[0]);assert int(a['windows'])==j['expected_val_windows'] and 1<=int(a['best_epoch'])<=30
 cp=ROOT/'checkpoints/common_bound_ablations_30'/(j['stem']+'.pth');assert cp.exists()
 return dict(job=j,effective_training_budget=30,source_run_epochs=30,test=tests,validation=a,checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
def collect(partial=False):
 guard();plan=json.loads((R/'plan.json').read_text());done=[];pending=[]
 for j in plan['jobs']:
  z=record(j)
  if z is None:pending.append(j['stem'])
  else:done.append(z)
 if not partial:assert not pending,'Pending '+str(len(pending))
 raw=[]
 for z in done:
  for a in z['test']:raw.append(dict(a,epochs=30,effective_training_budget=30,source_run_epochs=z['source_run_epochs']))
 raw+=plan['full100_test'];summary=[]
 for ds in ['EFP_long_sub1','EFP_long_sub2','EFP_long_sub3','CHO']:
  for variant in ['full']+list(dict.fromkeys(j['variant'] for j in plan['jobs'])):
   a=[x for x in raw if x['dataset']==ds and x['variant']==variant]
   if len(a)!=4:continue
   summary.append(dict(dataset=ds,variant=variant,epochs=100 if variant=='full' else 30,mse=statistics.mean(x['mse'] for x in a),mae=statistics.mean(x['mae'] for x in a),seed=2021,sd=None))
 if not partial:assert len(raw)==400 and len(done)==288 and len(summary)==100
 return dict(done=len(done),reused=sum(z['source_run_epochs']==100 for z in done),new_complete=sum(z['source_run_epochs']==30 for z in done),pending=len(pending),raw=raw,summary=summary,validation=[z['validation'] for z in done]+plan['full100_validation'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');p.add_argument('--raw',action='store_true');a=p.parse_args();d=collect(a.partial)
 if not a.raw:d={k:v for k,v in d.items() if k not in ['raw','validation']}
 print(json.dumps(d))
