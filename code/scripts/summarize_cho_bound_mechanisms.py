"""Complete-gate summary, stdout only; no retained JSON intermediates."""
from pathlib import Path
import re,json,math,statistics
from cho_data import ROOT

def fields(line):
 return dict(re.findall(r'(\w+)=([^\s]+)',line))
def mean(x):return statistics.mean(x)
def sd(x):return statistics.stdev(x)
def collect():
 out=ROOT/'test_results/cho_bound_mechanism';pipeline=(ROOT/'test_results/cho_bound_mechanism_pipeline.log').read_text()
 assert 'ALL_CHO_BOUND_MECHANISM_COMPLETE' in pipeline
 raw=[];validation=[];sensitivity=[];old=[]
 for variant in ['correct','contradictory','unrelated']:
  for seed in [2021,2022,2023]:
   for h in [6,12,24]:
    path=out/f'Fusion_h{h}_seed{seed}_{variant}.log';text=path.read_text();assert 'CHO_BOUND_JOB_COMPLETE' in text
    ranks=re.findall(r'CHO_DDP_RANK (\d+) LOCAL (\d+) WORLD (\d+)',text);assert set((int(a),int(b),int(c)) for a,b,c in ranks)=={(i,i,6) for i in range(6)}
    assert [int(e) for e in re.findall(r'^Epoch: (\d+)',text,re.M)]==list(range(1,101))
    vals=[fields(l) for l in text.splitlines() if l.startswith('CHO_BOUND_VALIDATION ')];sens=[fields(l) for l in text.splitlines() if l.startswith('CHO_BOUND_SENSITIVITY ')];tests=[fields(l) for l in text.splitlines() if l.startswith('CHO_RESULT ')]
    assert len(vals)==len(sens)==3 and len(tests)==(2 if h==24 else 1)
    for r in vals:
     assert int(r['windows'])=={6:598,12:574,24:526}[h];assert int(r['seed'])==seed and int(r['h'])==h and r['trained']==variant
     r.update(mse=float(r['mse']),mae=float(r['mae']),seed=seed,h=h,log=str(path));assert math.isfinite(r['mse']) and math.isfinite(r['mae']);validation.append(r)
    assert {r['input'] for r in vals}=={'correct','contradictory','unrelated'}
    for r in sens:
     r.update(prediction_delta_rms=float(r['prediction_delta_rms']),relative_prediction_delta_rms=float(r['relative_prediction_delta_rms']),seed=seed,h=h);assert math.isfinite(r['relative_prediction_delta_rms']);sensitivity.append(r)
    for r in tests:
     th=int(r['h']);assert int(r['test_windows'])=={6:435,12:417,24:381,48:309}[th] and int(r['completed_epochs'])==100
     r.update(mse=float(r['mse']),mae=float(r['mae']),seed=seed,h=th,log=str(path));assert math.isfinite(r['mse']) and math.isfinite(r['mae']);raw.append(r)
 for seed in [2021,2022,2023]:
  for h in [6,12,24]:
   text=(ROOT/f'test_results/cho/Fusion_h{h}_seed{seed}_full.log').read_text();loss=[float(v) for v in re.findall(r'Vali Loss: ([0-9.eE+-]+)',text)];assert len(loss)==100
   old.append(dict(seed=seed,h=h,mse=min(loss)))
 assert len(raw)==36 and len(validation)==len(sensitivity)==81
 val_summary=[];test_summary=[]
 for trained in ['correct','contradictory','unrelated']:
  for inp in ['correct','contradictory','unrelated']:
   q=[r for r in validation if r['trained']==trained and r['input']==inp]
   mse=[mean(r['mse'] for r in q if r['seed']==seed) for seed in [2021,2022,2023]];mae=[mean(r['mae'] for r in q if r['seed']==seed) for seed in [2021,2022,2023]]
   val_summary.append(dict(trained=trained,input=inp,mse=mean(mse),mse_sd=sd(mse),mse_seeds=mse,mae=mean(mae),mae_sd=sd(mae),mae_seeds=mae))
  for h in [6,12,24,48,'overall']:
   q=[r for r in raw if r['variant']==trained and (h=='overall' or r['h']==h)]
   mse=[mean(r['mse'] for r in q if r['seed']==seed) for seed in [2021,2022,2023]];mae=[mean(r['mae'] for r in q if r['seed']==seed) for seed in [2021,2022,2023]]
   test_summary.append(dict(variant=trained,h=h,mse=mean(mse),mse_sd=sd(mse),mse_seeds=mse,mae=mean(mae),mae_sd=sd(mae),mae_seeds=mae))
 lookup={(r['trained'],r['input']):r for r in val_summary};correct=lookup['correct','correct'];old_seeds=[mean(r['mse'] for r in old if r['seed']==seed) for seed in [2021,2022,2023]]
 checks={'improves_old_mean_validation':correct['mse']<mean(old_seeds)}
 for variant in ['contradictory','unrelated']:
  checks['beats_trained_'+variant+'_all_paired_seeds']=all(a<b for a,b in zip(correct['mse_seeds'],lookup[variant,variant]['mse_seeds']))
  checks['same_checkpoint_'+variant+'_worsens_all_paired_seeds']=all(a<b for a,b in zip(correct['mse_seeds'],lookup['correct',variant]['mse_seeds']))
 return dict(training_jobs=27,test_cells=36,validation_cells=81,validation=validation,sensitivity=sensitivity,raw=raw,validation_summary=val_summary,test_summary=test_summary,old_validation_mse=mean(old_seeds),old_validation_mse_seeds=old_seeds,checks=checks,eligible_main_update=all(checks.values()))
if __name__=='__main__':print(json.dumps(collect(),ensure_ascii=False))
