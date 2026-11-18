"""Strict completeness audit and in-memory aggregation; no JSON intermediate files."""
from pathlib import Path
import argparse,json,math,re,statistics
import pandas as pd
from common_bound_ablations import ROOT,VARIANTS,DEFINITIONS
OUT=ROOT/'test_results/common_bound_ablations'
def fields(line):return dict(x.split('=',1) for x in line.split() if '=' in x)
# Python3.8 compatibility: avoid dict union syntax at runtime.
def stats(rows):
 out={}
 for m in ['mse','mae']:out[m+'_mean']=statistics.mean(x[m] for x in rows);out[m+'_sd']=statistics.stdev(x[m] for x in rows)
 return out
def collect():
 counts={}
 for ds in ['CHO','EFP_long_sub1','EFP_long_sub2','EFP_long_sub3']:
  if ds=='CHO':
   manifest=pd.read_csv(ROOT/'dataset/CHO/batch_manifest.csv')
   lengths={split:manifest.loc[manifest.eligible & (manifest.split==split),'points'].tolist() for split in ['val','test']}
  else:
   df=pd.read_csv(ROOT/'dataset'/ds/(ds+'.csv'));bids=df.batch_id.drop_duplicates().tolist();lens=df.groupby('batch_id',sort=False).size().to_dict();eligible=[b for b in bids if lens[b]>=96];n=int(len(eligible)*.7);v=int(len(eligible)*.1)
   lengths={'val':[lens[b] for b in eligible[n:n+v]],'test':[lens[b] for b in eligible[n+v:]]}
  for split,ll in lengths.items():
   for h in [6,12,24,48]:counts[(ds,split,h)]=sum(int(x)-48-h+1 for x in ll)
 raw=[];validation=[];jobs=0
 for ds in ['CHO','EFP_long_sub1','EFP_long_sub2','EFP_long_sub3']:
  for variant in ['full']+VARIANTS:
   for seed in [2021,2022,2023]:
    for h in [6,12,24]:
     log=OUT/f'{ds}_h{h}_seed{seed}_{variant}.log';assert log.exists(),str(log)
     text=log.read_text(errors='replace')
     assert re.findall(r'^Epoch: (\d+) ',text,re.M)==[str(i) for i in range(1,101)],str(log)
     assert set(re.findall(r'(?:COMMON_ABLATION|CHO|EFP)_DDP_RANK (\d+) ',text))==set('012345'),str(log)
     assert any(x in text for x in ['COMMON_ABLATION_JOB_COMPLETE','CHO_BOUND_JOB_COMPLETE','EFP_BOUND_JOB_COMPLETE']),str(log)
     testlines=re.findall(r'^(?:COMMON_ABLATION_RESULT|EFP_BOUND_RESULT|CHO_RESULT) (.+)$',text,re.M)
     assert len(testlines)==(2 if h==24 else 1),str(log)
     for line in testlines:
      row=fields(line);hh=int(row['h']);assert int(row['seed'])==seed and hh in ([24,48] if h==24 else [h])
      assert int(row['completed_epochs'])==100 and int(row['test_windows'])==counts[(ds,'test',hh)]
      for metric in ['mse','mae']:assert math.isfinite(float(row[metric])) and float(row[metric])>=0
      expected='recursive' if hh==48 else 'direct';assert row['protocol']==expected
      raw.append(dict(dataset=ds,variant=variant,source_variant=row.get('variant'),seed=seed,h=hh,mse=float(row['mse']),mae=float(row['mae']),windows=int(row['test_windows']),best_epoch=int(row['best_epoch']),epochs=100,protocol=expected,source=str(log.resolve().relative_to(ROOT))))
     vl=re.findall(r'^(?:COMMON_ABLATION_VALIDATION|EFP_BOUND_VALIDATION|CHO_BOUND_VALIDATION) (.+)$',text,re.M)
     vl=[fields(x) for x in vl];vl=[x for x in vl if 'trained' not in x or x['trained']==x['input']]
     assert len(vl)==1,str(log);row=vl[0];assert int(row['h'])==h and int(row['seed'])==seed and int(row['windows'])==counts[(ds,'val',h)]
     assert all(math.isfinite(float(row[m])) for m in ['mse','mae'])
     validation.append(dict(dataset=ds,variant=variant,seed=seed,h=h,mse=float(row['mse']),mae=float(row['mae']),windows=int(row['windows']),best_epoch=int(row['best_epoch']),source=str(log.resolve().relative_to(ROOT))))
     jobs+=1
 assert jobs==900 and len(raw)==1200 and len(validation)==900
 assert 'ALL_COMMON_BOUND_ABLATION_PREFLIGHT_COMPLETE' in (OUT/'preflight.log').read_text()
 cells=[];overall=[]
 for ds in ['CHO','EFP_long_sub1','EFP_long_sub2','EFP_long_sub3']:
  for v in ['full']+VARIANTS:
   for h in [6,12,24,48]:
    rr=[x for x in raw if x['dataset']==ds and x['variant']==v and x['h']==h];assert len(rr)==3
    cells.append(dict(dataset=ds,variant=v,h=h,**stats(rr)))
 for family in ['EFP','CHO']:
  seeds_byvariant={}
  for v in ['full']+VARIANTS:
   seeds=[]
   for seed in [2021,2022,2023]:
    rr=[x for x in raw if x['dataset'].startswith(family) and x['variant']==v and x['seed']==seed];assert len(rr)==(12 if family=='EFP' else 4)
    seeds.append(dict(seed=seed,**{m:statistics.mean(x[m] for x in rr) for m in ['mse','mae']}))
   seeds_byvariant[v]=seeds
   overall.append(dict(dataset=family,variant=v,seeds=seeds,**stats(seeds)))
  reference=next(x for x in overall if x['dataset']==family and x['variant']=='full')
  for entry in [x for x in overall if x['dataset']==family]:
   for m in ['mse','mae']:entry[m+'_change_percent']=100*(entry[m+'_mean']/reference[m+'_mean']-1)
   entry['paired_seed_deltas']=[dict(seed=a['seed'],**{m:a[m]-b[m] for m in ['mse','mae']}) for a,b in zip(entry['seeds'],seeds_byvariant['full'])]
 return dict(training_jobs=jobs,test_cells=len(raw),validation_cells=len(validation),raw=raw,validation=validation,cells=cells,overall=overall,definitions=DEFINITIONS)
def main():
 p=argparse.ArgumentParser();p.add_argument('--summary-only',action='store_true');p.add_argument('--report',action='store_true');args=p.parse_args();d=collect()
 if args.report:
  lines=['# Common bound-model ablations: final verified results','All900fits100epochs/sixranks;1200testcells and900ownconditionvalidationcells; allwindows finite/exact.','All3seeds included. Main/control100epochs;baseline<=10epoch comparisons remain unequal budget. Main/controlH48recursive.','Positive change indicates worse error than full main. No test-score-driven recipe changes.','|Dataset|variant|MSE mean|sampleSD|MAE mean|sampleSD|MSE change%|MAE change%|','|---|---|---|---|---|---|---|---|']
  for x in d['overall']:lines.append('|'+str(x['dataset'])+'|'+str(x['variant'])+'|'+ '|'.join(str(x[k]) for k in ['mse_mean','mse_sd','mae_mean','mae_sd','mse_change_percent','mae_change_percent'])+'|')
  lines+=['## Definitions']+[f'- {k}: {v}' for k,v in DEFINITIONS.items()]
  (OUT/'final_report.md').write_text('\n'.join(lines)+'\n')
 if args.summary_only:d={k:v for k,v in d.items() if k not in ['raw','validation','cells']}
 print(json.dumps(d,ensure_ascii=False))
if __name__=='__main__':main()
