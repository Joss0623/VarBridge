"""Audit and aggregate common bound EFP results; stdout only, no JSON intermediate files."""
from pathlib import Path
import json,re,math,statistics,runpy,io,contextlib,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def fields(s):return dict(re.findall(r'(\w+)=([^\s]+)',s))
def collect():
 out=ROOT/'test_results/efp_bound_mechanism'
 assert 'ALL_EFP_BOUND_MECHANISM_COMPLETE' in (ROOT/'test_results/efp_bound_mechanism_pipeline.log').read_text()
 raw=[];val=[]
 for sub in [1,2,3]:
  ds=f'EFP_long_sub{sub}';frame=pd.read_csv(ROOT/f'dataset/{ds}/{ds}.csv');ids=frame.batch_id.drop_duplicates().tolist();sizes=frame.groupby('batch_id',sort=False).size();eligible=[i for i in ids if sizes[i]>=96];splits={'train':eligible[:int(.7*len(eligible))],'val':eligible[int(.7*len(eligible)):int(.7*len(eligible))+int(.1*len(eligible))],'test':eligible[int(.7*len(eligible))+int(.1*len(eligible)):]}
  assert [len(splits[k]) for k in ['train','val','test']]==[277,39,80]
  counts=lambda flag,h:sum(max(int(sizes[i])-48-h+1,0) for i in splits[flag])
  for seed in [2021,2022,2023]:
   for h in [6,12,24]:
    f=out/f'{ds}_h{h}_seed{seed}_correct.log';t=f.read_text();assert 'EFP_BOUND_JOB_COMPLETE' in t
    assert set(re.findall(r'EFP_DDP_RANK (\d+) LOCAL (\d+) WORLD (\d+)',t))=={(str(i),str(i),'6') for i in range(6)}
    assert list(map(int,re.findall(r'^Epoch: (\d+)',t,re.M)))==list(range(1,101))
    rows=[fields(s) for s in t.splitlines() if s.startswith('EFP_BOUND_RESULT ')];vs=[fields(s) for s in t.splitlines() if s.startswith('EFP_BOUND_VALIDATION ')]
    assert len(rows)==(2 if h==24 else 1) and len(vs)==1
    for x in rows+vs:
     x.update(sub=sub,seed=int(x['seed']),h=int(x['h']),mse=float(x['mse']),mae=float(x['mae']),best_epoch=int(x['best_epoch']),log=str(f.relative_to(ROOT)))
     assert x['seed']==seed and x['dataset']==ds and math.isfinite(x['mse']) and math.isfinite(x['mae'])
    for x in rows:
     assert x['h'] in ([24,48] if h==24 else [h]) and int(x['completed_epochs'])==100 and int(x['test_windows'])==counts('test',x['h'])
     assert x['protocol']==('recursive' if x['h']==48 else 'direct');raw.append(x)
    assert vs[0]['h']==h and int(vs[0]['windows'])==counts('val',h);val.extend(vs)
 assert len(raw)==36 and len(val)==27
 capture=io.StringIO()
 with contextlib.redirect_stdout(capture):runpy.run_path(str(ROOT/'scripts/summarize_efp_all_baselines_3seed.py'),run_name='__main__')
 historic=json.loads(capture.getvalue());old=[x for x in historic['raw'] if x['model']=='fusion']
 def aggregate(rows):
  seedvals=[dict(seed=s,**{k:statistics.mean(x[k] for x in rows if x['seed']==s) for k in ['mse','mae']}) for s in [2021,2022,2023]]
  return dict(seeds=seedvals,**{k+suffix:fn(x[k] for x in seedvals) for k in ['mse','mae'] for suffix,fn in [('_mean',statistics.mean),('_sd',statistics.stdev)]})
 cells=[dict(sub=sub,h=h,**aggregate([x for x in raw if x['sub']==sub and x['h']==h])) for sub in [1,2,3] for h in [6,12,24,48]]
 overall=aggregate(raw);oldoverall=aggregate(old);validation=aggregate(val)
 wins={}
 for model in ['fusion','DLinear','PatchTST','TimesNet','iTransformer','TimeMixer','Crossformer','TiDE','FEDformer','xPatch','TimeCMA','T3Time','TimeLLM']:
  wins[model]={k:sum(c[k+'_mean']<next(x for x in historic['cells'] if (x['model'],x['sub'],x['h'])==(model,c['sub'],c['h']))[k+'_mean'] for c in cells) for k in ['mse','mae']}
 return dict(raw=raw,validation=val,cells=cells,overall=overall,old=old,oldoverall=oldoverall,validation_overall=validation,baseline_overall=[x for x in historic['overall'] if x['model']!='fusion'],wins=wins,training_jobs=27,test_cells=36,validation_cells=27)
if __name__=='__main__':print(json.dumps(collect()))
