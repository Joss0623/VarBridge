import json,re,math,statistics,pathlib,sys
p=pathlib.Path('/home/home_new/qsmx/pycodes/GreyFerm')
assert 'ALL_EFP_ALL_BASELINES_3SEED_COMPLETE' in (p/'test_results/efp_all_baselines_3seed_pipeline.log').read_text()
logs=list((p/'test_results/efp_all_baselines_3seed').glob('TimeLLM_*.checkpoint_eval.log'))
assert len(logs)==36
raw=json.loads((p/'test_results/efp_3seed_baseline10/fusion_results.json').read_text())['raw']
for x in raw:
 t=(p/x['log']).read_text()
 assert max(map(int,re.findall(r'Epoch:\s*(\d+)',t)))==100, x['log']
for x in json.loads((p/'test_results/efp_all_baselines_3seed/manifest.json').read_text()):
 f=pathlib.Path(x['log'])
 if x['model']=='TimeLLM':
  f=f.with_suffix('.checkpoint_eval.log')
  assert 'ALL_TEST_WINDOWS' in f.read_text()
 t=f.read_text()
 scores=re.findall(r'mse:\s*([\d.eE+-]+),\s*mae:\s*([\d.eE+-]+)',t)
 assert scores, str(f)
 mse,mae=map(float,scores[-1])
 assert math.isfinite(mse) and math.isfinite(mae)
 train=pathlib.Path(x['log']).read_text()
 epochs=list(map(int,re.findall(r'epoch:\s*(\d+)',train,re.I)))
 assert epochs and max(epochs)<=10, str(f)
 raw.append(dict(model=x['model'],seed=x['seed'],sub=x['sub'],h=x['h'],mse=mse,mae=mae,log=str(f.relative_to(p)),epochs=max(epochs)))
assert len(raw)==468
keys={(x['model'],x['seed'],x['sub'],x['h']) for x in raw}
assert len(keys)==468
models=['fusion','DLinear','PatchTST','TimesNet','iTransformer','TimeMixer','Crossformer','TiDE','FEDformer','xPatch','TimeCMA','T3Time','TimeLLM']
cells=[]; overall=[]
for m in models:
 for sub in [1,2,3]:
  for h in [6,12,24,48]:
   a=[x for x in raw if (x['model'],x['sub'],x['h'])==(m,sub,h)]
   assert sorted(x['seed'] for x in a)==[2021,2022,2023]
   cells.append(dict(model=m,sub=sub,h=h,**{k+suffix:fn([x[k] for x in a]) for k in ['mse','mae'] for suffix,fn in [('_mean',statistics.mean),('_sd',statistics.stdev)]}))
 seeds=[dict(seed=s,**{k:statistics.mean(x[k] for x in raw if x['model']==m and x['seed']==s) for k in ['mse','mae']}) for s in [2021,2022,2023]]
 overall.append(dict(model=m,seeds=seeds,**{k+suffix:fn([x[k] for x in seeds]) for k in ['mse','mae'] for suffix,fn in [('_mean',statistics.mean),('_sd',statistics.stdev)]}))
wins={m:{k:sum(next(x for x in cells if (x['model'],x['sub'],x['h'])==('fusion',s,h))[k+'_mean'] < next(x for x in cells if (x['model'],x['sub'],x['h'])==(m,s,h))[k+'_mean'] for s in [1,2,3] for h in [6,12,24,48]) for k in ['mse','mae']} for m in models[1:]}
print(json.dumps(dict(raw=raw,cells=cells,overall=overall,wins=wins)))

