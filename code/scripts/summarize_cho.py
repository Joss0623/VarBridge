from pathlib import Path
import re,math,statistics,json
ROOT=Path('/home/home_new/qsmx/pycodes/GreyFerm');OUT=ROOT/'test_results/cho'
def collect():
 assert 'ALL_CHO_EXPERIMENTS_COMPLETE' in (ROOT/'test_results/cho_pipeline.log').read_text()
 rows=[];jobs=0
 for f in sorted(OUT.glob('*.log')):
  if f.name=='text_preparation.log':continue
  s=f.read_text();assert 'CHO_JOB_COMPLETE' in s,f.name
  assert set(re.findall(r'CHO_DDP_RANK (\d)',s))==set('012345'),f.name
  jobs+=1
  for line in s.splitlines():
   if not line.startswith('CHO_RESULT '):continue
   d=dict(re.findall(r'(\w+)=([^ ]+)',line))
   for k in ['seed','h','test_windows','best_epoch','completed_epochs']:d[k]=int(d[k])
   for k in ['mse','mae']:d[k]=float(d[k]);assert math.isfinite(d[k])
   d['log']=str(f)
   assert d['test_windows']=={6:435,12:417,24:381,48:309}[d['h']]
   assert 1<=d['best_epoch']<=d['completed_epochs']
   assert (d['completed_epochs']==100 if d['model']=='Fusion' else 1<=d['completed_epochs']<=10)
   assert d['protocol']==('recursive' if d['model']=='Fusion' and d['h']==48 else 'direct')
   rows.append(d)
 assert jobs==172 and len(rows)==180
 assert len(set((d['model'],d['variant'],d['seed'],d['h']) for d in rows))==180
 summary=[]
 for model,variant in sorted(set((d['model'],d['variant']) for d in rows)):
  r=[d for d in rows if d['model']==model and d['variant']==variant];assert len(r)==12
  for h in [6,12,24,48,'overall']:
   selected=[d for d in r if h=='overall' or d['h']==h];q={'model':model,'variant':variant,'h':h}
   assert set(d['seed'] for d in selected)=={2021,2022,2023}
   for metric in ['mse','mae']:
    v=[statistics.mean(d[metric] for d in selected if d['seed']==seed) for seed in [2021,2022,2023]]
    q[metric]=statistics.mean(v);q[metric+'_sd']=statistics.stdev(v);q[metric+'_seeds']=v
   summary.append(q)
 return {'jobs':jobs,'cells':len(rows),'raw':rows,'summary':summary}
if __name__=='__main__':print(json.dumps(collect(),separators=(',',':')))
