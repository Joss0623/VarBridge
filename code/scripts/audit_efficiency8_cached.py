"""Final audit of cached text latency; not a training or manuscript operation."""
from pathlib import Path
import csv,math,statistics,hashlib,io
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'test_results/efficiency8_4090';OUT=BASE/'cached_text_remeasure'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert (OUT/'ALL8_CACHED_INFERENCE_COMPLETE').exists()
 rows=list(csv.DictReader((OUT/'cached_inference_results.csv').open()))
 original={r['model']:r for r in csv.DictReader((BASE/'online_remeasure/online_inference_results.csv').open())}
 assert len(rows)==8 and {r['model'] for r in rows}==set(original)
 selected=next(csv.DictReader((OUT/'selected_gpu.csv').open()));hardware=selected['physical_gpu']
 report=io.StringIO();keylist=None
 for r in rows:
  m=r['model'];ref=original[m]
  for key,v in [('scenario','text_already_cached_CPU'),('physical_gpu',hardware),('epochs','100'),('batch_size','1'),('cpu_threads','2'),('precision','float32'),('lookback','48'),('horizon','24'),('seed','2021'),('test_windows','381'),('samples','160'),('matched_windows','32'),('repeats','5'),('warmups','20')]:assert r[key]==v,(m,key)
  assert r['checkpoint']==ref['checkpoint'] and sha(r['checkpoint'])==r['checkpoint_sha256']==ref['checkpoint_sha256']
  assert abs(float(r['mse'])-float(ref['mse']))<5e-6 and abs(float(r['mae'])-float(ref['mae']))<5e-6
  samples=list(csv.DictReader((OUT/(m+'_cached_samples.csv')).open()))
  old_samples=list(csv.DictReader((BASE/'online_remeasure'/(m+'_online_samples.csv')).open()))
  assert len(samples)==160 and {s['model'] for s in samples}=={m}
  keys=[tuple(s[k] for k in ['repeat','window_index','start','batch_id']) for s in samples]
  assert len(set(keys))==160
  assert keys==[tuple(s[k] for k in ['repeat','window_index','start','batch_id']) for s in old_samples]
  if keylist is None:keylist=keys
  else:assert keylist==keys
  values=[float(s['latency_ms']) for s in samples];assert all(math.isfinite(v) and v>0 for v in values)
  for k,v in [('latency_median_ms',statistics.median(values)),('latency_mean_ms',statistics.mean(values)),('latency_sd_ms',statistics.stdev(values))]:assert abs(float(r[k])-v)<1e-9,(m,k)
  for k in ['mse','mae','peak_allocated_MiB','peak_reserved_MiB']:assert math.isfinite(float(r[k])) and float(r[k])>0
  assert int(r['resident_parameters'])==int(r['forecaster_parameters']) and int(r['external_text_parameters'])==0
  assert int(r['forecaster_parameters'])==int(ref['forecaster_parameters'])
  contexts=list(csv.DictReader((OUT/(m+'_cache_metadata.csv')).open()));assert len(contexts)==32
  assert len(set(c['window_index'] for c in contexts))==32
  assert sum(int(c['cpu_cache_bytes']) for c in contexts)==int(r['total_32window_CPU_cache_bytes'])
  if m=='VarBridge':assert int(r['resident_parameters'])==1032026 and all(int(c['cpu_cache_bytes'])==5*8*768*4 for c in contexts)
  elif m=='TimeLLM':
   assert int(r['internal_text_parameters'])==81912576
   for c in contexts:assert int(c['cpu_cache_bytes'])==6*2*5*768*int(c['text_prefix_tokens'])*4
  else:assert all(int(c['cpu_cache_bytes'])==0 for c in contexts)
  log=(OUT/(m+'_cached.log')).read_text()
  assert 'EFF8_CACHED_COMPLETE '+m in log and 'CACHED_PREDICTION_IDENTITY_PASS '+m in log
  report.write(f'{m}: same checkpoint PASS; full381 prediction equivalence PASS; GPU{hardware} same32x5 samples PASS; median/mean/SD PASS; CPU cache scope/bytes PASS\n')
 for s in csv.DictReader((OUT/'source_audit.csv').open()):assert sha(s['source'])==s['sha256']
 report.write('Adopted source hashes and all8 checkpoint hashes unchanged. No retraining. No paper/Excel changes.\n')
 report.write('Cached scenario excludes observed-text cache generation; caches must be rebuilt for changed descriptions. CPU cache lookup/H2D, normalization, numerical forecasting and D2H remain timed. TimeLLM retains numerical patch reprogramming and GPT2 processing; no cached numerical forecasts.\n')
 report.write('FINAL_EFFICIENCY8_CACHED_AUDIT_PASS\n');(OUT/'final_audit.log').write_text(report.getvalue())
 print(report.getvalue())
 for r in rows:
  ref=original[r['model']]
  print(r['model'],'MSE',f"{float(r['mse']):.6f}",'cached_median_ms',f"{float(r['latency_median_ms']):.3f}",'prior_online_ms',f"{float(ref['latency_median_ms']):.3f}",'resident_params',r['resident_parameters'],'cache_KiB_per_window',f"{float(r['mean_CPU_cache_bytes'])/1024:.2f}")
if __name__=='__main__':main()
