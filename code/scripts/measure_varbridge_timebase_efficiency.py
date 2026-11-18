"""Fresh same-GPU cached inference, plus explicitly scoped operator MAC counts. No fit."""
import os,sys,argparse,re,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT))
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);args=ap.parse_args()
source=ROOT/'scripts'/('measure_timellm_embedding_cache.py' if args.model=='TimeLLM' else 'measure_efficiency8_cached.py')
s=source.read_text().split("if __name__=='__main__':")[0]
s=re.sub(r"OUT=SOURCE/[^;\n]+", "OUT=ROOT/'test_results/varbridge_timebase_analysis/efficiency'",s,count=1)
# Preserve native inference timing and capture its memory before profiler-only execution.
s=s.replace("peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20", "peak_allocated_MiB=measured_peak_alloc,peak_reserved_MiB=measured_peak_reserved")
needle=" v=np.array([r['latency_ms'] for r in samples])"
assert s.count(needle)==1
block='''
 measured_peak_alloc=torch.cuda.max_memory_allocated()/2**20
 measured_peak_reserved=torch.cuda.max_memory_reserved()/2**20
 # Operator accounting uses an equivalent unfused attention execution, outside timing.
 # Count dense matrix/conv multiply-accumulates only; FFT, softmax, norm and elementwise ops are excluded.
 mha_states=[(m,m.training,m.dropout) for m in net.modules() if isinstance(m,torch.nn.MultiheadAttention)]
 for m,_,_ in mha_states:m.training=True;m.dropout=0.0
 try:
  with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False):
   i=selected[0];p_profile=cached(i,contexts[i]);p_native=reference(i)
   assert torch.allclose(p_profile,p_native,atol=3e-5,rtol=3e-5),('profiling execution identity',model,float((p_profile-p_native).abs().max()))
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,with_flops=True) as prof:
    for i in selected:cached(i,contexts[i])
    torch.cuda.synchronize()
  op_rows=[dict(model=model,operator=e.key,calls=e.count,flops=int(e.flops or 0)) for e in prof.key_averages()]
 finally:
  for m,training,dropout in mha_states:m.training=training;m.dropout=dropout
 counted_macs=sum(r['flops'] for r in op_rows)/2/len(selected)
 assert counted_macs>0
 writecsv(OUT/(model+'_operator_counts.csv'),op_rows)
 # Cache generation and the full resident language encoder are separate from cached predictor cost.
 offline_text_parameters=0
 if model=='VarBridge':
  from transformers import AutoConfig
  cfg=AutoConfig.from_pretrained('FacebookAI/roberta-base',local_files_only=True)
  # Actual encoder instance, on CPU, counted once; its time is not added to cached inference.
  from transformers import AutoModel
  enc=AutoModel.from_pretrained('FacebookAI/roberta-base',local_files_only=True)
  offline_text_parameters=sum(p.numel() for p in enc.parameters());del enc
'''
s=s.replace(needle,block+needle)
s=s.replace("result=dict(model=model,scenario=", "result=dict(operator_macs_per_sample=counted_macs,operator_macs_scope='matmul/conv FLOPs divided by2; attention profiled with equivalent unfused execution; excludes FFT/softmax/norm/elementwise/cache-generation',profiling_prediction_max_abs_error=float((p_profile-p_native).abs().max()),offline_text_encoder_parameters=offline_text_parameters,model=model,scenario=")
# Preserve full-test outputs on the server for original-EXP grouped evaluation.
s=s.replace('contexts={};context_meta=[];max_error=0.;se=ae=0.;prep_seconds=0.', 'contexts={};context_meta=[];max_error=0.;se=ae=0.;prep_seconds=0.;full_predictions=[]')
s=s.replace('  delta=(pred-ds[i][1].unsqueeze(0)).double();assert torch.isfinite(delta).all()', '  full_predictions.append(pred.cpu())\n  delta=(pred-ds[i][1].unsqueeze(0)).double();assert torch.isfinite(delta).all()')
s=s.replace(' mse=se/(381*24*5);mae=ae/(381*24*5)', " torch.save(dict(prediction=torch.cat(full_predictions),starts=ds.starts,batch_ids=ds.ids,checkpoint_sha256=ckhash),OUT/(model+'_fulltest_predictions.pt'))\n mse=se/(381*24*5);mae=ae/(381*24*5)")
namespace={'__name__':'timebase_measure_worker','__file__':str(ROOT/'scripts/measure_varbridge_timebase_efficiency.py')}
exec(compile(s,str(source), 'exec'),namespace)
namespace['worker'](args.model)
