from pathlib import Path
import os,json,re,time,subprocess,fcntl
ROOT=Path('/home/home_new/qsmx/pycodes');P=ROOT/'GreyFerm';OUT=P/'test_results/efp_all_baselines_3seed';OUT.mkdir(exist_ok=True)
lock=open(OUT/'.lock','w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit(0)
PY='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python'
models=['DLinear','PatchTST','TimesNet','iTransformer','TimeMixer','Crossformer','TiDE','FEDformer','xPatch','TimeCMA','T3Time','TimeLLM']
def complete(log):
 if not log.exists():return False
 t=log.read_text(errors='replace');epochs=[int(x) for x in re.findall(r'(?:Epoch:\s*|epoch:)(\d+)',t)]
 return len(re.findall(r'^mse:[\d.eE+-]+,\s*mae:[\d.eE+-]+',t,re.M))==1 and 'Traceback' not in t and epochs and max(epochs)<=10
def command(m,sub,h,seed):
 ds=f'EFP_long_sub{sub}';mid=f'{ds}_h{h}_seed{seed}';data=str(P/'dataset'/ds)
 common=['--is_training','1','--model_id',mid,'--model',m,'--data','efp_batch','--root_path',data,'--data_path',ds+'.csv','--features','M','--seq_len','48','--label_len','24','--pred_len',str(h),'--enc_in','5','--batch_size','32','--num_workers','2','--train_epochs','10','--patience','3','--learning_rate','0.0001']
 if m in ['TimeCMA','T3Time']:return P,[PY,'scripts/run_efp_multimodal.py','--model',m,'--dataset',ds,'--horizon',str(h),'--epochs','10','--patience','3','--seed',str(seed),'--batch_size','32']
 if m=='TimeLLM':return ROOT/'Time-LLM',[PY,'-m','accelerate.commands.launch','--multi_gpu','--num_processes','6','--num_machines','1','--num_cpu_threads_per_process','4','--mixed_precision','no','--main_process_port','29667','run_main.py','--efp_six_gpu']+common+['--task_name','long_term_forecast','--model_comment','efp_multi10','--dec_in','5','--c_out','5','--d_model','16','--d_ff','32','--n_heads','4','--patch_len','16','--stride','8','--prompt_domain','1','--llm_model','GPT2','--llm_dim','768','--llm_layers','6','--use_amp','--lradj','COS','--des','efp_multi10','--checkpoints',str(P/'checkpoints/efp_multimodal/TimeLLM'),'--seed',str(seed)]
 if m=='xPatch':return ROOT/'xPatch',[PY,'-u','run.py']+common+['--patch_len','24','--stride','12','--ma_type','ema','--revin','1','--gpu','0','--des','efp_main10','--random_seed',str(seed)]
 return ROOT/'Time-Series-Library',[PY,'-u','run.py']+common+['--task_name','long_term_forecast','--with_curve','0','--dec_in','5','--c_out','5','--d_model','128','--d_ff','256','--n_heads','4','--e_layers','2','--d_layers','1','--down_sampling_layers','2','--down_sampling_window','2','--down_sampling_method','avg','--gpu','0','--des','efp_main10','--embed','timeF','--random_seed',str(seed)]
jobs=[]
for m in models:
 for seed in [2021,2022,2023]:
  for sub in [1,2,3]:
   for h in [6,12,24,48]:
    ds=f'EFP_long_sub{sub}';log=OUT/f'{m}_{ds}_h{h}_seed{seed}.log';reuse=None
    if m=='Crossformer':reuse=P/f'test_results/efp_3seed_baseline10/crossformer_{ds}_h{h}_seed{seed}.log'
    elif seed==2021:reuse=P/'test_results'/('efp_multimodal' if m in ['TimeCMA','T3Time','TimeLLM'] else 'efp_main_baselines')/f'{ds}_h{h}_{m}.log'
    if reuse and complete(reuse) and not log.exists():log.symlink_to(reuse)
    cwd,cmd=command(m,sub,h,seed);jobs.append(dict(model=m,seed=seed,sub=sub,h=h,log=str(log),reuse=str(reuse) if log.is_symlink() else None,cwd=str(cwd),command=cmd))
protocol=dict(seeds=[2021,2022,2023],baseline_max_epochs=10,baseline_patience=3,main_epochs=100,main_patience=100,baseline_models=models,result_cells=432,main_result_cells=36,gpus=list(range(2,8)),hyperparameters='Per-model historical seed2021 recipes retained for missing seeds. Crossformer reuses its completed three-seed batch126/Huber/sigmoid baseline10 recipe; other TSL baselines use batch32/MSE/type1 as historically run.',fusion_results='test_results/efp_3seed_baseline10/fusion_results.json',selection='Validation checkpoint selection; report all seeds. No outcome-dependent baseline hyperparameter changes.',main_h48='recursive H24 checkpoint',baseline_h48='direct')
protocol.update(distributed_transition='Remaining TimeLLM jobs use six actual DDP ranks on GPUs2-7, global batch32 partitioned6/6/5/5/5/5 with sample-weighted gradients. Existing single-GPU results reused. Deterministic epoch sampler seed+epoch. Validation and final test evaluate all windows on rank0; checkpoint filename checkpoint.',ddp_min_free_mib_per_gpu=8000)
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));(OUT/'manifest.json').write_text(json.dumps(jobs,indent=2));print('REUSED',sum(complete(Path(j['log'])) for j in jobs),'TOTAL',len(jobs),flush=True)
running={};retry={};handles={}
while True:
 for gpu,(proc,j) in list(running.items()):
  code=proc.poll()
  if code is None:continue
  handles.pop(gpu).close();del running[gpu];ok=complete(Path(j['log']));print('END',j['model'],j['seed'],j['sub'],j['h'],'exit',code,'complete',ok,flush=True)
  if not ok:retry[j['log']]=time.time()+120
 live_args=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
 pending=[j for j in jobs if not any(' '.join(j['command'][1:]) in line for line in live_args) and not complete(Path(j['log'])) and all(j['log']!=v[1]['log'] for v in running.values())]
 if not pending and not running:break
 data=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 free={int(a):int(b) for a,b in (line.split(',') for line in data.strip().splitlines())}
 # Drain the pre-existing single-GPU jobs before launching six-rank DDP.
 old_live=any('run_main.py --is_training 1' in line and '--model TimeLLM ' in line and '--model_comment efp_multi10' in line for line in live_args)
 if not running and not old_live and all(free[g]>=8000 for g in range(2,8)):
  j=next((j for j in pending if retry.get(j['log'],0)<=time.time()),None)
  if j is not None:
   assert j['model']=='TimeLLM', 'Only TimeLLM should remain; other models require their DDP adapters'
   log=Path(j['log'])
   if log.exists():
    assert not log.is_symlink(),'Refuse to overwrite a reused log'
    log.rename(str(log)+f'.failed.{int(time.time())}')
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2,3,4,5,6,7',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
   handle=open(log,'w');proc=subprocess.Popen(j['command'],cwd=j['cwd'],env=env,stdout=handle,stderr=subprocess.STDOUT)
   running[2]=(proc,j);handles[2]=handle
   print('START_DDP6',j['model'],j['seed'],j['sub'],j['h'],'GPUS 2,3,4,5,6,7 PID',proc.pid,'GLOBAL_BATCH32',flush=True)
 time.sleep(20)
print('ALL_EFP_ALL_BASELINES_3SEED_COMPLETE',flush=True)
