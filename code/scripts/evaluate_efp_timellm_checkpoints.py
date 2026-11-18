"""Re-evaluate original TimeLLM saved validation checkpoints on all test windows."""
from pathlib import Path
import os,sys,re,json,time,subprocess,fcntl
ROOT=Path('/home/home_new/qsmx/pycodes/GreyFerm')
OUT=ROOT/'test_results/efp_all_baselines_3seed'
PY='/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python'
HERE=Path(__file__).resolve()

def queue():
    lock=open(OUT/'.checkpoint_eval.lock','w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: return
    pipeline=ROOT/'test_results/efp_all_baselines_3seed_pipeline.log'
    while 'ALL_EFP_ALL_BASELINES_3SEED_COMPLETE' not in pipeline.read_text(errors='replace'):
        time.sleep(30)
    jobs=json.loads((OUT/'manifest.json').read_text())
    for j in jobs:
        if j['model']!='TimeLLM': continue
        source=Path(j['log']).read_text(errors='replace')
        log=OUT/(Path(j['log']).stem+'.checkpoint_eval.log')
        if log.exists() and re.search(r'^mse:[0-9.eE+-]+, mae:[0-9.eE+-]+$',log.read_text(),re.M): continue
        token=f"EFP_long_sub{j['sub']}_h{j['h']}"+(f"_seed{j['seed']}" if j['seed']!=2021 else '')+'_TimeLLM_'
        dirs=list((ROOT/'checkpoints/efp_multimodal/TimeLLM').glob('*'+token+'*'))
        assert len(dirs)==1, (token,dirs)
        checkpoint=dirs[0]/'checkpoint'
        assert checkpoint.exists(), checkpoint
        if (dirs[0]/'checkpoint.pth').exists():
            raise RuntimeError('Both checkpoint variants present; audit required before choosing one')
        command=j['command']; modelargs=command[command.index('run_main.py')+1:]
        modelargs=[a for a in modelargs if a!='--efp_six_gpu']
        cmd=[PY,'-m','accelerate.commands.launch','--multi_gpu','--num_processes','6','--num_machines','1','--num_cpu_threads_per_process','4','--mixed_precision','no','--main_process_port','29668',str(HERE),'--efp_checkpoint',str(checkpoint)]+modelargs
        for attempt in range(3):
            while True:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                free={int(a):int(b) for a,b in (line.split(',') for line in raw.strip().splitlines())}
                if min(free[g] for g in range(2,8))>=8000: break
                time.sleep(30)
            print('EVAL_CHECKPOINT_START',j['seed'],j['sub'],j['h'],'GPUS2-7',flush=True)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2,3,4,5,6,7',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
            with open(log,'w') as f: result=subprocess.run(cmd,cwd=ROOT.parent/'Time-LLM',env=env,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode==0 and re.search(r'^mse:[0-9.eE+-]+, mae:[0-9.eE+-]+$',log.read_text(),re.M): break
            if attempt==2: raise RuntimeError(f'Checkpoint evaluation failed: {log}')
            log.rename(str(log)+f'.failed.{int(time.time())}')
            time.sleep(30)
        print('EVAL_CHECKPOINT_DONE',j['seed'],j['sub'],j['h'],flush=True)
    print('ALL_EFP_TIMELLM_CHECKPOINT_EVALUATIONS_COMPLETE',flush=True)

if '--queue' in sys.argv:
    queue(); sys.exit(0)

# Reuse the source parser without executing its training loop or changing defaults.
import ast
source_dir=ROOT.parent/'Time-LLM'
os.chdir(source_dir);sys.path.insert(0,str(source_dir))
tree=ast.parse((source_dir/'run_main.py').read_text())
prefix=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in node.targets): break
    prefix.append(node)
namespace={}
exec(compile(ast.Module(body=prefix,type_ignores=[]),str(source_dir/'run_main.py'),'exec'),namespace)
parser=namespace['parser'];parser.add_argument('--efp_checkpoint',required=True)
args=parser.parse_args()
import torch,numpy as np,random
from accelerate import Accelerator
random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
accelerator=Accelerator()
assert accelerator.num_processes==6
print('EFP_CHECKPOINT_EVAL_RANK',accelerator.process_index,'DEVICE',accelerator.device,flush=True)
args.content=namespace['load_content'](args)
data,loader=namespace['data_provider'](args,'test')
# Evaluation must not inherit training drop_last=True; include all windows.
loader=torch.utils.data.DataLoader(data,batch_size=args.eval_batch_size,shuffle=False,num_workers=args.num_workers,drop_last=False)
model=namespace['TimeLLM'].Model(args).float()
state=torch.load(args.efp_checkpoint,map_location='cpu',weights_only=False)
model.load_state_dict(state,strict=True)
model,loader=accelerator.prepare(model,loader)
model.eval(); mse_sum=mae_sum=0.; elements=windows=0
with torch.no_grad():
    for bx,by,bxm,bym in loader:
        bx,by,bxm,bym=[x.float().to(accelerator.device) for x in (bx,by,bxm,bym)]
        decoder=torch.cat([by[:,:args.label_len,:],torch.zeros_like(by[:,-args.pred_len:,:])],dim=1)
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            pred=model(bx,bxm,decoder,bym)
            if args.output_attention: pred=pred[0]
        fdim=-1 if args.features=='MS' else 0
        pred=pred[:,-args.pred_len:,fdim:];true=by[:,-args.pred_len:,fdim:]
        pred,true=accelerator.gather_for_metrics((pred,true))
        if accelerator.is_main_process:
            error=(pred-true).double();mse_sum+=error.square().sum().item();mae_sum+=error.abs().sum().item()
            elements+=error.numel();windows+=len(error)
if accelerator.is_main_process:
    assert windows==len(data), (windows,len(data))
    print('EFP_CHECKPOINT',args.efp_checkpoint,'ALL_TEST_WINDOWS',windows,'SEED',args.seed,flush=True)
    print(f'mse:{mse_sum/elements:.9f}, mae:{mae_sum/elements:.9f}',flush=True)
accelerator.wait_for_everyone()
