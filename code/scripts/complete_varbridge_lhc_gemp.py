"""Finish the existing supplement pipeline after its current bounded API run."""
import os,sys,time,json,csv,subprocess
from pathlib import Path
ROOT=Path("/home/home_new/qsmx/pycodes/GreyFerm")
R=ROOT/"test_results/varbridge_lhc_gemp"
api_pid=2227668
while True:
 p=Path("/proc")/str(api_pid)/"cmdline"
 try:alive=b"varbridge_lhc_gemp_api.py" in p.read_bytes()
 except OSError:alive=False
 if not alive:break
 time.sleep(20)
env=os.environ.copy()
env.update(OMP_NUM_THREADS="2",HF_HUB_OFFLINE="1",TOKENIZERS_PARALLELISM="false",PYTHONDONTWRITEBYTECODE="1")
with (R/"logs/api_analysis.log").open("a") as f:
 subprocess.run([sys.executable,"-u",str(ROOT/"scripts/analyze_varbridge_lhc_gemp_api.py")],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
state=json.loads((R/"api/analysis_status.json").read_text())
if not state["complete"]:
 (R/"API_FOLLOWUP_REQUIRED.json").write_text(json.dumps(state,indent=2))
 raise RuntimeError("API study not complete; inspect sanitized API log and budget status. No completion claim.")
with (R/"logs/finalizer.log").open("a") as f:
 subprocess.run([sys.executable,"-u",str(ROOT/"scripts/finalize_varbridge_lhc_gemp.py")],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
print("SUPPLEMENTS_FINAL_AUDIT_COMPLETE; manuscript and formal result files unchanged",flush=True)
