"""Recover only missing API observations, then audit and summarize this supplement."""
import os,sys,time,json,subprocess,fcntl
from pathlib import Path
ROOT=Path("/home/home_new/qsmx/pycodes/GreyFerm");R=ROOT/"test_results/varbridge_lhc_gemp";A=R/"api"
lock=(R/"finish_six_models.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=os.environ.copy();env.update(OMP_NUM_THREADS="2",HF_HUB_OFFLINE="1",TOKENIZERS_PARALLELISM="false",PYTHONDONTWRITEBYTECODE="1")
def active_api():
 for p in Path("/proc").iterdir():
  if not p.name.isdigit():continue
  try:args=(p/"cmdline").read_bytes().split(b"\0")
  except OSError:continue
  if len(args)>1 and b"python" in args[0] and any(a.endswith(b"varbridge_lhc_gemp_api.py") for a in args[1:]):return True
 return False
while active_api():time.sleep(20)
for recovery in range(8):
 if (A/"ALL_API_RESPONSES_COMPLETE").exists():break
 rows=[json.loads(s) for s in (A/"receipts.jsonl").read_text().splitlines()]
 last=rows[-1]
 if (A/"BUDGET_PAUSE").exists():raise RuntimeError("5CNY study cap: no further calls")
 slot=(last["model"],last["dataset"],last["prompt_version"],last["trial"])
 attempts=[x for x in rows if (x["model"],x["dataset"],x["prompt_version"],x["trial"])==slot]
 if len(attempts)>=4:raise RuntimeError("Repeated failure of same slot; retain partial result, no false completion")
 if last["status"]=="ambiguous_failure":
  if last.get("error_type") not in ["timeout","TimeoutError"] or last.get("response"):raise RuntimeError("Unresolved API failure needs inspection")
  p=A/"technical_failure_review.json";review=json.loads(p.read_text()) if p.exists() else {}
  review[last["request_id"]]=dict(disposition="No recoverable reply; preserve and reserve full maximum possible billed cost",allow_replacement=True,held_cny=last["bound_cny"],new_attempt_must_have_unique_id=True)
  p.write_text(json.dumps(review,indent=2))
 elif last["status"]=="http_error":
  if last.get("http_status") not in [429,500,502,503,504]:raise RuntimeError("Nontransient provider rejection: "+str(last.get("http_status")))
 else:raise RuntimeError("API stopped unexpectedly; inspect logs")
 time.sleep(30)
 log=R/"logs/api.log";log.rename(log.with_name("api.recovery.%d.log"%time.time_ns()))
 with log.open("w") as f:
  subprocess.run([sys.executable,"-u",str(ROOT/"scripts/varbridge_lhc_gemp_api.py")],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
if not (A/"ALL_API_RESPONSES_COMPLETE").exists():raise RuntimeError("Remaining API slots incomplete")
with (R/"logs/api_analysis.log").open("a") as f:
 subprocess.run([sys.executable,"-u",str(ROOT/"scripts/analyze_varbridge_lhc_gemp_api.py")],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
with (R/"logs/finalizer.log").open("a") as f:
 subprocess.run([sys.executable,"-u",str(ROOT/"scripts/finalize_varbridge_lhc_gemp.py")],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
print("SIX_MODEL_FULL_AUDIT_COMPLETE",flush=True)
