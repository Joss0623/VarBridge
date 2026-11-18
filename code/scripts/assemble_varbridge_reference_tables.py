"""Assemble source-backed reference tables; no training, no canonical paper edits."""
from pathlib import Path
import csv,json,re,math,statistics,hashlib,ast
ROOT=Path("/home/home_new/qsmx/pycodes/GreyFerm");R=ROOT/"test_results/varbridge_lhc_gemp";D=R/"deliverables";D.mkdir(exist_ok=True)
MODELS=["VarBridge","DLinear","PatchTST","TimesNet","iTransformer","TimeMixer","Crossformer","TiDE","FEDformer","xPatch","TimeCMA","T3Time","TimeLLM"]
TASKS=["EFP_long_sub1","EFP_long_sub2","EFP_long_sub3","CHO"]
def out(name,rows):
 assert rows
 with (D/name).open("w") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def epochs(text):return max(map(int,re.findall(r"Epoch:\s*(\d+)",text,re.I)))
raw=[]
# Current adopted EFP model comes from bound-mechanism logs, not historical fusion JSON.
for sub in [1,2,3]:
 for seed in [2021,2022,2023]:
  for trained_h in [6,12,24]:
   p=ROOT/"test_results/efp_bound_mechanism"/("EFP_long_sub%d_h%d_seed%d_correct.log"%(sub,trained_h,seed))
   text=p.read_text();assert epochs(text)==100
   found=[]
   for line in text.splitlines():
    if line.startswith("EFP_BOUND_RESULT "):
     d=dict(re.findall(r"(\w+)=([^\s]+)",line))
     if d.get("variant")=="correct":
      H=int(d["h"]);assert H==trained_h or (H==48 and trained_h==24)
      assert int(d["completed_epochs"])==100
      raw.append(dict(dataset="EFP_long_sub"+str(sub),model="VarBridge",horizon=H,seed=seed,mse=float(d["mse"]),mae=float(d["mae"]),completed_epochs=100,budget_max_epochs=100,log=str(p),log_sha256=digest(p),protocol=d["protocol"]))
      found.append(H)
   assert sorted(found)==([24,48] if trained_h==24 else [trained_h]),(p,found)
for x in json.loads((ROOT/"test_results/efp_all_baselines_3seed/manifest.json").read_text()):
 p=Path(x["log"]);p=ROOT/p if not p.is_absolute() else p
 score=p.with_suffix(".checkpoint_eval.log") if x["model"]=="TimeLLM" else p
 t=score.read_text();values=re.findall(r"mse:\s*([\d.eE+-]+),\s*mae:\s*([\d.eE+-]+)",t);assert values,score
 mse,mae=map(float,values[-1]);e=epochs(p.read_text());assert e<=10
 if x["model"]=="TimeLLM":assert "ALL_TEST_WINDOWS" in t
 raw.append(dict(dataset="EFP_long_sub"+str(x["sub"]),model=x["model"],horizon=x["h"],seed=x["seed"],mse=mse,mae=mae,completed_epochs=e,budget_max_epochs=10,log=str(score),log_sha256=digest(score),protocol="direct"))
for model in MODELS:
 for seed in [2021,2022,2023]:
  for H in [6,12,24,48]:
   h=24 if model=="VarBridge" and H==48 else H
   p=ROOT/("test_results/cho_bound_mechanism"/Path("Fusion_h%d_seed%d_correct.log"%(h,seed)) if model=="VarBridge" else Path("test_results/cho")/("%s_h%d_seed%d_full.log"%(model,h,seed)))
   t=p.read_text();matches=[]
   for line in t.splitlines():
    if line.startswith("CHO_RESULT "):
     d=dict(re.findall(r"(\w+)=([^\s]+)",line))
     if int(d["h"])==H:matches.append(d)
   assert len(matches)==1,(p,H,len(matches))
   d=matches[0];e=int(d["completed_epochs"]);assert e==100 if model=="VarBridge" else e<=10
   raw.append(dict(dataset="CHO",model=model,horizon=H,seed=seed,mse=float(d["mse"]),mae=float(d["mae"]),completed_epochs=e,budget_max_epochs=100 if model=="VarBridge" else 10,log=str(p),log_sha256=digest(p),protocol=d["protocol"]))
keys=[(x["dataset"],x["model"],x["horizon"],x["seed"]) for x in raw]
assert len(raw)==len(set(keys))==624
assert all(math.isfinite(x["mse"]) and math.isfinite(x["mae"]) for x in raw)
out("main_three_seed_raw.csv",raw)
summary=[]
for task in TASKS:
 for model in MODELS:
  for H in [6,12,24,48,"Avg"]:
   q=[x for x in raw if x["dataset"]==task and x["model"]==model and (H=="Avg" or x["horizon"]==H)]
   for metric in ["mse","mae"]:
    v=[statistics.mean(x[metric] for x in q if x["seed"]==seed) for seed in [2021,2022,2023]]
    assert all(sum(x["seed"]==seed for x in q)==(4 if H=="Avg" else 1) for seed in [2021,2022,2023])
    m=statistics.mean(v);sd=statistics.stdev(v);half=4.302652729911275*sd/math.sqrt(3)
    summary.append(dict(dataset=task,model=model,horizon=H,metric=metric,n_seeds=3,mean=m,sample_sd=sd,ci95_lower=m-half,ci95_upper=m+half,seed2021=v[0],seed2022=v[1],seed2023=v[2],interval="Student t across three distinct training seeds; not batch-population confidence",budget_max_epochs=100 if model=="VarBridge" else 10))
out("main_three_seed_statistics.csv",summary)
selected=[x for x in summary if x["model"] in ["VarBridge",("PatchTST" if x["dataset"]=="CHO" else "Crossformer")]]
out("LHC_Table_V_statistics.csv",selected)
# Baseline inventory reflects actual numerical/text interfaces, not unverified CI/CD labels.
families={"DLinear":"Decomposition + linear projection","PatchTST":"Patch Transformer","TimesNet":"Periodic 2D temporal blocks","iTransformer":"Variable-token Transformer","TimeMixer":"Multiscale temporal mixing","Crossformer":"Cross-time and cross-variable attention","TiDE":"Dense temporal encoder-decoder","FEDformer":"Frequency-enhanced decomposition","xPatch":"Decomposition + patch MLP","TimeCMA":"Numerical/text representation alignment","T3Time":"Temporal/spectral/text fusion","TimeLLM":"Numerical patch reprogramming + GPT-2","VarBridge":"Variable-bound numerical-text forecast fusion"}
inv=[]
for m in MODELS:
 q=[x for x in raw if x["model"]==m]
 inv.append(dict(model=m,modality="Numerical + text" if m in ["TimeCMA","T3Time","TimeLLM","VarBridge"] else "Numerical",representation=families[m],budget_max_epochs=100 if m=="VarBridge" else 10,observed_completed_epochs_min=min(x["completed_epochs"] for x in q),observed_completed_epochs_max=max(x["completed_epochs"] for x in q),seeds="2021;2022;2023",source="main_three_seed_raw.csv"))
out("baseline_inventory.csv",inv)
# Preserve reference groups as researcher-defined assessment labels; not model architecture.
ref=json.loads((R/"api/researcher_reference.json").read_text());groups=[]
for task,record in ref.items():
 for i,name in enumerate(record["groups"]):
  groups.append(dict(dataset=task,group_index=i,group_name=name,variable_scores="; ".join(n+":"+str(record["reference"][j][i]) for j,n in enumerate(record["names"])),provenance="Researcher-constructed rubric; not expert annotations; not learned model partitions"))
out("researcher_functional_groups.csv",groups)
# Exact-epoch existing checkpoints for a new minimal numerical/LLM efficiency counterpart.
E=R/"budget_efficiency";E.mkdir(exist_ok=True);sources=[]
for m in ["DLinear","iTransformer","TimeLLM","VarBridge"]:
 x=next(x for x in raw if x["dataset"]=="CHO" and x["model"]==m and x["horizon"]==24 and x["seed"]==2021)
 assert x["completed_epochs"]==(100 if m=="VarBridge" else 10)
 ck=ROOT/("checkpoints/cho_bound_mechanism/Fusion_h24_seed2021_correct.pth" if m=="VarBridge" else "checkpoints/cho/%s_h24_seed2021_full.pth"%m)
 assert ck.exists()
 sources.append(dict(model=m,checkpoint=str(ck),checkpoint_sha256=digest(ck),epochs=x["completed_epochs"],mse=x["mse"],mae=x["mae"],log=x["log"],training_scope="Original training log, shared-server conditions; not controlled training speed"))
with (E/"sources.csv").open("w") as f:
 w=csv.DictWriter(f,fieldnames=list(sources[0]));w.writeheader();w.writerows(sources)
print(json.dumps(dict(raw_rows=len(raw),statistics_rows=len(summary),models=len(MODELS),datasets=len(TASKS),tableV_rows=len(selected),efficiency_models=[x["model"] for x in sources])))
