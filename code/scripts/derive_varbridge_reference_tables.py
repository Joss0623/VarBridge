"""Derive confidence/significance and paper-style LLM tables from immutable observations."""
from pathlib import Path
import csv,json,math,statistics,collections,hashlib
ROOT=Path("/home/home_new/qsmx/pycodes/GreyFerm");R=ROOT/"test_results/varbridge_lhc_gemp";D=R/"deliverables"
def read(p):return list(csv.DictReader(p.open()))
def write(name,rows):
 with (D/name).open("w") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
raw=read(D/"main_three_seed_raw.csv");models=list(dict.fromkeys(x["model"] for x in raw))
tests=[]
for task in ["EFP_long_sub1","EFP_long_sub2","EFP_long_sub3","CHO"]:
 for H in [6,12,24,48]:
  for model in [x for x in models if x!="VarBridge"]:
   for metric in ["mse","mae"]:
    diffs=[]
    for seed in [2021,2022,2023]:
     own=next(x for x in raw if x["dataset"]==task and x["model"]=="VarBridge" and int(x["horizon"])==H and int(x["seed"])==seed)
     base=next(x for x in raw if x["dataset"]==task and x["model"]==model and int(x["horizon"])==H and int(x["seed"])==seed)
     diffs.append(float(own[metric])-float(base[metric]))
    m=statistics.mean(diffs);sd=statistics.stdev(diffs);se=sd/math.sqrt(3);t=abs(m/se) if se else (math.inf if m else 0)
    root=math.sqrt(t*t+2);p=2/(root*(root+t)) if math.isfinite(t) else 0.
    tests.append(dict(dataset=task,horizon=H,baseline=model,metric=metric,n_seed_pairs=3,mean_difference=m,difference_sd=sd,difference_ci95_lower=m-4.302652729911275*se,difference_ci95_upper=m+4.302652729911275*se,p_two_sided=p,assumption="Exploratory paired t; normal seed differences cannot be established with n3; unequal training budgets remain"))
assert len(tests)==384
last=0
for rank,x in enumerate(sorted(tests,key=lambda x:x["p_two_sided"])):
 last=max(last,min(1,(len(tests)-rank)*x["p_two_sided"]));x["p_holm"]=last;x["fusion_lower_and_adjusted_p_below05"]=x["mean_difference"]<0 and last<.05
write("exploratory_seed_pair_tests.csv",tests)
# Definition-scoring tables: modes across the three complete trials for each wording.
receipts=[json.loads(l) for l in (R/"api/receipts.jsonl").read_text().splitlines()]
received=[x for x in receipts if x["status"]=="received" and x.get("format_valid")]
model_names=set(x["name"] for x in received)
slots={(x["dataset"],x["name"],x["prompt_version"],x["trial"]) for x in received}
N=max(n for n in range(6) if all((d,m,v,k) in slots for d in ["EFP_long_sub2","CHO"] for m in model_names for v in [0,1] for k in range(n)))
assert N>=3 and len(model_names)==6
rows=[x for x in received if x["trial"]<N];assert len(rows)==24*N
ref=json.loads((R/"api/researcher_reference.json").read_text())
def mode(v):
 c=collections.Counter(v);return min(c,key=lambda x:(-c[x],x))
def qwk(y,p):
 n=len(y);a=collections.Counter(y);b=collections.Counter(p)
 o=sum((u-v)**2 for u,v in zip(y,p))/n;e=sum((u-v)**2*a[u]*b[v] for u in range(3) for v in range(3))/n**2
 return 1-o/e
def ndcg(y,p):
 order=sorted(range(len(y)),key=lambda i:-p[i]);dcg=0.;j=0
 while j<len(order):
  k=j+1
  while k<len(order) and p[order[k]]==p[order[j]]:k+=1
  dcg+=sum(y[i] for i in order[j:k])*sum(1/math.log2(i+2) for i in range(j,k))/(k-j);j=k
 ideal=sum(v/math.log2(i+2) for i,v in enumerate(sorted(y,reverse=True)));return dcg/ideal
def fmi(y,p):
 a=collections.Counter(y);b=collections.Counter(p);ab=collections.Counter(zip(y,p));pair=lambda n:n*(n-1)/2
 return sum(pair(n) for n in ab.values())/math.sqrt(sum(pair(n) for n in a.values())*sum(pair(n) for n in b.values()))
table=[]
for task in ["EFP_long_sub2","CHO"]:
 y=ref[task]["reference"];yf=[z for r in y for z in r];yg=[max(range(3),key=lambda g:r[g]) for r in y]
 for model in dict.fromkeys(x["name"] for x in rows):
  for v in [0,1]:
   q=[x for x in rows if x["dataset"]==task and x["name"]==model and x["prompt_version"]==v];assert len(q)==N
   p=[[mode([x["parsed"]["scores"][j][g] for x in q]) for g in range(3)] for j in range(5)]
   pg=[mode([max(range(3),key=lambda g:x["parsed"]["scores"][j][g]) for x in q]) for j in range(5)]
   pf=[z for r in p for z in r];diff=[u-v for u,v in zip(yf,pf)]
   table.append(dict(dataset=task,model=model,prompt_version=v,replies=N,score_MAE=statistics.mean(abs(x) for x in diff),QWK=qwk(yf,pf),NDCG=statistics.mean(ndcg(a,b) for a,b in zip(y,p)),partition_FMI=fmi(yg,pg),SSim=1-math.sqrt(sum(x*x for x in diff))/(2*math.sqrt(15)),latency_mean_seconds=statistics.mean(x["elapsed_seconds"] for x in q),latency_sample_sd_seconds=statistics.stdev(x["elapsed_seconds"] for x in q),aggregation="Elementwise modal rating and modal argmax assignment over complete replies; ties use lowest index",reference="Researcher rubric, not expert ground truth",all_five_planned_trials_complete=(N==5)))
write("llm_modal_tables.csv",table)
# Classify every original entry by concrete deliverable and evidence; filled by renderer.
print(json.dumps(dict(raw_main=624,paired_exploratory_tests=len(tests),holm_family=384,llm_modal_rows=len(table),balanced_llm_calls=len(rows),planned_llm_calls=120,successful_llm_calls=len(received))))
