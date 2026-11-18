"""Unpack and inventory the verified full IndPenSim archive; no fitting."""
from pathlib import Path
import json,zipfile,shutil,csv,pandas as pd,numpy as np,hashlib
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"dataset/bioprocess_pretraining_sources/indpensim"
OUT=ROOT/"test_results/bioprocess_pretraining_inventory"
raw=BASE/"raw";raw.mkdir(exist_ok=True)
with zipfile.ZipFile(BASE/"100_Batches_IndPenSim.zip") as z:
 for info in z.infolist():
  if info.is_dir():continue
  name=Path(info.filename).name
  assert name in ["100_Batches_IndPenSim_V3.csv","100_Batches_IndPenSim_Statistics.csv"]
  dest=raw/name
  if not dest.exists():
   assert shutil.disk_usage(BASE).free-info.file_size>20*1024**3
   part=dest.with_suffix(".extracting")
   with z.open(info) as r,part.open("wb") as w:shutil.copyfileobj(r,w,4*1024**2)
   assert part.stat().st_size==info.file_size
   part.rename(dest)
  print("UNPACKED_CRC_VERIFIED",name,info.file_size,flush=True)
p=raw/"100_Batches_IndPenSim_V3.csv"
with p.open() as f:columns=next(csv.reader(f))
original_header=columns[:]
# The publisher's PAT label contains two unquoted commas. Rejoin its
# three fragments in the parser only; never rewrite the original CSV.
assert columns[33:36]==["1- No Raman spec"," 1-Raman spec recorded","2-PAT control(PAT_ref:PAT ref)"]
columns=columns[:33]+[",".join(columns[33:36])]+columns[36:]
start=next(i for i,c in enumerate(columns) if c=="2400")
process=columns[:start];spectral=columns[start:]
print("INDPENSIM_SCHEMA",len(columns),"non_spectral",len(process),"spectral",len(spectral),process,flush=True)
batch_col="Batch ID"
assert batch_col in process
acc={};total=0;missing={c:0 for c in process};nonfinite={c:0 for c in process}
for d in pd.read_csv(p,header=None,skiprows=1,names=columns,usecols=process,chunksize=4096):
 total+=len(d)
 for c in process:
  missing[c]+=int(d[c].isna().sum())
  if pd.api.types.is_numeric_dtype(d[c]):nonfinite[c]+=int(np.isinf(d[c]).sum())
 for bid,g in d.groupby(batch_col):
  key=str(bid);entry=acc.setdefault(key,{"batch_id":key,"rows":0,"time":[],"reference_ids":set(),"fault_values":set(),"control_values":set()})
  entry["rows"]+=len(g);entry["time"].extend(g["Time (h)"].tolist())
  entry["reference_ids"].update(g["Batch reference(Batch_ref:Batch ref)"].dropna().unique().tolist())
  entry["fault_values"].update(g["Fault flag"].dropna().unique().tolist())
  entry["control_values"].update(g["0 - Recipe driven 1 - Operator controlled(Control_ref:Control ref)"].dropna().unique().tolist())
rows=[]
for key,e in acc.items():
 t=np.sort(np.unique(np.array(e.pop("time"),dtype=float)));delta=np.diff(t)
 assert len(delta)>0, (key,e["rows"])
 e.update(time_min=float(t.min()),time_max=float(t.max()),median_interval_hours=float(np.median(delta)),min_interval_hours=float(delta.min()),max_interval_hours=float(delta.max()))
 for k in ["reference_ids","fault_values","control_values"]:e[k]=sorted(e[k])
 rows.append(e)
pd.DataFrame(rows).to_csv(OUT/"indpensim_batches.csv",index=False)
stats=pd.read_csv(raw/"100_Batches_IndPenSim_Statistics.csv")
report={"physical_simulated_batches":len(rows),"rows":total,"total_columns":len(columns),"original_header_fields":len(original_header),"header_correction":"Rejoined original fields 33-35 (unquoted commas in PAT label), parser only; raw file unchanged","non_spectral_columns":len(process),"raman_columns":len(spectral),"process_columns":process,"statistics_rows":len(stats),"statistics_columns":list(stats),"batch_records":rows,"missing_values":missing,"infinite_values":nonfinite,"license":"CC BY 4.0","doi":"10.17632/pdnjz7zz5x.2","full_original_archive_sha256":"bc434c6cf96444fee3c915d434ac84e2a9c6492be4daebcfa4aed9cdff181f00","issues":["Simulation; do not call real industrial measurement data","Metadata/control/fault/batch columns must not become forecasting targets or inputs leaking labels","Offline assays and simulated continuous internal states have different availability"]}
(OUT/"indpensim_audit.json").write_text(json.dumps(report,indent=2,default=str))
assert len(rows)==100 and total>100000
print("INDPENSIM_FULL_AUDIT_PASS",len(rows),total,len(columns),flush=True)
