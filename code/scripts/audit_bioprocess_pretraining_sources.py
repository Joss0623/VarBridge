"""Read-only source inventory; writes audit tables only. No model imports or fitting."""
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,re,csv
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"dataset/bioprocess_pretraining_sources"
OUT=ROOT/"test_results/bioprocess_pretraining_inventory"
OUT.mkdir(exist_ok=True)
def save(name,data):
 (OUT/(name+".json")).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str))
def intervals(values):
 v=np.sort(np.unique(np.asarray(values,dtype=float)))
 delta=np.diff(v);delta=delta[np.isfinite(delta)&(delta>0)]
 return None if not len(delta) else {"min":float(np.min(delta)),"median":float(np.median(delta)),"max":float(np.max(delta))}
def pichia():
 p=BASE/"pichia/Dataset_online.xlsx";book=pd.ExcelFile(p)
 batches=[]
 for sn in book.sheet_names:
  d=pd.read_excel(book,sheet_name=sn,header=1).dropna(how="all")
  batches.append({"batch_id":sn,"rows":len(d),"non_null":d.count().to_dict(),
                  "input_target_biomass_max_abs_difference":float(np.nanmax(np.abs(d.iloc[:,1]-d.iloc[:,5]))),
                  "input_target_product_max_abs_difference":float(np.nanmax(np.abs(d.iloc[:,2]-d.iloc[:,6])))})
 pd.DataFrame([{k:v for k,v in b.items() if k!="non_null"} for b in batches]).to_csv(OUT/"pichia_batches.csv",index=False)
 save("pichia_audit",{"batches":len(batches),"total_rows":sum(b["rows"] for b in batches),"columns":list(d),
  "declared_grid_seconds":60,"explicit_time_column":False,"physical_variables":["methanol feed Fs","biomass X","product P","culture volume V"],"derived_columns":["shock factor Sh","volumetric flow F (definition needs paper)"],
  "issues":["Target product header says gDCW/L while record says mg/L; retain original values and flag units",
  "Prepared 3900-row sequence per experiment, not demonstrated 3900 independent assays",
  "Input and target X/P need temporal alignment verification before forecasting; do not assume same-row targets are future",
  "No observed-versus-reconstructed mask in downloaded workbook"],"batch_records":batches,"license":"CC BY 4.0","record":"10.5281/zenodo.15855565"})
def nistcho():
 b=BASE/"nistcho/timeseries"
 d=pd.read_csv(b/"01_ViCell_growth_data.csv")
 batches=[]
 for keys,g in d.groupby(["Condition","Replicate","Batch"]):
  batches.append(dict(zip(["condition","replicate","campaign"],keys),rows=len(g),time_hours_min=float(g.Hours.min()),time_hours_max=float(g.Hours.max()),interval_hours=intervals(g.Hours)))
 pd.DataFrame(batches).to_csv(OUT/"nistcho_batches.csv",index=False)
 save("nistcho_audit",{"batches":len(batches),"campaigns":sorted(d.Batch.unique()),"conditions":sorted(d.Condition.unique()),"growth_rows":len(d),"columns":list(d),"batch_records":batches,
  "variables":["viability (%)","viable cell density (10^6 cells/mL)","average cell diameter","antibody titer (ug/mL)","lactate (mM)","glucose (g/L)","feeding and sampling metadata"],
  "scope":"Seven original/top-level cultivation tables extracted and CRC-verified; MS binaries excluded",
  "license":"CC BY 4.0","record":"10.5281/zenodo.18609042","issues":["Use individual replicate table, not averaged table as extra batches","Uneven approximately daily sampling, initial interval commonly 72h","Condition letters and strategy names must map using provided protocol and author scripts","MS technical repetitions do not create extra cultivation batches"]})
def biolector():
 p=BASE/"biolector/8X4PF4.csv";s=p.read_text(errors="replace")
 lines=s.splitlines()
 wells=re.findall(r"\[\d+_well\]\s*(\w+)",s)
 # Raw BioLector structured export: print data section descriptors into the audit.
 markers=[(i,l) for i,l in enumerate(lines) if l.startswith("====")]
 save("biolector_structure",{"wells":wells,"well_count":len(set(wells)),"markers":markers,"last_header_lines":lines[-4:]})
 print("BIOLECTOR_MARKERS",markers[-12:],flush=True)
 for i,l in markers[-8:]:print("BIOLECTOR_SECTION",i,lines[i:i+5],flush=True)
 for f in ["8X4PF4_eventlog.xlsx","8X4PF4_medium_composition.xlsx"]:
  w=pd.ExcelFile(BASE/"biolector"/f)
  print("BIOLECTOR_BOOK",f,w.sheet_names,flush=True)
  for sn in w.sheet_names:print(sn,pd.read_excel(w,sheet_name=sn).shape,pd.read_excel(w,sheet_name=sn).head(2).to_dict("records"),flush=True)
def ecoli():
 files=sorted((BASE/"ecoli").glob("*.csv"))
 if len(files)!=4:
  print("ECOLI pending",len(files),flush=True);return
 allb=[];schemas={}
 for p in files:
  d=pd.read_csv(p);schemas[p.name]=list(d)
  print("ECOLI_SCHEMA",p.name,d.shape,list(d),flush=True)
  for bid,g in d.groupby("Batch ID"):
   tv=pd.to_numeric(g["Time"],errors="coerce").dropna()
   allb.append({"campaign_file":p.name,"batch_id":str(bid),"rows":len(g),"time_min":float(tv.min()),"time_max":float(tv.max()),"interval_time_units":intervals(tv)})
 pd.DataFrame(allb).to_csv(OUT/"ecoli_batches.csv",index=False)
 save("ecoli_audit",{"batches":len(allb),"batch_id_unique_across_files":len(set(b["batch_id"] for b in allb)),"campaign_files":len(files),"columns_by_file":schemas,"batch_records":allb,"license":"CC BY 4.0","source_grid":"9-second interpolated/aligned release; Time units must be checked","record":"10.6084/m9.figshare.24999488.v1"})
def existing():
 rows=[]
 for p in sorted((ROOT/"dataset").glob("EFP_long_sub*/*.csv")):
  d=pd.read_csv(p);rows.append({"path":str(p.relative_to(ROOT)),"rows":len(d),"batches":int(d.batch_id.nunique()),"columns":list(d)})
 m=pd.read_csv(ROOT/"dataset/CHO/batch_manifest.csv")
 save("existing_sources",{"efp_views":rows,"efp_warning":"All three views overlap in batch/time and must be one source family","cho_raw_reactor_segments_manifest":m.to_dict("records"),"cho_warning":"Parallel reactors from the same EXP must remain grouped; use raw sources to extend to all variables"})
if __name__=="__main__":
 for f in [pichia,nistcho,biolector,ecoli,existing]:
  print("AUDIT",f.__name__,flush=True);f()
 print("CURRENT_AVAILABLE_SOURCE_AUDITS_COMPLETE",flush=True)
