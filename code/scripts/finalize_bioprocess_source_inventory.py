from pathlib import Path
import json,csv,hashlib,datetime,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"dataset/bioprocess_pretraining_sources"
OUT=ROOT/"test_results/bioprocess_pretraining_inventory"
def read(n):return json.loads((OUT/n).read_text())
a={n:read(n+"_audit.json") for n in ["pichia","ecoli","biolector","nistcho","indpensim"]}
assert [a["pichia"]["batches"],a["ecoli"]["batches"],a["biolector"]["wells"],a["nistcho"]["batches"],a["indpensim"]["physical_simulated_batches"]]==[21,48,48,25,100]
before=read("protected_sources_before.json")
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in before.items())
efp=BASE/"efp_public_full/EFP_long.csv"
assert hashlib.md5(efp.read_bytes()).hexdigest()=="6f65e6af4bc8f5e750414372b8cf81ca"
stats=pd.read_csv(BASE/"indpensim/raw/100_Batches_IndPenSim_Statistics.csv")
ids={int(float(b["batch_id"])) for b in a["indpensim"]["batch_records"]}
assert ids==set(stats["Batch ref"].astype(int))==set(range(1,101))
assert all(abs(b["median_interval_hours"]-.2)<1e-8 for b in a["indpensim"]["batch_records"])
rows=[
["EFP_public",406,50536,"24 process/target fields;27 total columns","released hourly","CC BY 4.0","one source family;three overlapping feature views","10.5281/zenodo.14619074"],
["DYG2024",688,106811,"36 raw fields;24 selected derived fields","241 hourly and447 minute-like batches;offline~12h","private;public redistribution unverified","14 reactors,one factory/product;local QC inventory only","user-supplied"],
["CHO_existing",24,"","38 inline+10 offline in raw source","asynchronous;offline up to twice/day","CC BY 4.0","12 EXP;keep paired reactors grouped;22 current eligible segments","10.5281/zenodo.20829178"],
["Pichia",21,81900,"Fs,X,P,V,Sh,F;8 columns with paired targets","declared60s grid;no timestamp column","CC BY 4.0","16 padded sheets;candidate66030 positive-volume rows;conditional use","10.5281/zenodo.15855565"],
["E_coli",48,1878942,"31 measurement/control/derived fields;33 total columns","9s interpolated grid;Time in hours","CC BY 4.0","4 campaigns x12;campaign+reactor key;interpolation limitation","10.6084/m9.figshare.24999488.v1"],
["BioLector",48,16128,"3 main channels plus environment,medium,event metadata","median780s/channel;112 cycles","repo AGPL-3.0;separate data-only license not found","one plate;truncate by each well harvest event","github.com/JuBiotech/bletl-paper"],
["NISTCHO",25,242,"VCD,viability,diameter,titer,lactate,glucose,feed metadata","~24h;initial gap~72h;9-10 growth samples","CC BY 4.0","2 campaigns;7 conditions;7 cultivation tables;MS binaries excluded","10.5281/zenodo.18609042"],
["IndPenSim_full",100,113935,"30 process variables;37 nonspectral+2200 Raman fields","0.2h=12min","CC BY 4.0","simulation;original header2239 corrected in parser to2237","10.17632/pdnjz7zz5x.2"]
]
with (OUT/"dataset_inventory.csv").open("w") as f:
 w=csv.writer(f);w.writerow(["dataset","trajectories","rows","variables","sampling","license","scope_and_limitations","source"]);w.writerows(rows)
with (OUT/"licenses_and_sources.csv").open("w") as f:
 w=csv.writer(f);w.writerow(["dataset","license","source","evidence"])
 for r in rows:w.writerow([r[0],r[5],r[7],"publisher metadata in source directory;BioLector LICENSE;DYG local inventory"])
manifest={"protected_sources_unchanged":before,"ecoli":read("ecoli_download_manifest.json"),"indpensim":read("indpensim_download_manifest.json"),"nistcho":json.loads((BASE/"nistcho/timeseries_download_manifest.json").read_text()),"small_files":[]}
for name in ["pichia","biolector"]:
 for p in (BASE/name).iterdir():
  if p.is_file():manifest["small_files"].append({"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
(OUT/"final_download_manifest.json").write_text(json.dumps(manifest,indent=2))
fragment=BASE/"nistcho/feeding_strategies_zenodo.zip.part"
if fragment.exists():fragment.unlink()
summary="# Verified bioprocess source inventory\n\nNo training, model changes, manuscript edits or Excel edits.\n\n"
summary+="|Source|Trajectories|Sampling|License|Scope|\n|---|---:|---|---|---|\n"
for r in rows:summary+="|%s|%s|%s|%s|%s|\n"%(r[0],r[1],r[4],r[5],r[6])
summary+="\nNISTCHO: all seven cultivation tables, not the 7.33GB MS binary archive. IndPenSim: full original archive including all Raman channels.\n"
summary+="\nStudy proposal: pretraining_and_transfer_plan_zh.md. Source-level holdout; source-only selection; separate zero-shot and1/3/5-batch target adaptation; compare generic backbone, domain adaptation,correct semantics and matched no/wrong semantics. No execution scheduled.\n"
summary+="\nCritical findings: Pichia zero-padding/target tails and ambiguous product units; E.coli interpolated grid;BioLector single-plate dependence and separate data license not found;IndPenSim publisher header contains two unquoted commas. Detailed audits retain these findings.\n"
(OUT/"inventory_summary.md").write_text(summary)
(OUT/"FINAL_AUDIT_PASS.json").write_text(json.dumps({"status":"PASS","no_training_started":True,"protected_sources_unchanged":True,"nistcho_scope":"complete cultivation table subset;MS binaries excluded","timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
print("BIOPROCESS_DATA_ACQUISITION_AND_INVENTORY_COMPLETE",flush=True)
for r in rows:print(r[0],r[1],r[4],flush=True)
