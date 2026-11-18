from pathlib import Path
import json,concurrent.futures
from acquire_bioprocess_pretraining_sources import download
ROOT=Path(__file__).resolve().parents[1]
d=json.loads((ROOT/"dataset/bioprocess_pretraining_sources/ecoli/download_routing.json").read_text())
jobs=[["ecoli",f["name"],f["url"],f["size"],"md5:"+f["md5"]] for f in d]
results=[]
with concurrent.futures.ThreadPoolExecutor(4) as ex:
 for row in ex.map(download,jobs):
  row["url"]="https://doi.org/10.6084/m9.figshare.24999488.v1"
  results.append(row)
  (ROOT/"test_results/bioprocess_pretraining_inventory/ecoli_download_manifest.json").write_text(json.dumps(results,indent=2))
print("ECOLI_SOURCE_DOWNLOAD_COMPLETE",flush=True)
