"""Acquire public data only; no model imports, GPU use, fitting, or preprocessing."""
from pathlib import Path
import requests,json,hashlib,shutil,time,concurrent.futures,fcntl,os
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"dataset/bioprocess_pretraining_sources"
AUDIT=ROOT/"test_results/bioprocess_pretraining_inventory"
def digest(path,kind="sha256"):
 h=hashlib.new(kind)
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def download(job):
 directory,name,url,size,checksum=job
 dest=BASE/directory/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  if size is not None and dest.stat().st_size!=size:raise RuntimeError("Existing size mismatch "+str(dest))
  if checksum and digest(dest,checksum.split(":")[0])!=checksum.split(":")[1]:raise RuntimeError("Existing digest mismatch "+str(dest))
  return {"path":str(dest.relative_to(ROOT)),"bytes":dest.stat().st_size,"sha256":digest(dest),"url":url,"status":"verified_existing","source_checksum":checksum}
 part=dest.with_name(dest.name+".part")
 for attempt in range(5):
  try:
   if shutil.disk_usage(BASE).free < 20*1024**3:raise RuntimeError("DISK_RESERVE: less than 20 GiB available")
   offset=part.stat().st_size if part.exists() else 0
   headers={"User-Agent":"bioprocess-data-audit/1.0"}
   if offset:headers["Range"]="bytes=%d-"%offset
   with requests.get(url,headers=headers,stream=True,timeout=(30,90)) as r:
    r.raise_for_status()
    resume=offset and r.status_code==206
    with part.open("ab" if resume else "wb") as f:
     last=time.monotonic()
     for chunk in r.iter_content(2*1024*1024):
      if chunk:f.write(chunk)
      if time.monotonic()-last>=25:
       f.flush()
       if shutil.disk_usage(BASE).free < 20*1024**3:raise RuntimeError("DISK_RESERVE")
       print("PROGRESS",directory,name,part.stat().st_size,"/",size,flush=True);last=time.monotonic()
   if size is not None and part.stat().st_size!=size:raise RuntimeError("Size mismatch "+str(part.stat().st_size))
   if checksum and digest(part,checksum.split(":")[0])!=checksum.split(":")[1]:
    raise RuntimeError("Checksum mismatch; retained .part for diagnosis")
   part.rename(dest)
   result={"path":str(dest.relative_to(ROOT)),"bytes":dest.stat().st_size,"sha256":digest(dest),"url":url,"status":"downloaded_verified","source_checksum":checksum}
   print("DONE",directory,name,result["bytes"],flush=True)
   return result
  except Exception as e:
   print("RETRY",directory,attempt+1,str(e),flush=True)
   if "DISK_RESERVE" in str(e) or "Checksum" in str(e):raise
   time.sleep(min(3*(attempt+1),12))
 raise RuntimeError("Download retries exhausted "+name)
def main():
 BASE.mkdir(exist_ok=True);AUDIT.mkdir(exist_ok=True)
 with (AUDIT/"download.lock").open("w") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  jobs=json.loads((AUDIT/"download_jobs.json").read_text())
  print("START files",len(jobs),"free_GiB",round(shutil.disk_usage(BASE).free/1024**3,2),flush=True)
  results=[];errors=[]
  with concurrent.futures.ThreadPoolExecutor(2) as ex:
   fs={ex.submit(download,j):j for j in jobs}
   for f in concurrent.futures.as_completed(fs):
    try:results.append(f.result())
    except Exception as e:errors.append({"job":fs[f],"error":str(e)})
    (AUDIT/"download_manifest.json").write_text(json.dumps({"files":results,"errors":errors},indent=2))
  if errors:raise RuntimeError(str(errors))
  print("ALL_BIOPROCESS_SOURCE_DOWNLOADS_COMPLETE",flush=True)
if __name__=="__main__":main()
