"""Bounded range downloader for a public archive; no model or training code."""
from pathlib import Path
import requests,json,hashlib,concurrent.futures,time,shutil,os,signal
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"dataset/bioprocess_pretraining_sources/indpensim"
OUT=ROOT/"test_results/bioprocess_pretraining_inventory"
SIZE=503792426
URL="https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/e349116a-7ee5-4daf-ba5e-162ceeebff6b"
EXPECTED="bc434c6cf96444fee3c915d434ac84e2a9c6492be4daebcfa4aed9cdff181f00"
CHUNK=2*1024**2
# Stop only the exact previous acquisition worker owned by this user.
p=Path("/proc/603873/cmdline")
if p.exists() and p.stat().st_uid==os.getuid():
 c=p.read_bytes().split(b"\0")
 if c[:3]==[b"/home/home_new/qsmx/anaconda3/envs/AutoTimes/bin/python",b"-u",b"-"]:
  os.kill(603873,signal.SIGTERM)
dest=BASE/"100_Batches_IndPenSim.zip"
parts=BASE/"archive_parts";parts.mkdir(exist_ok=True)
def get(i):
 lo=i*CHUNK;hi=min(SIZE,lo+CHUNK)-1
 target=parts/("%04d"%i)
 if target.exists() and target.stat().st_size==hi-lo+1:return i
 for attempt in range(4):
  try:
   if shutil.disk_usage(BASE).free<20*1024**3:raise RuntimeError("DISK_RESERVE")
   temp=target.with_suffix(".part")
   offset=temp.stat().st_size if temp.exists() else 0
   if offset==hi-lo+1:
    temp.rename(target);return i
   r=requests.get(URL+"?part=%d"%i,headers={"Range":"bytes=%d-%d"%(lo+offset,hi)},stream=True,timeout=(20,35))
   r.raise_for_status()
   expected="bytes %d-%d/%d"%(lo+offset,hi,SIZE)
   assert r.status_code==206 and r.headers.get("Content-Range")==expected,(r.status_code,r.headers.get("Content-Range"))
   temp=target.with_suffix(".part");start=time.monotonic()
   with temp.open("ab" if offset else "wb") as f:
    for b in r.iter_content(65536):
     f.write(b)
     if time.monotonic()-start>180:raise RuntimeError("Slow chunk")
   r.close();assert temp.stat().st_size==hi-lo+1
   temp.rename(target);return i
  except Exception as e:
   if attempt==3:raise
   time.sleep(2)
n=(SIZE+CHUNK-1)//CHUNK
done=0
with concurrent.futures.ThreadPoolExecutor(32) as ex:
 fs=[ex.submit(get,i) for i in range(n)]
 for f in concurrent.futures.as_completed(fs):
  f.result();done+=1
  if done%8==0:print("INDPENSIM_PARTS",done,"/",n,flush=True)
partial=dest.with_suffix(".assembled.part");h=hashlib.sha256()
with partial.open("wb") as w:
 for i in range(n):
  with (parts/("%04d"%i)).open("rb") as f:
   for b in iter(lambda:f.read(1024**2),b""):w.write(b);h.update(b)
assert partial.stat().st_size==SIZE and h.hexdigest()==EXPECTED
partial.rename(dest)
for f in parts.iterdir():f.unlink()
parts.rmdir()
legacy=BASE/"100_Batches_IndPenSim.zip.part"
if legacy.exists():legacy.unlink()
(OUT/"indpensim_download_manifest.json").write_text(json.dumps({"path":str(dest.relative_to(ROOT)),"bytes":SIZE,"sha256":h.hexdigest(),"source":"10.17632/pdnjz7zz5x.2","verification":"Exact full-archive SHA256 from official Mendeley file metadata"},indent=2))
print("INDPENSIM_FULL_DOWNLOAD_COMPLETE",flush=True)
