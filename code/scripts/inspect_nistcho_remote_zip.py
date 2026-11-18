"""Read the public NISTCHO ZIP directory using HTTP ranges; no training."""
from pathlib import Path
import io,zipfile,requests,json,csv,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"dataset/bioprocess_pretraining_sources/nistcho"
URL="https://zenodo.org/api/records/18609042/files/feeding_strategies_zenodo.zip/content"
SIZE=7330460280
class RangeFile(io.RawIOBase):
 def __init__(self):self.pos=0;self.cache={}
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else SIZE+offset
  return self.pos
 def read(self,n=-1):
  if n<0:n=SIZE-self.pos
  if n==0:return b""
  lo=self.pos;hi=min(SIZE,lo+n)-1;key=(lo,hi)
  if key not in self.cache:
   if hi-lo>32*1024**2:raise RuntimeError("Unexpected large directory read")
   for attempt in range(4):
    try:
     r=requests.get(URL,headers={"User-Agent":"bioprocess-data-audit/1.0","Range":"bytes=%d-%d"%(lo,hi)},timeout=(20,60))
     r.raise_for_status()
     if r.status_code!=206 or len(r.content)!=hi-lo+1:raise RuntimeError("Server ignored byte range")
     self.cache[key]=r.content;break
    except Exception:
     if attempt==3:raise
     time.sleep(2)
  b=self.cache[key];self.pos+=len(b);return b
with zipfile.ZipFile(RangeFile()) as z:
 rows=[{"name":i.filename,"bytes":i.file_size,"compressed_bytes":i.compress_size,"crc32":format(i.CRC,"08x")} for i in z.infolist()]
 with (OUT/"archive_members.csv").open("w") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print("ARCHIVE_MEMBERS",len(rows),flush=True)
 for r in rows:
  if not r["name"].endswith("/"):print(r,flush=True)
