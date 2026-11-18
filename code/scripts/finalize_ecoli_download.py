from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"dataset/bioprocess_pretraining_sources/ecoli"
d=json.loads((BASE/"repository_metadata.json").read_text())
manifest=[]
for f in d["files"]:
 dest=BASE/f["name"];part=dest.with_suffix(".assembled.part");chunks=BASE/(f["name"]+".chunks")
 md5=hashlib.md5();sha=hashlib.sha256()
 if not dest.exists():
  n=0
  with part.open("wb") as w:
   for i in range((f["size"]+1048575)//1048576):
    p=chunks/str(i).zfill(4)
    b=p.read_bytes();assert len(b)==min(1048576,f["size"]-i*1048576)
    w.write(b);md5.update(b);sha.update(b);n+=len(b)
  assert n==f["size"] and md5.hexdigest()==f["computed_md5"],f["name"]
  part.rename(dest)
 else:
  with dest.open("rb") as r:
   for b in iter(lambda:r.read(4*1024**2),b""):md5.update(b);sha.update(b)
  assert md5.hexdigest()==f["computed_md5"]
 manifest.append({"path":str(dest.relative_to(ROOT)),"bytes":f["size"],"md5":f["computed_md5"],"sha256":sha.hexdigest(),"source":"10.6084/m9.figshare.24999488.v1","route":"HTTPS range transport, memory-only relay, no local data files"})
 if chunks.exists():shutil.rmtree(chunks)
 old=BASE/(f["name"]+".relay.part")
 if old.exists():old.unlink()
 print("VERIFIED",f["name"],f["size"],flush=True)
(ROOT/"test_results/bioprocess_pretraining_inventory/ecoli_download_manifest.json").write_text(json.dumps(manifest,indent=2))
print("ECOLI_ALL_ORIGINAL_FILES_VERIFIED",flush=True)
