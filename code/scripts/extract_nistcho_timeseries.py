from pathlib import Path
import sys,json,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
# Importing the range reader also verifies and records the archive directory.
from inspect_nistcho_remote_zip import RangeFile,OUT
selected=[]
with zipfile.ZipFile(RangeFile()) as z:
 for i in z.infolist():
  path=Path(i.filename)
  if path.parent.as_posix()!="data" or path.suffix.lower() not in (".csv",".xlsx",".txt",".md"):continue
  assert not path.is_absolute() and ".." not in path.parts
  dest=OUT/"timeseries"/path.name
  dest.parent.mkdir(exist_ok=True)
  b=z.read(i) # zipfile validates the archive CRC32
  if dest.exists():assert dest.read_bytes()==b
  else:dest.write_bytes(b)
  selected.append({"member":i.filename,"path":str(dest.relative_to(ROOT)),"bytes":len(b),"crc32":format(i.CRC,"08x"),"sha256":hashlib.sha256(b).hexdigest()})
  print("EXTRACTED",i.filename,len(b),flush=True)
(OUT/"timeseries_download_manifest.json").write_text(json.dumps({"scope":"All top-level original and processed cultivation time-series tables; mass-spectrometry binaries excluded","source_record":"10.5281/zenodo.18609042","full_archive_bytes":7330460280,"full_archive_downloaded":False,"verification":"HTTP byte ranges, ZIP per-entry CRC32 and local SHA256","files":selected},indent=2))
print("NISTCHO_CULTIVATION_TABLES_COMPLETE",len(selected),flush=True)
