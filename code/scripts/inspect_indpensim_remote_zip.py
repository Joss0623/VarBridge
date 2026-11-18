from pathlib import Path
import ast,requests,time,io,zipfile,csv,json
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/"dataset/bioprocess_pretraining_sources/indpensim"
s=(ROOT/"scripts/inspect_nistcho_remote_zip.py").read_text()
node=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name=="RangeFile")
src=ast.get_source_segment(s,node)
ns=dict(io=io,requests=requests,time=time,URL="https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/e349116a-7ee5-4daf-ba5e-162ceeebff6b",SIZE=503792426)
exec(src,ns)
with zipfile.ZipFile(ns["RangeFile"]()) as z:
 rows=[{"name":i.filename,"bytes":i.file_size,"compressed_bytes":i.compress_size,"crc32":format(i.CRC,"08x")} for i in z.infolist()]
 (out/"archive_members.json").write_text(json.dumps(rows,indent=2))
 print("INDPENSIM_MEMBERS",rows,flush=True)
 for i in z.infolist():
  if i.filename.lower().endswith("statistics.csv"):
   b=z.read(i);p=out/Path(i.filename).name
   if p.exists():assert p.read_bytes()==b
   else:p.write_bytes(b)
   print("STATISTICS_EXTRACTED",len(b),flush=True)
  if i.filename.lower().endswith("v3.csv"):
   with z.open(i) as f:b=f.readline()
   (out/"original_header.txt").write_bytes(b)
   print("HEADER",b.decode(errors="replace")[:6500],flush=True)
print("INDPENSIM_REMOTE_DIRECTORY_AUDIT_COMPLETE",flush=True)
