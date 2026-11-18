from acquire_bioprocess_pretraining_sources import download
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
jobs=[['ecoli', 'DDB_PD_117_AMBR_online_stacked_9s.csv', 'https://ndownloader.figshare.com/files/44048477', 156587348, 'md5:ce9bfed813ec9069117c49b375792593'], ['ecoli', 'DDB_PD_121_AMBR_online_stacked_9s.csv', 'https://ndownloader.figshare.com/files/44048480', 151500198, 'md5:29d05042350babdcc59806ca55737d85'], ['ecoli', 'DDB_PD_122_AMBR_online_stacked_9s.csv', 'https://ndownloader.figshare.com/files/44048483', 146222422, 'md5:b1106d92b186beb274939c09c7db71ed'], ['ecoli', 'DDB_PD_123_AMBR_online_stacked_9s.csv', 'https://ndownloader.figshare.com/files/44048486', 174505765, 'md5:0ca545b453400a64a271eaf7176ed746'], ['ecoli', 'README.md', 'https://raw.githubusercontent.com/jonruh/Effects-of-glucose-oscillations-on-E.-coli/main/README.md', None, None]]
results=[]
for j in jobs:
 results.append(download(j))
 (root/"test_results/bioprocess_pretraining_inventory/ecoli_download_manifest.json").write_text(json.dumps(results,indent=2))
print("ECOLI_SOURCE_DOWNLOAD_COMPLETE",flush=True)
