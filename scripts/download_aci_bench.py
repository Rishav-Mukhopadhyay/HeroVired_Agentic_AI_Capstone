"""Download the ACI-Bench dialogues, reference notes and metadata (about 2 MB).

Run:  python scripts/download_aci_bench.py
"""

import requests

from mediscribe.data_sources import ACI_BASE_URL, ACI_SPLITS, load_aci, raw_dir

out = raw_dir()
for name in ACI_SPLITS.values():
    for suffix in ("", "_metadata"):
        fname = f"{name}{suffix}.csv"
        r = requests.get(f"{ACI_BASE_URL}/{fname}", timeout=60)
        r.raise_for_status()
        (out / fname).write_bytes(r.content)
        print(f"downloaded {fname:45s} {len(r.content) / 1024:7.1f} KB")

for split in ACI_SPLITS:
    df = load_aci(split)
    print(f"{split:6s}: {len(df):3d} encounters, e.g. {df.encounter_id.iloc[0]} "
          f"({df.cc.iloc[0]})")
