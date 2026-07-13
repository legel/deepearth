"""Fetch iNat photo URLs for obs missing from photo_manifest (the recovered out-of-tree species), so rembed can
embed their ground vision. obs occurrenceID = iNat observation URL -> obs id -> iNat API (200/req) -> photo URLs
-> append to photo_manifest.parquet. Resumable (skips obs already in the manifest)."""
import pandas as pd, requests, time, sys, csv
from pathlib import Path
GB=Path.home()/"deepearth/data/deepearth_gbif"; DER=Path.home()/"deepearth/data/cache/derived"
LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else 0
meta=pd.read_parquet(GB/"observations_meta.parquet").drop_duplicates("gbifID")
mani=pd.read_parquet(GB/"photo_manifest.parquet"); have=set(mani.gbifID.tolist())
b2g=set(r["binomial"] for r in csv.DictReader(open(DER/"species_index.csv")))
todo=meta[meta.species.isin(b2g) & ~meta.gbifID.isin(have) & meta.occurrenceID.notna()].copy()
todo["inat_id"]=todo.occurrenceID.str.extract(r"observations/(\d+)")[0]
todo=todo[todo.inat_id.notna()]
if LIMIT: todo=todo.head(LIMIT)
print(f"{len(todo)} obs need photos (out-of-tree)", flush=True)
sess=requests.Session()
def api(ids):
    for k in range(4):
        try:
            r=sess.get("https://api.inaturalist.org/v1/observations?per_page=200&id="+",".join(ids),timeout=30)
            if r.status_code==200:
                return {str(o["id"]):[p["url"].replace("/square.","/medium.") for p in o.get("photos",[]) if p.get("url")] for o in r.json().get("results",[])}
        except Exception: pass
        time.sleep(2*(k+1))
    return {}
id2r={str(r.inat_id):r for r in todo.itertuples()}; ids=list(id2r); rows=[]
for i in range(0,len(ids),200):
    for iid,purls in api(ids[i:i+200]).items():
        r=id2r[iid]
        for u in purls:
            rows.append(dict(gbifID=r.gbifID,photo_url=u,species=r.species,decimalLatitude=r.decimalLatitude,decimalLongitude=r.decimalLongitude,eventDate=str(r.eventDate),occurrenceID=r.occurrenceID))
    if i%4000==0: print(f"  {i}/{len(ids)} | {len(rows)} photo rows",flush=True)
    time.sleep(0.5)
if rows and not LIMIT:
    pd.concat([mani,pd.DataFrame(rows)],ignore_index=True).to_parquet(GB/"photo_manifest.parquet")
    print(f"extended photo_manifest: +{len(rows)} rows",flush=True)
else:
    print(f"TEST: got {len(rows)} photo rows for {len(ids)} obs; sample: {rows[0] if rows else None}",flush=True)
