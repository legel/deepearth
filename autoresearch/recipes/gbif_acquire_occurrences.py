#!/usr/bin/env python3
"""Scaled occurrence acquisition: CA vascular plants 2019-2024, partitioned year x month (each <100k),
concurrent, keep vocab species only -> gbif_densify_bulk.npz (gbifID, species_local, lat, lon)."""
import urllib.request, json, numpy as np, time
from concurrent.futures import ThreadPoolExecutor, as_completed
vocab=np.load("/workspace/deepearth/data/deepcal/gbif_vocab.npz",allow_pickle=True)
def norm(s):
    p=str(s).split(); return (p[0]+" "+p[1]).lower() if len(p)>=2 else str(s).lower()
V={norm(b):i for i,b in enumerate(vocab["binomial"])}
def url(y,m,off):
    return (f"https://api.gbif.org/v1/occurrence/search?country=US&stateProvince=California"
            f"&taxonKey=7707728&hasCoordinate=true&hasGeospatialIssue=false&year={y}&month={m}&offset={off}&limit=300")
def page(args):
    y,m,off=args
    for _ in range(3):
        try:
            d=json.load(urllib.request.urlopen(url(y,m,off),timeout=45)); out=[]
            for r in d.get("results",[]):
                sp=r.get("species");la=r.get("decimalLatitude");lo=r.get("decimalLongitude");gk=r.get("gbifID")
                if not(sp and la is not None and lo is not None and gk):continue
                si=V.get(norm(sp))
                if si is None:continue
                out.append((int(gk),si,float(la),float(lo)))
            return out,d.get("count",0)
        except Exception: time.sleep(2)
    return [],0
# build page tasks: for each year-month, offsets 0..min(count,99900)
tasks=[]
for y in range(2019,2025):
    for m in range(1,13):
        _,cnt=page((y,m,0)); n=min(cnt,99900)
        tasks+=[(y,m,o) for o in range(0,n+1,300)]
print(f"{len(tasks)} page-tasks across 2019-2024 year-months",flush=True)
rows=[]; t0=time.time(); done=0
with ThreadPoolExecutor(max_workers=32) as ex:
    for f in as_completed([ex.submit(page,t) for t in tasks]):
        r,_=f.result(); rows+=r; done+=1
        if done%200==0: print(f"  {done}/{len(tasks)} pages, {len(rows)} obs, {time.time()-t0:.0f}s",flush=True)
seen=set(); uniq=[r for r in rows if not(r[0] in seen or seen.add(r[0]))]
a=np.array(uniq,dtype=np.float64)
print(f"DONE {len(a)} unique vocab-species CA occurrences ({time.time()-t0:.0f}s)",flush=True)
np.savez_compressed("/workspace/deepearth/data/deepcal/gbif_densify_bulk.npz",
    gbifID=a[:,0].astype(np.int64),species_local=a[:,1].astype(np.int32),
    lat=a[:,2].astype(np.float32),lon=a[:,3].astype(np.float32))
print("wrote gbif_densify_bulk.npz",flush=True)
