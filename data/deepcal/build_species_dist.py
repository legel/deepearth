"""One-off: local species-abundance distributions around every observation at 3 scales (3km/300m/30m).
Ground truth for the distribution-prediction benchmarks (KL vs held-out). Cache -> gbif_species_dist.npz."""
import numpy as np, glob
from collections import defaultdict
gid=[]; sp=[]; lat=[]; lon=[]
for f in sorted(glob.glob("gbif_tokens/*.npz")):
    z=np.load(f); gid.append(z["gbifID"]); sp.append(z["species_local"]); lat.append(z["lat"]); lon.append(z["lon"])
gid=np.concatenate(gid); sp=np.concatenate(sp).astype(np.int32); lat=np.concatenate(lat); lon=np.concatenate(lon)
N=len(gid); print(f"{N} obs, {len(np.unique(sp))} species")
SCALES={"3km":0.027, "300m":0.0027, "30m":0.00027}   # deg ~ meters at CA latitude
TOPK=30
out={"gbifID":gid}
for name,d in SCALES.items():
    cell=(np.floor(lat/d).astype(np.int64)*4000000 + np.floor(lon/d).astype(np.int64))
    # per-cell species multiset
    cell_sp=defaultdict(lambda: defaultdict(int))
    for i in range(N): cell_sp[cell[i]][int(sp[i])]+=1
    # per-obs: top-K species idx + normalized freq of its cell (the local distribution to predict)
    idx=np.full((N,TOPK), -1, np.int32); frq=np.zeros((N,TOPK), np.float32); ncell=np.zeros(N,np.int32)
    for i in range(N):
        dd=cell_sp[cell[i]]; items=sorted(dd.items(), key=lambda kv:-kv[1])[:TOPK]
        tot=sum(dd.values()); ncell[i]=len(dd)
        for j,(s,c) in enumerate(items): idx[i,j]=s; frq[i,j]=c/tot
    out[f"idx_{name}"]=idx; out[f"frq_{name}"]=frq; out[f"nsp_{name}"]=ncell
    rich=ncell[ncell>1]
    print(f"  {name}: cells={len(cell_sp)} | obs-in-multispecies-cells={int((ncell>1).sum())} | median richness(>1)={int(np.median(rich)) if len(rich) else 0} | max={ncell.max()}")
np.savez_compressed("gbif_species_dist.npz", **out)
print("wrote gbif_species_dist.npz")
