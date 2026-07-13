"""Assign species_local to the pollinator obs from the staged 8111-taxon pollinator vocab (add_observation left them
-1 because it maps against the PLANT vocab). Species not in the vocab are written to pending_pollinator.json for
add_species (inductive placement on the pollinator dated tree). Idempotent; re-run after the ingest completes."""
import numpy as np, csv, json, glob
from pathlib import Path
D=Path("/home/legel/deepcal/data/deepcal")
pv={}
for r in csv.reader(open(D/"pollinator/pollinator_vocab.csv")):
    if r and r[0].isdigit(): pv[r[1].strip().lower()]=int(r[0])
shards=sorted(glob.glob(str(D/"gbif_pollinator_obs/*.npz")))
mapped=missing=0; missing_sp={}
for f in shards:
    z=dict(np.load(f, allow_pickle=True))
    sl=np.full(len(z["gbifID"]), -1, np.int32)
    for i,s in enumerate(z["species"]):
        k=str(s).strip().lower(); j=pv.get(k)
        if j is not None: sl[i]=j; mapped+=1
        else:
            missing+=1
            if k and k!="nan": missing_sp[k]=z
    z["species_local"]=sl
    np.savez(f, **z)                                      # overwrite shard with the pollinator species_local
print(f"mapped {mapped} obs to pollinator vocab | {missing} obs of {len(missing_sp)} species NOT in vocab")
json.dump(sorted(missing_sp), open(D/"derived/pending_pollinator_missing.json","w"))
print("wrote pending_pollinator_missing.json (species for add_species)")
