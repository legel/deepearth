import numpy as np, glob, csv, time
import torch, torch.nn as nn, torch.nn.functional as F
c="data/deepcal"; t0=time.time()
vocab=np.load(c+"/gbif_vocab.npz",allow_pickle=True); gidx=vocab["global_idx"]
rows=list(csv.DictReader(open(c+"/derived/species_index.csv")))
fam_id=np.unique(np.array([rows[i]["family"] for i in gidx]),return_inverse=True)[1]
lat=[];lon=[];sp=[];gid=[]
for f in sorted(glob.glob(c+"/gbif_tokens/*.npz"))[:16]:
    z=np.load(f); lat.append(z["lat"]);lon.append(z["lon"]);sp.append(z["species_local"]);gid.append(z["gbifID"])
lat=np.concatenate(lat);lon=np.concatenate(lon);sp=np.concatenate(sp);gid=np.concatenate(gid)
fam=fam_id[sp]; n_fam=int(fam_id.max())+1
z=np.load(c+"/gbif_alphaearth_tokens.npz"); m={int(g):i for i,g in enumerate(z["gbifID"])}; AEall=z["ae"]
idx=np.array([m.get(int(g),-1) for g in gid]); keep=idx>=0
lat,lon,fam,idx=lat[keep],lon[keep],fam[keep],idx[keep]; AE=AEall[idx].astype(np.float32)
blk=(np.floor(lat/0.5).astype(np.int64)*100000+np.floor(lon/0.5).astype(np.int64))
ub=np.unique(blk); rng=np.random.default_rng(0); rng.shuffle(ub); held=set(ub[:int(len(ub)*0.2)].tolist())
test=np.array([b in held for b in blk])
rn=np.stack([lat/90.,lon/180.],1).astype(np.float32)
pr=rn@(np.random.default_rng(0).normal(0,8.0,(2,32)).astype(np.float32)); RFF=np.concatenate([np.sin(pr),np.cos(pr)],1).astype(np.float32)
def ev(X):
    d="cuda"; X=torch.tensor(X); y=torch.tensor(fam); tr=~torch.tensor(test); te=torch.tensor(test)
    Xtr,ytr,Xte,yte=X[tr].to(d),y[tr].to(d),X[te].to(d),y[te].to(d)
    h=nn.Linear(X.shape[1],n_fam).to(d); o=torch.optim.Adam(h.parameters(),1e-2)
    for _ in range(4000):
        i=torch.randint(0,Xtr.shape[0],(4096,),device=d); l=F.cross_entropy(h(Xtr[i]),ytr[i]); o.zero_grad();l.backward();o.step()
    with torch.no_grad():
        lo=h(Xte); return (lo.argmax(-1)==yte).float().mean().item(), (lo.topk(5,-1).indices==yte[:,None]).any(-1).float().mean().item()
print(f"N={len(lat)} held-out={int(test.sum())} families={n_fam}")
for name,X in [("raw-coords",rn),("RFF",RFF),("AlphaEarth",AE),("raw+AlphaEarth",np.concatenate([rn,AE],1).astype(np.float32))]:
    a,t=ev(X); print(f"  {name:16s} held-out-block family  top1 {a:.4f}  top5 {t:.4f}")
print(f"done {time.time()-t0:.0f}s")
