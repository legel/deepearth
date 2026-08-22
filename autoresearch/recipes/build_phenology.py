import numpy as np, glob, os, subprocess, netCDF4
cache="/workspace/deepearth/data/deepcal"
# obs coords
gid,lat,lon=[],[],[]
for f in sorted(glob.glob(f"{cache}/gbif_tokens/*.npz")):
    d=np.load(f); gid.append(d["gbifID"]); lat.append(d["lat"]); lon.append(d["lon"])
gid=np.concatenate(gid); lat=np.concatenate(lat).astype(np.float64); lon=np.concatenate(lon).astype(np.float64)
keys=[l.strip() for l in open("/tmp/pheno_keys.txt") if l.strip()]
base="https://noaa-cdr-ndvi-pds.s3.amazonaws.com/"
months=[]
for i,k in enumerate(keys):
    dst=f"/tmp/pheno_{i:02d}.nc"
    subprocess.run(["curl","-sL",base+k,"-o",dst],check=True)
    ds=netCDF4.Dataset(dst)
    var="NDVI" if "NDVI" in ds.variables else [v for v in ds.variables if "ndvi" in v.lower()][0]
    ndvi=ds.variables[var][:]           # [1,lat,lon] or [lat,lon]
    ndvi=np.squeeze(np.asarray(ndvi, dtype=np.float32))
    latg=np.asarray(ds.variables["latitude"][:]); long=np.asarray(ds.variables["longitude"][:])
    li=np.clip(np.searchsorted(-latg, -lat), 0, len(latg)-1) if latg[0]>latg[-1] else np.clip(np.searchsorted(latg,lat),0,len(latg)-1)
    lj=np.clip(np.searchsorted(long, lon), 0, len(long)-1)
    vals=ndvi[li, lj]
    months.append(np.where(np.abs(vals)>2, np.nan, vals).astype(np.float32))
    ds.close(); os.remove(dst)
    print(f"month {i+1}/12 sampled", flush=True)
P=np.stack(months,1)                    # [N,12] annual NDVI cycle
bad=~np.isfinite(P); has=~bad.any(1); P[bad]=0.0
np.savez(f"{cache}/gbif_phenology_tokens.npz", gbifID=gid, phenology=P, has_phenology=has)
print(f"phenology[{P.shape}] coverage {100*has.mean():.1f}% -> gbif_phenology_tokens.npz", flush=True)
