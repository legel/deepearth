"""Per-observation hydrology (TWI, HAND, catchment) + wind exposure (Winstral Sx) from USGS 3DEP 1m DEM.
Microclimate niche channels (drainage + wind), physics-validated pure-numpy. Reuses build_topo's DEM fetch/cell
structure. Output: gbif_hydro_tokens.npz {gbifID, hydro[N,6]=[twi,hand,ln_sca,sx_w,sx_mean,tpi], has_hydro}. Resumable."""
import os, time, pickle, warnings, math
import numpy as np
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from rasterio.io import MemoryFile
from scipy import ndimage
from scipy.ndimage import map_coordinates
from skimage.morphology import reconstruction
from pyproj import Transformer
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
CELL = 0.002; PATCH_HALF = 256; WORKERS = 12
CKPT = os.path.join(HERE, "hydrowind_ckpt.pkl")
NBR = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
_tf = {}
def utm_epsg(lon): return 26910 if lon < -120.0 else 26911
def transformer(e):
    if e not in _tf: _tf[e] = Transformer.from_crs(4326, e, always_xy=True)
    return _tf[e]

def fetch_dem(xmin, ymin, xmax, ymax, epsg, n, retries=5):
    url = (f"{BASE}?bbox={xmin},{ymin},{xmax},{ymax}&bboxSR={epsg}&size={n},{n}&imageSR={epsg}"
           f"&format=tiff&pixelType=F32&noData=-9999&adjustAspectRatio=false&interpolation=RSP_BilinearInterpolation&f=image")
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r: data = r.read()
            with MemoryFile(data) as mf, mf.open() as ds: return ds.read(1).astype(np.float64)
        except Exception:
            if k == retries-1: return None
            time.sleep(2.0*(k+1))
    return None

def hydrowind_rasters(dem, res):
    ny, nx = dem.shape
    seed = dem.max()*np.ones_like(dem)
    seed[0,:]=dem[0,:]; seed[-1,:]=dem[-1,:]; seed[:,0]=dem[:,0]; seed[:,-1]=dem[:,-1]
    zf = reconstruction(seed, dem, method="erosion")
    dist = np.array([math.sqrt(2),1,math.sqrt(2),1,1,math.sqrt(2),1,math.sqrt(2)])*res
    r,c = np.indices(zf.shape); best = np.full(zf.shape,-np.inf); rr_=r.copy(); rc_=c.copy()
    for k,(dr,dc) in enumerate(NBR):
        rr=np.clip(r+dr,0,ny-1); cc=np.clip(c+dc,0,nx-1)
        slope=(zf-zf[rr,cc])/dist[k]
        slope=np.where((r+dr<0)|(r+dr>=ny)|(c+dc<0)|(c+dc>=nx),-np.inf,slope)
        upd=slope>best; best=np.where(upd,slope,best); rr_=np.where(upd,rr,rr_); rc_=np.where(upd,cc,rc_)
    recflat=(rr_*nx+rc_).ravel(); order=np.argsort(zf.ravel())[::-1]
    acc=np.ones(zf.size)
    for idx in order:
        rec=recflat[idx]
        if rec!=idx: acc[rec]+=acc[idx]
    acc=acc.reshape(zf.shape)
    dzdy,dzdx=np.gradient(zf,res); tanb=np.maximum(np.tan(np.arctan(np.hypot(dzdx,dzdy))),0.001)
    sca=acc*res; twi=np.log(sca/tanb)
    stream=(acc>2000).ravel(); zflat=zf.ravel(); se=zflat.copy()
    for idx in np.argsort(zflat):
        if not stream[idx]:
            rec=recflat[idx]
            if rec!=idx: se[idx]=se[rec]
    hand=np.clip(zf-se.reshape(zf.shape),0,None)
    def sx1(z, wf, dmax=100.0, step=2.0):
        rows,cols=np.mgrid[0:ny,0:nx].astype(np.float32)
        az=math.radians(wf); drow=-math.cos(az)/res; dcol=math.sin(az)/res; sx=np.full(z.shape,-np.inf,np.float32)
        for k in range(1,int(dmax/step)+1):
            d=k*step; zv=map_coordinates(z,[rows+drow*d,cols+dcol*d],order=1,mode="nearest",prefilter=False)
            np.maximum(sx,np.degrees(np.arctan((zv-z)/d)),out=sx)
        return sx
    sx_w=np.mean([sx1(dem,270.0+30.0*(k/4-0.5)) for k in range(5)],0)                # prevailing W sector
    sx_mean=np.mean([sx1(dem,a) for a in (0,90,180,270)],0)                          # omnidirectional
    tpi=zf-ndimage.uniform_filter(zf,101,mode="nearest")
    return dict(twi=twi, hand=hand, ln_sca=np.log(sca), sx_w=sx_w, sx_mean=sx_mean, tpi=tpi)

def process_cell(cell, obs):
    clat, clon = cell
    latc, lonc = (clat+0.5)*CELL, (clon+0.5)*CELL
    epsg=utm_epsg(lonc); tf=transformer(epsg); cx,cy=tf.transform(lonc,latc)
    if not (np.isfinite(cx) and np.isfinite(cy)): return None
    n=2*PATCH_HALF; xmin,xmax=cx-PATCH_HALF,cx+PATCH_HALF; ymin,ymax=cy-PATCH_HALF,cy+PATCH_HALF
    dem=fetch_dem(xmin,ymin,xmax,ymax,epsg,n)
    if dem is None or dem.shape!=(n,n): return None
    dem=np.where(np.isfinite(dem)&(dem>-1e4),dem,np.nan)
    if np.isnan(dem).mean()>0.5: return None
    dem=np.nan_to_num(dem,nan=np.nanmean(dem))
    R=hydrowind_rasters(dem,1.0); out={}
    for gid,la,lo in obs:
        x,y=tf.transform(lo,la); col=(x-xmin)/((xmax-xmin)/n); row=(ymax-y)/((ymax-ymin)/n)
        r=int(np.clip(round(row),0,n-1)); c=int(np.clip(round(col),0,n-1))
        out[gid]=np.array([R["twi"][r,c],R["hand"][r,c],R["ln_sca"][r,c],R["sx_w"][r,c],R["sx_mean"][r,c],R["tpi"][r,c]],np.float32)
    return out

def main():
    z=np.load(os.path.join(HERE,"obs_coords.npz")); gid,lat,lon=z["gbifID"],z["lat"],z["lon"]
    cells={}
    for i in range(len(gid)):
        cells.setdefault((int(np.floor(lat[i]/CELL)),int(np.floor(lon[i]/CELL))),[]).append((int(gid[i]),float(lat[i]),float(lon[i])))
    done=pickle.load(open(CKPT,"rb")) if os.path.exists(CKPT) else {}
    todo=[c for c in cells if not all(o[0] in done for o in cells[c])]
    print(f"{len(cells)} cells, {len(todo)} to fetch, {len(gid)} obs",flush=True)
    t0=time.time(); nok=0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs={ex.submit(process_cell,c,cells[c]):c for c in todo}
        for i,fut in enumerate(as_completed(futs)):
            r=fut.result()
            if r: done.update(r); nok+=1
            if (i+1)%200==0:
                pickle.dump(done,open(CKPT,"wb")); el=time.time()-t0
                print(f"  {i+1}/{len(todo)} cells | {len(done)} obs | {el:.0f}s | eta {el/(i+1)*(len(todo)-i-1)/60:.1f}min",flush=True)
    pickle.dump(done,open(CKPT,"wb"))
    hy=np.zeros((len(gid),6),np.float32); have=np.zeros(len(gid),bool)
    for i in range(len(gid)):
        g=int(gid[i])
        if g in done: hy[i]=done[g]; have[i]=True
    np.savez(os.path.join(HERE,"..","gbif_hydro_tokens.npz"),gbifID=gid,hydro=hy,has_hydro=have)
    print(f"DONE: {have.sum()}/{len(gid)} obs have hydro/wind ({100*have.mean():.1f}%)",flush=True)

if __name__=="__main__": main()
