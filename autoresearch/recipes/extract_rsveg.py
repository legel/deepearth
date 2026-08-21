#!/usr/bin/env python3
"""RS vegetation channel: per-observation MODIS NDVI/EVI phenology (2020), keyed by gbifID.
Continuous, transferable habitat/productivity signal (NOT climate/soil/terrain). Planetary Computer STAC.
Output features per obs: ndvi_mean, ndvi_amp, ndvi_std, ndvi_peak_frac, ndvi_min, evi_mean, evi_amp.
"""
import numpy as np, pystac_client, planetary_computer as pc, rasterio
from rasterio.warp import transform as wtransform
from rasterio.transform import rowcol
import warnings; warnings.filterwarnings("ignore")

c = np.load("/tmp/obs_coords.npz")
gid, lat, lon = c["gbifID"], c["lat"].astype(float), c["lon"].astype(float)
N = len(gid); print(f"{N} observations", flush=True)

cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
# all 2020 CA 13Q1 items, dedup by datetime (each date -> the CA tiles h08v04/h08v05)
items = list(cat.search(collections=["modis-13Q1-061"], bbox=[-124.5, 32.0, -114.0, 42.1],
                        datetime="2020-01-01/2020-12-31").items())
dates = sorted({(it.properties.get("datetime") or it.properties.get("start_datetime"))[:10] for it in items})
print(f"{len(items)} items over {len(dates)} dates", flush=True)

# pre-reproject points to MODIS sinusoidal ONCE (same CRS across tiles)
with rasterio.open(items[0].assets["250m_16_days_NDVI"].href) as ds0:
    modis_crs = ds0.crs
xs, ys = wtransform("EPSG:4326", modis_crs, lon, lat); xs = np.array(xs); ys = np.array(ys)

def sample_band(it, band):
    href = it.assets[band].href
    with rasterio.open(href) as ds:
        arr = ds.read(1)
        r, cc = rowcol(ds.transform, xs, ys); r = np.asarray(r); cc = np.asarray(cc)
        ok = (r >= 0) & (r < ds.height) & (cc >= 0) & (cc < ds.width)
        out = np.full(N, np.nan)
        vv = arr[r[ok], cc[ok]].astype(float)
        nod = ds.nodata
        if nod is not None: vv[vv == nod] = np.nan
        vv[vv <= -3000] = np.nan            # MODIS 13Q1 fill/water flag
        out[ok] = vv * 1e-4                  # scale to [-0.2,1]
        return out

ndvi = np.full((N, len(dates)), np.nan); evi = np.full((N, len(dates)), np.nan)
by_date = {}
for it in items: by_date.setdefault((it.properties.get("datetime") or it.properties.get("start_datetime"))[:10], []).append(it)
for di, d in enumerate(dates):
    for it in by_date[d]:
        for band, series in (("250m_16_days_NDVI", ndvi), ("250m_16_days_EVI", evi)):
            v = sample_band(it, band); m = np.isfinite(v); series[m, di] = v[m]
    print(f"  {d}: ndvi coverage {np.isfinite(ndvi[:,di]).mean():.2f}", flush=True)

def stats(a):
    mean = np.nanmean(a, axis=1); mn = np.nanmin(a, axis=1); mx = np.nanmax(a, axis=1)
    std = np.nanstd(a, axis=1)
    peak = np.full(N, np.nan); valid = np.isfinite(a).sum(1) > 0
    peak[valid] = np.nanargmax(np.where(np.isfinite(a[valid]), a[valid], -9), axis=1) / max(1, a.shape[1] - 1)
    return mean, mx - mn, std, peak, mn
nm, na, ns, npk, nmin = stats(ndvi); em, ea, *_ = stats(evi)
feat = np.stack([nm, na, ns, npk, nmin, em, ea], axis=1).astype(np.float32)  # [N,7]
# impute missing with train-median (non-spatial), so absence is not a location cue
med = np.nanmedian(feat, axis=0)
miss = ~np.isfinite(feat); feat[miss] = np.take(med, np.where(miss)[1])
cov = np.isfinite(ndvi).any(1).mean()
print(f"coverage {cov:.3f}; feat shape {feat.shape}; medians {np.round(med,3)}", flush=True)
np.savez_compressed("/workspace/deepearth/data/deepcal/gbif_rsveg_tokens.npz",
                    gbifID=gid, rsveg=feat,
                    feat_names=np.array(["ndvi_mean","ndvi_amp","ndvi_std","ndvi_peak","ndvi_min","evi_mean","evi_amp"]))
print("wrote gbif_rsveg_tokens.npz", flush=True)
