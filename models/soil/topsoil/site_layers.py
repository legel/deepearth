"""Surface-soil (0-30 cm) property grids for a site, at the 3DEP 1/3 arc-second (~10 m) DEM grid, from the recipe the
NEON benchmark supports (soils/benchmark.py, independent NEON plots, leave-one-site-out):

  pH, clay %, sand %   SOLUS100 (USDA-NRCS, 100 m), bilinear to 10 m.  NEON: pH R2 0.81 nationally, the only product
                       with positive within-site skill (0.18); SSURGO -0.35 within site.
  organic C %          10 ** (mean of log10 SOLUS100 and log10 SoilGrids) + a ridge correction on 10 m terrain fitted on
                       site-centred NEON residuals.  NEON: log R2 0.59 national, 0.13 within site (best of any source).
                       Where SoilGrids is masked (urban), log10 SOLUS100 + the NEON-calibrated offset between the two.
  texture class        USDA triangle (texture.py) from sand and clay.

fit() trains the organic-C terrain correction and offset on work/soils/benchmark/points_neon.parquet and writes
work/soils/benchmark/site_model.json; layers(lon, lat, half_m) returns the grids for a square window."""
import json, math, pathlib, sys
import numpy as np, pandas as pd, rasterio
from rasterio.warp import transform, reproject, Resampling
from affine import Affine

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import products as P, terrain as T
from texture import texture_class

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from deepearth_site import config as _cfg  # noqa: E402
MODEL = P_OUT = _cfg.WORK / "soils" / "benchmark" / "site_model.json"
TERR = [t for t in T.NAMES if t != "elev"]


def fit():
    from sklearn.linear_model import RidgeCV
    d = pd.read_parquet(MODEL.parent / "points_neon.parquet")
    d = d[d["oc"].notna() & d["solus_oc"].notna()].dropna(subset=TERR)
    ls, lg = np.log10(np.maximum(d["solus_oc"], 0.05)), np.log10(np.maximum(d["soilgrids_oc"], 0.05))
    offset = float((lg - ls).dropna().mean())            # SoilGrids minus SOLUS, log10
    b = pd.concat([ls, lg], axis=1).mean(axis=1, skipna=True)
    y = np.log10(np.maximum(d["oc"], 0.05))
    r = y - b; rc = r - r.groupby(d["site"]).transform("mean")
    X = d[TERR]; Xc = X - X.groupby(d["site"]).transform("mean"); sd = Xc.std()
    g = RidgeCV(alphas=np.logspace(-1, 4, 20)).fit(Xc / sd, rc)
    m = {"oc_offset_log10_soilgrids_minus_solus": offset, "terrain": TERR, "sd": sd.to_dict(),
         "coef": dict(zip(TERR, g.coef_.tolist())), "alpha": float(g.alpha_), "n_plots": int(len(d)),
         "n_sites": int(d["site"].nunique())}
    MODEL.write_text(json.dumps(m, indent=1))
    return m


def _warp(path, scal, dst_shape, dst_transform, nodata):
    """Resample a product raster onto the DEM grid (EPSG:4269 ~ 4326 at this scale), bilinear."""
    out = np.full(dst_shape, np.nan, dtype="float64")
    with rasterio.Env(**P.ENV), rasterio.open(path) as s:
        reproject(rasterio.band(s, 1), out, src_nodata=s.nodata if s.nodata is not None else nodata,
                  dst_transform=dst_transform, dst_crs="EPSG:4326", dst_nodata=np.nan, resampling=Resampling.bilinear)
    return out / scal


def layers(lon, lat, half_m=300):
    """{"ph", "clay", "sand", "oc", "texture", "lon", "lat"} grids on the ~10 m DEM lattice around (lon, lat)."""
    m = json.loads(MODEL.read_text()) if MODEL.exists() else fit()
    dy = T.PIX_DEG * 111320.0; dx = dy * math.cos(math.radians(lat))
    hy, hx = int(math.ceil(half_m / dy)), int(math.ceil(half_m / dx))
    pad = 60                                                       # 600 m of context for TPI / flow routing
    z = T.read_dem(lon, lat, hx + pad, hy + pad)
    with rasterio.Env(**T.ENV), rasterio.open(T.tile_url(lon, lat)) as s:
        r0, c0 = s.index(lon, lat)
        x0, y0 = s.xy(r0 - hy, c0 - hx, offset="ul")
    shape = (2 * hy + 1, 2 * hx + 1)
    tr = Affine(T.PIX_DEG, 0, x0, 0, -T.PIX_DEG, y0)
    g = T.grid_covariates(z, lat)
    crop = (slice(pad, pad + shape[0]), slice(pad, pad + shape[1]))
    out = {}
    for prop, (name, scal) in P.SOLUS.items():
        layers_ = [_warp(f"/vsicurl/https://storage.googleapis.com/solus100pub/{name}_{k}_cm_p.tif", scal, shape, tr, 65535)
                   for k in (0, 5, 15, 30)]
        prof = np.stack([(layers_[i] + layers_[i + 1]) / 2 for i in range(3)], -1)       # LAYERS 0-5, 5-15, 15-30
        out[prop] = P.interval_mean(prof.reshape(-1, 3), 0, 30).reshape(shape)
    ls = np.log10(np.maximum(out["oc"], 0.05))
    try:
        sg = np.stack([_warp(f"/vsicurl/https://files.isric.org/soilgrids/latest/data/soc/soc_{a}-{b}cm_mean.vrt", 100,
                             shape, tr, -32768) for a, b in P.LAYERS], -1)
        lsg = np.log10(np.maximum(P.interval_mean(sg.reshape(-1, 3), 0, 30).reshape(shape), 0.05))
    except Exception:
        lsg = np.full(shape, np.nan)
    lsg = np.where(np.isnan(lsg), ls + m["oc_offset_log10_soilgrids_minus_solus"], lsg)
    base = (ls + lsg) / 2
    corr = np.zeros(shape)
    for t in m["terrain"]:
        v = g[t][crop]
        corr += m["coef"][t] * (v - np.nanmean(v)) / m["sd"][t]   # site-centred, as fitted
    out["oc"] = 10 ** (base + corr)
    out["texture"] = texture_class(out["sand"], out["clay"])
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    out["lon"], out["lat"] = x0 + (cols + 0.5) * T.PIX_DEG, y0 - (rows + 0.5) * T.PIX_DEG
    return out


if __name__ == "__main__":
    print(json.dumps(fit(), indent=1)[:900])
    L = layers(-122.0115, 36.9655, 220)
    for k in ("ph", "clay", "sand", "oc"):
        print(k, L[k].shape, np.nanmin(L[k]).round(2), np.nanmedian(L[k]).round(2), np.nanmax(L[k]).round(2))
    import collections; print(collections.Counter(L["texture"].ravel()).most_common(4))


DEPTHS_CM = (0, 5, 15, 30, 60, 100, 150)        # SOLUS100 point depths


def depth_layers(lon, lat, half_m=300):
    """SOLUS100 pH, clay %, sand %, organic C % and USDA texture class at each point depth 0-150 cm, on the same ~10 m
    lattice as layers(): arrays shaped (len(DEPTHS_CM), ny, nx). Below 30 cm SOLUS100 is used as published (the NEON
    calibration of organic C covers 0-30 cm only); the surface (0-30 cm) values the Simulator scores with remain layers()."""
    dy = T.PIX_DEG * 111320.0; dx = dy * math.cos(math.radians(lat))
    hy, hx = int(math.ceil(half_m / dy)), int(math.ceil(half_m / dx))
    with rasterio.Env(**T.ENV), rasterio.open(T.tile_url(lon, lat)) as s:
        r0, c0 = s.index(lon, lat)
        x0, y0 = s.xy(r0 - hy, c0 - hx, offset="ul")
    shape = (2 * hy + 1, 2 * hx + 1)
    tr = Affine(T.PIX_DEG, 0, x0, 0, -T.PIX_DEG, y0)
    out = {}
    for prop, (name, scal) in P.SOLUS.items():
        out[prop] = np.stack([_warp(f"/vsicurl/https://storage.googleapis.com/solus100pub/{name}_{k}_cm_p.tif", scal, shape,
                                    tr, 65535) for k in DEPTHS_CM])
    out["texture"] = np.stack([texture_class(out["sand"][i], out["clay"][i]) for i in range(len(DEPTHS_CM))])
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    out["lon"], out["lat"] = x0 + (cols + 0.5) * T.PIX_DEG, y0 - (rows + 0.5) * T.PIX_DEG
    out["depth_cm"] = np.array(DEPTHS_CM)
    return out
