"""Terrain covariates from USGS 3DEP 1/3 arc-second (~10 m) elevation, read from the public cloud-optimized tiles
(prd-tnm.s3.amazonaws.com, one 1-degree tile per n{lat}w{lon}). One code path computes them on a grid
(grid_covariates); a point's covariates are the centre pixel of a 3 km window, so the model trained at pedons is applied
to a site's map with exactly the same arithmetic.

Covariates:
  elev               m
  slope              degrees (centred finite differences)
  northness/eastness cos/sin of aspect, weighted by sin(slope)
  tpi_{50,150,500}   elevation minus the mean within that radius (m): hilltop > 0, hollow < 0
  curv               Laplacian of the 1-pixel-smoothed elevation (1/m), concave < 0
  twi                ln(a / tan(slope)), a = D8 upslope area per unit contour width (window-limited)
  relief_500         max - min within 500 m
These are the classic soil-forming terrain drivers (Jenny's "relief" factor): water and fine material move downslope,
so hollows and footslopes are wetter, deeper and higher in organic carbon than shoulders and ridges."""
import math
import numpy as np, rasterio
from rasterio.windows import Window
from scipy import ndimage, signal

ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
           GDAL_HTTP_MAX_RETRY=4, GDAL_HTTP_RETRY_DELAY=2)
HALF = 150                     # pixels: 3 km window at ~10 m
PIX_DEG = 1 / 3 / 3600
NAMES = ("elev", "slope", "northness", "eastness", "tpi_50", "tpi_150", "tpi_500", "curv", "twi", "relief_500")


def tile_url(lon, lat):
    n, w = math.ceil(lat), math.ceil(-lon)
    return (f"/vsicurl/https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/"
            f"n{n:02d}w{w:03d}/USGS_13_n{n:02d}w{w:03d}.tif")


def read_dem(lon, lat, half_x, half_y):
    """DEM window centred on (lon, lat), (2*half_y+1, 2*half_x+1) pixels, NaN outside data."""
    with rasterio.Env(**ENV), rasterio.open(tile_url(lon, lat)) as s:
        r, c = s.index(lon, lat)
        nod = s.nodata if s.nodata is not None else -999999.0
        z = s.read(1, window=Window(c - half_x, r - half_y, 2 * half_x + 1, 2 * half_y + 1), boundless=True,
                   fill_value=nod).astype("float64")
    z[(z == nod) | (z < -1000)] = np.nan
    return z


def d8_area(z, cell):
    """D8 upslope contributing area (m2), cells processed from high to low."""
    h, w = z.shape
    zf = z.ravel()
    acc = np.full(z.size, cell * cell)
    pad = np.pad(z, 1, constant_values=np.inf)
    off = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    drops = np.stack([(z - pad[1 + a:1 + a + h, 1 + b:1 + b + w]) / math.hypot(a, b) for a, b in off])
    best = drops.argmax(0); down = drops.max(0) > 0
    rr, cc = np.divmod(np.arange(z.size), w)
    tgt = (rr + np.array([o[0] for o in off])[best.ravel()]) * w + cc + np.array([o[1] for o in off])[best.ravel()]
    tgt = np.where(down.ravel(), tgt, -1)
    for i in np.argsort(-zf, kind="stable"):
        if tgt[i] >= 0:
            acc[tgt[i]] += acc[i]
    return acc.reshape(h, w)


def _disc(radius_px_y, radius_px_x):
    ry, rx = int(math.ceil(radius_px_y)), int(math.ceil(radius_px_x))
    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
    k = ((yy / max(radius_px_y, 1e-9)) ** 2 + (xx / max(radius_px_x, 1e-9)) ** 2) <= 1
    return k / k.sum()


def grid_covariates(z, lat):
    """{name: array like z} for a DEM grid at 1/3 arc-second around latitude `lat`."""
    z = np.where(np.isnan(z), np.nanmean(z), z)
    dy = PIX_DEG * 111320.0
    dx = dy * math.cos(math.radians(lat))
    gy, gx = np.gradient(z, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.arctan2(-gx, gy)
    sw = np.sin(np.radians(slope))
    out = {"elev": z, "slope": slope, "northness": np.cos(aspect) * sw, "eastness": np.sin(aspect) * sw}
    for r in (50, 150, 500):
        k = _disc(r / dy, r / dx); py, px = k.shape[0] // 2, k.shape[1] // 2
        out[f"tpi_{r}"] = z - signal.fftconvolve(np.pad(z, ((py, py), (px, px)), mode="edge"), k, mode="valid")
    cell = (dx * dy) ** 0.5
    out["curv"] = ndimage.laplace(ndimage.gaussian_filter(z, 1.0)) / (cell * cell)
    a = d8_area(z, cell) / cell
    out["twi"] = np.log(a / np.tan(np.radians(np.maximum(slope, 0.1))))
    fp = _disc(500 / dy, 500 / dx) > 0
    out["relief_500"] = ndimage.maximum_filter(z, footprint=fp) - ndimage.minimum_filter(z, footprint=fp)
    return out


def covariates(lon, lat):
    """Point covariates at (lon, lat) from a 3 km 3DEP window (see point_covariates)."""
    return point_covariates(read_dem(lon, lat, HALF, HALF), lat)


def point_covariates(z, lat):
    """grid_covariates at the centre pixel of a (2*HALF+1)^2 window, computed for that pixel only: the same
    definitions as the grid, so a model fitted at points applies unchanged to a site's map (tests check equality)."""
    if np.isnan(z[HALF, HALF]):
        return {k: np.nan for k in NAMES}
    z = np.where(np.isnan(z), np.nanmean(z), z)
    dy = PIX_DEG * 111320.0
    dx = dy * math.cos(math.radians(lat))
    c = (slice(HALF - 1, HALF + 2), slice(HALF - 1, HALF + 2))
    gy, gx = np.gradient(z[c], dy, dx)
    slope = math.degrees(math.atan(math.hypot(gx[1, 1], gy[1, 1])))
    aspect = math.atan2(-gx[1, 1], gy[1, 1]); sw = math.sin(math.radians(slope))
    out = {"elev": float(z[HALF, HALF]), "slope": slope, "northness": math.cos(aspect) * sw,
           "eastness": math.sin(aspect) * sw}
    for r in (50, 150, 500):
        k = _disc(r / dy, r / dx); py, px = k.shape[0] // 2, k.shape[1] // 2
        out[f"tpi_{r}"] = float(z[HALF, HALF] - (z[HALF - py:HALF + py + 1, HALF - px:HALF + px + 1] * k).sum())
    cell = (dx * dy) ** 0.5
    sm = ndimage.gaussian_filter(z[HALF - 8:HALF + 9, HALF - 8:HALF + 9], 1.0)
    out["curv"] = float(ndimage.laplace(sm)[8, 8] / (cell * cell))
    fp = _disc(500 / dy, 500 / dx) > 0; py, px = fp.shape[0] // 2, fp.shape[1] // 2
    win = z[HALF - py:HALF + py + 1, HALF - px:HALF + px + 1][fp]
    out["relief_500"] = float(win.max() - win.min())
    sub = z[HALF - 50:HALF + 51, HALF - 50:HALF + 51]          # flow routing over the central 1 km
    a = d8_area(sub, cell)[50, 50] / cell
    out["twi"] = math.log(a / math.tan(math.radians(max(slope, 0.1))))
    return out


if __name__ == "__main__":
    import time
    t = time.time(); print(covariates(-122.0115, 36.9655), round(time.time() - t, 2), "s")
