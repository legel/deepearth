"""NEON AOP canopy height model (DP3.30015.001, 1 m, 1 km UTM tiles) around every NEON soil plot in the benchmark,
latest flight per site. Writes canopy-structure covariates per plot to $DEEPEARTH_SOIL_DATA/neon/chm_features.parquet:
  chm_mean_{10,30,100}  mean canopy height (m) within that radius
  chm_cover_{10,30,100} fraction of 1 m cells taller than 2 m
  chm_p95_30            95th percentile height within 30 m
Vegetation is a soil-forming factor in its own right (litter, roots, shade-moderated moisture), and the DeepEarth site
store carries the same LiDAR structure everywhere, so this measures what LiDAR could add to the soil model."""
import io, json, math, pathlib, time, urllib.request
import numpy as np, pandas as pd, rasterio

from .config import DATA
from pyproj import Transformer

W = DATA
API = "https://data.neonscience.org/api/v0"
TOK = pathlib.Path.home() / ".config/neon/token"
CACHE = W / "neon/chm_tiles"


def get(url, raw=False):
    req = urllib.request.Request(url, headers={"X-API-Token": TOK.read_text().strip()} if TOK.exists() else {})
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                b = r.read()
                return b if raw else json.loads(b)
        except Exception:
            time.sleep(3 + 5 * k)
    raise RuntimeError(url)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    pts = pd.read_parquet(W / "benchmark/points_neon.parquet")[["id", "site", "lat", "lon"]]
    prod = get(f"{API}/products/DP3.30015.001")["data"]
    months = {s["siteCode"]: sorted(s["availableMonths"]) for s in prod["siteCodes"]}
    rows = []
    for site, g in pts.groupby("site"):
        if site not in months:
            continue
        zone = int((g["lon"].mean() + 180) // 6) + 1
        tf = Transformer.from_crs(4326, 32600 + zone, always_xy=True)
        e, n = tf.transform(g["lon"].to_numpy(), g["lat"].to_numpy())
        need = {(int(x // 1000) * 1000, int(y // 1000) * 1000) for x, y in zip(e, n)}
        files = None
        for m in reversed(months[site]):                  # latest flight that covers the plots
            fl = get(f"{API}/data/DP3.30015.001/{site}/{m}")["data"]["files"]
            fl = {f["name"]: f["url"] for f in fl if f["name"].endswith("_CHM.tif")}
            have = {(int(k.split("_")[4]), int(k.split("_")[5])) for k in fl}
            if need & have:
                files = fl; break
        if not files:
            continue
        tiles = {}
        for name, url in files.items():
            key = (int(name.split("_")[4]), int(name.split("_")[5]))
            if key in need:
                p = CACHE / name
                if not p.exists():
                    p.write_bytes(get(url, raw=True))
                tiles[key] = p
        for pid, x, y in zip(g["id"], e, n):
            t = tiles.get((int(x // 1000) * 1000, int(y // 1000) * 1000))
            if not t:
                continue
            with rasterio.open(t) as s:
                r, c = s.index(x, y)
                win = s.read(1, window=((r - 100, r + 101), (c - 100, c + 101)), boundless=True, fill_value=-9999).astype(float)
            win[win < -1] = np.nan
            yy, xx = np.mgrid[-100:101, -100:101]; rad = np.hypot(yy, xx)
            f = {"id": pid}
            for R in (10, 30, 100):
                v = win[rad <= R]
                f[f"chm_mean_{R}"] = np.nanmean(v); f[f"chm_cover_{R}"] = np.nanmean(v > 2)
            f["chm_p95_30"] = np.nanpercentile(win[rad <= 30], 95)
            rows.append(f)
        print(site, len(tiles), "tiles", flush=True)
    pd.DataFrame(rows).to_parquet(W / "neon/chm_features.parquet", index=False)
    print(len(rows), "plots with canopy structure")


if __name__ == "__main__":
    main()
