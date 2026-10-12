"""Benchmark of gridded soil products against precise field observations.

Test sets
  neon   NEON DP1.10086.001 plot means of mineral-horizon cores (pH in water, organic C %), ~20 m coordinate
         uncertainty, measured by NEON's contract labs: independent of every product's training data.
  wosis  WoSIS 2023 snapshot, US profiles with layers overlapping 0-15 cm (pH H2O, clay, sand, organic C). Mostly NCSS
         pedons: SOLUS100 is trained on these (in-sample for SOLUS); POLARIS and SSURGO are not.

Two skills are reported per product and property:
  national   R2 / RMSE / bias / Spearman over all locations: does the product get the region right?
  (organic C is scored in log10(%): it is roughly log-normal, so a 1 % miss at 1 % matters as a 10 % miss at 10 %)
  within     the same on deviations from each NEON site's mean (sites span a few km): does it resolve the fine-scale
             variation a planting design depends on? This is the number a 1 m site model has to beat.

Output: $DEEPEARTH_SOIL_DATA/benchmark/{points_<set>.parquet, metrics_<set>.csv}."""
import argparse, concurrent.futures as cf
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from . import products as P
from . import terrain as T
from .config import DATA

W = DATA
OUT = W / "benchmark"


def neon_points():
    col = pd.read_parquet(W / "neon/collection.parquet")
    ph = pd.read_parquet(W / "neon/ph.parquet")
    ch = pd.read_parquet(W / "neon/chem.parquet")
    for d in (col, ph, ch):
        for c in d.columns:
            if c in ("decimalLatitude", "decimalLongitude", "sampleTopDepth", "sampleBottomDepth", "soilInWaterpH",
                     "organicCPercent", "coordinateUncertainty"):
                d[c] = pd.to_numeric(d[c], errors="coerce")
    col = col[col["horizon"] == "M"].drop_duplicates("sampleID")
    ch = ch[(ch["sampleType"] == "soil") & (ch.get("cnPercentQF", "OK").fillna("OK") == "OK")]
    ch = ch.assign(sampleID=ch["sampleID"].str.replace(r"-CN$", "", regex=True)).groupby("sampleID")["organicCPercent"].mean()
    ph = ph.groupby("sampleID")["soilInWaterpH"].mean()
    s = col.set_index("sampleID").join(ph.rename("ph")).join(ch.rename("oc"))
    s = s[s["ph"].notna() | s["oc"].notna()]
    g = s.groupby("plotID").agg(site=("siteID", "first"), lat=("decimalLatitude", "mean"), lon=("decimalLongitude", "mean"),
                                unc=("coordinateUncertainty", "max"), top=("sampleTopDepth", "median"),
                                bottom=("sampleBottomDepth", "median"), ph=("ph", "mean"), oc=("oc", "mean"),
                                n=("ph", "size"), nlcd=("nlcdClass", "first")).reset_index()
    g["clay"] = np.nan; g["sand"] = np.nan
    g["bottom"] = g["bottom"].clip(upper=30).fillna(15); g["top"] = g["top"].clip(lower=0).fillna(0)
    return g.rename(columns={"plotID": "id"})


def wosis_points(n=6000, seed=0):
    """US WoSIS profiles located to ~100 m: thickness-weighted mean of mineral layers over 0-30 cm for pH (water),
    clay %, sand %, organic C % (orgc g/kg / 10). One profile per 1 km cell (the first by id), then a random `n`."""
    f = W / "wosis/us_surface.parquet"
    if f.exists():
        return pd.read_parquet(f)
    src = W / "wosis/WoSIS_2023_December"
    prof = pd.read_csv(src / "wosis_202312_profiles.tsv", sep="\t", usecols=["profile_id", "dataset_code",
                       "positional_uncertainty", "country_name", "longitude", "latitude"])
    prof = prof[(prof["country_name"] == "United States of America") & (prof["positional_uncertainty"] == "Circa 100 m")]
    out = prof.set_index("profile_id")
    for prop, name, scale in (("ph", "phaq", 1), ("clay", "clay", 1), ("sand", "sand", 1), ("oc", "orgc", 0.1)):
        d = pd.read_csv(src / f"wosis_202312_{name}.tsv", sep="\t", usecols=["profile_id", "upper_depth", "lower_depth",
                        "organic_surface", "value_avg"])
        d = d[d["profile_id"].isin(out.index) & (d["organic_surface"] == "f") & (d["upper_depth"] < 30)]
        d["w"] = (d["lower_depth"].clip(upper=30) - d["upper_depth"].clip(lower=0)).clip(lower=0)
        d = d[d["w"] > 0]
        g = (d["value_avg"] * d["w"]).groupby(d["profile_id"]).sum() / d["w"].groupby(d["profile_id"]).sum()
        out[prop] = g * scale
    out = out.dropna(subset=["ph", "clay", "sand", "oc"], how="all").reset_index()
    out = out.rename(columns={"profile_id": "id", "longitude": "lon", "latitude": "lat", "dataset_code": "dataset"})
    cell = (np.floor(out["lat"] / 0.009).astype(int).astype(str) + "_" + np.floor(out["lon"] / 0.011).astype(int).astype(str))
    out = out.sort_values("id").groupby(cell.loc[out.sort_values("id").index]).head(1)
    out = out.sample(min(n, len(out)), random_state=seed)
    out["site"] = out["id"].astype(str); out["top"] = 0.0; out["bottom"] = 30.0
    out.to_parquet(f, index=False)
    return out


def sample_all(pts, workers=8, grids=("solus", "polaris", "soilgrids")):
    lon, lat = pts["lon"].to_numpy(), pts["lat"].to_numpy()
    top, bottom = pts["top"].to_numpy(), pts["bottom"].to_numpy()
    out = pts.copy()
    # gridded products: the 0-30 cm profile once per product and property, then each point's own sample interval
    for prop in P.PROPS:
        for name in grids:
            try:
                out[f"{name}_{prop}"] = P.interval_mean(P.profile(name, prop, lon, lat), top, bottom)
            except Exception as e:
                print("sample error", name, prop, e, flush=True)
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda i: P._sda_point(round(float(lon[i]), 6), round(float(lat[i]), 6),
                                                   int(round(top[i])), int(round(bottom[i]))), range(len(pts))))
    for prop in P.PROPS:
        out[f"ssurgo_{prop}"] = [np.nan if r is None else r[prop] for r in res]

    def terr(i):
        try:
            return T.covariates(float(lon[i]), float(lat[i]))
        except Exception:                      # outside 3DEP 1/3" coverage
            return {k: np.nan for k in T.NAMES}
    with cf.ThreadPoolExecutor(workers) as ex:
        tv = pd.DataFrame(list(ex.map(terr, range(len(pts)))), index=out.index)
    return pd.concat([out, tv], axis=1)


def metrics(df, products=("solus", "polaris", "soilgrids", "ssurgo")):
    rows = []
    for prop in P.PROPS:
        for prod in products:
            c = f"{prod}_{prop}"
            if c not in df:
                continue
            for scope in ("national", "within"):
                d = df[[prop, c, "site"]].dropna()
                if prop == "oc":
                    d = d.assign(**{prop: np.log10(np.maximum(d[prop], 0.05)), c: np.log10(np.maximum(d[c], 0.05))})
                if scope == "within":
                    d = d[d.groupby("site")[prop].transform("size") >= 5]
                    d = d.assign(**{prop: d[prop] - d.groupby("site")[prop].transform("mean"),
                                    c: d[c] - d.groupby("site")[c].transform("mean")})
                if len(d) < 20:
                    continue
                y, p = d[prop].to_numpy(), d[c].to_numpy()

                rows.append(dict(property=prop, product=prod, scope=scope, n=len(d),
                                 r2=1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
                                 rmse=float(np.sqrt(np.mean((y - p) ** 2))), bias=float(np.mean(p - y)),
                                 spearman=float(spearmanr(y, p).statistic)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("neon", "wosis"), default="neon")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--grids", default="solus,polaris,soilgrids")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pts = neon_points() if a.set == "neon" else wosis_points()
    if a.limit:
        pts = pts.sample(a.limit, random_state=0) if len(pts) > a.limit else pts
    print(len(pts), "locations", flush=True)
    df = sample_all(pts, grids=tuple(a.grids.split(",")))
    df.to_parquet(OUT / f"points_{a.set}.parquet", index=False)
    m = metrics(df)
    m.to_csv(OUT / f"metrics_{a.set}.csv", index=False)
    print(m.round(3).to_string())


if __name__ == "__main__":
    main()
