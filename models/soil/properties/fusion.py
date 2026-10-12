"""A fused soil model: learns how to combine the gridded products (SOLUS100, POLARIS, SSURGO, SoilGrids) with 10 m
terrain (terrain.py) to predict surface (0-30 cm) pH, clay, sand and organic C at a point.

Why fuse: each product is a different estimate with different errors (SSURGO = map-unit polygons with component
representative values, so no variation inside a polygon; SOLUS/POLARIS/SoilGrids = statistical maps at 100/30/250 m).
Terrain adds the within-polygon variation the map units cannot express: hollows collect water, fines and carbon.

Validation, two ways, never on training locations:
  * trained on WoSIS US pedons (thinned to one per 1 km cell), tested on every NEON plot (an independent lab and
    sampling design);
  * leave-one-NEON-site-out on NEON itself, reported both nationally and within site (deviation from the site mean),
    the scale a planting design works at.
Model: gradient-boosted trees (LightGBM), monotone-free, small (200 trees, 31 leaves), features missing -> native NaN
handling. Output: $DEEPEARTH_SOIL_DATA/benchmark/fusion_metrics.csv and fusion_<prop>.txt models trained on everything."""
import pathlib, sys
import numpy as np, pandas as pd, lightgbm as lgb

from . import benchmark as B
from .terrain import NAMES as TERRAIN

OUT = B.OUT
PRODUCTS = ("solus", "polaris", "soilgrids", "ssurgo")
PARAMS = dict(n_estimators=200, num_leaves=31, learning_rate=0.05, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4)


def features(df, prop):
    cols = [f"{p}_{prop}" for p in PRODUCTS if f"{p}_{prop}" in df] + [t for t in TERRAIN if t in df]
    return cols


def score(y, p, site=None, log=False):
    if log:
        y, p = np.log10(np.maximum(y, 0.05)), np.log10(np.maximum(p, 0.05))
    out = {"n": len(y), "r2": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
           "rmse": float(np.sqrt(np.mean((y - p) ** 2)))}
    if site is not None:
        d = pd.DataFrame({"y": y, "p": p, "s": site})
        d = d[d.groupby("s")["y"].transform("size") >= 5]
        yw = d["y"] - d.groupby("s")["y"].transform("mean"); pw = d["p"] - d.groupby("s")["p"].transform("mean")
        out.update(within_n=len(d), within_r2=1 - np.sum((yw - pw) ** 2) / np.sum(yw ** 2))
    return out


def main():
    neon = pd.read_parquet(OUT / "points_neon.parquet")
    wosis = pd.read_parquet(OUT / "points_wosis.parquet") if (OUT / "points_wosis.parquet").exists() and "--neon-only" not in sys.argv else None
    rows = []
    for prop in ("ph", "oc", "clay", "sand"):
        cols = features(neon, prop)
        nt = neon[neon[prop].notna()]
        lg = prop == "oc"                         # organic C: fit and score in log10(%)
        T = (lambda v: np.log10(np.maximum(v, 0.05))) if lg else (lambda v: v)
        Ti = (lambda v: 10 ** v) if lg else (lambda v: v)
        # (1) WoSIS -> NEON transfer
        if wosis is not None and prop in wosis and wosis[prop].notna().sum() > 200:
            wt = wosis[wosis[prop].notna()]
            m = lgb.LGBMRegressor(**PARAMS).fit(wt[cols], T(wt[prop]))
            rows.append(dict(property=prop, model="fusion", test="wosis->neon", **score(nt[prop].to_numpy(),
                             Ti(m.predict(nt[cols])), nt["site"].to_numpy(), lg)))
            m.booster_.save_model(str(OUT / f"fusion_{prop}.txt"))
        # (2) leave-one-NEON-site-out
        if len(nt) > 100:
            pred = np.full(len(nt), np.nan)
            for s in nt["site"].unique():
                tr, te = nt["site"] != s, nt["site"] == s
                train = pd.concat([nt[tr], wosis[wosis[prop].notna()]]) if wosis is not None and prop in wosis else nt[tr]
                pred[te.to_numpy()] = Ti(lgb.LGBMRegressor(**PARAMS).fit(train[cols], T(train[prop])).predict(nt.loc[te, cols]))
            rows.append(dict(property=prop, model="fusion", test="leave-site-out", **score(nt[prop].to_numpy(), pred,
                                                                                            nt["site"].to_numpy(), lg)))
            nt = nt.assign(fusion=pred)
        for p in PRODUCTS:          # the products alone on the same rows, for the comparison
            c = f"{p}_{prop}"
            if c in nt:
                d = nt[nt[c].notna()]
                if len(d) > 20:
                    rows.append(dict(property=prop, model=p, test="product", **score(d[prop].to_numpy(), d[c].to_numpy(),
                                                                                     d["site"].to_numpy(), lg)))
                    if "fusion" in d:      # fusion on exactly the rows this product covers
                        rows.append(dict(property=prop, model="fusion", test=f"leave-site-out on {p} rows",
                                         **score(d[prop].to_numpy(), d["fusion"].to_numpy(), d["site"].to_numpy(), lg)))
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "fusion_metrics.csv", index=False)
    print(m.round(3).to_string())


if __name__ == "__main__":
    main()
