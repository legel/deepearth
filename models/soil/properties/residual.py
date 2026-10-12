"""Residual soil model: keep the best gridded product as the base and learn only a terrain correction.
  prediction = base(x) + g(terrain(x)),   base = SOLUS100 for pH / clay / sand, mean of SOLUS100 and SoilGrids in
  log10 for organic C; g = ridge regression on standardized terrain covariates (small, smooth, cannot erase the base's
  spatial detail the way a tree ensemble on raw targets does).
Validated leave-one-NEON-site-out (national and within-site R2), optionally adding WoSIS pedons to training."""
import pathlib, sys
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .fusion import score, OUT
from .terrain import NAMES as TERRAIN

TERR = [t for t in TERRAIN if t != "elev"]          # elevation is regional, not a within-site driver


def base(df, prop):
    if prop == "oc":
        l = [np.log10(np.maximum(df[c], 0.05)) for c in ("solus_oc", "soilgrids_oc") if c in df]
        return pd.concat(l, axis=1).mean(axis=1, skipna=True) if l else pd.Series(np.nan, index=df.index)
    return df[f"solus_{prop}"]


def run(neon, wosis=None):
    rows = []
    for prop in ("ph", "oc", "clay", "sand"):
        lg = prop == "oc"
        d = neon[neon[prop].notna()].copy()
        d["b"] = base(d, prop); d["y"] = np.log10(np.maximum(d[prop], 0.05)) if lg else d[prop]
        d = d.dropna(subset=["b", "y"] + TERR)
        if len(d) < 100:
            continue
        extra = None
        if wosis is not None and prop in wosis:
            w = wosis[wosis[prop].notna()].copy()
            if f"solus_{prop}" in w or lg:
                w["b"] = base(w, prop); w["y"] = np.log10(np.maximum(w[prop], 0.05)) if lg else w[prop]
                extra = w.dropna(subset=["b", "y"] + TERR)
        pred = np.full(len(d), np.nan)
        for s in d["site"].unique():
            tr, te = (d["site"] != s).to_numpy(), (d["site"] == s).to_numpy()
            T = d[tr] if extra is None else pd.concat([d[tr], extra])
            g = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20))).fit(T[TERR], T["y"] - T["b"])
            pred[te] = d["b"].to_numpy()[te] + g.predict(d.loc[te, TERR])
        inv = (lambda v: 10 ** v) if lg else (lambda v: v)
        y = d[prop].to_numpy()
        rows.append(dict(property=prop, model="base", **score(y, inv(d["b"].to_numpy()), d["site"].to_numpy(), lg)))
        rows.append(dict(property=prop, model="base+terrain ridge" + (" (+wosis)" if extra is not None else ""),
                         **score(y, inv(pred), d["site"].to_numpy(), lg)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    neon = pd.read_parquet(OUT / "points_neon.parquet")
    w = OUT / "points_wosis.parquet"
    print(run(neon).round(3).to_string())
    if w.exists():
        print(run(neon, pd.read_parquet(w)).round(3).to_string())
