"""Plants as soil indicators, learned from data (the idea behind Ellenberg indicator values, fitted rather than
assigned): each species' indicator value u_s is the shrunken mean, over the training plots where it grows, of the soil
residual the gridded products miss (observation - SOLUS100/ensemble base, centred within each site). A plot's
correction is the cover-weighted mean of its species' u_s. Validated leave-one-NEON-site-out, so every species value
used at a site was learned only at other sites. Reports national and within-site R2 against the product base."""
import pathlib, sys
import numpy as np, pandas as pd

from .config import DATA

from .residual import base, OUT
from .fusion import score

W = DATA
K = 5.0          # shrinkage: a species seen on n plots keeps n / (n + K) of its mean residual


def community(level="species"):
    """plot x taxon mean cover; level species | genus (taxa resolved at least to genus)."""
    c = pd.read_parquet(W / "neon/plant_cover.parquet")
    c["percentCover"] = pd.to_numeric(c["percentCover"], errors="coerce").fillna(0.5)
    if level == "species":
        c = c[c["taxonRank"].isin(["species", "subspecies", "variety"])]
        c["sp"] = c["scientificName"].str.split().str[:2].str.join(" ")
    else:
        c = c[c["taxonRank"].isin(["genus", "species", "subspecies", "variety"])]
        c["sp"] = c["scientificName"].str.split().str[0]
    return c.groupby(["plotID", "sp"])["percentCover"].mean().unstack(fill_value=0.0)


def run(level="species"):
    neon = pd.read_parquet(OUT / "points_neon.parquet")
    M = community(level)
    rows = []
    for prop in ("ph", "oc"):
        lg = prop == "oc"
        d = neon[neon[prop].notna()].copy(); d["b"] = base(d, prop)
        d["y"] = np.log10(np.maximum(d[prop], 0.05)) if lg else d[prop]
        d = d.dropna(subset=["b", "y"]); d = d[d["id"].isin(M.index)]
        d = d[d.groupby("site")["y"].transform("size") >= 5]
        r = d["y"] - d["b"]; d["rc"] = r - r.groupby(d["site"]).transform("mean")
        C = M.loc[d["id"]].to_numpy(); P = (C > 0).astype(float)
        pred_c = np.zeros(len(d))
        for s in d["site"].unique():
            tr, te = (d["site"] != s).to_numpy(), (d["site"] == s).to_numpy()
            n = P[tr].sum(0); u = (P[tr] * d["rc"].to_numpy()[tr, None]).sum(0) / (n + K)
            w = C[te] * (n > 0)
            corr = (w * u).sum(1) / np.maximum(w.sum(1), 1e-9)
            pred_c[te] = corr - corr.mean()            # within-site deviations only; the site level stays the base's
        inv = (lambda v: 10 ** v) if lg else (lambda v: v)
        y = d[prop].to_numpy(); site = d["site"].to_numpy()
        rows.append(dict(property=prop, model="base", **score(y, inv(d["b"].to_numpy()), site, lg)))
        for a in (0.5, 1.0, 1.5):
            rows.append(dict(property=prop, model=f"base + {level} indicators x{a}",
                             **score(y, inv(d["b"].to_numpy() + a * pred_c), site, lg)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for level in ("species", "genus"):
        print(run(level).round(3).to_string())
