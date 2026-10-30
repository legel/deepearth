"""Benchmark tab data (data/benchmark.json): what FluxFill-Bench is made of, from the files the benchmark is built
from. Towers and splits (towers/split.csv, split_v2.csv), measured-only provenance per variable
(towers/qc_counts.parquet), measured half-hours per year (towers/coverage.parquet), real outages at training towers
(bench/outages_train.parquet), the frozen test release 0.2 (bench/test_r02: manifest, windows, and the hidden values
of every mask from the FluxFill v3 test scores, which cover every family and window) and the validation sets.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FLUXES = ("NEE", "LE", "H", "FCH4")
QC_VARS = ("NEE", "LE", "H", "FCH4", "TA", "SW_IN", "VPD", "WS", "PA", "USTAR")
SPLITS = {"train": "train", "val": "validation", "test": "test"}
FAMILY_ORDER = ["M1", "M2-2h", "M2-6h", "M2-1d", "M2-7d", "M2-30d", "M2-90d", "M2-180d", "M3",
                "M4-no-reanalysis+M2-30d", "M5"]
GAP_CLASSES = ["1-1", "2-8", "9-64", "65-576", "577-inf"]


def split_counts(s: pd.DataFrame) -> dict:
    return {SPLITS[k]: {"towers": int(len(g)), "methane": int(g.has_fch4.sum())} for k, g in s.groupby("split")}


def build(root: Path) -> dict:
    t = root / "towers"
    out = {}
    s = pd.read_csv(t / "split.csv")
    out["splits"] = {"frozen": split_counts(s)}
    if (t / "split_v2.csv").exists():
        out["splits"]["v2"] = split_counts(pd.read_csv(t / "split_v2.csv"))
    out["igbp"] = [{"key": k, **{SPLITS[sp]: int((g.split == sp).sum()) for sp in SPLITS}}
                   for k, g in sorted(s.groupby("igbp"), key=lambda x: -len(x[1]))]
    out["koppen"] = [{"key": k, **{SPLITS[sp]: int((g.split == sp).sum()) for sp in SPLITS}}
                     for k, g in sorted(s.groupby("koppen_main"), key=lambda x: -len(x[1]))]
    m = s[s.has_fch4]
    out["igbp_methane"] = [{"key": k, **{SPLITS[sp]: int((g.split == sp).sum()) for sp in SPLITS}}
                           for k, g in sorted(m.groupby("igbp"), key=lambda x: -len(x[1]))]
    if (t / "split_v2.csv").exists():
        m2 = pd.read_csv(t / "split_v2.csv")
        m2 = m2[m2.has_fch4]
        out["igbp_methane_v2"] = [{"key": k, **{SPLITS[sp]: int((g.split == sp).sum()) for sp in SPLITS}}
                                  for k, g in sorted(m2.groupby("igbp"), key=lambda x: -len(x[1]))]
    # Measured-only rule: what each variable loses between the source files and the benchmark.
    q = pd.read_parquet(t / "qc_counts.parquet")
    q = q[q.site_id.isin(s.site_id) & q.variable.isin(QC_VARS)]
    g = q.groupby("variable")[["steps", "measured_at_source", "filled", "quality", "range", "stuck", "spike",
                                "ustar", "site", "final"]].sum()
    out["qc"] = [{"variable": v, "steps": int(r.steps), "final": int(r.final), "filled_at_source": int(r.filled),
                  "qc_removed": int(r.measured_at_source - r.final),
                  "not_reported": int(max(r.steps - r.measured_at_source - r.filled, 0)),
                  "by_test": {k: int(r[k]) for k in ("quality", "range", "stuck", "spike", "ustar", "site")}}
                 for v, r in g.reindex([v for v in QC_VARS if v in g.index]).iterrows()]
    # Measured flux half-hours per year and split.
    c = pd.read_parquet(t / "coverage.parquet", columns=["site_id", "variable", "year", "measured"])
    c = c[c.variable.isin(FLUXES)].merge(s[["site_id", "split"]], on="site_id")
    cy = c.groupby(["variable", "year", "split"]).measured.sum().unstack(fill_value=0)
    out["coverage_years"] = {v: [{"year": int(y), **{SPLITS[sp]: int(r.get(sp, 0)) for sp in SPLITS}}
                                 for y, r in cy.loc[v].iterrows()] for v in FLUXES if v in cy.index.get_level_values(0)}
    # Real outages at training towers (all tower channels missing together), in half-hours.
    o = pd.read_parquet(root / "bench" / "outages_train.parquet")
    L = np.concatenate([np.asarray(x) for x in o.lengths]) if len(o) else np.zeros(0)
    edges = [1, 2, 4, 8, 16, 48, 144, 336, 1440, 4320, 8640, 17520]
    h, _ = np.histogram(L, bins=edges + [np.inf])
    hv, _ = np.histogram(L, bins=edges + [np.inf], weights=L)
    out["outages"] = {"n": int(L.size), "tower_years": int(len(o)), "edges": edges,
                      "count": [int(x) for x in h], "halfhours": [int(x) for x in hv]}
    # Test release 0.2.
    tr = root / "bench" / "test_r02"
    man = json.loads((tr / "MANIFEST.json").read_text())
    w = json.loads((tr / "windows_test.json").read_text())
    w = w["windows"] if isinstance(w, dict) else w
    wd = pd.DataFrame(w)
    wd["year"] = wd.start.str[:4].astype(int)
    out["test"] = {"version": man["version"], "n_windows": man["n_windows"], "n_sites": man["n_sites"],
                   "split_sha256": man["split_sha256"][:12], "windows_sha256": man["windows_sha256"][:12],
                   "per_site": {str(k): int(v) for k, v in wd.site.value_counts().value_counts().sort_index().items()},
                   "season": {k: int(v) for k, v in wd.season.value_counts().reindex(["DJF", "MAM", "JJA", "SON"]).fillna(0).items()},
                   "year": {str(k): int(v) for k, v in wd.year.value_counts().sort_index().items()},
                   "masks": int(sum(len(v) for v in man["masks"].values()))}
    f = tr / "parts" / "fluxfill_v3.parquet"
    if f.exists():
        d = pd.read_parquet(f, columns=["family", "site", "window", "variable", "kind", "gap_class", "n"])
        d = d[d.kind == "values"]
        a = d[d.gap_class == "all"]
        fam = []
        for k in FAMILY_ORDER:
            x = a[a.family == k]
            if not len(x):
                continue
            gc = d[(d.family == k) & d.gap_class.isin(GAP_CLASSES)].groupby("gap_class").n.sum()
            fam.append({"family": k, "windows": int(x[["site", "window"]].drop_duplicates().shape[0]),
                        "hidden": int(x.n.sum()), "flux": {v: int(x[x.variable == v].n.sum()) for v in FLUXES},
                        "gap_classes": {c: int(gc.get(c, 0)) for c in GAP_CLASSES}})
        out["test"]["families"] = fam
        out["test"]["hidden_total"] = int(a.n.sum())
    # Validation sets.
    vl = root / "bench" / "val" / "windows_val_list.json"
    if vl.exists():
        v = json.loads(vl.read_text())
        v = v["windows"] if isinstance(v, dict) else v
        out["validation"] = {"windows": len(v), "sites": len({x["site"] for x in v})}
    for name, fn in (("methane", "windows_val_fch4.json"), ("methane_v2", "windows_val_fch4_v2.json")):
        p = root / "bench" / "val" / fn
        if p.exists():
            v = json.loads(p.read_text())["windows"]
            vc = pd.Series([x["site"] for x in v]).value_counts()
            out.setdefault("validation", {})[name] = {"windows": len(v), "sites": int(vc.size),
                                                      "per_site": {k: int(n) for k, n in vc.items()}}
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(build(Path(sys.argv[1] if len(sys.argv) > 1 else "/home/photon/4tb/fluxfill")), indent=1)[:3000])
