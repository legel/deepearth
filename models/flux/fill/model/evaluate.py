"""Score FluxFill against the climatology floor on validation sites with the benchmark runner (bench/run.py).

Windows: up to `--per-site` usable 180-day windows per validation site, evenly spaced over the record, starting at local
midnight. Masks: M2 all-tower blocks of 1, 7, 30, 90 and 180 days and M4 reanalysis dropout with a 30-day block, seeded
as in the benchmark. The regime is R1-like (sites never trained on, the window's own surrounding data visible).
Output: the runner's parquet rows and a summary table (CRPS, point scores, coverage of values by gap class and of
daily, 30-day and window sums).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bench"))

from data import load_records, load_split, table_dir, valid_starts  # noqa: E402
from method import FluxFill  # noqa: E402
import run as bench  # noqa: E402


def val_windows(per_site: int, max_sites: int = 0):
    sites = sorted(p.stem for p in table_dir().glob("*.parquet"))
    split, name = load_split(sites)
    val = [s for s, v in split.items() if v == "val"]
    if max_sites:
        val = val[:max_sites]
    recs = load_records(val, verbose=False)
    specs = []
    for s in val:
        if s not in recs:
            continue
        st = valid_starts(recs[s], exclude_heldout=False)
        if st.size == 0:
            continue
        for j in np.unique(np.linspace(0, st.size - 1, per_site + 2).astype(int)[1:-1]):
            specs.append({"site": s, "start": str((recs[s].t0 + pd.Timedelta(days=int(st[j]))).date()),
                          "table": str(table_dir() / f"{s}.parquet")})
    return specs, name


def summarise(df: pd.DataFrame, matched: bool = True) -> pd.DataFrame:
    """Rows of (method, table, key, n, crps, mae, bias, cov50, cov80, cov95, skill).

    - `values-gap`: all tower variables pooled per gap class; coverage is unit-free; `skill` is the mean CRPS skill
      1 - sum(n CRPS) / sum(n CRPS(climatology)) over matched (window, mask, variable, gap class) cells, computed per
      variable and averaged over variables (so units cancel).
    - `values-var`: per variable, all gap classes, in the variable's units.
    - `sum-<block>`: per flux, CRPS, bias and 95 % coverage of daily, 30-day and window sums (variable units x steps).
    """
    key = ["site", "window", "family", "variable", "gap_class", "kind"]
    df = df[df.n > 0].copy()
    if matched and "fill_rate" in df and (df.method == "climatology").any():
        # cells (window, mask, variable) where climatology fills every target: all methods score identical values
        ck = ["site", "window", "family", "variable"]
        full = df[(df.method == "climatology") & (df.kind == "values") & (df.gap_class == "all")
                  & (df.fill_rate >= 0.999)][ck].drop_duplicates()
        df = df.merge(full, on=ck)
    clim = df[df.method == "climatology"].set_index(key)["crps"]
    clim = clim[~clim.index.duplicated()]
    df["crps_clim"] = df.set_index(key).index.map(clim).to_numpy(dtype=float)

    def skill(g):
        g = g[np.isfinite(g.crps_clim) & np.isfinite(g.crps)]
        per = [1 - (h.n * h.crps).sum() / (h.n * h.crps_clim).sum() for _, h in g.groupby("variable")
               if (h.n * h.crps_clim).sum() > 0]
        return float(np.mean(per)) if per else np.nan
    out = []
    v = df[df.kind == "values"]
    for (m, gc), g in v.groupby(["method", "gap_class"]):
        w = g.n.to_numpy()
        out.append(dict(method=m, table="values-gap", key=gc, n=int(w.sum()), skill=skill(g),
                        **{k: float(np.average(g[k], weights=w)) for k in ("cov50", "cov80", "cov95")}))
    for (m, var), g in v[v.gap_class == "all"].groupby(["method", "variable"]):
        w = g.n.to_numpy()
        out.append(dict(method=m, table="values-var", key=var, n=int(w.sum()), skill=skill(g),
                        **{k: float(np.average(g[k], weights=w)) for k in ("crps", "mae", "bias", "cov95")}))
    s = df[df.kind.str.startswith("sum")]
    for (m, k, var), g in s.groupby(["method", "kind", "variable"]):
        w = g.n.to_numpy()
        out.append(dict(method=m, table=k, key=var, n=int(w.sum()), skill=skill(g),
                        **{c: float(np.average(g[c], weights=w)) for c in ("crps", "mae", "bias", "cov95")}))
    return pd.DataFrame(out)


def main():
    import torch
    torch.set_num_threads(2)            # machine rule: at most 2 CPU threads per job
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", nargs="*", default=["marginal"])
    ap.add_argument("--imagery-ablation", action="store_true", help="also score each mode with imagery removed")
    ap.add_argument("--sites", nargs="*", default=[], help="restrict to these validation sites")
    ap.add_argument("--per-site", type=int, default=2)
    ap.add_argument("--max-sites", type=int, default=0)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--families", nargs="+", default=["M2", "M4-reanalysis"])
    ap.add_argument("--no-climatology", action="store_true")
    ap.add_argument("--climatology-from", default="",
                    help="take climatology rows from an earlier output parquet with the same windows and masks")
    ap.add_argument("--windows-from", default="", help="reuse the (site, window) list of an earlier output parquet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=16, help="joint samples generated per batch (GPU memory)")
    a = ap.parse_args()
    t0 = time.time()
    specs, split_name = val_windows(a.per_site, a.max_sites)
    if a.windows_from:
        prev = pd.read_parquet(a.windows_from)[["site", "window"]].drop_duplicates()
        keep = set(map(tuple, prev.to_numpy()))
        sp, split_name = load_split(sorted({s for s, _ in keep}))
        specs = [{"site": s, "start": w, "table": str(table_dir() / f"{s}.parquet")} for s, w in sorted(keep)
                 if sp.get(s) == "val"]                     # never a test site, and only current validation sites
    if a.sites:
        specs = [x for x in specs if x["site"] in set(a.sites)]
    print(f"{len(specs)} validation windows ({split_name} split)", flush=True)
    tables = {s["site"]: pd.read_parquet(s["table"]) for s in specs}
    methods = [] if (a.no_climatology or a.climatology_from) else [bench.Climatology(a.samples)]
    for m in a.modes:
        methods.append(FluxFill(a.ckpt, mode=m, n_samples=a.samples, steps=a.steps, chunk=a.chunk, device=a.device))
        if a.imagery_ablation:
            mm = FluxFill(a.ckpt, mode=m, n_samples=a.samples, steps=a.steps, chunk=a.chunk, device=a.device,
                          imagery=False)
            mm.name += "-noimg"
            methods.append(mm)
    rows = []
    for i, spec in enumerate(specs):
        rows.append(bench.run(methods, [spec], tables, "R1-val", a.families))
        print(f"{i + 1}/{len(specs)} {spec['site']} {spec['start']} {time.time() - t0:.0f} s", flush=True)
        df = pd.concat(rows, ignore_index=True)
        df["split"] = split_name
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(a.out, index=False)
    if a.climatology_from:
        cl = pd.read_parquet(a.climatology_from)
        cl = cl[cl.method == "climatology"].merge(df[["site", "window", "family"]].drop_duplicates())
        df = pd.concat([df, cl], ignore_index=True)
        df.to_parquet(a.out, index=False)
    summ = summarise(df)
    summ.to_csv(Path(a.out).with_suffix(".summary.csv"), index=False)
    summarise(df, matched=False).to_csv(Path(a.out).with_suffix(".summary_all.csv"), index=False)
    pd.set_option("display.width", 200)
    for t, g in summ.groupby("table"):
        print(f"== {t}")
        print(g.drop(columns="table").sort_values(["key", "method"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
