"""Protocol gate: a method's prediction for a mask depends only on that mask's visible copy and the context.

For one real window and two masks A and B, a fresh method predicts B alone, and another predicts A then B; the
predictions for B must be identical. The prediction is compared through `check_state`: the predictive parameters for
FluxFill, the fixed-seed samples otherwise. A second run moves every hidden value of B by 100 and must change nothing,
which guards the construction of the visible copy. Any state kept across masks fails here, as the site adaptation of
October 2026 did: it adapted once per window, on the first mask's visible values.

Usage: python leak_check.py <site_table.parquet> <site> <start> --methods climatology mds fluxfill
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "baselines")]

from windows import Window, load_window, lst_index  # noqa: E402


def visible_of(w, m, shift_hidden=0.0):
    v = w.values + shift_hidden * m
    return Window(w.site, w.start, w.t, w.keys, np.where(m, np.nan, v), w.measured & ~m, w.provenance, w.meta)


def state(meth, out):
    """What must be identical: the FluxFill head output, else the prediction."""
    last = getattr(meth, "_params_last", None)
    return [np.asarray(u, np.float64) for u in last] if last else [np.asarray(out, np.float64)]


def same(a, b, tol=1e-5):
    return all(np.allclose(x, y, atol=tol, equal_nan=True) for x, y in zip(a, b))


def check(make, w, ma, mb, ctx, name):
    alone = make()
    s0 = state(alone, alone.predict(visible_of(w, mb), mb, ctx))
    after = make()
    after.predict(visible_of(w, ma), ma, ctx)
    s1 = state(after, after.predict(visible_of(w, mb), mb, ctx))
    moved = make()
    s2 = state(moved, moved.predict(visible_of(w, mb, 100.0), mb, ctx))
    print(f"{name}: mask order {'PASS' if same(s0, s1) else 'FAIL'}, hidden values moved {'PASS' if same(s0, s2) else 'FAIL'}", flush=True)
    return same(s0, s1) and same(s0, s2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("site"); ap.add_argument("start")
    ap.add_argument("--methods", nargs="+", default=["climatology", "mds"])
    ap.add_argument("--lat", type=float, default=0.0); ap.add_argument("--lon", type=float, default=0.0)
    a = ap.parse_args()
    from run import masks_for, method_registry
    tab = pd.read_parquet(a.table)
    w = load_window(tab, a.site, a.start)
    ms = masks_for(w, ["M2"])
    ma, mb = ms["M2-7d"], ms["M2-30d"]
    idx = lst_index(tab)
    years = list(range(w.t[0].year, w.t[-1].year + 1))
    ctx = {"table": tab[~idx.year.isin(years)], "index": idx[~idx.year.isin(years)]}
    assert not set(ctx["index"].year) & set(years), "the context holds a year of the window"
    reg = method_registry()
    ok = True
    for name in a.methods:
        if name == "mds":
            from mds import MDS
            make = lambda: MDS(lat=a.lat, lon=a.lon, utc_offset=round(a.lon / 15))
        else:
            make = reg[name]
        ok &= check(make, w, ma, mb, ctx, name)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
