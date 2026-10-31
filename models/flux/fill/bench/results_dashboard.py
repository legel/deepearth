"""Turn runner outputs (bench/run.py parquet) into the dashboard's results.json: tables, chart data, generated
headlines and findings.

Usage: python results_dashboard.py <scores.parquet> [<more.parquet> ...] --out results.json --title "..." --note "..."
       [--truth truth_stats.parquet] [--split validation]

Every number is pooled over cells (site, window, mask family, variable) from the per-cell rows of the runner:
- CRPS, MAE, bias and interval coverage: means weighted by the number of scored values n.
- MSE = sum n rmse^2 / sum n (exact); RMSE = sqrt(MSE).
- R2 = 1 - sum n rmse^2 / (sum y^2 - (sum y)^2 / N), with the pooled truth sums of the same cells.
- WAPE = 100 sum n mae / sum |y|.
The truth sums come from the parts' sum_y, sum_y2, sum_abs_y columns (bench/run.py writes them) or, for older parts,
from --truth (bench/truth_stats.py), joined per cell.

Matched cells. A column compares methods on identical values: the cells (site, window, family, variable) that every
compared method fills completely (fill_rate = 1). For a gap length, the compared methods are those that fill at least
MIN_FILL of its hidden measured values; the others are shown with their fill rate and not scored. Because every
scored cell is filled completely, R2 and WAPE always rest on complete cells.

Headlines (one declarative sentence per table and chart) and the findings block are generated from the same numbers,
so they stay true when the numbers change.
"""

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["M1", "M2-2h", "M2-6h", "M2-1d", "M2-7d", "M2-30d", "M2-90d", "M2-180d", "M3", "M4-no-reanalysis+M2-30d", "M5"]
METHOD_ORDER = ["climatology", "mds", "irvin_rf", "vekuri_de", "liu_xgb", "saits", "csdi", "chronos2", "fluxfill-v0",
                "fluxfill-v1", "fluxfill-ref", "fluxfill", "fluxfill-adapt"]
LABEL = {"climatology": "climatology", "mds": "MDS (REddyProc)", "irvin_rf": "RF ensemble (Irvin 2021)",
         "vekuri_de": "deep ensemble (Vekuri 2025)", "liu_xgb": "XGBoost (Liu 2025)",
         "saits": "SAITS + residual Gaussian", "csdi": "CSDI", "chronos2": "Chronos-2 zero-shot",
         "fluxfill-v0": "FluxFill v0 (Student-t marginals)", "fluxfill-v1": "FluxFill v1 (joint, masked generation)",
         "fluxfill-ref": "FluxFill v2 (width 192, depth 6)",
         "fluxfill": "FluxFill"}
SHORT = {"climatology": "climatology", "mds": "MDS", "irvin_rf": "the RF ensemble", "vekuri_de": "the deep ensemble",
         "liu_xgb": "XGBoost", "saits": "SAITS", "csdi": "CSDI", "chronos2": "Chronos-2", "fluxfill-v0": "FluxFill v0",
         "fluxfill-v1": "FluxFill v1", "fluxfill-ref": "FluxFill v2", "fluxfill": "FluxFill"}
OURS = ("fluxfill-v0", "fluxfill-v1", "fluxfill-ref", "fluxfill", "fluxfill-adapt")
FLUXES = ("NEE", "LE", "H", "FCH4")
UNITS = {"NEE": "µmol m⁻² s⁻¹", "LE": "W m⁻²", "H": "W m⁻²", "FCH4": "nmol m⁻² s⁻¹"}
NAMES = {"NEE": "net ecosystem exchange of CO₂", "LE": "latent heat flux", "H": "sensible heat flux",
         "FCH4": "methane flux"}
# A sum of half-hourly values over the hidden measured half-hours of a block, in physical units: factor per
# (unit x half-hour). NEE: 1800 s x 12.011 ug C per umol; LE, H: 1800 s, J to MJ; FCH4: 1800 s x 16.04 ng per nmol.
SUM_UNITS = {"NEE": ("g C m⁻²", 1800 * 12.011e-6), "LE": ("MJ m⁻²", 1800e-6), "H": ("MJ m⁻²", 1800e-6),
             "FCH4": ("mg CH₄ m⁻²", 1800 * 16.04e-6)}
SUM_KINDS = {"sum-daily": "daily", "sum-30-day": "30-day", "sum-window": "180-day window"}
MIN_FILL = 0.9

METRICS = {
    "crps": dict(name="CRPS", long="continuous ranked probability score", better="lower", units="unit",
                 role="primary", formula="CRPS(F, y) = E|X − y| − ½ E|X − X′|, with X, X′ independent draws from the "
                 "predictive distribution F",
                 text="The primary score. Computed from 64 samples per value with the fair ensemble estimator "
                      "(Ferro 2014). It is a proper scoring rule: a method minimises its expected CRPS only by "
                      "predicting the true distribution. For a point forecast it equals the absolute error."),
    "cov95": dict(name="95% coverage", long="coverage of the central 95% interval", better="nominal", nominal=0.95,
                  units="share", role="calibration",
                  formula="share of true values inside [q2.5%, q97.5%] of the method's samples",
                  text="Calibration. 0.95 is calibrated; below 0.95 the intervals are too narrow, above too wide."),
    "cov80": dict(name="80% coverage", long="coverage of the central 80% interval", better="nominal", nominal=0.80,
                  units="share", role="calibration", formula="share of true values inside [q10%, q90%]",
                  text="Calibration at 80%; 0.80 is calibrated."),
    "cov50": dict(name="50% coverage", long="coverage of the central 50% interval", better="nominal", nominal=0.50,
                  units="share", role="calibration", formula="share of true values inside [q25%, q75%]",
                  text="Calibration at 50%; 0.50 is calibrated."),
    "mae": dict(name="MAE", long="mean absolute error", better="lower", units="unit", role="point",
                formula="MAE = (1/N) Σ |ŷ − y|", text="Point accuracy of the predictive mean ŷ."),
    "mse": dict(name="MSE", long="mean squared error", better="lower", units="unit²", role="point",
                formula="MSE = (1/N) Σ (ŷ − y)², pooled exactly as Σ n·RMSE² / Σ n over cells",
                text="Point accuracy of the predictive mean, in squared units."),
    "rmse": dict(name="RMSE", long="root mean squared error", better="lower", units="unit", role="point",
                 formula="RMSE = √MSE", text="Point accuracy of the predictive mean, in the variable's units."),
    "bias": dict(name="bias", long="mean error", better="zero", units="unit", role="point",
                 formula="bias = (1/N) Σ (ŷ − y)",
                 text="Positive: the predictive mean is above the truth. For NEE, where uptake is negative, a "
                      "positive bias means uptake is underestimated."),
    "r2": dict(name="R²", long="coefficient of determination", better="higher", units="none", role="point",
               formula="R² = 1 − Σ (ŷ − y)² / Σ (y − ȳ)², ȳ the mean of all pooled true values",
               text="1 is perfect; 0 equals predicting the pooled mean ȳ; negative is worse than that. Pooled over "
                    "all values of the table cell (sites together), so between-site differences count as "
                    "explained variance."),
    "wape": dict(name="WAPE", long="weighted absolute percentage error", better="lower", units="%", role="point",
                 formula="WAPE = 100 × Σ |ŷ − y| / Σ |y|",
                 text="The absolute error as a percentage of the measured magnitude, 0% is perfect. It is not "
                      "capped at 100%: errors larger in total than the measured magnitudes score above 100%. "
                      "Per-value MAPE (the mean of |ŷ − y| / |y|) is not used, because NEE, H and FCH4 cross zero "
                      "and single terms with |y| near 0 are unbounded."),
    "fill": dict(name="filled", long="fill rate", better="higher", units="share", role="coverage",
                 formula="hidden measured values the method filled / hidden measured values",
                 text="A method is scored only on gap lengths where it fills at least 90% of the hidden values."),
}
VALUE_METRICS = ["crps", "cov95", "cov80", "cov50", "mae", "rmse", "mse", "bias", "r2", "wape"]
SUM_METRICS = ["crps", "cov95", "mae", "rmse", "bias", "r2", "wape"]
SUMMARY_METRICS = ["crps", "cov95", "mae", "rmse", "mse", "bias", "r2", "wape"]
CELL = ["site", "window", "family", "variable"]


# ----------------------------------------------------------------------------------------------- pooling

def pool(x: pd.DataFrame, scale: float = 1.0) -> dict:
    """Pooled metrics of rows (one per cell) with n, crps, mae, rmse, bias, cov*, sum_y, sum_y2, sum_abs_y."""
    x = x[x.n > 0]
    N = float(x.n.sum())
    if N == 0:
        return {}
    w = x.n.to_numpy(float)
    avg = lambda c: float(np.average(x[c].to_numpy(float), weights=w)) if x[c].notna().all() else float("nan")
    sse = float((w * x.rmse.to_numpy(float) ** 2).sum())
    out = dict(n=int(N), crps=avg("crps") * scale, mae=avg("mae") * scale, bias=avg("bias") * scale,
               mse=sse / N * scale ** 2, rmse=math.sqrt(sse / N) * scale,
               cov95=avg("cov95"), cov80=avg("cov80") if "cov80" in x else float("nan"),
               cov50=avg("cov50") if "cov50" in x else float("nan"))
    if {"sum_y", "sum_y2", "sum_abs_y"} <= set(x) and x[["sum_y", "sum_y2", "sum_abs_y"]].notna().all().all():
        sst = float(x.sum_y2.sum()) - float(x.sum_y.sum()) ** 2 / N
        out["r2"] = 1 - sse / sst if sst > 0 else float("nan")
        sa = float(x.sum_abs_y.sum())
        out["wape"] = 100 * float((w * x.mae.to_numpy(float)).sum()) / sa if sa > 0 else float("nan")
    else:
        out["r2"] = out["wape"] = float("nan")
    return out


def with_truth(d: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Fill sum_y, sum_y2, sum_abs_y from the truth table where a part lacks them. Truth stats hold the hidden
    measured values of a cell, which equal its scored values only when fill_rate = 1, so only those rows take them."""
    for c in ("sum_y", "sum_y2", "sum_abs_y"):
        if c not in d:
            d[c] = np.nan
    if truth is None:
        return d
    k = CELL + ["kind", "gap_class"]
    t = truth[k + ["n", "sum_y", "sum_y2", "sum_abs_y"]].rename(columns={c: c + "_t" for c in
                                                                         ["n", "sum_y", "sum_y2", "sum_abs_y"]})
    d = d.merge(t, on=k, how="left")
    ok = (d.fill_rate == 1) & (d.n == d.n_t)
    bad = (d.fill_rate == 1) & d.n_t.notna() & (d.n != d.n_t)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} fully filled rows disagree with the truth table on n")
    for c in ("sum_y", "sum_y2", "sum_abs_y"):
        d[c] = d[c].where(d[c].notna(), d[c + "_t"].where(ok))
    return d.drop(columns=[c + "_t" for c in ["n", "sum_y", "sum_y2", "sum_abs_y"]])


# ----------------------------------------------------------------------------------------------- wording

def gap_label(f: str) -> str:
    if f.startswith("M2-") and f.endswith("d"):
        return f"{f[3:-1]} d"
    if f.startswith("M2-") and f.endswith("h"):
        return f"{f[3:-1]} h"
    if f.startswith("M4-no-reanalysis+M2-"):
        return f"{f.split('M2-')[1][:-1]} d, no reanalysis"
    return {"M1": "short gaps (M1)", "M3": "real outages (M3)", "M5": "mixed per-variable (M5)"}.get(f, f)


def days_phrase(fams) -> str:
    """'1, 7 and 30 days' from M2 family names (other families by their label)."""
    ds = [f[3:-1] for f in fams if f.startswith("M2-") and f.endswith("d")]
    other = [gap_label(f) for f in fams if not (f.startswith("M2-") and f.endswith("d"))]
    parts = []
    if ds:
        parts.append((", ".join(ds[:-1]) + " and " + ds[-1] if len(ds) > 1 else ds[0]) + (" days" if ds != ["1"] else " day"))
    return join(parts + other)


def join(xs) -> str:
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1] if xs else ""


def fmtv(v: float, metric: str) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    if metric.startswith("cov") or metric == "fill" or metric == "r2":
        return f"{v:.3f}" if metric != "fill" else f"{100 * v:.0f}%"
    if metric == "wape":
        return f"{v:.1f}%"
    a = abs(v)
    d = 3 if a < 10 else 2 if a < 100 else 1 if a < 1000 else 0
    return f"{v:,.{d}f}"


def best_index(vals, metric: str):
    """Index of the best finite value under the metric's direction (None if none)."""
    m = METRICS[metric]
    arr = np.array([v if v is not None and np.isfinite(v) else np.nan for v in vals], float)
    if np.isnan(arr).all():
        return None
    if m["better"] == "lower":
        return int(np.nanargmin(arr))
    if m["better"] == "higher":
        return int(np.nanargmax(arr))
    if m["better"] == "zero":
        return int(np.nanargmin(np.abs(arr)))
    return int(np.nanargmin(np.abs(arr - m["nominal"])))


def best_indices(vals, metric: str):
    """Every index whose value, rounded as displayed (fmtv), ties the best under the metric's direction."""
    m = METRICS[metric]
    shown = []
    for v in vals:
        t = fmtv(v, metric) if v is not None else "n/a"
        shown.append(np.nan if t == "n/a" else float(t.replace(",", "").rstrip("%")))
    arr = np.array(shown, float)
    if np.isnan(arr).all():
        return []
    score = {"lower": arr, "higher": -arr, "zero": np.abs(arr)}.get(m["better"], np.abs(arr - m.get("nominal", 0.0)))
    top = np.nanmin(score)
    return [int(i) for i in np.flatnonzero(np.isclose(score, top, rtol=0, atol=1e-9))]


def verb(metric: str) -> str:
    m = METRICS[metric]
    if m["better"] == "nominal":
        return f"the {m['name']} closest to {m['nominal']:.2f}"
    return {"lower": f"the lowest {m['name']}", "higher": f"the highest {m['name']}",
            "zero": "the smallest absolute bias"}[m["better"]]


def column_headline(methods, cols, values, metric: str, what: str) -> str:
    """'<m> has <best> for <what> at every gap length from 1 to 180 days' or the per-column winners."""
    wins = {}
    for j, f in enumerate(cols):
        b = best_index([values[i][j] for i in range(len(methods))], metric)
        if b is not None:
            wins.setdefault(methods[b], []).append(f)
    if not wins:
        return f"No method is scored for {what}."
    m0 = sorted(wins, key=lambda m: (-len(wins[m]), cols.index(wins[m][0])))[0]
    order = [m0] + sorted((m for m in wins if m != m0), key=lambda m: cols.index(wins[m][0]))
    if len(order) == 1 and len(wins[m0]) == len(cols) and len(cols) > 1:
        if all(c.startswith("M2-") and c.endswith("d") for c in cols):
            return (f"{cap(SHORT[m0])} has {verb(metric)} for {what} at every gap length from {cols[0][3:-1]} to "
                    f"{days_phrase([cols[-1]])}.")
        return f"{cap(SHORT[m0])} has {verb(metric)} for {what} in every column."
    first = f"{cap(SHORT[m0])} has {verb(metric)} for {what} at {days_phrase(wins[m0])}"
    rest = [f"{SHORT[m]} at {days_phrase(wins[m])}" for m in order[1:]]
    return first + ("; " + "; ".join(rest) if rest else "") + "."


def plural(k: int, word: str) -> str:
    return f"{k} {word}" + ("" if k == 1 else "s")


def cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


# ----------------------------------------------------------------------------------------------- tables

def family_fill(v: pd.DataFrame, m: str, f: str) -> float:
    y = v[(v.method == m) & (v.family == f)]
    return float(np.average(y.fill_rate, weights=y.n_target)) if len(y) and y.n_target.sum() > 0 else float("nan")


def matched_cells(v: pd.DataFrame, methods, families=None):
    """Cells (site, window, family, variable) that every method in `methods` fills completely."""
    x = v if families is None else v[v.family.isin(families)]
    full = x[x.fill_rate == 1].groupby(CELL).method.apply(set)
    keep = full[full.apply(lambda s: set(methods) <= s)]
    return set(keep.index)


def cell_rows(d: pd.DataFrame, cells, m: str, kind: str) -> pd.DataFrame:
    x = d[(d.method == m) & (d.kind == kind) & (d.gap_class == "all")]
    idx = pd.MultiIndex.from_frame(x[CELL])
    return x[idx.isin(list(cells))] if cells else x.iloc[:0]


def by_gap(d: pd.DataFrame, v: pd.DataFrame, flux: str, methods, fams, kind: str, scale: float):
    """Per family: compared methods, matched cells, pooled metrics per method."""
    out = {"families": fams, "n": [], "cells": [], "windows": [], "compared": [], "pooled": {m: [] for m in methods},
           "fill": {m: [] for m in methods}}
    vf = v[v.variable == flux]
    for f in fams:
        fills = {m: family_fill(vf, m, f) for m in methods}
        comp = [m for m in methods if np.isfinite(fills[m]) and fills[m] >= MIN_FILL]
        cells = {c for c in matched_cells(vf, comp, [f])} if comp else set()
        out["compared"].append(comp)
        out["cells"].append(len(cells))
        out["windows"].append(len({(c[0], c[1]) for c in cells}))
        n = None
        for m in methods:
            out["fill"][m].append(fills[m])
            p = pool(cell_rows(d[d.variable == flux], cells, m, kind), scale) if m in comp and cells else {}
            out["pooled"][m].append(p)
            if p and n is None:
                n = p["n"]
        out["n"].append(n or 0)
    return out


def table_from(g, methods, metric: str, units: str, title: str, what: str, note: str, kind_label: str):
    cols = g["families"]
    vals = [[(g["pooled"][m][j] or {}).get(metric, float("nan")) for j in range(len(cols))] for m in methods]
    rows = []
    for i, m in enumerate(methods):
        cells = []
        for j in range(len(cols)):
            fr = g["fill"][m][j]
            if not np.isfinite(fr):
                cells.append({"v": None, "t": "not run"})
            elif m not in g["compared"][j]:
                cells.append({"v": None, "t": f"fills {100 * fr:.0f}%"})
            else:
                cells.append({"v": vals[i][j]})
        rows.append({"method": m, "label": LABEL.get(m, m), "ours": m in OURS, "cells": cells})
    best = [best_index([vals[i][j] if m in g["compared"][j] else None for i, m in enumerate(methods)], metric)
            for j in range(len(cols))]
    best_all = [best_indices([vals[i][j] if m in g["compared"][j] else None for i, m in enumerate(methods)], metric)
                for j in range(len(cols))]
    mm = METRICS[metric]
    u = units if mm["units"] == "unit" else (units + "²" if mm["units"] == "unit²" else
                                            {"share": "share of values", "%": "%", "none": "no unit"}[mm["units"]])
    return {"metric": metric, "title": title, "headline": column_headline(methods, cols, vals, metric, what),
            "units": u, "kind": kind_label,
            "columns": [{"id": f, "label": gap_label(f), "n": g["n"][j], "cells": g["cells"][j],
                         "windows": g["windows"][j]} for j, f in enumerate(cols)],
            "rows": rows, "best": best, "best_all": best_all, "note": note}


def summary_table(d, v, flux, methods, units):
    """Every metric on the cells that all methods fill completely, over every gap length."""
    vf = v[v.variable == flux]
    cells = matched_cells(vf, methods)
    fams = sorted({c[2] for c in cells}, key=lambda f: ORDER.index(f) if f in ORDER else 99)
    rows, vals = [], {}
    for m in methods:
        p = pool(cell_rows(d[d.variable == flux], cells, m, "values")) if cells else {}
        vals[m] = p
    n = next((vals[m]["n"] for m in methods if vals[m]), 0)
    for m in methods:
        p = vals[m]
        rows.append({"method": m, "label": LABEL.get(m, m), "ours": m in OURS,
                     "cells": [{"v": p.get(k, float("nan"))} for k in SUMMARY_METRICS]})
    best = [best_index([vals[m].get(k) for m in methods], k) for k in SUMMARY_METRICS]
    best_all = [best_indices([vals[m].get(k) for m in methods], k) for k in SUMMARY_METRICS]
    nw = len({(c[0], c[1]) for c in cells})
    ns = len({c[0] for c in cells})
    if cells and best[0] is not None:
        order = np.argsort([vals[m].get("crps", np.inf) for m in methods])
        b, s = methods[order[0]], methods[order[1]]
        head = (f"{cap(SHORT[b])} has the lowest CRPS for {flux} on the {plural(len(cells), 'cell')} ({n:,} values) "
                f"all {len(methods)} methods fill: {fmtv(vals[b]['crps'], 'crps')} against {fmtv(vals[s]['crps'], 'crps')} for {SHORT[s]} "
                f"({units}).")
        r2b = methods[best[SUMMARY_METRICS.index("r2")]] if best[SUMMARY_METRICS.index("r2")] is not None else None
        if r2b is not None:
            head += (f" It also has the highest R² ({fmtv(vals[r2b]['r2'], 'r2')})." if r2b == b else
                     f" {cap(SHORT[r2b])} has the highest R² ({fmtv(vals[r2b]['r2'], 'r2')}).")
    else:
        head = f"No cell of {flux} is filled completely by all {len(methods)} methods."
    return {"flux": flux, "title": f"{flux}: every metric on the cells all methods fill",
            "headline": head, "units": units, "n": n, "cells": len(cells), "windows": nw, "sites": ns,
            "families": fams, "metrics": SUMMARY_METRICS, "rows": rows, "best": best,
            "best_all": best_all}, vals, cells


def fill_table(v, flux, methods, fams):
    vf = v[v.variable == flux]
    rows, short = [], []
    for m in methods:
        fr = [family_fill(vf, m, f) for f in fams]
        rows.append({"method": m, "label": LABEL.get(m, m), "ours": m in OURS,
                     "cells": [{"v": x} if np.isfinite(x) else {"v": None, "t": "not run"} for x in fr]})
        part = [f for f, x in zip(fams, fr) if np.isfinite(x) and x < 0.995]
        if part:
            short.append(f"{SHORT[m]} fills " + join(f"{100 * x:.0f}% at {gap_label(f)}" for f, x in zip(fams, fr)
                                                    if np.isfinite(x) and x < 0.995))
    n_full = sum(1 for r in rows if all((c["v"] or 0) >= 0.995 for c in r["cells"]))
    head = (cap("; ".join(short)) + f"; the other {n_full} methods fill every hidden measured value of {flux}."
            if short else f"Every method fills every hidden measured value of {flux}.")
    return {"metric": "fill", "title": f"{flux}: share of hidden measured values filled", "headline": head,
            "units": "share of values", "columns": [{"id": f, "label": gap_label(f)} for f in fams], "rows": rows,
            "best": [None] * len(fams), "note": "All cells. A method is scored on a gap length only where it fills at "
                                                f"least {100 * MIN_FILL:.0f}% of its hidden measured values."}


# ----------------------------------------------------------------------------------------------- findings

def findings(summ, gaps, sums, methods, scope):
    out = []
    S = {t["flux"]: (t, vals) for t, vals in summ}
    # 1. overall CRPS winners on all-method cells
    by = {}
    for flux, (t, vals) in S.items():
        if t["cells"] and vals:
            b = min((m for m in methods if vals[m]), key=lambda m: vals[m]["crps"])
            by.setdefault(b, []).append(f"{flux} ({fmtv(vals[b]['crps'], 'crps')} {t['units']}, {t['n']:,} values)")
    if by:
        t0 = next(iter(S.values()))[0]
        fams = sorted({f for t, _ in S.values() for f in t["families"]}, key=lambda f: ORDER.index(f) if f in ORDER else 99)
        parts = [f"{SHORT[m]} has the lowest CRPS for {join(v)}" for m, v in
                 sorted(by.items(), key=lambda kv: -len(kv[1]))]
        out.append({"head": "Lowest CRPS on identical values.",
                    "text": cap("; ".join(parts)) + ".",
                    "data": f"Cells (window × gap × variable) that all {len(methods)} methods fill completely: gaps of "
                            f"{days_phrase(fams)}; {scope['windows']} {scope['split']} windows at {scope['sites']} "
                            f"{scope['split']} sites, regime {', '.join(scope.get('regimes', ['R1']))}."})
    # 2. winners by gap length
    segs = []
    for flux in FLUXES:
        if flux not in gaps:
            continue
        g = gaps[flux]
        wins = {}
        for j, f in enumerate(g["families"]):
            vals = [(g["pooled"][m][j] or {}).get("crps") if m in g["compared"][j] else None for m in methods]
            b = best_index(vals, "crps")
            if b is not None:
                wins.setdefault(methods[b], []).append(f)
        lab = flux if max(g["n"]) >= 1000 else f"{flux} ({min(x for x in g['n'] if x):,} to {max(g['n']):,} values per gap length)"
        segs.append(f"{lab}: " + "; ".join(f"{SHORT[m]} at {days_phrase(fs)}" for m, fs in
                                            sorted(wins.items(), key=lambda kv: g["families"].index(kv[1][0]))))
    if segs:
        out.append({"head": "Lowest CRPS by gap length.", "text": ". ".join(segs) + ".",
                    "data": f"Per gap length, the cells that every method filling at least {100 * MIN_FILL:.0f}% of "
                            f"that gap length fills completely; same {scope['windows']} windows."})
    # 3. calibration of values
    cov = []
    for flux, (t, vals) in S.items():
        if not vals or not t["cells"]:
            continue
        ours = [m for m in OURS if vals.get(m)]
        oth = [m for m in methods if m not in OURS and vals.get(m)]
        lo = min(oth, key=lambda m: vals[m]["cov95"])
        cov.append(f"{flux}: " + ", ".join(f"{SHORT[m]} {vals[m]['cov95']:.3f}" for m in ours) +
                   f"; baselines {vals[lo]['cov95']:.3f} ({SHORT[lo]}) to "
                   f"{max(vals[m]['cov95'] for m in oth):.3f}")
    if cov:
        out.append({"head": "Calibration of half-hourly values (target 0.95).", "text": ". ".join(cov) + ".",
                    "data": "95% interval coverage on the same cells as the first finding."})
    # 4. sums
    ss = []
    for flux in ("NEE", "LE", "H"):
        g = sums.get((flux, "sum-30-day"))
        if not g:
            continue
        rng = {}
        for m in methods:
            xs = [p.get("cov95") for j, p in enumerate(g["pooled"][m]) if p and m in g["compared"][j]]
            if xs:
                rng[m] = (min(xs), max(xs))
        if not rng:
            continue
        top = min(rng, key=lambda m: np.mean([abs(x - 0.95) for x in rng[m]]))
        show = [top] + [m for m in OURS if m in rng and m != top]
        oth = [m for m in rng if m not in show]
        txt = f"{flux}: " + ", ".join(f"{SHORT[m]} {rng[m][0]:.2f} to {rng[m][1]:.2f}" for m in show)
        if oth:
            txt += f"; the other methods {min(rng[m][0] for m in oth):.2f} to {max(rng[m][1] for m in oth):.2f}"
        ss.append(txt)
    if ss:
        out.append({"head": "Calibration of 30-day sums (target 0.95).", "text": ". ".join(ss) + ".",
                    "data": "95% coverage of sums over the hidden measured half-hours of each 30-day block, range "
                            "over gap lengths; sums from each method's own samples, so methods with independent "
                            "half-hour samples give intervals that are too narrow."})
    # 5. point accuracy
    pt = []
    for flux, (t, vals) in S.items():
        if not vals or not t["cells"]:
            continue
        ms = [m for m in methods if vals.get(m) and np.isfinite(vals[m].get("r2", np.nan))]
        if not ms:
            continue
        r = max(ms, key=lambda m: vals[m]["r2"])
        w = min(ms, key=lambda m: vals[m]["wape"])
        s = f"{flux}: highest R² {SHORT[r]} {vals[r]['r2']:.3f}, lowest WAPE {SHORT[w]} {vals[w]['wape']:.1f}%"
        pt.append(s)
    if pt:
        out.append({"head": "Point accuracy of the predictive mean (secondary).", "text": ". ".join(pt) + ".",
                    "data": "Same cells as the first finding; R² pooled over sites; WAPE = 100 × Σ|ŷ − y| / Σ|y|."})
    return out


def chart_headline(gaps, methods):
    wins, total = {}, 0
    for flux, g in gaps.items():
        for j in range(len(g["families"])):
            vals = [(g["pooled"][m][j] or {}).get("crps") if m in g["compared"][j] else None for m in methods]
            b = best_index(vals, "crps")
            if b is not None:
                wins[methods[b]] = wins.get(methods[b], 0) + 1
                total += 1
    if not wins:
        return "No method is scored yet."
    order = sorted(wins, key=lambda m: -wins[m])
    return (f"{cap(SHORT[order[0]])} has the lowest CRPS in {wins[order[0]]} of {total} flux and gap-length "
            f"combinations" + ("; " + "; ".join(f"{SHORT[m]} in {wins[m]}" for m in order[1:]) if len(order) > 1
                               else "") + ".")


# ----------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", nargs="+")
    ap.add_argument("--out", default=None, help="results.json (or give it as the last positional argument)")
    ap.add_argument("--title", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--truth", default=None, help="bench/truth_stats.py output, for parts without truth sums")
    ap.add_argument("--split", default="validation", help="the site split the windows come from")
    ap.add_argument("--keep-other", action="store_true", help="accepted for compatibility; the file holds one scope")
    a = ap.parse_args()
    if a.out is None and a.scores[-1].endswith(".json"):
        a.out = a.scores.pop()
    d = pd.concat([pd.read_parquet(p) for p in a.scores], ignore_index=True)
    d = with_truth(d, pd.read_parquet(a.truth) if a.truth else None)
    methods = [m for m in METHOD_ORDER if m in set(d.method)] + sorted(set(d.method) - set(METHOD_ORDER))
    n_win = d[["site", "window"]].drop_duplicates()
    scope = {"split": a.split, "windows": int(len(n_win)), "sites": int(n_win.site.nunique()),
             "site_list": sorted(n_win.site.unique()), "regimes": sorted(d.regime.unique()),
             "families": [f for f in ORDER if f in set(d.family)] + sorted(set(d.family) - set(ORDER)),
             "methods": len(methods)}
    v = d[(d.kind == "values") & (d.gap_class == "all")]
    summ, gaps, sums, by_gap_tables, sum_tables, fills, chart = [], {}, {}, {}, {}, {}, {}
    for flux in FLUXES:
        if v[v.variable == flux].empty:
            continue
        u = UNITS[flux]
        fams = [f for f in scope["families"] if f in set(v[v.variable == flux].family)]
        t, vals, _ = summary_table(d, v, flux, methods, u)
        summ.append((t, vals))
        g = by_gap(d, v, flux, methods, fams, "values", 1.0)
        gaps[flux] = g
        note = (f"Half-hourly values; units {u}. Each column compares the methods on the cells they all fill "
                f"completely; n is the number of values in the column.")
        by_gap_tables[flux] = {mt: table_from(g, methods, mt, u, f"{flux}: {METRICS[mt]['long']} of half-hourly values",
                                              f"{flux}", note, "values") for mt in VALUE_METRICS}
        fills[flux] = fill_table(v, flux, methods, fams)
        chart[flux] = {"families": [gap_label(f) for f in fams], "units": u,
                       "n": g["n"], "series": [{"method": m, "label": LABEL.get(m, m), "short": SHORT.get(m, m),
                                                "ours": m in OURS,
                                                "crps": [(p or {}).get("crps") if m in g["compared"][j] else None
                                                         for j, p in enumerate(g["pooled"][m])]} for m in methods]}
        su, f = SUM_UNITS[flux]
        for kind, kl in SUM_KINDS.items():
            if d[(d.kind == kind) & (d.variable == flux)].empty:
                continue
            gs = by_gap(d, v, flux, methods, fams, kind, f)
            sums[(flux, kind)] = gs
            note = (f"Sums of the half-hourly values over the hidden measured half-hours of each {kl} block, in {su}; "
                    f"n is the number of block sums. Sums are formed per sample; for methods whose samples are "
                    f"independent per half-hour (climatology, MDS, RF, XGBoost, SAITS, Chronos-2, FluxFill v0) the "
                    f"spread of a sum is the independence spread.")
            sum_tables.setdefault(flux, {})[kind] = {
                mt: table_from(gs, methods, mt, su, f"{flux}: {METRICS[mt]['long']} of {kl} sums",
                               f"{kl} sums of {flux}", note, kind) for mt in SUM_METRICS}
    out = {
        "title": a.title, "label": cap(a.split), "note": a.note, "scope": scope,
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "methods": [{"id": m, "label": LABEL.get(m, m), "short": SHORT.get(m, m), "ours": m in OURS} for m in methods],
        "metrics": METRICS, "min_fill": MIN_FILL,
        "findings": findings(summ, gaps, sums, methods, scope),
        "summary": [t for t, _ in summ],
        "by_gap": by_gap_tables, "sums": sum_tables, "fill": fills,
        "chart": {"headline": chart_headline(gaps, methods), "fluxes": chart},
        "fluxes": [{"id": f, "name": NAMES[f], "units": UNITS[f], "sum_units": SUM_UNITS[f][0]} for f in FLUXES
                   if f in chart],
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(clean(out), separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    print(f"wrote {a.out}: {len(summ)} summary tables, {sum(len(x) for x in by_gap_tables.values())} gap tables, "
          f"{len(out['findings'])} findings")
    for f in out["findings"]:
        print("-", f["head"], f["text"])


def clean(o):
    """NaN and infinities to null, numpy scalars to Python, for strict JSON."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(x) for x in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


if __name__ == "__main__":
    main()
