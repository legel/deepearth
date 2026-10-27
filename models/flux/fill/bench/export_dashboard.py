"""Export benchmark windows, masks and method predictions as compact JSON for the dashboard's Windows tab.

Usage: python export_dashboard.py <site_table.parquet> <site> <window_start> <out_dir> [--method climatology]
Values are rounded to 4 significant figures. Predictions are stored per masked span only (`a` = first index):
{"a", "mean", "q025", "q10", "q90", "q975"}; the page expands them onto the window.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "baselines"))

import climatology  # noqa: E402
from run import Climatology  # noqa: E402
from windows import Window  # noqa: E402
from masks import contiguous_block, seed_for, STEPS_PER_DAY  # noqa: E402
from metrics import coverage_samples, crps_samples  # noqa: E402
from windows import LABELS, UNITS, load_window, lst_index  # noqa: E402

FLUX_ROWS = ("NEE", "LE", "H", "FCH4")


def r4(a):
    return [None if not np.isfinite(x) else float(f"{x:.4g}") for x in a]


def spans(row_mask: np.ndarray):
    e = np.flatnonzero(np.diff(np.concatenate(([0], row_mask.astype(np.int8), [0]))))
    return [[int(a), int(b)] for a, b in zip(e[::2], e[1::2])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("site"); ap.add_argument("start"); ap.add_argument("out")
    ap.add_argument("--days", type=int, nargs="+", default=[7, 30])
    ap.add_argument("--note", default="")
    ap.add_argument("--methods", nargs="+", default=["climatology", "mds"])
    ap.add_argument("--lat", type=float, default=0.0); ap.add_argument("--lon", type=float, default=0.0)
    a = ap.parse_args()
    tab = pd.read_parquet(a.table)
    from run import method_registry
    registry = method_registry()
    methods = []
    for mname in a.methods:
        if mname == "mds":
            from mds import MDS
            methods.append(MDS(lat=a.lat, lon=a.lon, utc_offset=round(a.lon / 15)))
        else:
            methods.append(registry[mname]())
    w = load_window(tab, a.site, a.start)
    tower = w.rows("tower")
    V, T = w.values.shape
    out = {"site": w.site, "start": str(w.start.date()), "end": str((w.t[-1]).date()),
           "t": [x.strftime("%Y-%m-%d %H:%M") for x in w.t],
           "day_labels": [(w.start + pd.Timedelta(days=d)).strftime("%d %b") for d in range(181)],
           "variables": [], "masks": {}, "predictions": {}, "scores": {}, "note": a.note}
    for i, k in enumerate(w.keys):
        base = k.replace("ERA5_", "")
        out["variables"].append({"key": k, "label": LABELS.get(base, base) + (" (ERA5)" if k.startswith("ERA5_") else ""),
                                 "units": UNITS.get(base, ""), "provenance": w.provenance[i], "values": r4(w.values[i])})
    for days in a.days:
        name = f"M2 {days} d, all tower channels"
        # The block hides every tower channel for its whole length (as an outage does); it is shaded and filled in
        # full, and only its measured values are scored.
        m = contiguous_block(tower, V, T, days, np.random.default_rng(seed_for(a.site, a.start, "M2", days)))
        out["masks"][name] = {w.keys[i]: spans(m[i]) for i in tower if (m[i] & w.measured[i]).any()}
        visible = Window(w.site, w.start, w.t, w.keys, np.where(m, np.nan, w.values), w.measured & ~m, w.provenance)
        idx = lst_index(tab)
        years = list(range(w.t[0].year, w.t[-1].year + 1))
        ctx = {"table": tab[~idx.year.isin(years)], "index": idx[~idx.year.isin(years)]}
        for meth in methods:
            s_all = meth.predict(visible, m, ctx)
            preds, scores = {}, {}
            for i in tower:
                if not (m[i] & w.measured[i]).any():
                    continue
                k, s = w.keys[i], s_all[:, i]
                if not np.isfinite(s[:, m[i]]).any():
                    scores[k] = {"filled": 0.0}
                    continue
                segs = []
                for lo, hi in spans(m[i]):
                    ss = s[:, lo:hi]
                    with np.errstate(all="ignore"):
                        q = np.nanquantile(ss, [0.025, 0.1, 0.9, 0.975], axis=0)
                        mu = np.nanmean(ss, axis=0)
                    segs.append({"a": lo, "mean": r4(mu), "q025": r4(q[0]), "q10": r4(q[1]), "q90": r4(q[2]), "q975": r4(q[3])})
                preds[k] = segs
                tgt = m[i] & w.measured[i]
                hid = tgt & np.isfinite(s).all(axis=0)
                if hid.any():
                    y, ss = w.values[i, hid], s[:, hid]
                    cov = coverage_samples(ss, y, (0.8, 0.95))
                    scores[k] = {"crps": float(f"{crps_samples(ss, y).mean():.4g}"),
                                 "mae": float(f"{np.abs(ss.mean(0) - y).mean():.4g}"),
                                 "cov80": round(cov[0.8], 3), "cov95": round(cov[0.95], 3), "n": int(hid.sum()),
                                 "filled": round(float(hid.sum() / tgt.sum()), 3)}
            out["predictions"].setdefault(meth.name, {})[name] = preds
            out["scores"].setdefault(meth.name, {})[name] = scores
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    fn = f"{a.site}_{out['start']}.json"
    (od / fn).write_text(json.dumps(out, separators=(",", ":")))
    idx_path = od / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {"windows": []}
    idx["windows"] = [x for x in idx["windows"] if x["file"] != fn] + [
        {"site": a.site, "start": out["start"], "end": out["end"], "file": fn, "masks": list(out["masks"]),
         "methods": list(out["predictions"])}]
    idx_path.write_text(json.dumps(idx, indent=1))
    print(f"wrote {od / fn} ({(od / fn).stat().st_size / 1e6:.2f} MB), {len(w.keys)} variables")


if __name__ == "__main__":
    main()
