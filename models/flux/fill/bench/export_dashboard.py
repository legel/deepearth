"""Export benchmark windows, masks and method predictions as compact JSON for the dashboard's Windows tab.

Usage: python export_dashboard.py <site_table.parquet> <site> <window_start> <out_dir> --split validation|test
           [--masks M1 M2-6h M2-1d M2-7d M2-30d M2-90d M4-no-reanalysis+M2-30d M5] [--methods climatology mds fluxfill]
The masks are those of the benchmark (run.masks_for, the same seeds as the release). Every method reads the window
with the hidden values set to NaN and the site's other years as context (regime R1); the window's own years are never
in the context. Values are rounded to 4 significant figures. Predictions are stored per masked span only (`a` = first
index): {"a", "mean", "q025", "q10", "q90", "q975"}; the page expands them onto the window. fluxfill reads its
checkpoint from FLUXFILL_CKPT (run.fluxfill_method).
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "baselines"))

from windows import Window  # noqa: E402
from metrics import coverage_samples, crps_samples  # noqa: E402
from windows import LABELS, UNITS, load_window, lst_index  # noqa: E402

FLUX_ROWS = ("NEE", "LE", "H", "FCH4")


def r4(a):
    return [None if not np.isfinite(x) else float(f"{x:.4g}") for x in a]


def spans(row_mask: np.ndarray):
    e = np.flatnonzero(np.diff(np.concatenate(([0], row_mask.astype(np.int8), [0]))))
    return [[int(a), int(b)] for a, b in zip(e[::2], e[1::2])]


MASK_LABELS = {"M1": "M1: short gaps in the fluxes (distribution of real gaps, Irvin et al. 2021)",
               "M2-2h": "M2: 2 h, all tower channels", "M2-6h": "M2: 6 h, all tower channels",
               "M2-1d": "M2: 1 d, all tower channels", "M2-7d": "M2: 7 d, all tower channels",
               "M2-30d": "M2: 30 d, all tower channels", "M2-90d": "M2: 90 d, all tower channels",
               "M2-180d": "M2: 180 d, all tower channels (whole window)",
               "M3": "M3: a real outage transplanted from another tower",
               "M4-no-reanalysis+M2-30d": "M4: reanalysis rows removed, plus 30 d of all tower channels",
               "M5": "M5: mixed per-variable blocks, plus one whole modality hidden"}
MODALITY_LABELS = {"tower_met": "tower meteorology", "reanalysis": "reanalysis", "fluxes": "all fluxes"}
FAMILY_OF = {"M2-2h": "M2-sub", "M2-6h": "M2-sub", "M2-1d": "M2", "M2-7d": "M2", "M2-30d": "M2", "M2-90d": "M2",
             "M2-180d": "M2", "M4-no-reanalysis+M2-30d": "M4-reanalysis"}
DEFAULT_MASKS = ("M1", "M2-6h", "M2-1d", "M2-7d", "M2-30d", "M2-90d", "M4-no-reanalysis+M2-30d", "M5")


def mask_label(name: str, w: Window) -> str:
    lab = MASK_LABELS.get(name, name)
    if name == "M5" and w.meta.get("M5_modality"):
        lab += f" ({MODALITY_LABELS.get(w.meta['M5_modality'], w.meta['M5_modality'])})"
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("site"); ap.add_argument("start"); ap.add_argument("out")
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--masks", nargs="+", default=list(DEFAULT_MASKS))
    ap.add_argument("--note", default="")
    ap.add_argument("--methods", nargs="+", default=["climatology", "mds", "fluxfill"])
    ap.add_argument("--lat", type=float, default=0.0); ap.add_argument("--lon", type=float, default=0.0)
    a = ap.parse_args()
    from run import masks_for, method_registry
    tab = pd.read_parquet(a.table)
    registry = method_registry()
    methods = []
    for mname in a.methods:
        if mname == "mds":
            from mds import MDS
            methods.append(MDS(lat=a.lat, lon=a.lon, utc_offset=round(a.lon / 15)))
        else:
            methods.append(registry[mname]())
    w = load_window(tab, a.site, a.start)
    fams = {FAMILY_OF.get(n, n) for n in a.masks}
    masks = {}
    for fam in fams:
        try:
            masks.update({n: m for n, m in masks_for(w, [fam]).items() if n in a.masks})
        except FileNotFoundError as e:      # M1 record masks exist for the release sites only
            print(f"{a.site} {a.start}: {fam} skipped ({e})", flush=True)
    out = {"site": w.site, "start": str(w.start.date()), "end": str((w.t[-1]).date()), "split": a.split,
           "t": [x.strftime("%Y-%m-%d %H:%M") for x in w.t],
           "day_labels": [(w.start + pd.Timedelta(days=d)).strftime("%d %b") for d in range(181)],
           "variables": [], "masks": {}, "predictions": {}, "scores": {}, "note": a.note,
           "context_years": sorted({int(y) for y in lst_index(tab).year.unique()} - set(range(w.t[0].year, w.t[-1].year + 1)))}
    for i, k in enumerate(w.keys):
        base = k.replace("ERA5_", "")
        out["variables"].append({"key": k, "label": LABELS.get(base, base) + (" (ERA5)" if k.startswith("ERA5_") else ""),
                                 "units": UNITS.get(base, ""), "provenance": w.provenance[i], "values": r4(w.values[i])})
    idx = lst_index(tab)
    years = list(range(w.t[0].year, w.t[-1].year + 1))
    ctx = {"table": tab[~idx.year.isin(years)], "index": idx[~idx.year.isin(years)]}
    for mkey in a.masks:
        if mkey not in masks:
            continue
        m = masks[mkey]
        name = mask_label(mkey, w)
        # A hidden span is shaded and filled in full; only its measured values are scored.
        out["masks"][name] = {w.keys[i]: spans(m[i]) for i in range(len(w.keys)) if m[i].any()}
        visible = Window(w.site, w.start, w.t, w.keys, np.where(m, np.nan, w.values), w.measured & ~m, w.provenance, w.meta)
        rows = [i for i in range(len(w.keys)) if (m[i] & w.measured[i]).any()]
        for meth in methods:
            s_all = meth.predict(visible, m, ctx)
            if os.environ.get("FLUXFILL_SAMPLES_DUMP") and mkey.startswith("M2") and "30 d" in name:   # raw samples,
                d = Path(os.environ["FLUXFILL_SAMPLES_DUMP"]); d.mkdir(parents=True, exist_ok=True)      # report figures
                np.savez_compressed(d / f"{w.site}_{str(w.start)[:10]}_{meth.name}_M2-30d.npz", samples=s_all[:, rows].astype(np.float32),
                                    keys=np.array([w.keys[i] for i in rows]), mask=m[rows])
            preds, scores = {}, {}
            for i in rows:
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
        print(f"{a.site} {a.start} {name}: done", flush=True)
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    fn = f"{a.site}_{out['start']}.json"
    (od / fn).write_text(json.dumps(out, separators=(",", ":")))
    idx_path = od / "index.json"
    ix = json.loads(idx_path.read_text()) if idx_path.exists() else {"windows": []}
    ix["windows"] = [x for x in ix["windows"] if x["file"] != fn] + [
        {"site": a.site, "start": out["start"], "end": out["end"], "split": a.split, "file": fn,
         "masks": list(out["masks"]), "methods": list(out["predictions"])}]
    idx_path.write_text(json.dumps(ix, indent=1))
    print(f"wrote {od / fn} ({(od / fn).stat().st_size / 1e6:.2f} MB), {len(w.keys)} variables")


if __name__ == "__main__":
    main()
