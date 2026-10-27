"""Add FluxFill predictions to dashboard windows (the format of bench/export_dashboard.py).

For each (site, start), the climatology export is written first if the window file does not exist yet; then the
FluxFill method fills the same M2 masks (7 and 30 days, all tower channels, benchmark seeds), and its mean, 2.5, 10,
90 and 97.5 % sample quantiles and scores are added under the method name. Run deploy.sh afterwards to publish.

Usage: python export.py --ckpt best.pt --windows SITE:YYYY-MM-DD ... [--mode marginal|joint] [--name fluxfill-v0]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bench"))

from data import table_dir  # noqa: E402
from export_dashboard import r4, spans  # noqa: E402
from masks import contiguous_block, seed_for  # noqa: E402
from method import FluxFill  # noqa: E402
from metrics import coverage_samples, crps_samples  # noqa: E402
from windows import Window, load_window, lst_index  # noqa: E402

DASH = Path("/home/photon/4tb/fluxfill/bench/dashboard/windows")


def main():
    import torch
    torch.set_num_threads(2)            # machine rule: at most 2 CPU threads per job
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows", nargs="+", required=True, help="SITE:YYYY-MM-DD")
    ap.add_argument("--mode", default="marginal")
    ap.add_argument("--name", default="fluxfill-v0")
    ap.add_argument("--days", type=int, nargs="+", default=[7, 30])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--steps", type=int, default=16, help="generation rounds (joint mode)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    m = FluxFill(a.ckpt, mode=a.mode, n_samples=a.samples, name=a.name, steps=a.steps, chunk=8)
    for spec in a.windows:
        site, start = spec.split(":")
        path = table_dir() / f"{site}.parquet"
        fn = DASH / f"{site}_{start}.json"
        if not fn.exists():
            subprocess.run([sys.executable, str(HERE.parent / "bench" / "export_dashboard.py"), str(path), site, start,
                            str(DASH), "--days", *map(str, a.days), "--note", a.note, "--methods", "climatology"], check=True)
        out = json.loads(fn.read_text())
        if a.note:
            out["note"] = a.note
        tab = pd.read_parquet(path)
        idx = lst_index(tab)
        w = load_window(tab, site, start)
        years = set(range(w.t[0].year, w.t[-1].year + 1))
        context = {"table": tab[~idx.year.isin(list(years))], "index": idx[~idx.year.isin(list(years))]}
        tower = w.rows("tower")
        V, T = w.values.shape
        out["predictions"][a.name], out["scores"][a.name] = {}, {}
        for days in a.days:
            name = f"M2 {days} d, all tower channels"
            mk = contiguous_block(tower, V, T, days, np.random.default_rng(seed_for(site, start, "M2", days)))
            vis = Window(w.site, w.start, w.t, w.keys, np.where(mk, np.nan, w.values), w.measured & ~mk, w.provenance, w.meta)
            s = m.predict(vis, mk, context)
            preds, scores = {}, {}
            for i in tower:
                if not (mk[i] & w.measured[i]).any():
                    continue
                k = w.keys[i]
                segs = []
                for lo, hi in spans(mk[i]):
                    ss = s[:, i, lo:hi]
                    q = np.quantile(ss, [0.025, 0.1, 0.9, 0.975], axis=0)
                    segs.append({"a": lo, "mean": r4(ss.mean(0)), "q025": r4(q[0]), "q10": r4(q[1]), "q90": r4(q[2]),
                                 "q975": r4(q[3])})
                preds[k] = segs
                hid = mk[i] & w.measured[i]
                y, ss = w.values[i, hid], s[:, i, hid]
                cov = coverage_samples(ss, y, (0.8, 0.95))
                scores[k] = {"crps": float(f"{crps_samples(ss, y).mean():.4g}"),
                             "mae": float(f"{np.abs(ss.mean(0) - y).mean():.4g}"),
                             "cov80": round(cov[0.8], 3), "cov95": round(cov[0.95], 3), "n": int(hid.sum())}
            out["predictions"][a.name][name] = preds
            out["scores"][a.name][name] = scores
        fn.write_text(json.dumps(out, separators=(",", ":")))
        ip = DASH / "index.json"
        ix = json.loads(ip.read_text())
        for x in ix["windows"]:
            if x["file"] == fn.name:
                x["methods"] = list(out["predictions"])
        ip.write_text(json.dumps(ix, indent=1))
        print(f"{fn}: methods {list(out['predictions'])}", flush=True)


if __name__ == "__main__":
    main()
