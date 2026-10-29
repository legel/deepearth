"""Exact interval coverage and PIT of a checkpoint on the main validation windows (train.val_examples), CPU.

For a continuous predictive CDF F the central 95 % interval [F^-1(0.025), F^-1(0.975)] contains y exactly when
0.025 <= F(y) <= 0.975, so coverage and the PIT come from F(y) without sampling: for the single Student-t head
F(y) = T((y - loc) / scale; df); for the mixture heads F(y) = sum_k w_k T((y - loc_k) / scale_k; df_k). The exact
predictive standard deviation of a mixture is sqrt(sum_k w_k (scale_k^2 df_k / (df_k - 2) + loc_k^2) - mean^2).

Per hidden measured value it writes: variable, family (M2 block), window, the PIT, the exact predictive mean and
standard deviation (standardised units), the truth, and for gated heads the local-component weight, both component
scales and the distance to the nearest visible value of the variable.

Usage: python coverage_exact.py --ckpt CK --out rows.parquet [--imagery-sites ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sst

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import VARS, Stats, heldout_year, load_records, load_split, table_dir  # noqa: E402
from imagery import load_imagery  # noqa: E402
from net import FluxFillNet, nearest_visible_distance  # noqa: E402
from sampler import collate  # noqa: E402
import train  # noqa: E402


def main():
    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imagery-sites", default="")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    c = ck["config"]
    net = FluxFillNet(d=c["d"], depth=c["depth"], heads=c["heads"], memory=c.get("memory", False),
                      mem_residual=c.get("mem_residual", False), components=c.get("components", 1),
                      local_base=c.get("local_base", False), gated_base=c.get("gated_base", False),
                      gate_scale=c.get("gate_scale", False), edge_continue=c.get("edge_continue", False),
                      memory_skew=c.get("memory_skew", False), driver_anomaly=c.get("driver_anomaly", False),
                      memory_skew_fluxes_only=c.get("memory_skew_fluxes", False),
                      scale_anchor=c.get("scale_anchor", False), year_anomaly=c.get("year_anomaly", False),
                      sinr_place=c.get("sinr_place", -1), sinr_replace=c.get("sinr_replace", False), img_nll=c.get("img_nll", False))
    net.load_state_dict(ck["model"]); net.eval()
    stats = Stats.from_dict(c["stats"])
    sites = sorted(p.stem for p in table_dir().glob("*.parquet"))
    split, _ = load_split(sites)
    val = [s for s, v in split.items() if v == "val"]
    records = load_records(val, verbose=False)
    allowed = set(a.imagery_sites.split(",")) if a.imagery_sites else None
    imagery = {}
    for s, r in records.items():
        if allowed is None or s in allowed:
            im = load_imagery(s, r.utc_offset_h)
            if im is not None:
                imagery[s] = im
    vex = train.val_examples(records, val, stats, imagery, memory=c.get("memory", False),
                             memory_skew=c.get("memory_skew", False) or c.get("memory_skew_fluxes", False))
    rows = []
    for e in vex:
        b = collate([{k: v for k, v in e.items() if not k.startswith("_")}])
        with torch.no_grad():
            _, p = net(b)
        tm = e["tmask"]
        y = e["tgt"][tm]
        vi = np.nonzero(tm)[0]
        if len(p) == 3:
            loc, sc, df = (u[0].numpy().astype(np.float64)[tm] for u in p)
            F = sst.t.cdf((y - loc) / sc, df)
            mean, sd = loc, sc * np.sqrt(df / (df - 2))
            extra = {}
        else:
            loc, sc, df, lw = (u[0].numpy().astype(np.float64)[tm] for u in p)     # (n, M)
            w = np.exp(lw)
            F = (w * sst.t.cdf((y[:, None] - loc) / sc, df)).sum(1)
            mean = (w * loc).sum(1)
            sd = np.sqrt(np.maximum((w * (sc ** 2 * df / (df - 2) + loc ** 2)).sum(1) - mean ** 2, 0))
            extra = {"w_local": w[:, 0], "scale_local": sc[:, 0], "scale_memory": sc[:, 1],
                     "loc_local": loc[:, 0], "loc_memory": loc[:, 1]}
            if c.get("gated_base", False):
                extra["dist"] = nearest_visible_distance(b["obs"])[0].numpy()[tm]
        rows.append(pd.DataFrame({"window": "|".join(map(str, e["_id"])), "family": e["family"],
                                  "variable": np.array(VARS)[vi], "pit": F, "mean": mean, "sd": sd, "truth": y, **extra}))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(a.out, index=False)
    out["cov95"] = (out.pit >= 0.025) & (out.pit <= 0.975)
    print(f"{len(out)} values, {out.window.nunique()} windows; exact 95 % coverage {out.cov95.mean():.3f}")


if __name__ == "__main__":
    main()
