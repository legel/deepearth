"""Cross-variable inference: hide one group of tower channels for a block and reconstruct it from everything else.

For each validation window and each channel group (radiation, humidity and water, wind and momentum, soil heat,
temperature, fluxes), the group's tower rows are hidden for one 30-day block (start seeded per window and group, as
the benchmark's M2), every other row stays visible, and FluxFill's marginal predictive distribution of the hidden,
measured values is scored (fair CRPS from samples) against climatology on the same values. With
`--hide-reanalysis` the reanalysis rows are hidden as well for the whole window, so the group is inferred from the
other tower channels alone (and from site memory), not from ERA5's copy of the same quantity.

The method protocol of bench/run.py is used throughout (visible copy of the window, context = the site's other
calendar years), validation sites only.

Usage: python cross_var.py --ckpt CK --windows list.json --out rows.parquet [--hide-reanalysis] [--extra-tower]
       [--samples 64] [--device cpu]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bench"))
if "--extra-tower" in sys.argv:                  # an extended-channel checkpoint needs the extended row layout
    import os
    os.environ.setdefault("FLUXFILL_CHANNELS", "extended")

from masks import contiguous_block, seed_for  # noqa: E402
from metrics import crps_samples  # noqa: E402
from windows import EXTRA_TOWER_VARS, TOWER_VARS, Window, load_window, lst_index, with_extra_columns  # noqa: E402

GROUPS = {
    "radiation": ("SW_IN", "LW_IN", "SW_OUT", "LW_OUT", "NETRAD", "PPFD_IN", "PPFD_OUT"),
    "humidity_water": ("VPD", "RH", "P", "SWC_1", "SWC_2", "SWC_3", "SWC_4", "SWC_5", "WTD"),
    "wind_momentum": ("WS", "USTAR", "WD_SIN", "WD_COS"),
    "soil_heat": ("G", "TS_1", "TS_2", "TS_3", "TS_4", "TS_5"),
    "temperature": ("TA",),
    "fluxes": ("NEE", "LE", "H", "FCH4"),
    "co2_pressure": ("CO2", "PA"),
}
BLOCK_DAYS = 30


def main():
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hide-reanalysis", action="store_true")
    ap.add_argument("--extra-tower", action="store_true")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    from method import FluxFill
    from run import Climatology
    specs = json.loads(Path(a.windows).read_text())
    specs = specs["windows"] if isinstance(specs, dict) else specs
    tvars = TOWER_VARS + (EXTRA_TOWER_VARS if a.extra_tower else ())
    ff = FluxFill(a.ckpt, mode="marginal", n_samples=a.samples, device=a.device)
    clim = Climatology()
    rows = []
    for spec in specs:
        tab = pd.read_parquet(spec["table"])
        if a.extra_tower:
            tab = with_extra_columns(tab)
        w = load_window(tab, spec["site"], spec["start"], tower_vars=tvars)
        idx = lst_index(tab)
        years = set(range(w.t[0].year, w.t[-1].year + 1))
        ctx = {"table": tab[~idx.year.isin(list(years))], "index": idx[~idx.year.isin(list(years))]}
        V, T = w.values.shape
        rean = w.rows("reanalysis")
        for g, members in GROUPS.items():
            rws = [i for i, k in enumerate(w.keys) if k in members and w.provenance[i] == "tower"]
            if not rws:
                continue
            mk = contiguous_block(rws, V, T, BLOCK_DAYS, np.random.default_rng(seed_for(w.site, spec["start"], f"X{g}")))
            if a.hide_reanalysis and rean:
                mk[rean] = True
            vis = Window(w.site, w.start, w.t, w.keys, np.where(mk, np.nan, w.values), w.measured & ~mk,
                         w.provenance, w.meta)
            s_ff = ff.predict(vis, mk, ctx)
            s_cl = clim.predict(vis, mk, ctx)
            for i in rws:
                sel = mk[i] & w.measured[i]
                if sel.sum() < 5:
                    continue
                y = w.values[i, sel]
                c_ff = crps_samples(s_ff[:, i, sel], y)
                ok = np.isfinite(s_cl[:, i, sel]).all(0)
                c_cl = crps_samples(s_cl[:, i, sel][:, ok], y[ok]) if ok.any() else np.array([np.nan])
                rows.append({"model": a.name or Path(a.ckpt).parent.name, "site": w.site, "window": spec["start"],
                             "group": g, "variable": w.keys[i], "hide_reanalysis": a.hide_reanalysis,
                             "n": int(sel.sum()), "crps": float(c_ff.mean()),
                             "crps_on_clim_rows": float(c_ff[ok].mean()) if ok.any() else np.nan,
                             "crps_climatology": float(c_cl.mean()), "n_clim": int(ok.sum())})
        print(spec["site"], spec["start"], len(rows), flush=True)
        pd.DataFrame(rows).to_parquet(a.out)
    d = pd.DataFrame(rows)
    d["skill"] = 1 - d.crps_on_clim_rows / d.crps_climatology
    print(d.groupby(["group", "variable"]).agg(n=("n", "sum"), skill=("skill", "median")).round(3).to_string())


if __name__ == "__main__":
    main()
