"""Independent validation on VegBank plots for every rendered species, eastern US included. Each range card is
decoded only at the plot cells (``codec.decode_cells``), so no map needs to be stored. Scored over all plots
in CONUS (cells outside the calibration area count as unsuitable / absent) for species with >= 20 presence
plots: AUC of suitability, TSS of the P5 and replicate-vote binaries; Daru's published raster at the same plots
where Dryad has one. Range cards are read from the configured directories per mode (``per_species.range_cards``).
Writes work/natives/vegbank_validation.csv.

usage: validate_vegbank.py [--modes daru,occurrences] [--config configs/conus.json]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import codec, config, validate  # noqa: E402
from ranges.predictors import ConusStack  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default="daru,occurrences")
    ap.add_argument("--min-presence", type=int, default=20)
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = config.load(a.config)
    R = cfg.root
    truth = validate.PlotTruth(vegbank_dir=R / "raw/validation/vegbank", wcvp_dir=R / "raw/wcvp")
    locs = truth.vb_locs.reset_index(drop=True)
    print(f"{len(locs)} VegBank plots usable; {len(truth.vb_sp)} species recorded", flush=True)
    stack = ConusStack(R / "work/conus240")
    t = stack.profile["transform"]
    x, y = Transformer.from_crs(4326, 5070, always_xy=True).transform(locs.lon.values, locs.lat.values)
    rows, cols = np.floor((t.f - y) / -t.e).astype(np.int64), np.floor((x - t.c) / t.a).astype(np.int64)
    daru = {p.stem for p in (R / "daru_ref_all/rasters").glob("*.tif")}
    out = []
    for mode in a.modes.split(","):
        for card in sorted(c for d in cfg["per_species"]["range_cards"].get(mode, [])
                           for c in cfg.path(d).glob("*/*.rangecard")):
            name = card.stem.replace("_", " ")
            tr = truth.vegbank(name)
            if tr is None or tr.present.sum() < a.min_presence:
                continue
            yv = tr.present.values
            v = codec.decode_cells(card, stack, rows, cols)
            s = np.where(v["suitability"] > 0, (v["suitability"].astype(float) - 1) / 254, 0.0)
            rec = {"mode": mode, "species": name, "n_presence": int(yv.sum()), "n_plots": len(yv),
                   "auc": validate._scores(yv, v["binary_p5"] == 2, s).get("AUC"),
                   "tss_p5": validate._scores(yv, v["binary_p5"] == 2, None)["TSS"],
                   "tss_vote": validate._scores(yv, v["binary_vote"] == 2, None)["TSS"]}
            if name in daru:
                dr = validate._sample(R / f"daru_ref_all/raw_rasters/{name}.tif", locs.lon.values, locs.lat.values)
                db = validate._sample(R / f"daru_ref_all/rasters/{name}.tif", locs.lon.values, locs.lat.values)
                rec.update(daru_auc=validate._scores(yv, np.nan_to_num(db) == 1, np.nan_to_num(dr)).get("AUC"),
                           daru_tss=validate._scores(yv, np.nan_to_num(db) == 1, None)["TSS"])
            out.append(rec)
            print(rec, flush=True)
    d = pd.DataFrame(out)
    d.to_csv(R / "work/natives/vegbank_validation.csv", index=False)
    for mode, g in d.groupby("mode"):
        print(f"{mode}: {len(g)} species, median AUC {g.auc.median():.3f}, TSS P5 {g.tss_p5.median():.3f}, vote {g.tss_vote.median():.3f}")
        b = g.dropna(subset=["daru_auc"]) if "daru_auc" in g else g.iloc[0:0]
        if len(b):
            print(f"   with Daru maps ({len(b)}): AUC {b.auc.median():.3f} vs Daru {b.daru_auc.median():.3f} (better for {(b.auc > b.daru_auc).mean():.0%}); "
                  f"TSS P5 {b.tss_p5.median():.3f} / vote {b.tss_vote.median():.3f} vs Daru {b.daru_tss.median():.3f}")


if __name__ == "__main__":
    main()
