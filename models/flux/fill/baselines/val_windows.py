"""Validation windows for the baseline development runs (validation sites only; never test sites).

Rule, fixed before any baseline was scored:
1. Sites: the validation sites of `model/data.load_split` (the frozen `towers/split.csv` once it exists, else the
   provisional 20 % hold-out by SHA-256 of the site id, the rule the FluxFill model uses).
2. Candidate windows: the model's validation windows (`model/evaluate.val_windows`, two 180-day windows per site,
   evenly spaced over the usable starts), so that FluxFill and the baselines are scored on the same windows.
3. A window is kept when the site's other calendar years (the runner's `context`) hold at least 365 days' worth
   (17,520) of measured NEE half-hours, so that per-site methods have a context to fit and to calibrate on.
4. Scoring sites: the kept sites ordered by SHA-256 of `fluxfill-baselines-val-v0|<site>`, first `--n-sites`; all
   their kept windows. US-Ha1 (the only validation site with Clay embeddings) is always included when it qualifies.
5. Calibration sites: every other validation site. They are used only for spreads and early stopping of the
   cross-site imputers (SAITS, CSDI), never scored.

Usage: python val_windows.py --out /home/photon/4tb/fluxfill/bench/val/windows_val.json [--n-sites 12]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model"))
sys.path.insert(0, str(HERE.parent / "bench"))

MIN_CONTEXT_NEE = 17520
ALWAYS = ("US-Ha1",)


def order_key(site: str) -> int:
    return int.from_bytes(hashlib.sha256(f"fluxfill-baselines-val-v0|{site}".encode()).digest()[:8], "little")


def context_nee(rec, start: str) -> int:
    """Measured NEE half-hours of the record outside the calendar years the window touches."""
    import pandas as pd
    from data import VARS
    s = pd.Timestamp(start)
    years = set(range(s.year, (s + pd.Timedelta(days=179)).year + 1))
    t = rec.t0 + pd.to_timedelta(np.arange(rec.values.shape[1]) * 30, unit="min")
    keep = ~np.isin(t.year, list(years))
    return int(np.isfinite(rec.values[VARS.index("NEE"), keep]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sites", type=int, default=12)
    ap.add_argument("--per-site", type=int, default=2)
    a = ap.parse_args()
    from data import load_records, load_split, table_dir
    from evaluate import val_windows
    specs, split_name = val_windows(a.per_site)
    sites = sorted({s["site"] for s in specs})
    recs = load_records(sites, verbose=False)
    kept = [dict(s, context_nee=context_nee(recs[s["site"]], s["start"])) for s in specs if s["site"] in recs]
    kept = [s for s in kept if s["context_nee"] >= MIN_CONTEXT_NEE]
    ok_sites = sorted({s["site"] for s in kept}, key=order_key)
    score = [s for s in ALWAYS if s in ok_sites]
    score += [s for s in ok_sites if s not in score][: max(0, a.n_sites - len(score))]
    all_val = sorted(load_split(sorted(p.stem for p in table_dir().glob("*.parquet")))[0].items())
    calib = [s for s, v in all_val if v == "val" and s not in score]
    out = {"split": split_name, "rule": __doc__.split("Usage")[0].strip(),
           "windows": [s for s in kept if s["site"] in score], "scoring_sites": score, "calibration_sites": calib}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    # the runner reads a plain list of {site, start, table}
    Path(a.out).with_name(Path(a.out).stem + "_list.json").write_text(
        json.dumps([{k: s[k] for k in ("site", "start", "table")} for s in out["windows"]], indent=1))
    print(f"{split_name} split: {len(out['windows'])} windows at {len(score)} scoring sites; "
          f"{len(calib)} calibration sites")
    for s in out["windows"]:
        print(s["site"], s["start"], s["context_nee"])


if __name__ == "__main__":
    main()
