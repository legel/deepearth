"""Freeze the FluxFill-Bench test release: windows, masks and their hashes (BENCHMARK.md section 9, rule 1).

Test windows: up to 4 usable 180-day windows per test site (towers/split.csv), one per start season where
available, starting on or after 2015-01-01. Within a season, the window with the most measured flux is chosen, so
every scored value is a real measurement. Masks: every family in `run.masks_for` (M2 blocks, M3 transplanted
outages, M4 reanalysis dropout), generated from fixed seeds.

The manifest records the SHA-256 of split.csv, of the window list, and of every mask array. A later run recomputes the
masks and refuses to score if a hash differs. Writes <out>/windows_test.json and <out>/MANIFEST.json.

Usage: python freeze.py <towers dir> <out dir>
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run import masks_for  # noqa: E402
from windows import load_window  # noqa: E402

FAMILIES = ("M2", "M3", "M4-reanalysis")
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
          9: "SON", 10: "SON", 11: "SON"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def choose(win: pd.DataFrame, test_sites) -> pd.DataFrame:
    w = win[win.usable & win.site_id.isin(test_sites) & (pd.to_datetime(win.start_lst) >= "2015-01-01")].copy()
    w["season"] = pd.to_datetime(w.start_lst).dt.month.map(SEASON)
    w["flux"] = w[["NEE", "LE", "H"]].fillna(0).sum(axis=1) + w["FCH4"].fillna(0)
    return (w.sort_values("flux", ascending=False).groupby(["site_id", "season"], as_index=False).head(1)
             .sort_values(["site_id", "start_lst"]))


def main():
    towers, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(towers / "split.csv")
    test = set(split[split.split == "test"].site_id)
    win = pd.read_parquet(towers / "windows.parquet")
    sel = choose(win, test)
    specs, masks = [], {}
    for r in sel.itertuples():
        f = towers / "hh" / f"{r.site_id}.parquet"
        start = str(pd.Timestamp(r.start_lst).date())
        w = load_window(pd.read_parquet(f), r.site_id, start)
        m = masks_for(w, FAMILIES)
        key = f"{r.site_id}|{start}"
        masks[key] = {name: sha(np.packbits(a).tobytes()) for name, a in sorted(m.items())}
        if "M3_donor" in w.meta:
            masks[key]["M3_donor"] = w.meta["M3_donor"]
        specs.append({"site": r.site_id, "start": start, "table": str(f), "season": r.season})
    wl = json.dumps(specs, indent=1).encode()
    (out / "windows_test.json").write_bytes(wl)
    manifest = {"version": "0.1", "split_sha256": sha((towers / "split.csv").read_bytes()),
                "windows_sha256": sha(wl), "families": FAMILIES, "n_windows": len(specs),
                "n_sites": len({s["site"] for s in specs}), "masks": masks}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"froze {len(specs)} test windows at {manifest['n_sites']} test sites; manifest {out / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
