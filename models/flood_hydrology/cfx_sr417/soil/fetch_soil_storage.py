#!/usr/bin/env python3
"""Fetch the SSURGO fields needed to give infiltration a finite capacity.

Why this exists
---------------
The solver's Horton infiltration is a RATE that decays to `fc` and then stays there
indefinitely, with nothing limiting the cumulative volume. Over a 72-hour event at site3's
`fc_eff` of 23.3 mm/hr that is 1,678 mm of capacity against 392 mm of rain, so essentially all
rainfall infiltrates. Measured against the Gee Creek gauge for Hurricane Ian, the simulated
runoff coefficient was 1.0% where the observed was 19.6% — a 19.3x gap that fully accounts for
the 13x peak-discharge shortfall.

Real soil has finite storage. Once the profile fills, infiltration stops and further rainfall
becomes runoff. That is saturation-excess runoff, and it dominates over infiltration-excess on
flat terrain with a shallow water table — exactly this landscape.

This is the "maximum deficit" parameter of the HEC-RAS / HEC-HMS Deficit and Constant loss
method, conventionally computed as effective porosity x active layer depth and then calibrated
against observations.

Fields fetched (SSURGO `muaggatt`, map-unit aggregated attributes)
-----------------------------------------------------------------
wtdepannmin     depth to the seasonal-high water table, annual minimum [cm]. This sets the
                active layer: the soil above it is what can still accept water.
wtdepaprjunmin  the same for April-June, kept for reference.
aws0150wta      available water storage 0-150 cm [cm]. Plant-available water only (field
                capacity minus wilting point), so it is a LOWER bound on drainable storage and
                is recorded for comparison rather than used directly.
brockdepmin     depth to bedrock [cm], where present.

Usage:
    python3 soil/fetch_soil_storage.py --site main_aoi
    python3 soil/fetch_soil_storage.py --site site3
"""
import os
import sys
import csv
import json
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, PROJ_DIR)

from ssurgo_download import query_ssurgo_tabular   # noqa: E402

FIELDS = ["mukey", "muname", "wtdepannmin", "wtdepaprjunmin", "brockdepmin",
          "aws025wta", "aws050wta", "aws0100wta", "aws0150wta"]


def fetch_storage(mukeys):
    """Query muaggatt for the storage-relevant attributes of the given map units."""
    in_list = ",".join(f"'{m}'" for m in mukeys)
    sql = f"SELECT {', '.join(FIELDS)} FROM muaggatt WHERE mukey IN ({in_list})"
    result = query_ssurgo_tabular(sql)
    if not result:
        return None
    rows = result["Table"] if isinstance(result, dict) else result
    header, *data = rows
    return [dict(zip(header, r)) for r in data]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="main_aoi", help="site name from the registry")
    args = ap.parse_args()

    try:
        import site_registry
        data_root = site_registry.data_root_for(args.site)
    except Exception:
        data_root = PROJ_DIR if args.site == "main_aoi" else os.path.join(PROJ_DIR, "site3_gee_creek")

    soil_dir = os.path.join(data_root, "soil", "data")
    params_path = os.path.join(soil_dir, "soil_parameters.json")
    if not os.path.exists(params_path):
        raise SystemExit(f"{params_path} not found — run soil/ssurgo_download.py --site {args.site} first.")

    mukeys = list(json.load(open(params_path)).keys())
    print(f"Fetching soil storage attributes for {len(mukeys)} map units ({args.site}) …")

    rows = fetch_storage(mukeys)
    if rows is None:
        raise SystemExit("SDA query failed — no storage table written; the solver will fall back "
                         "to unbounded infiltration.")

    out = os.path.join(soil_dir, "soil_storage.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_wt = sum(1 for r in rows if (r.get("wtdepannmin") or "").strip())
    n_surface = sum(1 for r in rows if (r.get("wtdepannmin") or "").strip() in ("0", "0.0"))
    print(f"  wrote {out}  ({len(rows)} map units)")
    print(f"  water table reported for {n_wt}/{len(rows)}; {n_surface} sit at the surface "
          f"(depressional — zero storage, they run off immediately)")


if __name__ == "__main__":
    main()
