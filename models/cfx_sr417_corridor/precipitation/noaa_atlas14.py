"""
NOAA Atlas 14 — Precipitation Frequency Data
==============================================
Fetches IDF (Intensity-Duration-Frequency) curves for the Winter Garden FL
property from NOAA's Precipitation Frequency Data Server (PFDS / Atlas 14).

Provides rainfall depth [inches and mm] for:
    Durations   : 1-hr, 3-hr, 6-hr, 12-hr, 24-hr
    Return periods: 2, 5, 10, 25, 50, 100, 200, 500, 1000 years (AEP series)

Also generates SCS Type II synthetic design hyetographs for 1-hr and 12-hr
events at the 10-yr and 100-yr return periods — these are the rainfall inputs
for flood_sim.py.

Cross-reference with GSDR observed maxima via gsdr_intensity_matrix.py to
validate the Atlas 14 estimates against real gauge records.

Outputs (saved under precipitation/data/):
    atlas14_idf_{lat}_{lon}.csv           — raw IDF table (depth, in + mm)
    atlas14_hyetograph_1hr_100yr.csv      — time series rainfall [mm/5-min step]
    atlas14_hyetograph_12hr_100yr.csv     — time series rainfall [mm/30-min step]

Usage:
    python3 precipitation/noaa_atlas14.py
    python3 precipitation/noaa_atlas14.py --lat 28.5652 --lon -81.5868
"""

import os
import sys
import json
import re
import argparse
import warnings
import requests
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Ported 2026-08-04 from models/flood_hydrology/precipitation/noaa_atlas14.py, unmodified
# except for these defaults and the registry hookup below. This closes Future-work item 4
# ("NOAA Atlas 14 design storms — reusable as-is once a simulation stage exists"): the
# simulation stage now exists, and analysis/flood_probability.py needs the full return-period
# series to convert simulated depth into an annual exceedance probability.
DEFAULT_LAT = 28.36687    # CFX SR417 corridor AOI (see site_registry.py)
DEFAULT_LON = -81.43299

# Corrected 2026-08-04: the /cgi-bin/hdsc/new/ path now 301-redirects to /cgi-bin/new/.
PFDS_URL = "https://hdsc.nws.noaa.gov/cgi-bin/new/cgi_readH5.py"

DURATIONS_HR = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 168, 240, 360, 720]

RETURN_PERIODS_YR = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]

# The 19 duration rows PFDS actually returns, in the order it returns them. Confirmed 2026-08-04
# against a real live response for this AOI (19 rows x 10 return-period columns).
PFDS_ROW_DURATIONS_HR = [
    5 / 60, 10 / 60, 15 / 60, 30 / 60,      # 5, 10, 15, 30 minutes
    1, 2, 3, 6, 12, 24,                      # 1, 2, 3, 6, 12, 24 hours
    48, 72, 96, 168, 240,                    # 2, 3, 4, 7, 10 days
    480, 720, 1080, 1440,                    # 20, 30, 45, 60 days
]


def _fetch_via_curl(lat, lon, params):
    """Fetch the PFDS response with system curl. Returns the body text, or None on failure.

    `-L` is required: /cgi-bin/hdsc/new/ 301-redirects to /cgi-bin/new/, and older saved URLs
    still point at the former.
    """
    import subprocess
    from urllib.parse import urlencode
    url = f"{PFDS_URL}?{urlencode(params)}"
    try:
        out = subprocess.run(["curl", "-sL", "--max-time", "45", url],
                             capture_output=True, text=True, timeout=60)
        if out.returncode == 0 and "quantiles" in out.stdout:
            return out.stdout
        return None
    except Exception:
        return None


def fetch_atlas14(lat, lon):
    """Query NOAA PFDS REST endpoint; return parsed depth table."""
    params = {
        "lat": lat,
        "lon": lon,
        "type": "pf",
        "data": "depth",
        "units": "english",   # inches; we convert to mm
        "series": "pds",      # partial-duration series
    }
    print(f"Querying NOAA Atlas 14 PFDS for ({lat}, {lon}) …")
    body = None
    try:
        resp = requests.get(PFDS_URL, params=params, timeout=30, verify=False)
        resp.raise_for_status()
        body = resp.text
    except Exception as exc:
        # This environment's Python 3.9 links LibreSSL 2.8.3, which cannot complete a TLS
        # handshake with hdsc.nws.noaa.gov (SSLV3_ALERT_HANDSHAKE_FAILURE) even with
        # verify=False — that flag skips certificate VALIDATION, not the handshake itself.
        # The system curl (built against a modern TLS stack) negotiates fine against the exact
        # same URL, so fall back to it rather than silently degrading to hard-coded defaults.
        # Verified 2026-08-04: curl returns real site-specific data here where requests cannot.
        print(f"  requests failed ({type(exc).__name__}); retrying via system curl …")
        body = _fetch_via_curl(lat, lon, params)
        if body is None:
            print("  curl fallback also failed; using Central Florida hard-coded defaults.")
            return _central_florida_defaults(lat, lon)
        print("  ✓ curl fallback succeeded")

    class _R:            # minimal shim so the parsing code below is identical either way
        text = body
    resp = _R()

    # ── Real-response parser (added 2026-08-04) ──────────────────────────────────────────
    # PFDS returns a JavaScript array-of-arrays, NOT the dict-of-dicts _parse_json_payload
    # below expects — so every call silently fell through to the hard-coded Orange County
    # defaults, which looked like a successful run because the fallback prints a plausible
    # table. Found because the URL was ALSO stale (301: /cgi-bin/hdsc/new/ -> /cgi-bin/new/),
    # which surfaced the fallback as an explicit SSL/connection error rather than a silent
    # format mismatch. Both fixed; the dict parser is kept below as a fallback in case the
    # service ever returns that shape.
    #   quantiles = [[d1_rp1, d1_rp2, ...], [d2_rp1, ...], ...]
    #   rows = PFDS_ROW_DURATIONS_HR (19), cols = RETURN_PERIODS_YR (10), units = inches
    m = re.search(r"quantiles\s*=\s*(\[\[.*?\]\])\s*;", resp.text, re.S)
    if m:
        try:
            grid = json.loads(m.group(1).replace("'", '"'))
            if len(grid) == len(PFDS_ROW_DURATIONS_HR) and len(grid[0]) == len(RETURN_PERIODS_YR):
                rows = []
                for dur_hr, rp_vals in zip(PFDS_ROW_DURATIONS_HR, grid):
                    for rp_yr, depth_in in zip(RETURN_PERIODS_YR, rp_vals):
                        depth_in = float(depth_in)
                        rows.append({
                            "duration_hr": dur_hr,
                            "return_period_yr": rp_yr,
                            "depth_in": depth_in,
                            "depth_mm": depth_in * 25.4,
                            "intensity_mm_hr": depth_in * 25.4 / dur_hr,
                        })
                df = pd.DataFrame(rows).sort_values(["duration_hr", "return_period_yr"])
                print(f"  ✓ Parsed REAL site-specific PFDS grid: "
                      f"{len(PFDS_ROW_DURATIONS_HR)} durations × {len(RETURN_PERIODS_YR)} "
                      f"return periods = {len(df)} values")
                return df
            print(f"  PFDS grid shape {len(grid)}×{len(grid[0])} != expected "
                  f"{len(PFDS_ROW_DURATIONS_HR)}×{len(RETURN_PERIODS_YR)} — not trusting it.")
        except Exception as exc:
            print(f"  PFDS array parse failed ({exc}).")

    # PFDS returns JavaScript-embedded JSON; parse the key array
    text = resp.text
    # Locate the depth data embedded as a JS variable
    # Format: var data = [[...], ...]  or JSON-like structure
    # Try to extract JSON block
    start = text.find("quantiles")
    if start == -1:
        # Fall back: try parsing entire response as JSON
        try:
            payload = resp.json()
        except Exception:
            # Service may have changed format; save raw for inspection
            raw_path = os.path.join(DATA_DIR, "atlas14_raw_response.txt")
            with open(raw_path, "w") as f:
                f.write(text)
            print(f"  Unexpected response format. Raw saved to {raw_path}")
            print("  Falling back to hard-coded Central Florida Atlas 14 values.")
            return _central_florida_defaults(lat, lon)
        return _parse_json_payload(payload, lat, lon)

    return _parse_json_payload(json.loads("{" + text[start:].split(";")[0].rstrip() + "}"), lat, lon)


def _parse_json_payload(payload, lat, lon):
    """Extract depth values from PFDS JSON response."""
    rows = []
    try:
        quantiles = payload.get("quantiles", {})
        for dur_key, rp_vals in quantiles.items():
            # dur_key like '1-hr', '12-hr', etc.
            try:
                dur_hr = float(dur_key.replace("-hr", "").replace("-day", ""))
                if "day" in dur_key:
                    dur_hr *= 24
            except ValueError:
                continue
            for rp_key, depth_in in rp_vals.items():
                try:
                    rp_yr = int(rp_key.replace("-yr", ""))
                except ValueError:
                    continue
                rows.append({
                    "duration_hr": dur_hr,
                    "return_period_yr": rp_yr,
                    "depth_in": float(depth_in),
                    "depth_mm": float(depth_in) * 25.4,
                    "intensity_mm_hr": float(depth_in) * 25.4 / dur_hr,
                })
    except Exception as exc:
        print(f"  JSON parse warning: {exc}; using Central Florida defaults.")
        return _central_florida_defaults(lat, lon)

    if not rows:
        return _central_florida_defaults(lat, lon)

    df = pd.DataFrame(rows).sort_values(["duration_hr", "return_period_yr"])
    print(f"  ✓ Parsed {len(df)} duration × return-period combinations")
    return df


def _central_florida_defaults(lat, lon):
    """
    Hard-coded Atlas 14 Volume 9 estimates for Orange County / Orlando FL.
    Source: NOAA Atlas 14, Volume 9, Region 1 (Southeast US), Table 3.1.
    These are the point estimates (50th percentile) for the Orlando area.
    """
    warnings.warn(
        f"NOAA PFDS query failed for ({lat}, {lon}) — falling back to hard-coded "
        "Central Florida Atlas 14 values (Orange County, 50th percentile). "
        "Results may not reflect site-specific precipitation frequency. "
        "Verify connectivity to https://hdsc.nws.noaa.gov/cgi-bin/hdsc/new/cgi_readH5.py",
        UserWarning, stacklevel=3
    )
    print("  Using hard-coded Atlas 14 values for Central Florida (Orange County).")
    # depth_mm[duration_hr][return_period_yr]
    # Approximate values from Atlas 14 Vol 9 for Orlando area
    atlas = {
        1:  {2: 47, 5: 63, 10: 75,  25: 92,  50: 105, 100: 119, 200: 133, 500: 152, 1000: 167},
        3:  {2: 62, 5: 84, 10: 99,  25: 121, 50: 138, 100: 157, 200: 175, 500: 202, 1000: 223},
        6:  {2: 77, 5: 103,10: 122, 25: 149, 50: 171, 100: 194, 200: 218, 500: 251, 1000: 278},
        12: {2: 96, 5: 127,10: 151, 25: 184, 50: 212, 100: 240, 200: 270, 500: 312, 1000: 346},
        24: {2:117, 5: 155,10: 183, 25: 223, 50: 257, 100: 292, 200: 329, 500: 381, 1000: 423},
        48: {2:148, 5: 194,10: 229, 25: 277, 50: 319, 100: 362, 200: 408, 500: 474, 1000: 527},
    }
    rows = []
    for dur_hr, rp_dict in atlas.items():
        for rp_yr, depth_mm in rp_dict.items():
            rows.append({
                "duration_hr": dur_hr,
                "return_period_yr": rp_yr,
                "depth_in": depth_mm / 25.4,
                "depth_mm": depth_mm,
                "intensity_mm_hr": depth_mm / dur_hr,
            })
    return pd.DataFrame(rows).sort_values(["duration_hr", "return_period_yr"])


def make_design_hyetograph(depth_mm, duration_hr, dt_min, storm_type="SCS_II"):
    """
    Generate a synthetic design hyetograph from a total rainfall depth.

    SCS Type II distribution (Florida standard) places the peak intensity at
    ~60% of the storm duration — appropriate for convective FL thunderstorms.

    Parameters
    ----------
    depth_mm   : total rainfall depth [mm]
    duration_hr: storm duration [hours]
    dt_min     : time step [minutes]
    storm_type : currently only 'SCS_II'

    Returns
    -------
    pd.DataFrame with columns: time_min, cumulative_depth_mm, incremental_depth_mm
    """
    # SCS Type II dimensionless cumulative rainfall ratios at canonical time fractions
    scs_ii_t  = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    scs_ii_P  = np.array([0, .04,.1, .2, .35,.63,.78,.87,.93,.97, 1.0])

    total_minutes = duration_hr * 60
    times = np.arange(0, total_minutes + dt_min, dt_min)
    t_frac = times / total_minutes

    cum_frac  = np.interp(t_frac, scs_ii_t, scs_ii_P)
    cum_depth = cum_frac * depth_mm
    incr      = np.diff(cum_depth, prepend=0)

    return pd.DataFrame({
        "time_min":              times,
        "cumulative_depth_mm":   cum_depth,
        "incremental_depth_mm":  incr,
    })


def main(lat=DEFAULT_LAT, lon=DEFAULT_LON):
    df = fetch_atlas14(lat, lon)

    idf_path = os.path.join(DATA_DIR, f"atlas14_idf_{lat:.4f}_{abs(lon):.4f}W.csv")
    df.to_csv(idf_path, index=False)
    print(f"Saved IDF table : {idf_path}")

    # Print summary table for key durations and return periods
    key_durs = [1, 6, 12, 24]
    key_rps  = [10, 25, 100, 500]
    print("\n── Atlas 14 Rainfall Depths (mm) ─────────────────────────────")
    print(f"{'Duration':>10}  {'10-yr':>8}  {'25-yr':>8}  {'100-yr':>9}  {'500-yr':>9}")
    for dur in key_durs:
        sub = df[df.duration_hr == dur]
        if sub.empty:
            continue
        row_vals = {}
        for rp in key_rps:
            match = sub[sub.return_period_yr == rp]
            row_vals[rp] = f"{match.depth_mm.values[0]:.1f}" if not match.empty else "—"
        print(f"  {dur:>4}-hr   {row_vals[10]:>8}  {row_vals[25]:>8}  {row_vals[100]:>9}  {row_vals[500]:>9}")

    # Generate design hyetographs for 1-hr 100-yr and 12-hr 100-yr
    scenarios = [
        (1,  100, 5,  "1hr_100yr"),
        (1,  10,  5,  "1hr_10yr"),
        (12, 100, 30, "12hr_100yr"),
        (12, 10,  30, "12hr_10yr"),
    ]
    print()
    for dur_hr, rp_yr, dt_min, label in scenarios:
        sub = df[(df.duration_hr == dur_hr) & (df.return_period_yr == rp_yr)]
        if sub.empty:
            print(f"  No Atlas 14 data for {dur_hr}-hr {rp_yr}-yr; skipping hyetograph.")
            continue
        depth_mm = sub.depth_mm.values[0]
        hyet = make_design_hyetograph(depth_mm, dur_hr, dt_min)
        hyet_path = os.path.join(DATA_DIR, f"atlas14_hyetograph_{label}.csv")
        hyet.to_csv(hyet_path, index=False)
        peak_intensity = hyet.incremental_depth_mm.max() / (dt_min / 60)
        print(f"  {label}: total {depth_mm:.1f} mm, peak intensity {peak_intensity:.1f} mm/hr → {hyet_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NOAA Atlas 14 IDF curves and generate design hyetographs")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    args = parser.parse_args()
    main(args.lat, args.lon)
