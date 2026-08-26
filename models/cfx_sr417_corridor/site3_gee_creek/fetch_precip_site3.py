"""
Fetch ASOS KSFB hourly precipitation for site3 (Gee Creek) -> site3_gee_creek/precipitation/data/
====================================================================================================
Real gap this fills: `asos_hourly_SFB.csv` already exists on disk (120 hourly rows,
2022-09-26 00:00 - 2022-09-30 23:00 UTC, the Hurricane Ian window `run_site3_ian.py` reads) but
was fetched ad hoc via `precipitation/fetch_asos_hourly.py`'s underlying functions with no
saved invocation.

Station choice: KSFB (Orlando Sanford Intl), 10.8km from site3, chosen on PROXIMITY only, unlike
the main AOI's MCO station (chosen after a real reliability cross-check that rejected the
closer ISM station). Per CLAUDE.md's dataset-selection-logic audit (2026-07-27), site3 still has
no equivalent local-gauge reliability check -- this script fetches the same window that's
already in use, it does not add that missing cross-validation (tracked as future-work item 21
in CLAUDE.md).

Reuses `fetch_iem_asos()` / `extract_hourly()` directly -- both are already station-agnostic
(take a station code + date range), so no monkey-patching is needed, just a different station
code and this project's site3 output directory.

Usage:
    python3 site3_gee_creek/fetch_precip_site3.py
    python3 site3_gee_creek/fetch_precip_site3.py --start 2022-09-26 --end 2022-10-01
"""
import os, sys, argparse

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ_DIR, "precipitation"))
import fetch_asos_hourly as asos  # noqa: E402

SITE3_PRECIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precipitation", "data")
STATION = "SFB"  # Orlando Sanford Intl (KSFB), 10.8km from site3 -- see docstring above


def main(start="2022-09-26", end="2022-10-01"):
    os.makedirs(SITE3_PRECIP_DIR, exist_ok=True)
    print(f"Fetching ASOS {STATION} hourly precip, {start} to {end} (Hurricane Ian window)")
    raw = asos.fetch_iem_asos(STATION, start, end)
    hourly_mm = asos.extract_hourly(raw)

    out_path = os.path.join(SITE3_PRECIP_DIR, f"asos_hourly_{STATION}.csv")
    hourly_mm.to_frame("prcp_mm").rename_axis("datetime_utc").to_csv(out_path)
    total = hourly_mm.sum()
    peak = hourly_mm.max()
    print(f"  Saved {len(hourly_mm)} hourly rows -> {out_path}")
    print(f"  Total: {total:.1f} mm, peak: {peak:.1f} mm/hr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ASOS KSFB hourly precipitation for site3")
    parser.add_argument("--start", default="2022-09-26")
    parser.add_argument("--end", default="2022-10-01")
    args = parser.parse_args()
    main(args.start, args.end)
