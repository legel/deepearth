"""Wet (Jun-Sep) vs dry (Oct-May) season split of OWM water area, Johns Lake.

Reads sentinel2/data/water_extent_timeseries.csv (152 OWM scenes, 2016-2026)
and checks whether the "authoritative" 144.47 ha figure in
dem/data/lake_volume.csv is seasonally biased. Read-only: does not touch the
existing OWM/lake_mask pipeline.
"""
import csv
import os
import statistics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TS_CSV = os.path.join(BASE_DIR, "..", "sentinel2", "data", "water_extent_timeseries.csv")
AUTHORITATIVE_HA = 144.47

WET_MONTHS = {6, 7, 8, 9}


def load_rows():
    rows = []
    with open(TS_CSV) as f:
        for r in csv.DictReader(f):
            date = r["date"]
            month = int(date[4:6])
            rows.append({
                "date": date,
                "month": month,
                "area_ha": float(r["water_area_ha"]),
                "season": "wet" if month in WET_MONTHS else "dry",
            })
    return rows


def summarize(rows):
    wet = [r["area_ha"] for r in rows if r["season"] == "wet"]
    dry = [r["area_ha"] for r in rows if r["season"] == "dry"]
    return wet, dry


def main():
    rows = load_rows()
    wet, dry = summarize(rows)

    print(f"Total scenes: {len(rows)}  (wet={len(wet)}, dry={len(dry)})")
    print()
    print(f"{'season':6} {'n':>4} {'mean_ha':>9} {'median_ha':>10} {'stdev_ha':>9} {'min_ha':>8} {'max_ha':>8}")
    for label, vals in [("wet", wet), ("dry", dry)]:
        print(f"{label:6} {len(vals):>4} {statistics.mean(vals):>9.2f} "
              f"{statistics.median(vals):>10.2f} {statistics.stdev(vals):>9.2f} "
              f"{min(vals):>8.2f} {max(vals):>8.2f}")

    overall_mean = statistics.mean(wet + dry)
    bias_ha = statistics.mean(wet) - statistics.mean(dry)
    print()
    print(f"Overall mean (unweighted, all scenes): {overall_mean:.2f} ha")
    print(f"Wet-minus-dry mean difference: {bias_ha:+.2f} ha "
          f"({100*bias_ha/statistics.mean(dry):+.1f}% relative to dry-season mean)")
    print(f"Authoritative lake_volume.csv figure: {AUTHORITATIVE_HA} ha")
    print(f"Authoritative vs overall mean: {AUTHORITATIVE_HA - overall_mean:+.2f} ha")
    print(f"Authoritative vs dry-season mean: {AUTHORITATIVE_HA - statistics.mean(dry):+.2f} ha")
    print(f"Authoritative vs wet-season mean: {AUTHORITATIVE_HA - statistics.mean(wet):+.2f} ha")


if __name__ == "__main__":
    main()
