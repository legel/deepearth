"""
GSDR vs NOAA Atlas 14 — Precipitation Comparison
==================================================
Compares the all-time observed GSDR gauge maxima (within 50 km of the site)
against NOAA Atlas 14 frequency estimates for Winter Garden FL.

This serves as an independent validation of the Atlas 14 design storms used
in flood_sim.py and shows where on the return-period spectrum the observed
extreme events fall.

Usage:
    python3 precipitation/compare_gsdr_atlas14.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
GSDR_CSV  = os.path.join(BASE_DIR, "..", "gsdr", "outputs",
                         "gsdr_intensity_28p5216_W81p6570.csv")
A14_CSV   = os.path.join(DATA_DIR, "atlas14_idf_28.5216_81.6570W.csv")
OUT_PNG   = os.path.join(DATA_DIR, "gsdr_vs_atlas14.png")

SITE_LAT  = 28.521592
SITE_LON  = -81.656981

# GSDR all-time observed maxima within 50 km — from gsdr_intensity_matrix.py run
# Nearest station: US_086638 (31.8 km), dataset ~1948-2014
GSDR_50KM = {
    1:  143.5,   # mm — US_086638, 1960
    3:  207.3,   # mm — US_086638, 1960
    6:  232.4,   # mm — US_084625, 1956
    12: 300.0,   # mm — US_084625, 1956
    24: 363.0,   # mm — US_084625, 1956
}

DURATIONS = [1, 3, 6, 12, 24]
RETURN_PERIODS = [10, 25, 100, 500]
RP_COLORS = {10: "#74b9ff", 25: "#0984e3", 100: "#2d3436", 500: "#6c5ce7"}
RP_LABELS = {10: "10-yr", 25: "25-yr", 100: "100-yr (design)", 500: "500-yr"}


def load_atlas14():
    df = pd.read_csv(A14_CSV)
    return df


def main():
    a14 = load_atlas14()

    # Build Atlas 14 lookup: duration → {rp: depth_mm}
    a14_lut = {}
    for dur in DURATIONS:
        a14_lut[dur] = {}
        for rp in RETURN_PERIODS:
            row = a14[(a14["duration_hr"] == dur) & (a14["return_period_yr"] == rp)]
            if not row.empty:
                a14_lut[dur][rp] = float(row["depth_mm"].values[0])

    # ── Print comparison table ──────────────────────────────────────────────
    print("=" * 76)
    print(f"GSDR observed all-time max  vs  NOAA Atlas 14 — Winter Garden FL")
    print(f"Site: {SITE_LAT}°N  {abs(SITE_LON)}°W")
    print(f"GSDR radius: ≤50 km  |  Nearest gauge: US_086638 (31.8 km)")
    print("=" * 76)
    print(f"{'Duration':<10} {'GSDR max':>10} {'A14 10yr':>10} {'A14 25yr':>10} "
          f"{'A14 100yr':>10} {'A14 500yr':>10}  {'GSDR equiv. RP':>15}")
    print("-" * 76)
    for dur in DURATIONS:
        gsdr_mm = GSDR_50KM[dur]
        a14_vals = a14_lut.get(dur, {})
        # Find which return-period bracket the GSDR max falls into
        rps_sorted = sorted(a14_vals.keys())
        bracket = "< 10-yr"
        for rp in rps_sorted:
            if gsdr_mm >= a14_vals[rp]:
                bracket = f"> {rp}-yr"
        # Interpolate approximate return period (log-linear)
        below_rp, above_rp, below_v, above_v = None, None, None, None
        for i, rp in enumerate(rps_sorted):
            if a14_vals[rp] <= gsdr_mm:
                below_rp, below_v = rp, a14_vals[rp]
            elif above_rp is None:
                above_rp, above_v = rp, a14_vals[rp]
        if below_rp and above_rp:
            t = (np.log(gsdr_mm) - np.log(below_v)) / (np.log(above_v) - np.log(below_v))
            interp_rp = int(np.exp(np.log(below_rp) + t * (np.log(above_rp) - np.log(below_rp))))
            equiv = f"~{interp_rp}-yr"
        elif below_rp:
            equiv = f"> {max(rps_sorted)}-yr"
        else:
            equiv = f"< {min(rps_sorted)}-yr"

        print(f"  {dur:>3}-hr   {gsdr_mm:>8.1f} mm"
              + "".join(f"{a14_vals.get(rp, float('nan')):>10.1f}" for rp in RETURN_PERIODS)
              + f"  {equiv:>15}")

    print("=" * 76)
    print("\nKey finding:")
    gsdr_1hr  = GSDR_50KM[1]
    gsdr_12hr = GSDR_50KM[12]
    a14_1hr_100  = a14_lut[1][100]
    a14_12hr_100 = a14_lut[12][100]
    print(f"  1-hr  GSDR max = {gsdr_1hr:.1f} mm  vs  Atlas14 100-yr = {a14_1hr_100:.1f} mm  "
          f"(+{100*(gsdr_1hr/a14_1hr_100 - 1):.0f}% above 100-yr design storm)")
    print(f"  12-hr GSDR max = {gsdr_12hr:.1f} mm  vs  Atlas14 100-yr = {a14_12hr_100:.1f} mm  "
          f"(+{100*(gsdr_12hr/a14_12hr_100 - 1):.0f}% above 100-yr design storm)")
    print(f"\n  The flood model's 100-yr design storms are conservative relative to observed")
    print(f"  extremes — the historical all-time max events exceed the 100-yr estimate by")
    print(f"  ~20% (1-hr) and ~25% (12-hr). Atlas 14 100-yr values remain the engineering")
    print(f"  standard; GSDR confirms they are in the right order of magnitude.")

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"GSDR Observed vs NOAA Atlas 14 — Winter Garden FL ({SITE_LAT}°N, {abs(SITE_LON)}°W)\n"
        f"GSDR: all-time observed maximum within 50 km (9 stations, ~1942–2014); "
        f"nearest gauge US_086638 at 31.8 km",
        fontsize=10, fontweight="bold"
    )

    dur_labels = [f"{d}-hr" for d in DURATIONS]
    x = np.arange(len(DURATIONS))
    width = 0.18

    for ax_idx, (ax, unit, scale, ylabel) in enumerate(zip(
        axes,
        ["mm", "in"],
        [1.0, 1/25.4],
        ["Precipitation depth [mm]", "Precipitation depth [inches]"],
    )):
        # Atlas 14 bars
        for i, rp in enumerate(RETURN_PERIODS):
            vals = [a14_lut[d].get(rp, np.nan) * scale for d in DURATIONS]
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, vals, width, label=RP_LABELS[rp],
                          color=RP_COLORS[rp], alpha=0.85, edgecolor="white", lw=0.5)

        # GSDR observed line
        gsdr_vals = [GSDR_50KM[d] * scale for d in DURATIONS]
        ax.plot(x, gsdr_vals, "r^-", ms=9, lw=2.2, zorder=5,
                label="GSDR all-time observed max\n(≤50 km, raw gauge record)",
                markeredgecolor="darkred", markeredgewidth=0.7)
        for xi, yv, dur in zip(x, gsdr_vals, DURATIONS):
            yr = {1: 1960, 3: 1960, 6: 1956, 12: 1956, 24: 1956}[dur]
            ax.text(xi, yv + (2 if unit == "mm" else 0.08),
                    f"{yv:.1f}{unit}\n({yr})", ha="center", va="bottom",
                    fontsize=7.5, color="darkred", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(dur_labels)
        ax.set_xlabel("Storm duration")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{'mm depth' if unit == 'mm' else 'inch depth'} by duration\n"
            "Design storms (bars) vs. GSDR historical max (red triangles)",
            fontsize=9
        )
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

        # Annotate the two design storm durations used in flood_sim
        for xi, dur in zip(x, DURATIONS):
            if dur in (1, 12):
                ax.axvline(xi, color="gray", lw=0.7, ls=":", alpha=0.6)
        ax.text(x[0], ax.get_ylim()[1] * 0.02, "← used in\nflood_sim",
                fontsize=7, color="gray", ha="center")
        ax.text(x[3], ax.get_ylim()[1] * 0.02, "← used in\nflood_sim",
                fontsize=7, color="gray", ha="center")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved comparison figure → {OUT_PNG}")


if __name__ == "__main__":
    main()
