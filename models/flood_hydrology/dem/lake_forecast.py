"""
Lake Water Level Forecast — Level-Pool Routing
================================================
Forecasts how lake water surface elevation (WSE) responds to a precipitation
event using the standard level-pool (storage routing) method.

Physics
-------
The lake is a storage reservoir. The continuity equation is:

    dS/dt = Q_in(t) − Q_out(h)          [m³/s]

where:
    S(h)      = lake storage volume at WSE h  [m³]  → from hypsometric curve
    Q_in(t)   = inflow hydrograph [m³/s]     → from watershed runoff
    Q_out(h)  = outflow via spillway/weir [m³/s]:
                Q_out = Cd · L · max(h − z_outlet, 0)^1.5
                (broad-crested weir, Cd ≈ 1.7, L = outlet crest width)

Discretized with 4th-order Runge-Kutta (RK4) at dt = 60 s timestep.

Inputs
------
    dem/data/lake_volume.csv           — baseline volume + area + depth
    dem/data/lake_volume_seasonal.csv  — seasonal S2-derived volumes (optional)
    dem/data/lake_bed_dem_*.tif        — bathymetry for hypsometric curve
    dem/data/lake_mask_s2.tif          — authoritative lake boundary
    dem/data/winter_garden_dem.tif     — terrain for hypsometric curve
    precipitation/data/atlas14_hyetograph_*.csv  — design storm time series
    soil/data/soil_parameters.json     — Horton infiltration parameters

Outputs
-------
    dem/lake_level_forecast.csv   — time series: t, Q_in, Q_out, h_wse, S, inundation_ha
    dem/lake_level_forecast.png   — 4-panel visualization

Usage:
    python3 dem/lake_forecast.py
    python3 dem/lake_forecast.py --scenario flash_1hr_100yr --initial_offset 0.0
    python3 dem/lake_forecast.py --scenario sustained_12hr_100yr --initial_offset -0.5
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
PRECIP_DIR = os.path.join(BASE_DIR, "..", "precipitation", "data")
SOIL_DIR   = os.path.join(BASE_DIR, "..", "soil", "data")
S2_DIR     = os.path.join(BASE_DIR, "..", "sentinel2", "data")

PROPERTY_LAT = 28.521592
PROPERTY_LON = -81.656981

# Broad-crested weir parameters for the lake outlet (spillway / control structure)
WEIR_CD      = 1.7    # discharge coefficient (broad-crested)
WEIR_L_M     = 3.0    # outlet crest width [m] (typical stormwater outfall)
DT_S         = 60.0   # integration timestep [seconds]
MIN_DEPTH_M  = 0.01   # numerical zero depth threshold


def load_hypsometric_curve(dem_path, lake_mask_path, lake_bed_path, n_steps=200):
    """
    Build hypsometric S(h) and A(h) curves from DEM + bathymetry.

    Returns (z_levels, S_m3, A_ha) arrays where z_levels goes from
    (water_surface - max_depth) to (water_surface + 2m freeboard).
    These are the physically correct curves — no assumptions, pure geometry.
    """
    import rasterio

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        cell_m = abs(src.transform.a)
    with rasterio.open(lake_mask_path) as src:
        lake_mask = src.read(1).astype(np.uint8)
    with rasterio.open(lake_bed_path) as src:
        lake_bed = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None:
            lake_bed[lake_bed == nd] = np.nan

    lake_bool = (lake_mask == 1) & np.isfinite(dem) & np.isfinite(lake_bed)
    if lake_bool.sum() == 0:
        raise RuntimeError("No valid lake cells found for hypsometric curve")

    water_surface = float(np.nanmean(dem[lake_bool]))
    max_depth = float(np.nanmax(np.maximum(water_surface - lake_bed[lake_bool], 0)))
    cell_area_m2 = cell_m ** 2

    z_min = water_surface - max(max_depth, 0.5)
    z_max = water_surface + 2.0   # 2m freeboard above current surface
    z_levels = np.linspace(z_min, z_max, n_steps)

    S_m3 = np.zeros(n_steps)
    A_ha = np.zeros(n_steps)
    for i, z in enumerate(z_levels):
        submerged = lake_bool & (lake_bed < z)
        depth_at_z = np.where(submerged, np.maximum(z - lake_bed, 0.0), 0.0)
        S_m3[i] = float(depth_at_z.sum() * cell_area_m2)
        A_ha[i] = float(submerged.sum() * cell_area_m2 / 1e4)

    print(f"  Hypsometric curve: {n_steps} levels, z={z_min:.1f}–{z_max:.1f} m NAVD88")
    print(f"  Water surface: {water_surface:.2f} m, max depth: {max_depth:.2f} m")
    return z_levels, S_m3, A_ha, water_surface


def interp_storage(h, z_levels, S_m3):
    """Interpolate storage volume at WSE h."""
    return float(np.interp(h, z_levels, S_m3))


def interp_area(h, z_levels, A_ha):
    """Interpolate lake surface area [ha] at WSE h."""
    return float(np.interp(h, z_levels, A_ha))


def wse_from_storage(S, z_levels, S_m3):
    """Inverse interpolation: storage → WSE."""
    return float(np.interp(S, S_m3, z_levels))


def weir_outflow(h, z_outlet, Cd=WEIR_CD, L=WEIR_L_M):
    """Broad-crested weir outflow [m³/s]: Q = Cd·L·max(h−z_out, 0)^1.5"""
    head = max(h - z_outlet, 0.0)
    return Cd * L * head ** 1.5


def compute_watershed_area_km2(dem_path, lake_mask_path):
    """Estimate watershed contributing area from the watershed.tif if available."""
    watershed_path = os.path.join(DATA_DIR, "watershed.tif")
    if os.path.exists(watershed_path):
        import rasterio
        with rasterio.open(watershed_path) as src:
            ws = src.read(1)
            cell_m = abs(src.transform.a)
        area_km2 = float((ws > 0).sum() * cell_m ** 2 / 1e6)
        print(f"  Watershed area from watershed.tif: {area_km2:.3f} km²")
        return area_km2
    # Fallback: 2×2 km AOI minus lake area
    import rasterio
    with rasterio.open(dem_path) as src:
        cell_m = abs(src.transform.a)
        total_cells = src.width * src.height
    with rasterio.open(lake_mask_path) as src:
        lake_cells = int(src.read(1).sum())
    land_cells = total_cells - lake_cells
    area_km2 = land_cells * cell_m ** 2 / 1e6
    print(f"  Watershed area (AOI − lake): {area_km2:.3f} km²")
    return area_km2


def load_inflow_hydrograph(scenario, watershed_area_km2):
    """
    Convert Atlas 14 design hyetograph to watershed inflow Q_in(t) [m³/s].

    Uses SCS curve number method:
        Pe = (P - 0.2S)² / (P + 0.8S)   [effective rainfall, mm]
        S  = 25400/CN - 254              [retention parameter, mm]
    Then: Q_in = Pe × watershed_area / storm_duration (simplified uniform routing)
    """
    # Map scenario to hyetograph file
    _scenario_map = {
        "flash_1hr_100yr":      "atlas14_hyetograph_1hr_100yr.csv",
        "flash_1hr_10yr":       "atlas14_hyetograph_1hr_10yr.csv",
        "sustained_12hr_100yr": "atlas14_hyetograph_12hr_100yr.csv",
        "sustained_12hr_10yr":  "atlas14_hyetograph_12hr_10yr.csv",
    }
    fname = _scenario_map.get(scenario, f"atlas14_hyetograph_{scenario}.csv")
    hyeto_path = os.path.join(PRECIP_DIR, fname)

    if not os.path.exists(hyeto_path):
        print(f"  ⚠ Hyetograph not found: {hyeto_path}")
        print(f"  Run precipitation/noaa_atlas14.py first.")
        return None, None

    df = pd.read_csv(hyeto_path)
    time_min = df["time_min"].values.astype(float)
    rain_mm  = df["rainfall_mm"].values.astype(float)

    # SCS curve number from soil parameters
    cn = 79.0  # default HSG-C residential (Orange County)
    soil_json = os.path.join(SOIL_DIR, "soil_parameters.json")
    if os.path.exists(soil_json):
        import json
        with open(soil_json) as f:
            sp = json.load(f)
        # dominant HSG → CN
        dominant_hsg = sp.get("dominant_hsg", "C")
        cn_map = {"A": 51, "B": 68, "C": 79, "D": 84}
        cn = cn_map.get(dominant_hsg, 79)
    print(f"  SCS CN={cn} (HSG-{chr(65 + list('ABCD').index(chr(64 + int(cn>51) + int(cn>68) + int(cn>79) + 1)))})")

    S_mm = 25400.0 / cn - 254.0  # retention parameter [mm]

    # Compute cumulative effective rainfall per timestep
    P_cum = np.cumsum(rain_mm)  # cumulative rainfall [mm]
    Pe_cum = np.where(P_cum > 0.2 * S_mm,
                      (P_cum - 0.2 * S_mm) ** 2 / (P_cum + 0.8 * S_mm),
                      0.0)
    # Incremental effective rainfall [mm per timestep]
    Pe_incr = np.diff(Pe_cum, prepend=0.0)

    # Convert to inflow [m³/s]: Q = Pe [m/s] × area [m²]
    dt_s = np.diff(time_min, prepend=time_min[0]) * 60.0  # seconds per step
    dt_s[0] = dt_s[1]
    area_m2 = watershed_area_km2 * 1e6
    Q_in = np.where(dt_s > 0, Pe_incr * 1e-3 * area_m2 / dt_s, 0.0)  # m³/s

    time_s = time_min * 60.0
    print(f"  Inflow hydrograph: {len(time_s)} steps, peak {Q_in.max():.2f} m³/s at t={time_s[Q_in.argmax()]/60:.0f} min")
    return time_s, Q_in


def level_pool_routing(time_s, Q_in, z_levels, S_m3, A_ha,
                       h_initial, z_outlet,
                       Cd=WEIR_CD, L=WEIR_L_M, dt=DT_S):
    """
    Solve dS/dt = Q_in(t) - Q_out(h(S)) using explicit RK4.

    Returns DataFrame with columns: time_s, Q_in_m3s, Q_out_m3s, h_wse_m,
    S_m3, lake_area_ha, lake_rise_m.
    """
    # Interpolate Q_in to uniform dt grid
    t_end = float(time_s[-1]) + 6 * 3600  # simulate 6 hours after storm ends
    t_uniform = np.arange(0, t_end + dt, dt)
    Q_in_uniform = np.interp(t_uniform, time_s, Q_in, left=0.0, right=0.0)

    h = h_initial
    S = interp_storage(h, z_levels, S_m3)
    baseline_h = h_initial

    records = []
    for i, t in enumerate(t_uniform):
        Qin = Q_in_uniform[i]
        Qout = weir_outflow(h, z_outlet, Cd, L)

        # RK4 integration
        def dSdt(S_):
            h_ = wse_from_storage(max(S_, 0.0), z_levels, S_m3)
            return Qin - weir_outflow(h_, z_outlet, Cd, L)

        k1 = dSdt(S)
        k2 = dSdt(S + 0.5 * dt * k1)
        k3 = dSdt(S + 0.5 * dt * k2)
        k4 = dSdt(S + dt * k3)
        S_new = max(S + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4), 0.0)

        h_new = wse_from_storage(S_new, z_levels, S_m3)
        area  = interp_area(h_new, z_levels, A_ha)

        records.append({
            "time_s":        t,
            "time_hr":       t / 3600.0,
            "Q_in_m3s":      Qin,
            "Q_out_m3s":     Qout,
            "h_wse_m":       h_new,
            "S_m3":          S_new,
            "lake_area_ha":  area,
            "lake_rise_m":   h_new - baseline_h,
        })

        S = S_new
        h = h_new

    df = pd.DataFrame(records)
    max_rise = df["lake_rise_m"].max()
    t_peak = df.loc[df["lake_rise_m"].idxmax(), "time_hr"]
    print(f"  Peak lake rise: {max_rise:.3f} m at t={t_peak:.2f} hr")
    return df


def visualize(results_dict, water_surface, scenario_labels=None):
    """4-panel forecast visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Lake Water Level Forecast — Level-Pool Routing\n"
        "17801 Champagne Dr, Winter Garden FL | Physics-based: dS/dt = Q_in − Q_out",
        fontsize=12, fontweight="bold")

    colors = ["steelblue", "tomato", "seagreen", "darkorange"]

    for idx, (scenario, df) in enumerate(results_dict.items()):
        c = colors[idx % len(colors)]
        label = (scenario_labels or {}).get(scenario, scenario)

        # Panel 1: Lake WSE over time
        ax = axes[0, 0]
        ax.plot(df["time_hr"], df["h_wse_m"], color=c, lw=2, label=label)
        ax.axhline(water_surface, color="royalblue", lw=1, linestyle="--", alpha=0.5,
                   label="Baseline WSE" if idx == 0 else "")
        ax.set_xlabel("Time [hr]"); ax.set_ylabel("Lake WSE [m NAVD88]")
        ax.set_title("Lake Water Surface Elevation")
        ax.legend(fontsize=8)

        # Panel 2: Lake rise above baseline
        ax = axes[0, 1]
        ax.plot(df["time_hr"], df["lake_rise_m"] * 100, color=c, lw=2, label=label)
        ax.axhline(0, color="gray", lw=0.8, linestyle=":")
        ax.set_xlabel("Time [hr]"); ax.set_ylabel("Lake rise [cm]")
        ax.set_title("Lake Level Rise Above Baseline")
        ax.legend(fontsize=8)

        # Panel 3: Inflow vs outflow
        ax = axes[1, 0]
        ax.plot(df["time_hr"], df["Q_in_m3s"], color=c, lw=2, label=f"Q_in {label}")
        ax.plot(df["time_hr"], df["Q_out_m3s"], color=c, lw=1.5, linestyle="--",
                label=f"Q_out {label}")
        ax.set_xlabel("Time [hr]"); ax.set_ylabel("Discharge [m³/s]")
        ax.set_title("Inflow (solid) vs Outflow (dashed)")
        ax.legend(fontsize=7)

        # Panel 4: Lake area change
        ax = axes[1, 1]
        ax.plot(df["time_hr"], df["lake_area_ha"], color=c, lw=2, label=label)
        ax.set_xlabel("Time [hr]"); ax.set_ylabel("Lake area [ha]")
        ax.set_title("Lake Surface Area During Event")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "lake_level_forecast.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def main(scenario=None, initial_offset=0.0, weir_l=WEIR_L_M):
    import rasterio

    dem_path = os.path.join(DATA_DIR, "winter_garden_dem.tif")
    mask_path = (os.path.join(DATA_DIR, "lake_mask_s2.tif")
                 if os.path.exists(os.path.join(DATA_DIR, "lake_mask_s2.tif"))
                 else os.path.join(DATA_DIR, "lake_mask.tif"))
    bed_path = (os.path.join(DATA_DIR, "lake_bed_dem_fwc.tif")
                if os.path.exists(os.path.join(DATA_DIR, "lake_bed_dem_fwc.tif"))
                else os.path.join(DATA_DIR, "lake_bed_dem_survey.tif")
                if os.path.exists(os.path.join(DATA_DIR, "lake_bed_dem_survey.tif"))
                else os.path.join(DATA_DIR, "lake_bed_dem_estimated.tif"))

    for p in [dem_path, mask_path, bed_path]:
        if not os.path.exists(p):
            sys.exit(f"Missing: {p}\nRun dem_download.py and dem_process.py first.")

    print(f"Lake bed source: {os.path.basename(bed_path)}")

    print("\n── Building hypsometric curve ────────────────────────────────")
    z_levels, S_m3, A_ha, water_surface = load_hypsometric_curve(
        dem_path, mask_path, bed_path)

    # Outlet elevation: default = current water surface (normal pool = outlet crest)
    z_outlet = water_surface + initial_offset
    print(f"  Outlet elevation (weir crest): {z_outlet:.2f} m NAVD88")
    print(f"  Weir width L={weir_l:.1f} m, Cd={WEIR_CD:.2f}")

    watershed_area_km2 = compute_watershed_area_km2(dem_path, mask_path)

    # Scenarios to run
    scenarios = ([scenario] if scenario else
                 ["flash_1hr_100yr", "sustained_12hr_100yr"])

    results = {}
    out_dfs = {}
    for sc in scenarios:
        print(f"\n── Scenario: {sc} ─────────────────────────────────────────")
        time_s, Q_in = load_inflow_hydrograph(sc, watershed_area_km2)
        if Q_in is None:
            continue
        df = level_pool_routing(time_s, Q_in, z_levels, S_m3, A_ha,
                                h_initial=water_surface,
                                z_outlet=z_outlet,
                                L=weir_l)
        results[sc] = df
        out_dfs[sc] = df

    if not results:
        print("\nNo scenarios completed. Run precipitation/noaa_atlas14.py first.")
        return

    # Save CSV (all scenarios combined)
    combined = pd.concat(
        [df.assign(scenario=sc) for sc, df in results.items()], ignore_index=True)
    out_csv = os.path.join(BASE_DIR, "lake_level_forecast.csv")
    combined.to_csv(out_csv, index=False)
    print(f"\nSaved forecast table → {out_csv}")

    # Save hypsometric curve
    hyps_df = pd.DataFrame({"z_m": z_levels, "volume_m3": S_m3, "area_ha": A_ha})
    hyps_path = os.path.join(DATA_DIR, "lake_hypsometric_curve.csv")
    hyps_df.to_csv(hyps_path, index=False)
    print(f"Saved hypsometric curve → {hyps_path}")

    visualize(results, water_surface)

    # Print summary
    print("\n── Forecast Summary ──────────────────────────────────────────")
    for sc, df in results.items():
        print(f"  {sc}:")
        print(f"    Peak inflow    : {df['Q_in_m3s'].max():.3f} m³/s")
        print(f"    Peak outflow   : {df['Q_out_m3s'].max():.3f} m³/s")
        print(f"    Max lake rise  : {df['lake_rise_m'].max()*100:.1f} cm")
        print(f"    Max lake WSE   : {df['h_wse_m'].max():.3f} m NAVD88")
        print(f"    Max lake area  : {df['lake_area_ha'].max():.2f} ha")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lake water level forecast via level-pool routing")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Scenario name (e.g. flash_1hr_100yr). Default: run all.")
    parser.add_argument("--initial_offset", type=float, default=0.0,
                        help="Initial lake level offset from normal pool [m], e.g. -0.5 for dry season")
    parser.add_argument("--weir_l", type=float, default=WEIR_L_M,
                        help=f"Weir/outlet crest width [m] (default: {WEIR_L_M})")
    args = parser.parse_args()
    main(args.scenario, args.initial_offset, args.weir_l)
