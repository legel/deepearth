"""
Flood Simulation Engine — Winter Garden FL Property
====================================================
2D raster-based physically-based flood model for the 2×2 km study area.

Governing equations implemented
---------------------------------
1. INFILTRATION — Horton exponential decay model
   f(t) = fc + (f0 − fc) × exp(−k × t)
   Cumulative: F(t) = fc·t + (f0−fc)/k × (1 − exp(−k·t))
   Net rainfall (runoff source): Pe(t) = max(P(t) − f(t), 0)  [mm/hr → m/s]

2. OVERLAND FLOW — Manning's equation in 2D diffusive wave form
   Unit discharge between adjacent cells:
     q = (1/n) × h^(5/3) × |S|^(1/2) × sign(S)     [m²/s]
   where S = Δη/Δx, η = h + z (water surface elevation)

3. 2D SHALLOW WATER EQUATIONS (local inertia approximation)
   This is the Bates et al. (2010) local inertia scheme — the core of
   LISFLOOD-FP — which simplifies the full SWE momentum equation by
   dropping the advection term (valid for Fr << 1, i.e., slow floodwater):

   Continuity:
     ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = Pe(t) − f(t)

   Momentum (simplified, per interface between cells i and j):
     q_new = [q_prev − g·h_flow·Δt·∂η/∂x] / [1 + g·Δt·n²·|q_prev|/h_flow^(7/3)]
     where h_flow = max(η_i, η_j) − max(z_i, z_j)

   Depth update (continuity on grid cell):
     h[i,j]_new = h[i,j] + (Δt/Δx)·(q_x_in − q_x_out + q_y_in − q_y_out) + Δt·Pe

4. LAKE STORAGE & OUTFLOW — broad-crested weir equation
   Volume-area-elevation (VAE) curve from lake-bed DEM.
   Outflow when lake level exceeds outlet elevation:
     Q_out = Cd · L · (h_lake − z_outlet)^(3/2)     [m³/s]
   where Cd ≈ 1.7 (broad crest), L = outlet width [m]

5. TIME STEP STABILITY — CFL condition
   Δt ≤ α · Δx / √(g · h_max)   (α = 0.5)

Scenarios simulated
-------------------
  flash_1hr_100yr   : 1-hr SCS Type II storm, 100-year return period
  sustained_12hr_100yr: 12-hr SCS Type II storm, 100-year return period
  extreme_12hr_500yr: 12-hr SCS Type II storm, 500-year return period

Inputs (from upstream pipeline)
--------------------------------
  dem/data/winter_garden_dem.tif       — terrain elevation [m NAVD88]
  dem/data/lake_bed_dem.tif            — lake bed elevation [m]
  dem/data/lake_mask.tif               — binary lake mask
  soil/data/soil_parameters.json       — Horton f0, fc, k (or defaults)
  precipitation/data/atlas14_hyetograph_*.csv — design storm time series

Outputs (saved under simulation/outputs/)
-----------------------------------------
  inundation_depth_{scenario}.tif      — peak water depth [m]
  flow_velocity_{scenario}.tif         — peak velocity magnitude [m/s]
  flood_extent_{scenario}.geojson      — inundation polygon (depth > 0.05m)
  hydrograph_{scenario}.csv            — time series: flooded_ha, lake_rise_m
  lake_vae_curve.csv                   — volume-area-elevation curve

Usage:
    python3 simulation/flood_sim.py
    python3 simulation/flood_sim.py --scenario flash_1hr_100yr --dt 5
    python3 simulation/flood_sim.py --scenario all
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = os.path.join(BASE_DIR, "outputs")
DEM_DIR     = os.path.join(os.path.dirname(BASE_DIR), "dem", "data")
SOIL_DIR    = os.path.join(os.path.dirname(BASE_DIR), "soil", "data")
PRECIP_DIR  = os.path.join(os.path.dirname(BASE_DIR), "precipitation", "data")

os.makedirs(OUT_DIR, exist_ok=True)

G         = 9.81    # gravity [m/s²]
CFL_ALPHA = 0.5     # Courant number safety factor
MIN_DEPTH = 1e-4    # [m] minimum depth below which flow is zero (wet/dry front)
DEPTH_THRESHOLD = 0.05  # [m] threshold for "flooded" in output GeoJSON

MANNING_N = {
    "water":        0.020,
    "residential":  0.040,  # suburban — houses, lawns, driveways mixed
    "grass_lawn":   0.030,
    "forest":       0.120,
    "pavement":     0.011,
    "default":      0.040,
}

# Scenarios: (label, hyetograph_file, description)
SCENARIOS = {
    "flash_1hr_100yr":        ("atlas14_hyetograph_1hr_100yr.csv",  "1-hr, 100-year return period"),
    "sustained_12hr_100yr":   ("atlas14_hyetograph_12hr_100yr.csv", "12-hr, 100-year return period"),
    "flash_1hr_10yr":         ("atlas14_hyetograph_1hr_10yr.csv",   "1-hr, 10-year return period"),
    "sustained_12hr_10yr":    ("atlas14_hyetograph_12hr_10yr.csv",  "12-hr, 10-year return period"),
}


# ── Raster I/O ────────────────────────────────────────────────────────────────

def load_raster(path, dtype=np.float32):
    import rasterio
    if not os.path.exists(path):
        return None, None
    with rasterio.open(path) as src:
        arr     = src.read(1).astype(dtype)
        profile = src.profile.copy()
    return arr, profile


def save_raster(array, profile, out_path, dtype=np.float32):
    import rasterio
    p = profile.copy()
    p.update(count=1, dtype=dtype, compress="lzw")
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(array.astype(dtype), 1)


def save_geojson(binary_mask, profile, out_path, properties=None):
    import rasterio.features
    import json
    transform = profile["transform"]
    shapes = list(rasterio.features.shapes(
        binary_mask.astype(np.uint8),
        mask=(binary_mask > 0).astype(np.uint8),
        transform=transform,
    ))
    features = []
    for geom, val in shapes:
        if val != 1:
            continue
        feat_props = properties.copy() if properties else {}
        features.append({"type": "Feature", "geometry": geom, "properties": feat_props})
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": str(profile.get("crs","EPSG:4326"))}},
          "features": features}
    with open(out_path, "w") as f:
        json.dump(fc, f)


# ── Horton Infiltration ────────────────────────────────────────────────────────

class HortonInfiltration:
    """
    Spatially uniform Horton infiltration model.
    f(t) = fc + (f0 - fc) * exp(-k * t)
    """

    def __init__(self, f0_mm_hr=112.5, fc_mm_hr=25.0, k_hr=1.8):
        self.f0  = f0_mm_hr / 1000 / 3600   # convert to m/s
        self.fc  = fc_mm_hr  / 1000 / 3600
        self.k   = k_hr      / 3600          # convert to s⁻¹

    def rate(self, t_s):
        """Infiltration rate at time t_s [seconds] since storm start [m/s]."""
        return self.fc + (self.f0 - self.fc) * np.exp(-self.k * t_s)

    def cumulative(self, t_s):
        """Cumulative infiltration F(t) [m]."""
        if self.k == 0:
            return self.fc * t_s
        return self.fc * t_s + (self.f0 - self.fc) / self.k * (1 - np.exp(-self.k * t_s))

    def effective_rainfall(self, rainfall_rate_ms, t_s):
        """Net rainfall = max(P - f, 0) [m/s]."""
        return np.maximum(rainfall_rate_ms - self.rate(t_s), 0.0)


# ── Lake VAE Curve ─────────────────────────────────────────────────────────────

class LakeStorageModel:
    """
    Volume-Area-Elevation (VAE) curve for a lake, derived from the lake-bed DEM.
    Supports computing water level from stored volume and vice versa.
    """

    def __init__(self, lake_bed_arr, lake_mask_arr, cell_area_m2, water_surface_elev_m):
        self.cell_area  = cell_area_m2
        self.wse0       = water_surface_elev_m   # initial water surface elevation
        self.bed        = lake_bed_arr[lake_mask_arr > 0]
        self.n_cells    = lake_mask_arr.sum()

        # Build VAE curve: for each elevation level, compute volume
        z_min = float(self.bed.min())
        z_max = water_surface_elev_m + 2.0  # allow 2m above initial WSE
        # Remove any NaN from bed (fall back to water surface so depth = 0)
        self.bed = np.where(np.isfinite(self.bed), self.bed, water_surface_elev_m)

        self.z_levels = np.linspace(z_min, z_max, 200)
        self.volumes  = np.array([
            self.cell_area * float(np.nansum(np.maximum(z - self.bed, 0)))
            for z in self.z_levels
        ])
        self.areas = np.array([
            self.cell_area * float(np.nansum(z > self.bed))
            for z in self.z_levels
        ])
        # Ensure monotonically non-decreasing volumes (interp requirement)
        self.volumes = np.maximum.accumulate(self.volumes)

        # Initial volume at water_surface_elev_m
        self.V0 = float(np.interp(water_surface_elev_m, self.z_levels, self.volumes))

    def wse_from_volume(self, V):
        """Return water surface elevation [m] for a given stored volume [m³]."""
        return float(np.interp(V, self.volumes, self.z_levels))

    def area_from_wse(self, wse):
        """Return lake area [m²] for a given WSE."""
        return float(np.interp(wse, self.z_levels, self.areas))

    def outflow(self, wse, z_outlet, L_outlet=5.0, Cd=1.7):
        """
        Broad-crested weir outflow [m³/s].
        Q = Cd · L · (h_lake - z_outlet)^(3/2)   if h_lake > z_outlet, else 0
        """
        head = wse - z_outlet
        if head <= 0:
            return 0.0
        return Cd * L_outlet * head ** 1.5

    def save_vae_table(self, out_path):
        df = pd.DataFrame({
            "elevation_m_navd88": self.z_levels,
            "volume_m3":          self.volumes,
            "area_m2":            self.areas,
        })
        df.to_csv(out_path, index=False)


# ── Local Inertia 2D Solver ───────────────────────────────────────────────────

def adaptive_dt(h, dx, alpha=CFL_ALPHA):
    """CFL-stable time step [s]."""
    h_max = float(np.max(h))
    if h_max < MIN_DEPTH:
        return 30.0  # no water → use large step
    return alpha * dx / np.sqrt(G * h_max)


def local_inertia_step(h, qx, qy, z, n_grid, dx, dt):
    """
    One time step of the 2D local inertia equations (Bates et al. 2010).

    State variables
    ---------------
    h   : water depth [m],        shape (ny, nx)
    qx  : unit discharge x [m²/s], shape (ny, nx)  (defined at x-interfaces)
    qy  : unit discharge y [m²/s], shape (ny, nx)  (defined at y-interfaces)
    z   : bed elevation [m],       shape (ny, nx)
    n_grid: Manning's n,           shape (ny, nx)
    dx  : cell size [m]
    dt  : time step [s]

    Returns updated (h, qx, qy).
    """
    # Guard against NaN pollution from nodata cells
    h = np.where(np.isfinite(h), h, 0.0)
    z = np.where(np.isfinite(z), z, float(np.nanmean(z)))

    eta = h + z   # water surface elevation [m]

    # ── X-direction fluxes (between columns j and j+1) ─────────────────────
    # h_flow at x-interface: depth available to flow
    eta_i = eta[:, :-1]
    eta_j = eta[:,  1:]
    z_i   = z[:,   :-1]
    z_j   = z[:,    1:]
    hf_x  = np.maximum(eta_i, eta_j) - np.maximum(z_i, z_j)
    hf_x  = np.maximum(hf_x, 0.0)

    n_i = n_grid[:, :-1]
    n_j = n_grid[:,  1:]
    n_x = 0.5 * (n_i + n_j)   # average n at interface

    deta_dx = (eta_j - eta_i) / dx   # water surface slope

    # Flow is zero where insufficient depth
    wet_x = hf_x > MIN_DEPTH
    q_prev_x = qx[:, :-1]

    # Local inertia momentum update
    numerator_x   = q_prev_x - G * hf_x * dt * deta_dx
    denominator_x = 1.0 + G * dt * n_x**2 * np.abs(q_prev_x) / (hf_x ** (7/3) + 1e-12)
    qx_new        = np.where(wet_x, numerator_x / denominator_x, 0.0)

    # Volume-conservation limiter (prevents draining cells below zero on steep slopes)
    # q cannot remove more than h_upstream * dx / (4*dt) from the source cell
    # Factor 4 accounts for up to 4 interfaces (x+, x-, y+, y-) per cell
    h_up_x  = np.where(qx_new >= 0, h[:, :-1], h[:, 1:])
    q_lim_x = h_up_x * dx / (4.0 * dt)
    qx_new   = np.clip(qx_new, -q_lim_x, q_lim_x)

    # ── Y-direction fluxes (between rows i and i+1) ─────────────────────────
    eta_i = eta[:-1, :]
    eta_j = eta[ 1:, :]
    z_i   = z[  :-1, :]
    z_j   = z[   1:, :]
    hf_y  = np.maximum(eta_i, eta_j) - np.maximum(z_i, z_j)
    hf_y  = np.maximum(hf_y, 0.0)

    n_i = n_grid[:-1, :]
    n_j = n_grid[ 1:, :]
    n_y = 0.5 * (n_i + n_j)

    deta_dy = (eta_j - eta_i) / dx

    wet_y = hf_y > MIN_DEPTH
    q_prev_y = qy[:-1, :]

    numerator_y   = q_prev_y - G * hf_y * dt * deta_dy
    denominator_y = 1.0 + G * dt * n_y**2 * np.abs(q_prev_y) / (hf_y ** (7/3) + 1e-12)
    qy_new        = np.where(wet_y, numerator_y / denominator_y, 0.0)

    # Volume-conservation limiter for y-direction
    h_up_y  = np.where(qy_new >= 0, h[:-1, :], h[1:, :])
    q_lim_y = h_up_y * dx / (4.0 * dt)
    qy_new   = np.clip(qy_new, -q_lim_y, q_lim_y)

    # ── Continuity: update water depth ─────────────────────────────────────
    # qx_new[:, j] = flow from cell j to cell j+1 (positive = eastward)
    # Net flux into interior cell j (1..nx-2): inflow from left - outflow to right
    flux_x = np.zeros_like(h)
    flux_x[:, 1:-1] += (qx_new[:, :-1] - qx_new[:, 1:]) * dt / dx
    # West boundary cell 0: only outflow to east → loses water when qx_new[:,0] > 0
    flux_x[:, 0]    -= qx_new[:, 0]  * dt / dx
    # East boundary cell nx-1: only inflow from west → gains water when qx_new[:,-1] > 0
    flux_x[:, -1]   += qx_new[:, -1] * dt / dx

    # qy_new[i, :] = flow from cell i to cell i+1 (positive = southward)
    flux_y = np.zeros_like(h)
    flux_y[1:-1, :] += (qy_new[:-1, :] - qy_new[1:, :]) * dt / dx
    # North boundary row 0: only outflow southward
    flux_y[0,    :]  -= qy_new[0,  :] * dt / dx
    # South boundary row ny-1: only inflow from north
    flux_y[-1,   :]  += qy_new[-1, :] * dt / dx

    # Assemble full qx/qy arrays (same shape as h)
    qx_out = np.zeros_like(h)
    qy_out = np.zeros_like(h)
    qx_out[:, :-1] = qx_new
    qy_out[:-1, :] = qy_new

    h_new = np.maximum(h + flux_x + flux_y, 0.0)

    # Absorbing (outflow) boundaries: water that reaches the domain edge exits freely.
    # Any remaining depth at boundary cells is cleared each step.
    h_new[0, :]  = 0.0   # north edge
    h_new[-1, :] = 0.0   # south edge
    h_new[:, 0]  = 0.0   # west edge
    h_new[:, -1] = 0.0   # east edge

    return h_new, qx_out, qy_out


# ── Main Simulation ───────────────────────────────────────────────────────────

def run_scenario(scenario_key, scenario_label, hyetograph_path, soil_params,
                 dem_arr, lake_bed_arr, lake_mask_arr, profile, dt_override=None):
    """
    Run a single flood scenario end-to-end.

    Parameters
    ----------
    scenario_key    : string identifier for output files
    scenario_label  : human-readable description
    hyetograph_path : path to atlas14_hyetograph_*.csv
    soil_params     : dict from soil_parameters.json
    dem_arr         : DEM array [m]
    lake_bed_arr    : lake bed DEM array [m]
    lake_mask_arr   : binary lake mask
    profile         : rasterio profile dict
    dt_override     : fixed time step [s] (None = adaptive CFL)
    """
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_label}")
    print(f"{'='*60}")

    if not os.path.exists(hyetograph_path):
        print(f"  ⚠ Hyetograph not found: {hyetograph_path}")
        print("  Run precipitation/noaa_atlas14.py first.")
        return None

    # Load hyetograph
    hyet = pd.read_csv(hyetograph_path)
    times_min  = hyet["time_min"].values
    incr_mm    = hyet["incremental_depth_mm"].values
    dt_hyet_s  = (times_min[1] - times_min[0]) * 60   # hyetograph time step [s]
    total_rain_mm = float(hyet["incremental_depth_mm"].sum())
    print(f"  Total rainfall : {total_rain_mm:.1f} mm over {times_min[-1]/60:.1f} hrs")
    print(f"  Peak intensity : {incr_mm.max() / (dt_hyet_s/3600):.1f} mm/hr")

    # Grid setup
    cell_m = abs(profile["transform"].a)
    ny, nx = dem_arr.shape
    total_cells = ny * nx
    print(f"  Grid           : {ny}×{nx} ({total_cells:,} cells, {cell_m:.1f}m resolution)")

    # Manning's n grid (uniform; can be refined with land cover data)
    n_grid = np.full((ny, nx), MANNING_N["residential"], dtype=np.float32)
    n_grid[lake_mask_arr > 0] = MANNING_N["water"]

    # Soil infiltration (use dominant map unit or default)
    if soil_params:
        first = next(iter(soil_params.values()))
        f0_mm_hr = first.get("f0_mm_hr", 112.5)
        fc_mm_hr = first.get("fc_mm_hr",  25.0)
        k_hr     = first.get("k_hr",       1.8)
    else:
        f0_mm_hr, fc_mm_hr, k_hr = 112.5, 25.0, 1.8  # Tavares fine sand defaults

    horton = HortonInfiltration(f0_mm_hr=f0_mm_hr, fc_mm_hr=fc_mm_hr, k_hr=k_hr)
    print(f"  Horton params  : f0={f0_mm_hr:.1f} mm/hr, fc={fc_mm_hr:.1f} mm/hr, k={k_hr:.2f} hr⁻¹")
    print(f"  Manning's n    : {MANNING_N['residential']:.3f} (residential), {MANNING_N['water']:.3f} (water)")

    # Lake storage model (if lake bed DEM available)
    # Only model lakes that are plausibly water bodies (< 40% of domain)
    lake_model = None
    n_lake_cells = int(lake_mask_arr.sum())
    domain_cells  = lake_mask_arr.size
    lake_fraction = n_lake_cells / domain_cells
    if n_lake_cells > 0 and lake_bed_arr is not None:
        if lake_fraction > 0.40:
            print(f"  ⚠ Lake mask covers {100*lake_fraction:.0f}% of domain — likely over-detection.")
            print("    Re-run dem_process.py to regenerate lake_mask.tif, or treat as no-lake domain.")
            n_lake_cells = 0  # disable lake model for this run
        else:
            wse0 = float(dem_arr[lake_mask_arr > 0].mean())
            cell_area = cell_m ** 2
            lake_model = LakeStorageModel(lake_bed_arr, lake_mask_arr, cell_area, wse0)
            vae_path = os.path.join(OUT_DIR, "lake_vae_curve.csv")
            lake_model.save_vae_table(vae_path)
            print(f"  Lake           : {n_lake_cells} cells ({100*lake_fraction:.1f}% of domain), WSE₀={wse0:.2f} m NAVD88")
            print(f"  Initial volume : {lake_model.V0:.0f} m³ ({lake_model.V0/1e6:.3f} × 10⁶ m³)")

    # Initialize state arrays
    z   = dem_arr.copy().astype(np.float64)
    h   = np.zeros((ny, nx), dtype=np.float64)       # water depth
    qx  = np.zeros((ny, nx), dtype=np.float64)       # x unit discharge
    qy  = np.zeros((ny, nx), dtype=np.float64)       # y unit discharge

    # Track peak values for output
    h_peak   = np.zeros((ny, nx), dtype=np.float32)
    vel_peak = np.zeros((ny, nx), dtype=np.float32)

    # Simulation loop — iterate over hyetograph time steps
    hydrograph_rows = []
    storm_start_s   = 0.0
    V_lake = lake_model.V0 if lake_model else 0.0
    z_outlet = float(dem_arr[lake_mask_arr > 0].max()) if n_lake_cells > 0 else 0.0

    t_wall_start = time.time()
    print(f"\n  Running simulation …")

    for step_i, t_min in enumerate(times_min):
        t_s = t_min * 60   # storm time [seconds]
        rain_rate_ms = incr_mm[step_i] / 1000 / dt_hyet_s   # [m/s]

        # Effective rainfall after Horton infiltration
        Pe_ms = horton.effective_rainfall(rain_rate_ms, t_s)

        # Sub-step the hydrodynamic solver within each hyetograph step
        t_substep = 0.0
        while t_substep < dt_hyet_s:
            dt = adaptive_dt(h, cell_m) if dt_override is None else float(dt_override)
            dt = min(dt, dt_hyet_s - t_substep)

            # Add effective rainfall as source term to water depth
            h += Pe_ms * dt

            # Zero rainfall on lake cells (lake model handles separately)
            if lake_model is not None:
                V_lake_inflow = float(Pe_ms * n_lake_cells * cell_m**2 * dt)
                V_lake += V_lake_inflow
                wse_lake = lake_model.wse_from_volume(V_lake)
                # Weir outflow: assume outlet at max shoreline elevation
                Q_out = lake_model.outflow(wse_lake, z_outlet, L_outlet=5.0)
                V_lake -= Q_out * dt
                V_lake  = max(V_lake, 0.0)

            # 2D flow routing
            h, qx, qy = local_inertia_step(h, qx, qy, z, n_grid, cell_m, dt)

            # Update peak tracking (velocity only where h > 1cm to avoid h≈0 division noise)
            h_f32 = h.astype(np.float32)
            vel   = np.where(h_f32 > 0.01,
                             np.sqrt(qx**2 + qy**2).astype(np.float32) / np.maximum(h_f32, 0.01),
                             0.0).astype(np.float32)
            h_peak   = np.maximum(h_peak,   h_f32)
            vel_peak = np.maximum(vel_peak, vel)

            t_substep += dt

        # Record hydrograph at each hyetograph step
        flooded_cells = int((h > DEPTH_THRESHOLD).sum())
        flooded_ha    = flooded_cells * cell_m**2 / 1e4
        lake_rise_m   = (lake_model.wse_from_volume(V_lake) - lake_model.wse0) if lake_model else 0.0
        mean_depth_m  = float(h[h > DEPTH_THRESHOLD].mean()) if flooded_cells > 0 else 0.0
        rain_int_mmhr = rain_rate_ms * 1000 * 3600

        hydrograph_rows.append({
            "time_min":        t_min,
            "rain_mm_hr":      round(rain_int_mmhr, 2),
            "Pe_mm_hr":        round(Pe_ms * 1000 * 3600, 2),
            "infilt_mm_hr":    round(horton.rate(t_s) * 1000 * 3600, 2),
            "flooded_ha":      round(flooded_ha, 3),
            "mean_depth_m":    round(mean_depth_m, 3),
            "lake_rise_m":     round(lake_rise_m, 3),
        })

        if step_i % max(1, len(times_min)//10) == 0 or step_i == len(times_min)-1:
            elapsed = time.time() - t_wall_start
            print(f"    t={t_min:5.0f}min  rain={rain_int_mmhr:5.1f} mm/hr  "
                  f"Pe={Pe_ms*3.6e6:.1f} mm/hr  "
                  f"flooded={flooded_ha:.2f} ha  lake_rise={lake_rise_m:+.3f}m  "
                  f"[{elapsed:.0f}s elapsed]")

    # Post-simulation: run-out (drain for 30 min after rain stops)
    print(f"  Post-storm drainage …")
    for _ in range(60):   # ~60 sub-steps of adaptive dt
        h, qx, qy = local_inertia_step(h, qx, qy, z, n_grid, cell_m,
                                        adaptive_dt(h, cell_m) if dt_override is None else float(dt_override))
        h_peak = np.maximum(h_peak, h.astype(np.float32))

    # ── Save outputs ──────────────────────────────────────────────────────────
    print(f"\n  Saving outputs for {scenario_key} …")

    depth_path = os.path.join(OUT_DIR, f"inundation_depth_{scenario_key}.tif")
    vel_path   = os.path.join(OUT_DIR, f"flow_velocity_{scenario_key}.tif")
    geojson_path = os.path.join(OUT_DIR, f"flood_extent_{scenario_key}.geojson")
    hydro_path   = os.path.join(OUT_DIR, f"hydrograph_{scenario_key}.csv")

    save_raster(h_peak,   profile, depth_path, dtype=np.float32)
    save_raster(vel_peak, profile, vel_path,   dtype=np.float32)

    flood_mask = (h_peak > DEPTH_THRESHOLD).astype(np.uint8)
    flooded_ha_peak = float(flood_mask.sum()) * cell_m**2 / 1e4
    save_geojson(flood_mask, profile, geojson_path, properties={
        "scenario": scenario_key,
        "total_rain_mm": round(total_rain_mm, 1),
        "flooded_ha": round(flooded_ha_peak, 2),
        "depth_threshold_m": DEPTH_THRESHOLD,
    })

    pd.DataFrame(hydrograph_rows).to_csv(hydro_path, index=False)

    max_depth = float(h_peak.max())
    max_vel   = float(vel_peak.max())
    peak_lake_rise = max(r["lake_rise_m"] for r in hydrograph_rows) if hydrograph_rows else 0.0

    print(f"\n── Results: {scenario_label} ─────────────────────────────")
    print(f"  Total rainfall     : {total_rain_mm:.1f} mm")
    print(f"  Peak flooded area  : {flooded_ha_peak:.2f} ha")
    print(f"  Max water depth    : {max_depth:.2f} m")
    print(f"  Max flow velocity  : {max_vel:.2f} m/s")
    print(f"  Max lake rise      : {peak_lake_rise:+.3f} m")
    print(f"  Depth raster       : {os.path.basename(depth_path)}")
    print(f"  Flood extent GeoJSON: {os.path.basename(geojson_path)}")

    return {
        "scenario": scenario_key,
        "total_rain_mm": total_rain_mm,
        "flooded_ha_peak": flooded_ha_peak,
        "max_depth_m": max_depth,
        "max_velocity_ms": max_vel,
        "max_lake_rise_m": peak_lake_rise,
    }


def main(run_scenarios="all", dt_override=None):
    print("Loading terrain and soil inputs …")

    dem_arr, profile = load_raster(os.path.join(DEM_DIR, "winter_garden_dem.tif"))
    if dem_arr is None:
        sys.exit("DEM not found. Run dem/dem_download.py first.")

    # Replace NaN/nodata with the domain mean (NaN propagates and breaks the solver)
    nodata_val = profile.get("nodata", None)
    nan_mask = ~np.isfinite(dem_arr)
    if nodata_val is not None:
        nan_mask |= (dem_arr == nodata_val)
    if nan_mask.any():
        fill_val = float(np.nanmean(dem_arr))
        dem_arr[nan_mask] = fill_val
        print(f"  Filled {nan_mask.sum():,} nodata cells in DEM with mean elevation {fill_val:.1f}m")

    # Prefer FWC bathymetric survey; fall back to shoreline-slope estimate
    _fwc_path = os.path.join(DEM_DIR, "lake_bed_dem_fwc.tif")
    _est_path = os.path.join(DEM_DIR, "lake_bed_dem_estimated.tif")
    _bed_path = _fwc_path if os.path.exists(_fwc_path) else _est_path
    lake_bed_arr, _ = load_raster(_bed_path)
    if lake_bed_arr is not None:
        print(f"  Lake bed source: {os.path.basename(_bed_path)}")
    lake_mask_arr, _= load_raster(os.path.join(DEM_DIR, "lake_mask.tif"), dtype=np.uint8)
    if lake_mask_arr is None:
        lake_mask_arr = np.zeros_like(dem_arr, dtype=np.uint8)

    if lake_bed_arr is None:
        lake_bed_arr = dem_arr.copy()
        print("  ⚠ lake_bed_dem_estimated.tif not found; using DEM as lake bed (run dem_process.py first)")
    else:
        # Fix NaN in lake bed
        lb_nan = ~np.isfinite(lake_bed_arr)
        if lb_nan.any():
            lake_bed_arr[lb_nan] = dem_arr[lb_nan]

    # Load soil parameters
    soil_params = None
    soil_path   = os.path.join(SOIL_DIR, "soil_parameters.json")
    if os.path.exists(soil_path):
        with open(soil_path) as f:
            soil_params = json.load(f)
        print(f"  Soil parameters : loaded from {soil_path}")
    else:
        print("  ⚠ soil_parameters.json not found; using Sandy soil defaults (Orange County FL)")

    # Select which scenarios to run
    if run_scenarios == "all":
        to_run = list(SCENARIOS.keys())
    else:
        to_run = [run_scenarios] if run_scenarios in SCENARIOS else list(SCENARIOS.keys())

    results = []
    for key in to_run:
        hyet_file, label = SCENARIOS[key]
        hyet_path = os.path.join(PRECIP_DIR, hyet_file)
        res = run_scenario(
            scenario_key=key,
            scenario_label=label,
            hyetograph_path=hyet_path,
            soil_params=soil_params,
            dem_arr=dem_arr.copy(),
            lake_bed_arr=lake_bed_arr.copy() if lake_bed_arr is not None else None,
            lake_mask_arr=lake_mask_arr.copy(),
            profile=profile,
            dt_override=dt_override,
        )
        if res:
            results.append(res)

    if results:
        summary = pd.DataFrame(results)
        summary_path = os.path.join(OUT_DIR, "simulation_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\n\nAll scenarios saved to {OUT_DIR}/")
        print(f"Summary table: {summary_path}")
        print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2D flood simulation for Winter Garden FL property")
    parser.add_argument("--scenario", type=str, default="all",
                        choices=list(SCENARIOS.keys()) + ["all"],
                        help="Scenario to run (default: all)")
    parser.add_argument("--dt", type=float, default=None,
                        help="Fixed time step [s] (default: adaptive CFL)")
    args = parser.parse_args()
    main(args.scenario, args.dt)
