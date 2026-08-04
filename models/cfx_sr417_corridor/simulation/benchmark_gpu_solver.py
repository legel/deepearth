"""
Benchmark run_sim_gpu() (torch/MPS) against the real, already-measured CPU number
=====================================================================================
Real CPU baseline, measured 2026-07-27 on site3_crop's 710,228-triangle mesh (site3's GNN-
training crop): 1414.2s / 44,618 steps = 31.7ms/step. This script runs the SAME mesh through
run_sim_gpu() for a short smoke test and reports real per-step timing + mass-balance residual,
so the GPU port is trusted based on a measured number, not assumed faster just because it's a
GPU — same "benchmark before committing" discipline as the rest of this project.

Usage:
    .venv/bin/python3 simulation/benchmark_gpu_solver.py
"""
import os, sys, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))

from mesh_shallow_water import (  # noqa: E402
    build_combined_mesh, compute_ground_impervious_mask, compute_lake_mask,
    load_spatial_horton_points, load_nlcd_impervious_fraction_points,
    run_sim, run_sim_gpu, build_ground_surface, build_building_surfaces,
)
from cache_bbox_points import load_cached_points  # noqa: E402
from build_lidar_pointcloud import bbox_from_center  # noqa: E402
from test_sites import get_site  # noqa: E402

SITE = "site3_crop"


def build_mesh_and_masks():
    site = get_site(SITE)
    lon_min, lat_min, lon_max, lat_max = bbox_from_center(site["lat"], site["lon"], site["radius_km"])
    ground = build_ground_surface(lon_min, lat_min, lon_max, lat_max,
                                   dem_cond_path=site.get("dem_cond_path"),
                                   decimate=site.get("ground_decimate", 1))
    pts = load_cached_points(lon_min, lat_min, lon_max, lat_max, site["bbox_cache_dir"])
    gxmin, gymin = ground.verts[:, :2].min(axis=0)
    gxmax, gymax = ground.verts[:, :2].max(axis=0)
    buildings, building_polys = build_building_surfaces(
        pts, (gxmin, gymin, gxmax, gymax), buildings_path=site.get("buildings_path"),
        max_points_per_building=site.get("roof_max_points"))
    mesh = build_combined_mesh(ground, buildings)

    ground_xy = mesh["xy"][:mesh["Tg"]]
    ground_impervious_mask = compute_ground_impervious_mask(ground_xy, roads_path=site.get("roads_path"))
    lake_mask = compute_lake_mask(ground_xy, site.get("pond_id3dhp"))
    ground_horton = load_spatial_horton_points(
        ground_xy, mukey_map_path=site.get("mukey_map_path"),
        mukey_legend_path=site.get("mukey_legend_path"), soil_json_path=site.get("soil_json_path"))
    already_hard = ground_impervious_mask.copy()
    if lake_mask is not None:
        already_hard |= lake_mask
    ground_nlcd_grade = load_nlcd_impervious_fraction_points(
        ground_xy, already_hard, nlcd_path=site.get("nlcd_path"))
    return mesh, ground_impervious_mask, lake_mask, ground_horton, ground_nlcd_grade


def main():
    print("Building site3_crop mesh + masks (shared by both benchmark runs) …")
    mesh, gim, lm, gh, gng = build_mesh_and_masks()
    print(f"  {mesh['T']:,} triangles, {len(mesh['edges']['i']):,} edges")

    common = dict(dt_target=0.15, total_s=30.0, rain_duration_s=20.0, peak_mm_hr=100.0,
                  frame_interval_s=5.0, ground_impervious_mask=gim, lake_mask=lm,
                  ground_horton=gh, ground_nlcd_grade=gng, verbose=False)

    print("\n[CPU] 30s-sim-time smoke test …")
    t0 = time.time()
    r_cpu = run_sim(mesh, **common)
    print(f"  {r_cpu['n_steps']} steps in {r_cpu['wall_s']:.2f}s "
          f"({r_cpu['wall_s']/r_cpu['n_steps']*1000:.2f} ms/step)")
    mb = r_cpu["mass_balance"]
    resid = mb["rain_vol_m3"] - mb["infil_vol_m3"] - mb["outflow_vol_m3"] - mb["stored_vol_m3"]
    print(f"  mass balance residual: {resid:.4f} m^3 "
          f"({100*resid/mb['rain_vol_m3'] if mb['rain_vol_m3'] else 0:.3f}%)")

    print("\n[GPU/MPS] 30s-sim-time smoke test …")
    t0 = time.time()
    r_gpu = run_sim_gpu(mesh, **common, device="mps")
    print(f"  {r_gpu['n_steps']} steps in {r_gpu['wall_s']:.2f}s "
          f"({r_gpu['wall_s']/r_gpu['n_steps']*1000:.2f} ms/step)")
    mb = r_gpu["mass_balance"]
    resid = mb["rain_vol_m3"] - mb["infil_vol_m3"] - mb["outflow_vol_m3"] - mb["stored_vol_m3"]
    print(f"  mass balance residual: {resid:.4f} m^3 "
          f"({100*resid/mb['rain_vol_m3'] if mb['rain_vol_m3'] else 0:.3f}%)")

    print("\n[compare] h_max CPU vs GPU:", float(r_cpu["h_max"].max()), "vs", float(r_gpu["h_max"].max()))
    print("[compare] steps CPU vs GPU:", r_cpu["n_steps"], "vs", r_gpu["n_steps"])

    real_cpu_ms_per_step = 31.7   # measured 2026-07-27, full 8-min site3_crop run, 44,618 steps
    gpu_ms_per_step = r_gpu["wall_s"] / r_gpu["n_steps"] * 1000
    speedup = real_cpu_ms_per_step / gpu_ms_per_step
    print(f"\n[verdict] real full-run CPU baseline: {real_cpu_ms_per_step:.1f} ms/step")
    print(f"[verdict] GPU/MPS measured this run:   {gpu_ms_per_step:.2f} ms/step")
    print(f"[verdict] speedup: {speedup:.2f}x {'(GPU faster)' if speedup > 1 else '(GPU SLOWER — do not use for the sweep)'}")


if __name__ == "__main__":
    main()
