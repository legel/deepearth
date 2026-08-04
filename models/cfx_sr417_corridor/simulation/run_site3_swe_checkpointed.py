"""
Checkpointed shallow-water solver run for site3 (Gee Creek gauge-matched validation site)
==========================================================================================
Reuses the already-built ground.pkl/buildings.pkl checkpoints from
lidar/build_site3_mesh_checkpointed.py (same Surface objects the droplet-prototype mesh used)
instead of rebuilding them via build_ground_surface()/build_building_surfaces() — those stages
are already verified correct and cached; re-running them here would just cost the same 4s+67.5s
for no benefit.

This site is much larger than site1/site2 (5.7M total triangles vs. site2's largest at 682,768,
~8.4x), and mesh_shallow_water.py's run_sim() is a single uninterrupted Python while-loop with
no internal checkpointing — a kill mid-run (see this project's CLAUDE.md on the never-fully-
diagnosed "long background process gets killed" environment behavior) would lose all progress.
Added --total-min/--smoke-test support here specifically to measure REAL per-step wall time at
this mesh's actual scale before committing to a long run, rather than extrapolating from site2's
numbers (different triangle/edge density, different decimate settings).

Usage:
    # smoke test: short sim, prints real per-step timing, does NOT export anything
    python3 simulation/run_site3_swe_checkpointed.py --smoke-test

    # full run (only after smoke-test timing looks tractable)
    python3 simulation/run_site3_swe_checkpointed.py --peak-rain-mm-hr 100
"""
import os, sys, json, pickle, time, argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))

from mesh_shallow_water import (  # noqa: E402
    build_combined_mesh, compute_ground_impervious_mask, compute_lake_mask,
    load_spatial_horton_points, load_nlcd_impervious_fraction_points, run_sim,
    run_flow_tracers, export_frames_bin, export_heightmap_bin, export_tracer_paths_drop,
    GEO_META, OUT_DIR, POND_OUTLET_ORIFICE_D_M,
)
from test_sites import get_site  # noqa: E402

SITE = "site3"
CKPT_DIR = os.path.join(PROJ_DIR, "site3_gee_creek", "lidar", "data", "checkpoints")
GROUND_CKPT = os.path.join(CKPT_DIR, "ground.pkl")
BUILDINGS_CKPT = os.path.join(CKPT_DIR, "buildings.pkl")


def load_pickle_checkpoint(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"  WARNING: {path} exists but is corrupt/truncated ({e})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true",
                     help="very short sim (30s sim-time), prints real per-step timing, no export")
    ap.add_argument("--peak-rain-mm-hr", type=float, default=100.0)
    ap.add_argument("--rain-duration-min", type=float, default=4.0)
    ap.add_argument("--total-min", type=float, default=8.0)
    ap.add_argument("--dt", type=float, default=0.15)
    ap.add_argument("--frame-interval-s", type=float, default=3.0)
    args = ap.parse_args()

    site = get_site(SITE)
    suffix = f"_{SITE}"

    ground = load_pickle_checkpoint(GROUND_CKPT)
    buildings, building_polys = load_pickle_checkpoint(BUILDINGS_CKPT)
    if ground is None or buildings is None:
        print("ERROR: ground.pkl/buildings.pkl not found — run "
              "lidar/build_site3_mesh_checkpointed.py first")
        sys.exit(1)
    print(f"Loaded checkpoints: {len(ground.simplices):,} ground triangles, "
          f"{len(buildings)} buildings")

    t0 = time.time()
    mesh = build_combined_mesh(ground, buildings)
    print(f"[mesh] {mesh['T']:,} triangles ({mesh['Tg']:,} ground + {mesh['T']-mesh['Tg']:,} roof)  "
          f"{len(mesh['edges']['i']):,} flux edges  ({time.time()-t0:.1f}s)")

    t0 = time.time()
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
    print(f"[masks] {int(ground_impervious_mask.sum())}/{mesh['Tg']} impervious ground triangles  "
          f"({time.time()-t0:.1f}s)")

    if args.smoke_test:
        print("\n[smoke test] running 30s sim-time to measure real per-step wall time …")
        t0 = time.time()
        result = run_sim(mesh, dt_target=args.dt, total_s=30.0, rain_duration_s=20.0,
                          peak_mm_hr=args.peak_rain_mm_hr, frame_interval_s=5.0,
                          ground_impervious_mask=ground_impervious_mask, lake_mask=lake_mask,
                          ground_horton=ground_horton, ground_nlcd_grade=ground_nlcd_grade,
                          pond_outlet_diameter_m=None)
        wall_s = result["wall_s"]
        n_steps = result["n_steps"]
        print(f"\n[smoke test] {n_steps} steps in {wall_s:.2f}s wall "
              f"({wall_s/n_steps*1000:.2f} ms/step, sim-dt~{30.0/n_steps*1000:.2f}ms)")
        steps_per_simsec = n_steps / 30.0
        ms_per_step = wall_s / n_steps * 1000
        for total_min in [4.0, 8.0]:
            est_steps = steps_per_simsec * total_min * 60
            est_wall_s = est_steps * ms_per_step / 1000
            print(f"  extrapolated: --total-min {total_min} -> ~{est_steps:.0f} steps, "
                  f"~{est_wall_s:.0f}s wall (~{est_wall_s/60:.1f} min)")
        return

    print(f"\n[run] {args.peak_rain_mm_hr:.0f}mm/hr peak, {args.rain_duration_min:.1f}min rain, "
          f"{args.total_min:.1f}min total …")
    result = run_sim(mesh, dt_target=args.dt, total_s=args.total_min * 60,
                      rain_duration_s=args.rain_duration_min * 60,
                      peak_mm_hr=args.peak_rain_mm_hr, frame_interval_s=args.frame_interval_s,
                      ground_impervious_mask=ground_impervious_mask, lake_mask=lake_mask,
                      ground_horton=ground_horton, ground_nlcd_grade=ground_nlcd_grade,
                      pond_outlet_diameter_m=None)

    mb = result["mass_balance"]
    residual = mb["rain_vol_m3"] - mb["infil_vol_m3"] - mb["outflow_vol_m3"] - mb["stored_vol_m3"]
    residual_pct = 100 * residual / mb["rain_vol_m3"] if mb["rain_vol_m3"] > 0 else 0.0
    print(f"\nDone: {result['n_steps']} steps in {result['wall_s']:.1f}s wall time")
    print(f"Mass balance (m^3): rain={mb['rain_vol_m3']:.3f}  infil={mb['infil_vol_m3']:.3f}  "
          f"outflow={mb['outflow_vol_m3']:.3f}  stored={mb['stored_vol_m3']:.3f}  "
          f"residual={residual:.4f} ({residual_pct:.2f}%)")

    n_tracers = 2500
    print(f"\nAdvecting {n_tracers} physics-driven flow tracers …")
    tracer_paths, tracer_reason = run_flow_tracers(
        ground, buildings, building_polys, mesh, result["frames_vel"], result["frame_times"],
        rain_duration_s=args.rain_duration_min * 60, peak_mm_hr=args.peak_rain_mm_hr,
        total_s=args.total_min * 60, n_tracers=n_tracers,
    )

    print("\nExporting …")
    lidar_data_dir = os.path.join(PROJ_DIR, "lidar", "data")
    export_frames_bin(mesh, result["frames_h"], result["frame_times"],
                       os.path.join(OUT_DIR, f"swe_mesh_frames{suffix}.bin"))
    with open(GEO_META) as fh:
        geo_meta = json.load(fh)
    export_heightmap_bin(mesh, geo_meta,
                          os.path.join(lidar_data_dir, f"swe_surface_heightmap{suffix}.bin"))
    export_tracer_paths_drop(tracer_paths, tracer_reason, geo_meta,
                              os.path.join(OUT_DIR, f"flow_tracer_paths{suffix}.bin"))

    summary = {
        "site": SITE, "site_label": site["label"],
        "n_triangles": mesh["T"], "n_ground_triangles": mesh["Tg"],
        "n_roof_triangles": mesh["T"] - mesh["Tg"], "n_buildings": len(buildings),
        "n_edges": len(mesh["edges"]["i"]), "n_frames": len(result["frame_times"]),
        "n_steps": result["n_steps"], "wall_s": result["wall_s"],
        "peak_rain_mm_hr": args.peak_rain_mm_hr, "rain_duration_min": args.rain_duration_min,
        "total_min": args.total_min,
        "ground_h_max_m": float(result["h_max"][~mesh["is_roof"]].max()),
        "roof_h_max_m": float(result["h_max"][mesh["is_roof"]].max()),
        "mass_balance_m3": mb, "mass_balance_residual_pct": residual_pct,
        "n_impervious_road_triangles": int(ground_impervious_mask.sum()),
        "flow_tracers": {
            "n_seeded": len(tracer_paths),
            "n_local_min": int((tracer_reason == "local_min").sum()),
            "n_left_mesh": int((tracer_reason == "left_mesh").sum()),
            "n_max_steps": int((tracer_reason == "max_steps").sum()),
        },
        "test_area": {"lat": site["lat"], "lon": site["lon"], "radius_km": site["radius_km"]},
    }
    with open(os.path.join(OUT_DIR, f"swe_mesh_summary{suffix}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  swe_mesh_summary{suffix}.json")
    print("\nDONE.")


if __name__ == "__main__":
    main()
