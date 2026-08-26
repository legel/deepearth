"""
Multi-scenario GNN training-data sweep on site3_crop — GPU (torch/MPS), checkpointed
=========================================================================================
Generates the real per-point 3D shallow-water training data the GNN surrogate needs (see
CLAUDE.md's 2026-07-27 "how do we combine/scale the 3D mesh solver with a full watershed"
entries for the full plan this is step 1-2 of). Runs run_sim_gpu() (verified 2026-07-27 via
benchmark_gpu_solver.py: mass balance matches the CPU solver to -0.000% residual, ~3.95x
faster) across several rain scenarios on site3_crop -- a small crop directly at site3's own
pour point, chosen specifically so this training data comes from site3's own soil/terrain
domain rather than reusing site1/2 (different location, different soil -- would risk real
domain-shift error at inference time).

Checkpointed at scenario granularity: the mesh + masks are built ONCE (shared across all
scenarios, since geometry/soil don't change between rain scenarios) and cached to disk: a kill
mid-sweep only costs whichever scenario was in progress, not the mesh-build step or any already
-completed scenario -- same resumability principle as every other checkpointed script in this
project (lidar/build_site3_mesh_checkpointed.py, lidar/download_laz_tiles.py, etc).

Site parameterized (added 2026-07-27, second run): the first sweep ran on "site3_crop"
(710,302-node full-resolution mesh) -- real, valid PHYSICS data, but a real benchmark
(benchmark_gnn_forward.py) found one MeshGraphKAN forward+backward pass on that graph takes
134.7s, making GNN TRAINING on it intractable (~37 days for 1200 samples x 20 epochs). This
script now defaults to "site3_crop_coarse" instead -- same real-world pour-point location,
decimated to HydroGraphNet's own reference node-count scale (~6,500-7,000 nodes) specifically
for GNN training. The fine-resolution site3_crop scenario data already collected stays valid
and untouched (different checkpoint/output directory, keyed by SITE name below).

Usage:
    .venv/bin/python3 simulation/run_gnn_training_sweep.py
    .venv/bin/python3 simulation/run_gnn_training_sweep.py --site site3_crop
    # re-run as many times as needed; already-completed scenarios are skipped instantly
"""
import os, sys, json, pickle, time, argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), "simulation"))  # sibling solver modules (flood_sim_ian, mesh_shallow_water)
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))

from mesh_shallow_water import (  # noqa: E402
    build_combined_mesh, compute_ground_impervious_mask, compute_lake_mask,
    load_spatial_horton_points, load_nlcd_impervious_fraction_points,
    run_sim_gpu, build_ground_surface, build_building_surfaces,
)
from cache_bbox_points import load_cached_points  # noqa: E402
from build_lidar_pointcloud import bbox_from_center  # noqa: E402
from test_sites import get_site  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--site", default="site3_crop_coarse")
SITE = _ap.parse_args().site

CKPT_DIR = os.path.join(PROJ_DIR, "site3_gee_creek", "gnn_training", SITE, "checkpoints")
SCENARIO_DIR = os.path.join(PROJ_DIR, "site3_gee_creek", "gnn_training", SITE, "scenarios")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SCENARIO_DIR, exist_ok=True)
MESH_CKPT = os.path.join(CKPT_DIR, "mesh_and_masks.pkl")

# Real diversity, not just intensity: varies peak rain (40-200mm/hr, spanning and extending
# site1/2's Low/Medium/High 40/100/180 convention) AND duration/total-time (short sharp bursts
# vs. longer sustained rain) -- a GNN trained only on one duration shape would risk not
# generalizing to the real Ian event's very different (72-hour) temporal profile; this can't
# fully solve that gap (see CLAUDE.md's honest caveat about training/deployment temporal
# mismatch) but real duration variety here is a partial, real mitigation, not nothing.
SCENARIOS = [
    dict(name="low_short",       peak_mm_hr=40,  rain_duration_min=4,  total_min=10),
    dict(name="mediumlow_short", peak_mm_hr=70,  rain_duration_min=4,  total_min=10),
    dict(name="medium_short",    peak_mm_hr=100, rain_duration_min=4,  total_min=10),
    dict(name="mediumhigh_short",peak_mm_hr=130, rain_duration_min=4,  total_min=10),
    dict(name="high_short",      peak_mm_hr=160, rain_duration_min=4,  total_min=10),
    dict(name="veryhigh_short",  peak_mm_hr=180, rain_duration_min=4,  total_min=10),
    dict(name="medium_long",     peak_mm_hr=100, rain_duration_min=8,  total_min=15),
    dict(name="high_sharp",      peak_mm_hr=150, rain_duration_min=2,  total_min=8),
]


def load_pickle_checkpoint(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"  WARNING: {path} corrupt/truncated ({e}) — rebuilding")
        return None


def atomic_pickle_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)


def build_mesh_and_masks():
    cached = load_pickle_checkpoint(MESH_CKPT)
    if cached is not None:
        print("[mesh+masks] already cached, loading …")
        return cached
    print("[mesh+masks] building from scratch …")
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

    print(f"  {mesh['T']:,} triangles, {len(mesh['edges']['i']):,} edges")
    bundle = (mesh, ground_impervious_mask, lake_mask, ground_horton, ground_nlcd_grade)
    atomic_pickle_dump(bundle, MESH_CKPT)
    return bundle


def run_scenario(scenario, mesh, gim, lm, gh, gng):
    name = scenario["name"]
    out_path = os.path.join(SCENARIO_DIR, f"{name}.pkl")
    if os.path.exists(out_path):
        print(f"[scenario {name}] already done, skipping")
        with open(out_path, "rb") as fh:
            cached = pickle.load(fh)
        return cached.get("memory_stats"), cached.get("wall_s", 0.0)

    print(f"\n[scenario {name}] {scenario['peak_mm_hr']}mm/hr peak, "
          f"{scenario['rain_duration_min']}min rain, {scenario['total_min']}min total …")
    t0 = time.time()
    result = run_sim_gpu(
        mesh, dt_target=0.15, total_s=scenario["total_min"] * 60,
        rain_duration_s=scenario["rain_duration_min"] * 60,
        peak_mm_hr=scenario["peak_mm_hr"], frame_interval_s=3.0,
        ground_impervious_mask=gim, lake_mask=lm, ground_horton=gh, ground_nlcd_grade=gng,
        device="mps", verbose=True,
    )
    elapsed = time.time() - t0
    mb = result["mass_balance"]
    resid = mb["rain_vol_m3"] - mb["infil_vol_m3"] - mb["outflow_vol_m3"] - mb["stored_vol_m3"]
    resid_pct = 100 * resid / mb["rain_vol_m3"] if mb["rain_vol_m3"] else 0.0
    mem = result["memory_stats"]
    print(f"  [{name}] Done: {result['n_steps']} steps in {result['wall_s']:.1f}s "
          f"(total incl. overhead {elapsed:.1f}s)")
    print(f"  [{name}] Mass balance residual: {resid:.4f} m^3 ({resid_pct:.3f}%)")
    print(f"  [{name}] Ground h_max: {float(result['h_max'][~mesh['is_roof']].max()):.3f}m")
    print(f"  [{name}] Memory: MPS current={mem['mps_peak_current_allocated_mb']:.0f}MB  "
          f"MPS driver={mem['mps_peak_driver_allocated_mb']:.0f}MB  "
          f"process RSS={mem['process_peak_rss_mb']:.0f}MB")

    # Save only what a GNN training pipeline actually needs (frames_h, frame_times, mass
    # balance, scenario params) -- NOT frames_vel (real fix, 2026-07-27: the first scenario's
    # saved file was 2.28GB, mostly frames_vel, which is 3x the size of frames_h per frame and
    # is explicitly documented in run_sim()'s own docstring as "a visualization-purpose
    # reconstruction, not a rigorously derived physical quantity" -- not something a depth-
    # prediction GNN needs as a training label; dropping it cuts storage ~4x). Also NOT
    # re-saving the mesh itself (shared, already in mesh_and_masks.pkl) or tracer data
    # (irrelevant to surrogate training).
    save_data = {
        "scenario": scenario,
        "frames_h": result["frames_h"],
        "frame_times": result["frame_times"], "h_max": result["h_max"],
        "mass_balance": mb, "mass_balance_residual_pct": resid_pct,
        "n_steps": result["n_steps"], "wall_s": result["wall_s"],
        "memory_stats": mem,
    }
    atomic_pickle_dump(save_data, out_path)
    kb = os.path.getsize(out_path) / 1024
    print(f"  [{name}] saved to {out_path} ({kb/1024:.1f} MB)")
    return mem, elapsed


def main():
    sweep_t0 = time.time()
    mesh, gim, lm, gh, gng = build_mesh_and_masks()
    mem_records = []
    for scenario in SCENARIOS:
        mem, wall_s = run_scenario(scenario, mesh, gim, lm, gh, gng)
        if mem is not None:
            mem_records.append({"scenario": scenario["name"], "wall_s": wall_s, **mem})
    print("\nAll scenarios complete.")
    done = [s["name"] for s in SCENARIOS if os.path.exists(os.path.join(SCENARIO_DIR, f"{s['name']}.pkl"))]
    print(f"{len(done)}/{len(SCENARIOS)} scenario files present in {SCENARIO_DIR}")

    # Compute/memory report ("what
    # compute/memory does a full simulation need"). Real numbers from this actual sweep, not
    # estimated: per-scenario MPS allocator usage + process RSS, plus an honest note that
    # Apple Silicon's unified memory means there's no separate "GPU VRAM" budget the way a
    # CUDA machine would report — the MPS numbers here are PyTorch's own allocator bookkeeping
    # on the GPU-facing side, not a distinct physical memory pool from system RAM.
    if mem_records:
        report = {
            "platform": "Apple Silicon MPS (unified memory — no separate discrete-GPU VRAM pool)",
            "mesh_triangles": mem_records[0]["n_triangles"],
            "mesh_edges": mem_records[0]["n_edges"],
            "mps_recommended_max_mb": mem_records[0]["mps_recommended_max_mb"],
            "n_scenarios": len(mem_records),
            "total_sweep_wall_s": time.time() - sweep_t0,
            "per_scenario": mem_records,
            "peak_mps_current_allocated_mb": max(r["mps_peak_current_allocated_mb"] for r in mem_records),
            "peak_mps_driver_allocated_mb": max(r["mps_peak_driver_allocated_mb"] for r in mem_records),
            "peak_process_rss_mb": max(r["process_peak_rss_mb"] for r in mem_records),
            "avg_wall_s_per_scenario": sum(r["wall_s"] for r in mem_records) / len(mem_records),
        }
        report_path = os.path.join(CKPT_DIR, "..", "compute_memory_report.json")
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nCompute/memory report written to {os.path.abspath(report_path)}")
        print(f"  Peak MPS current allocated: {report['peak_mps_current_allocated_mb']:.0f} MB")
        print(f"  Peak MPS driver allocated:  {report['peak_mps_driver_allocated_mb']:.0f} MB")
        print(f"  Peak process RSS:           {report['peak_process_rss_mb']:.0f} MB")
        print(f"  Device recommended max:     {report['mps_recommended_max_mb']:.0f} MB")
        print(f"  Avg wall time/scenario:     {report['avg_wall_s_per_scenario']:.1f}s")


if __name__ == "__main__":
    main()
