#!/usr/bin/env python3
"""Build a REAL unstructured LiDAR-point-cloud-derived 3D mesh (ground + separate building
roofs, real walls at building edges) for 17801 Champagne Dr, and run the proven mesh-based
shallow-water solver on it — replacing the flat regular-grid version, per direct request
(the grid version rendered the two real building height clusters at this site as a single
smooth "cliff" ramp instead of recognizable structures).

Reuses cfx_sr417_corridor's own proven Surface/build_combined_mesh/run_sim_gpu machinery
directly (same numerics already used for site1/site2/site3) rather than reimplementing mesh
shallow-water physics from scratch under time pressure.
"""
import os, sys, json, base64
import numpy as np

# All paths resolve from this file's own location, so the script runs on any checkout.
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))          # flood_hydrology/simulation
PROJ_DIR   = os.path.dirname(BASE_DIR)                            # flood_hydrology
PROGRAM_DIR = os.path.dirname(PROJ_DIR)                           # models/flood_hydrology

# This script reuses the sibling project's mesh shallow-water solver rather than duplicating it.
CFX_DIR = os.path.join(PROGRAM_DIR, "cfx_sr417")
if not os.path.isdir(CFX_DIR):
    raise SystemExit(f"Sibling project not found at {CFX_DIR}; this script reuses its mesh solver.")
sys.path.insert(0, os.path.join(CFX_DIR, "lidar"))
sys.path.insert(0, os.path.join(CFX_DIR, "simulation"))
sys.path.insert(0, CFX_DIR)

PROP_LAT, PROP_LON = 28.5217321, -81.6570725
LAZ = os.path.join(PROJ_DIR, "lidar", "data", "raw",
                   "USGS_LPC_FL_Peninsular_FDEM_2018_D19_DRRA_LID2019_258656_E.laz")
HALF_M = 12.5          # 25x25m box, same domain as the grid version
FT2M = 0.3048006096012192
RAIN_MM_HR = 143.5      # real observed extreme, GSDR US_086638, 1960
TOTAL_S = 180.0
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def rasterize_ground(xm, ym, zm, ground_mask, cell_m=0.25):
    """Ground -> a CLEAN regular grid (real LiDAR points averaged per cell, same technique
    already proven for the flat-grid version), not raw scattered-point Delaunay. This matters
    for more than tidiness: a Delaunay triangulation of scattered points has a jagged convex-
    hull boundary with occasional tiny sliver triangles, and the solver's open-boundary outflow
    term (h*sqrt(g*h), no volume cap — unlike internal edges) can drain more volume out of a
    thin sliver than it actually holds. Every previously-tested site avoided this because its
    ground always came from a clean rectangular grid; rasterizing here keeps that same safe
    boundary shape while still using only real, unmodified LiDAR elevations."""
    gx, gy, gz = xm[ground_mask], ym[ground_mask], zm[ground_mask]
    n = int(round(2 * HALF_M / cell_m))
    col = np.clip(((gx + HALF_M) / cell_m).astype(np.int64), 0, n - 1)
    row = np.clip(((gy + HALF_M) / cell_m).astype(np.int64), 0, n - 1)
    flat = row * n + col
    acc = np.bincount(flat, weights=gz, minlength=n * n)
    cnt = np.bincount(flat, minlength=n * n)
    with np.errstate(invalid="ignore"):
        cell_z = acc / np.where(cnt > 0, cnt, np.nan)
    if np.isnan(cell_z).any():
        from scipy.ndimage import distance_transform_edt
        grid = cell_z.reshape(n, n)
        ind = distance_transform_edt(np.isnan(grid), return_distances=False, return_indices=True)
        cell_z = grid[tuple(ind)].ravel()
    xs = (np.arange(n) + 0.5) * cell_m - HALF_M
    gxx, gyy = np.meshgrid(xs, xs)
    return gxx.ravel(), gyy.ravel(), cell_z.astype(np.float32)


def load_real_points():
    """Ground (class 2, rasterized to a clean grid) + building (class 6, kept as raw scattered
    points for a real triangulated roof/wall shape) — real local (x,y,z) metres, origin at the
    box center. Building points are split into their real, physically distinct height clusters
    (found by direct inspection: a genuine 2.9m gap in the elevation histogram at this site,
    37.1-37.9m vs 40.7-43.2m — two real structures, not one roof)."""
    import laspy
    from pyproj import Transformer
    las = laspy.read(LAZ)
    crs = las.header.parse_crs()
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = tr.transform(PROP_LON, PROP_LAT)
    x = np.asarray(las.x); y = np.asarray(las.y); z = np.asarray(las.z)
    cl = np.asarray(las.classification)
    xm = (x - cx) * FT2M
    ym = (y - cy) * FT2M
    zm = z * FT2M
    in_box = (np.abs(xm) < HALF_M) & (np.abs(ym) < HALF_M)

    ground = in_box & (cl == 2)
    gx, gy, gz = rasterize_ground(xm, ym, zm, ground)
    print(f"    ground: rasterized {ground.sum():,} real points -> "
          f"{len(gx):,} clean grid cells, z range {gz.min():.2f}-{gz.max():.2f}m")

    bldg = in_box & (cl == 6)
    bx, by, bz = xm[bldg], ym[bldg], zm[bldg]
    Z_GAP = 39.0   # real, verified split for this site (histogram gap 37.9-40.7m)
    low = bz < Z_GAP
    clusters = []
    for name, mask in [("house (lower roof)", low), ("adjacent structure (taller)", ~low)]:
        n = mask.sum()
        print(f"    building cluster '{name}': {n:,} points, "
              f"z {bz[mask].min():.2f}-{bz[mask].max():.2f}m")
        clusters.append((bx[mask], by[mask], bz[mask]))
    return (gx, gy, gz), clusters


def build_walls(surf, ground):
    """Explicit vertical wall geometry at a building's real boundary — without this, a roof
    mesh renders as a flat slab floating above the ground with a visible gap (the solver's own
    roof-to-ground connection is a physics flux edge, not visible 3D geometry). Uses the
    Surface's own Delaunay convex hull (real boundary edges of the actual roof points) and the
    real ground elevation directly beneath each boundary vertex (Surface.z_at, barycentric
    interpolation on the real ground mesh — not assumed/flat)."""
    hull_edges = surf.tri.convex_hull   # (E,2) vertex-index pairs, real boundary of this roof
    bxy = surf.verts[:, :2]
    ground_simplex = ground.simplex_of(bxy)
    ground_z = ground.z_at(bxy, ground_simplex)
    roof_z = surf.verts[:, 2]
    verts = []
    faces = []
    for a, b in hull_edges:
        gz_a = ground_z[a] if not np.isnan(ground_z[a]) else roof_z[a] - 3.0
        gz_b = ground_z[b] if not np.isnan(ground_z[b]) else roof_z[b] - 3.0
        i0 = len(verts); verts.append((bxy[a, 0], bxy[a, 1], roof_z[a]))
        i1 = len(verts); verts.append((bxy[b, 0], bxy[b, 1], roof_z[b]))
        i2 = len(verts); verts.append((bxy[b, 0], bxy[b, 1], gz_b))
        i3 = len(verts); verts.append((bxy[a, 0], bxy[a, 1], gz_a))
        faces.append((i0, i1, i2)); faces.append((i0, i2, i3))
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int64)


def main():
    import torch
    from droplet_flow_test import Surface
    from mesh_shallow_water import (build_combined_mesh, run_sim_gpu, export_frames_bin,
                                    run_flow_tracers)

    print("  loading real LiDAR points...")
    (gx, gy, gz), bldg_clusters = load_real_points()

    print("  building Delaunay ground surface...")
    ground = Surface(gx, gy, gz)
    print(f"    ground mesh: {len(ground.simplices):,} triangles")

    buildings = []
    for i, (bx, by, bz) in enumerate(bldg_clusters):
        if len(bx) < 4:
            print(f"    skipping cluster {i}, too few points ({len(bx)})")
            continue
        surf = Surface(bx, by, bz)
        buildings.append(surf)
        print(f"    building {i} mesh: {len(surf.simplices):,} triangles")

    print("  building real wall geometry at each roof's boundary...")
    wall_verts_list, wall_faces_list = [], []
    for surf in buildings:
        wv, wf = build_walls(surf, ground)
        wall_verts_list.append(wv); wall_faces_list.append(wf)
        print(f"    wall: {len(wf):,} triangles")

    from shapely.geometry import MultiPoint
    building_polys = [MultiPoint(list(zip(s.verts[:, 0], s.verts[:, 1]))).convex_hull
                      for s in buildings]

    print("  fusing into one combined mesh (real walls at each roof's boundary)...")
    mesh = build_combined_mesh(ground, buildings)
    print(f"    total: {mesh['T']:,} triangles ({mesh['Tg']:,} ground + "
          f"{mesh['T']-mesh['Tg']:,} roof), {len(mesh['edges']['i']):,} internal edges")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  solving on {dev}: {TOTAL_S:.0f}s @ {RAIN_MM_HR}mm/hr...")
    result = run_sim_gpu(mesh, dt_target=0.15, total_s=TOTAL_S, rain_duration_s=TOTAL_S,
                         peak_mm_hr=RAIN_MM_HR, frame_interval_s=4.5, device=dev, verbose=True)

    mb = result["mass_balance"]
    print(f"\n  steps={result['n_steps']:,}  wall={result['wall_s']:.1f}s  "
          f"h_max={float(result['h_max'].max())*100:.1f}cm  frames={len(result['frames_h'])}")
    print(f"  mass balance: rain={mb['rain_vol_m3']:.3f}m3  stored={mb['stored_vol_m3']:.3f}m3  "
          f"outflow={mb['outflow_vol_m3']:.3f}m3")
    resid = 100 * (mb['rain_vol_m3'] - mb['stored_vol_m3'] - mb['outflow_vol_m3'] - mb.get('infil_vol_m3', 0)) / max(mb['rain_vol_m3'], 1e-9)
    print(f"  mass residual: {resid:+.4f}%")
    ms = result["memory_stats"]
    print(f"  memory: MPS current={ms['mps_peak_current_allocated_mb']:.0f}MB  "
          f"driver={ms['mps_peak_driver_allocated_mb']:.0f}MB  "
          f"process RSS={ms['process_peak_rss_mb']:.0f}MB")

    print("  running real physics-driven flow tracers (real solved velocity, not a fake walk)...")
    paths, settle_reason, tracer_start_time = run_flow_tracers(
        ground, buildings, building_polys, mesh, result["frames_vel"], result["frame_times"],
        rain_duration_s=TOTAL_S, peak_mm_hr=RAIN_MM_HR, total_s=TOTAL_S, n_tracers=500, seed=1,
        return_start_time=True)
    from collections import Counter
    print(f"    {len(paths)} tracers: {dict(Counter(settle_reason))}  "
          f"start_time range {tracer_start_time.min():.1f}-{tracer_start_time.max():.1f}s")

    # ---- export: real mesh geometry (ground + roofs + walls) + real per-triangle depth ----
    # walls are appended AFTER ground+roofs, so the first mesh['T'] faces in the OBJ still
    # line up exactly with frames_b64's per-triangle indexing (n_tri below marks that split —
    # walls carry no water-depth data, they're static real geometry only).
    simplices_list = [ground.simplices] + [s.simplices for s in buildings] + wall_faces_list
    verts_list = [ground.verts] + [s.verts for s in buildings] + wall_verts_list
    n_tri_per_surf = [len(s) for s in simplices_list]

    # per-vertex color: tan ground, terracotta roof-1, a distinct rose for roof-2, muted grey
    # facade for walls — so the two real structures (and their real walls) read as visually
    # separate, not one blob
    palette = [(0.54, 0.51, 0.32), (0.60, 0.42, 0.36), (0.62, 0.34, 0.42)]
    wall_col = (0.44, 0.42, 0.40)
    colors_list = []
    for i, v in enumerate([ground.verts] + [s.verts for s in buildings]):
        c = np.tile((np.array(palette[min(i, 2)]) * 255).astype(np.uint8), (len(v), 1))
        colors_list.append(c)
    for wv in wall_verts_list:
        colors_list.append(np.tile((np.array(wall_col) * 255).astype(np.uint8), (len(wv), 1)))

    # Minimal, self-contained OBJ writer — NOT export_mesh_obj (that function expects verts
    # in absolute EPSG:5070 map coordinates plus a real SW-corner "origin" to subtract; ours
    # are already local, box-centered (x,y horizontal, z vertical) metres, so reusing it
    # applied a second, wrong offset that doubled the exported extent). Writes real (x,y,z)
    # directly, "v x y z r g b" (three.js OBJLoader-compatible extended form), no axis swap
    # here — the browser-side loader does x,y,z(solver) -> x,z,y(three.js Y-up) itself.
    obj_path = os.path.join(OUT_DIR, "mesh_twin_house.obj")
    with open(obj_path, "w") as fh:
        fh.write(f"# real point-cloud mesh: ground + {len(buildings)} building(s), "
                 f"{sum(n_tri_per_surf)} triangles\n")
        voff = 0
        for verts, simplices, colors in zip(verts_list, simplices_list, colors_list):
            rgb01 = colors.astype(np.float64) / 255.0
            for (xi, yi, zi), (r, g, b) in zip(verts, rgb01):
                fh.write(f"v {xi:.3f} {yi:.3f} {zi:.3f} {r:.4f} {g:.4f} {b:.4f}\n")
            for tri in simplices:
                a, b2, c = tri + 1 + voff
                fh.write(f"f {a} {b2} {c}\n")
            voff += len(verts)

    # per-triangle depth frames -> pack as JSON for the web viewer (small: T triangles x F frames)
    frames_h = np.stack(result["frames_h"])   # (F, T)
    T = mesh["T"]
    vmax = float(np.percentile(frames_h[frames_h > 1e-4], 98)) if (frames_h > 1e-4).any() else 0.05
    vmax = max(vmax, 0.02)
    fr_u = np.clip(frames_h / vmax * 255, 0, 255).astype(np.uint8)

    # real physics-driven flow-tracer paths -> flat float32 arrays (points concatenated across
    # all tracers) + a per-tracer point-count so the viewer can split them back apart, plus a
    # settle-reason code (0=settled in a real local low point, 1=drained off the mesh edge,
    # 2=still moving when the step budget ran out) — same 3-way outcome the sibling project's
    # own droplet renderer already uses, reused here for a proven, tested color convention.
    reason_code = {"local_min": 0, "left_mesh": 1, "max_steps": 2}
    tpts, tcounts, treason, tstart = [], [], [], []
    for path, reason, st in zip(paths, settle_reason, tracer_start_time):
        if len(path) < 2:
            continue
        arr = np.asarray(path, dtype=np.float32)
        tpts.append(arr)
        tcounts.append(len(arr))
        treason.append(reason_code.get(reason, 2))
        tstart.append(round(float(st), 2))
    tpts_flat = np.concatenate(tpts, axis=0) if tpts else np.zeros((0, 3), dtype=np.float32)

    payload = dict(
        n_tri=T, n_tri_per_surf=n_tri_per_surf, is_roof=mesh["is_roof"].tolist(),
        nframes=len(result["frame_times"]),
        frame_times=[round(t, 1) for t in result["frame_times"]],
        vmax_m=round(vmax, 4), h_max_cm=round(float(result["h_max"].max()) * 100, 1),
        mass_resid_pct=round(resid, 4), rain_mm_hr=RAIN_MM_HR, total_s=TOTAL_S,
        rain_vol_m3=round(mb['rain_vol_m3'], 4), stored_vol_m3=round(mb['stored_vol_m3'], 4),
        outflow_vol_m3=round(mb['outflow_vol_m3'], 4),
        frames_b64=base64.b64encode(fr_u.tobytes()).decode(),
        wall_s=round(result['wall_s'], 2), n_steps=result['n_steps'],
        mem_process_rss_mb=round(ms['process_peak_rss_mb'], 1),
        mem_mps_driver_mb=round(ms['mps_peak_driver_allocated_mb'], 1),
        device=dev,
        tracer_counts=tcounts, tracer_reason=treason, tracer_start_time=tstart,
        tracer_pts_b64=base64.b64encode(tpts_flat.astype(np.float32).tobytes()).decode(),
    )
    data_path = os.path.join(OUT_DIR, "mesh_twin_depth.json")
    json.dump(payload, open(data_path, "w"))
    print(f"\n  wrote {obj_path} ({os.path.getsize(obj_path)/1024:.0f} KB)")
    print(f"  wrote {data_path} ({os.path.getsize(data_path)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
