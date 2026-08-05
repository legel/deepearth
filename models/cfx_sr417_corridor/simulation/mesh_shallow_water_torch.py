"""
GPU (PyTorch) port of mesh_shallow_water.py's solver loop — benchmark-only companion script.
=============================================================================================
Answers the team-lead question "has PyTorch + GPU been evaluated?" directly: this ports the
same Bates et al. (2010) local-inertial time-stepping loop (identical physics, identical mesh,
identical per-edge Froude cap + per-cell volume limiter + roof anti-pond drain) from NumPy/CPU
to torch tensors, runnable on Apple's MPS backend (this machine has no CUDA GPU — an M-series
Mac — MPS is the only non-CPU torch backend available here; the port is written against plain
torch ops with no MPS-specific code, so it would run on CUDA unchanged if this project ever
moves to an NVIDIA box).

Mesh construction (build_ground_surface/build_building_surfaces/build_combined_mesh) is reused
UNCHANGED from mesh_shallow_water.py — it's one-time CPU setup (Delaunay triangulation, DEM
I/O), not the per-step hot loop, so porting it buys nothing. Only run_sim's per-step loop
(edge flux, volume limiter, rain/infiltration, roof drain) is ported, since that's the part
that actually runs thousands of times and is where a GPU could plausibly help.

Scope limits vs. the NumPy version (deliberate, for a clean apples-to-apples speed/mass-balance
comparison rather than a full feature port):
  - No velocity-field reconstruction / flow-tracer export (frames_vel, run_flow_tracers) — that
    machinery is a visualization concern, not part of what "the solver" means for a compute
    benchmark, and porting it would muddy the timing comparison with unrelated work.
  - Frames are still saved (frames_h) so the two versions' outputs can be diffed for a real
    correctness check, not just eyeballed.

Usage:
    python3 simulation/mesh_shallow_water_torch.py --site site1 --device mps
    python3 simulation/mesh_shallow_water_torch.py --site site1 --device cpu   # torch-CPU,
        useful as a third data point: NumPy/CPU vs torch/CPU vs torch/MPS, isolating "GPU
        helped" from "torch's vectorized kernels are just different from NumPy's."
"""
import os, sys, time, json, argparse
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)

sys.path.insert(0, BASE_DIR)
from mesh_shallow_water import (  # noqa: E402  (reuse mesh construction + constants as-is)
    build_ground_surface, build_building_surfaces, build_combined_mesh,
    compute_ground_impervious_mask, compute_lake_mask, synthetic_rain_rate_ms,
    GROUND_MANNING_N, ROOF_MANNING_N, CFL_ALPHA, L_FLOOR_M, MAX_DEPTH_CAP,
    AREA_MIN_ACTIVE, ROOF_POND_MAX_M, MIN_DEPTH,
)
sys.path.insert(0, os.path.join(PROJ_DIR, "lidar"))
from build_lidar_pointcloud import bbox_from_center  # noqa: E402
from test_sites import SITES, get_site  # noqa: E402
sys.path.insert(0, os.path.join(PROJ_DIR, "simulation"))
from flood_sim_ian import HORTON, G  # noqa: E402  (horton_rate itself is np.exp-based —
                                       # not torch-tensor-safe on a GPU device, see horton_rate_torch)


def horton_rate_torch(t_s, f0_si, fc_si, k_si):
    """torch.exp equivalent of flood_sim_ian.horton_rate — that function calls np.exp, which
    silently round-trips a GPU tensor through .numpy() (raises on mps, and would silently
    force a CPU sync even where it doesn't raise), so this is a separate, GPU-safe copy of the
    same formula, not a refactor of the original (kept CPU/NumPy for the main solver)."""
    return fc_si + (f0_si - fc_si) * torch.exp(-k_si * t_s)


def run_sim_torch(mesh, dt_target, total_s, rain_duration_s, peak_mm_hr, frame_interval_s,
                   device, ground_impervious_mask=None, lake_mask=None, verbose=True):
    """Same physics as mesh_shallow_water.run_sim, ported to torch tensors on `device`.
    See module docstring for exactly what's in/out of scope vs. the NumPy original."""
    dev = torch.device(device)
    # float32 throughout, not float64: Apple's MPS backend has NO float64 support at all
    # (confirmed by hitting a hard TypeError attempting it), and consumer/prosumer CUDA GPUs
    # are also typically 8-32x slower at float64 than float32 — so float64 isn't just
    # unavailable here, it wouldn't be the realistic choice on a future CUDA box either. This
    # is a genuine precision/GPU tradeoff worth flagging: the NumPy solver's near-exact mass
    # balance (residuals ~1e-13%) partly reflects float64 accumulation, not just the algorithm;
    # expect this port's residual to be larger (still small, but not that small) from float32
    # rounding alone, independent of any logic difference.
    dt64 = torch.float32

    T = mesh["T"]
    Tg = mesh["Tg"]
    area = torch.as_tensor(mesh["area"], dtype=dt64, device=dev)
    z = torch.as_tensor(mesh["z"], dtype=dt64, device=dev)
    is_roof = torch.as_tensor(mesh["is_roof"], dtype=torch.bool, device=dev)
    i_all = torch.as_tensor(mesh["edges"]["i"], dtype=torch.int64, device=dev)
    j_all = torch.as_tensor(mesh["edges"]["j"], dtype=torch.int64, device=dev)
    L_all = torch.as_tensor(mesh["edges"]["L"], dtype=dt64, device=dev)
    W_all = torch.as_tensor(mesh["edges"]["W"], dtype=dt64, device=dev)
    L_min = float(L_all.min().item())

    # Per-triangle edge-width sum -> effective Courant cell scale (same fix as the NumPy
    # version: many small roof-eave edges converging on one small ground cell need a per-cell
    # capacity term, not just per-edge length, or dt collapses toward zero mesh-wide).
    edge_width_sum = torch.zeros(T, dtype=dt64, device=dev)
    edge_width_sum.index_add_(0, i_all, W_all)
    edge_width_sum.index_add_(0, j_all, W_all)
    has_edges = edge_width_sum > 0
    cell_scale = torch.where(has_edges, area / torch.clamp(edge_width_sum, min=1e-9),
                              torch.full_like(area, float("inf")))
    cell_scale_min = float(cell_scale[has_edges].min().item()) if has_edges.any() else L_min
    L_min = max(min(L_min, cell_scale_min), L_FLOOR_M)

    n_arr = torch.where(is_roof, torch.tensor(ROOF_MANNING_N, dtype=dt64, device=dev),
                         torch.tensor(GROUND_MANNING_N, dtype=dt64, device=dev))
    n_edge = 0.5 * (n_arr[i_all] + n_arr[j_all])

    is_impervious = is_roof.clone()
    if ground_impervious_mask is not None:
        gim = torch.as_tensor(ground_impervious_mask, dtype=torch.bool, device=dev)
        is_impervious[:Tg] |= gim
    if lake_mask is not None:
        lm = torch.as_tensor(lake_mask, dtype=torch.bool, device=dev)
        is_impervious[:Tg] |= lm

    zero64 = torch.zeros(T, dtype=dt64, device=dev)
    f0_si = torch.where(is_impervious, zero64, torch.full_like(zero64, HORTON["f0"] / 1000 / 3600))
    fc_si = torch.where(is_impervious, zero64, torch.full_like(zero64, HORTON["fc"] / 1000 / 3600))
    k_si = torch.where(is_impervious, zero64, torch.full_like(zero64, HORTON["k"] / 3600))

    bnd_owner = torch.as_tensor(mesh["ground_open_boundary"]["owner"], dtype=torch.int64, device=dev)
    bnd_len = torch.as_tensor(mesh["ground_open_boundary"]["length"], dtype=dt64, device=dev)
    roof_below_tri = torch.as_tensor(mesh["roof_below"]["roof_tri"], dtype=torch.int64, device=dev)
    roof_below_ground = torch.as_tensor(mesh["roof_below"]["ground_tri"], dtype=torch.int64, device=dev)

    h = torch.zeros(T, dtype=dt64, device=dev)
    q = torch.zeros(len(i_all), dtype=dt64, device=dev)
    h_max = torch.zeros(T, dtype=dt64, device=dev)

    cum_rain_vol = 0.0
    cum_infil_vol = 0.0
    cum_outflow_vol = 0.0
    area_sum = float(area.sum().item())

    frames_h, frame_times = [], []
    lake_level_ts, lake_time_ts = [], []
    lake_area = float(area[:Tg][torch.as_tensor(lake_mask, device=dev)].sum().item()) if lake_mask is not None else 0.0

    t_s = 0.0
    last_frame_t = -1e9
    step = 0
    t0 = time.time()

    while t_s < total_s:
        h_max_local = float(h.max().item())   # CPU/GPU sync point — same dt-adaptivity cost
        dt = min(dt_target, CFL_ALPHA * L_min / (G * h_max_local) ** 0.5) if h_max_local > MIN_DEPTH else dt_target
        dt = min(dt, total_s - t_s)

        eta = z + h
        hf = torch.clamp(torch.maximum(eta[i_all], eta[j_all]) - torch.maximum(z[i_all], z[j_all]), min=0.0)
        num = q - G * hf * dt * (eta[j_all] - eta[i_all]) / L_all
        # 7/3, not 4/3 — see MANNING_EXP in flood_sim_ian.py for the derivation and the
        # numerical Manning steady-state check. Corrected 2026-08-04.
        denom = 1.0 + G * dt * n_edge ** 2 * torch.abs(q) / (hf ** (7.0 / 3.0) + 1e-10)
        q = torch.where(hf > MIN_DEPTH, num / denom, torch.zeros_like(num))
        q_cap = 0.9 * hf * torch.sqrt(G * torch.clamp(hf, min=MIN_DEPTH))
        q = torch.clamp(q, -q_cap, q_cap)

        vol_edge = q * W_all * dt
        out_i = torch.zeros(T, dtype=dt64, device=dev)
        out_i.index_add_(0, i_all, torch.clamp(vol_edge, min=0.0))
        out_j = torch.zeros(T, dtype=dt64, device=dev)
        out_j.index_add_(0, j_all, torch.clamp(-vol_edge, min=0.0))
        outflow_vol = out_i + out_j
        available_vol = h * area
        scale = torch.clamp(available_vol / torch.clamp(outflow_vol, min=1e-12), max=1.0)
        scale_send = torch.where(vol_edge >= 0, scale[i_all], scale[j_all])
        vol_edge = vol_edge * scale_send

        dh = torch.zeros(T, dtype=dt64, device=dev)
        dh.index_add_(0, i_all, -vol_edge / area[i_all])
        dh.index_add_(0, j_all, vol_edge / area[j_all])

        hf_b = torch.clamp(h[bnd_owner], min=0.0)
        q_out = torch.where(hf_b > MIN_DEPTH, hf_b * torch.sqrt(G * torch.clamp(hf_b, min=MIN_DEPTH)),
                             torch.zeros_like(hf_b))
        out_vol = float((q_out * bnd_len * dt).sum().item())
        cum_outflow_vol += out_vol
        dh.index_add_(0, bnd_owner, -q_out * bnd_len * dt / area[bnd_owner])

        rain_ms = synthetic_rain_rate_ms(t_s, rain_duration_s, peak_mm_hr)
        inf_ms = horton_rate_torch(t_s, f0_si, fc_si, k_si)
        Pe = torch.clamp(torch.as_tensor(rain_ms, dtype=dt64, device=dev) - inf_ms, min=0.0)
        cum_rain_vol += rain_ms * dt * area_sum
        rain_ms_t = torch.as_tensor(rain_ms, dtype=dt64, device=dev)
        cum_infil_vol += float((torch.minimum(rain_ms_t, inf_ms) * area).sum().item()) * dt
        dh = dh + Pe * dt

        h = torch.clamp(h + dh, 0.0, MAX_DEPTH_CAP)

        h_roof_below = h[roof_below_tri]
        roof_excess = torch.clamp(h_roof_below - ROOF_POND_MAX_M, min=0.0)
        if roof_excess.any():
            h[roof_below_tri] = h[roof_below_tri] - roof_excess
            excess_vol = roof_excess * area[roof_below_tri]
            add = torch.zeros(T, dtype=dt64, device=dev)
            add.index_add_(0, roof_below_ground, excess_vol)
            h = h + add / area

        h_max = torch.maximum(h_max, h)

        if lake_mask is not None:
            lm_t = torch.as_tensor(lake_mask, device=dev)
            lake_level_ts.append(float(torch.sum(h[:Tg][lm_t] * area[:Tg][lm_t]).item() / lake_area))
            lake_time_ts.append(t_s)

        if t_s - last_frame_t >= frame_interval_s:
            frames_h.append(h.to("cpu", torch.float32).numpy().copy())
            frame_times.append(t_s)
            last_frame_t = t_s

        if verbose and step % 200 == 0:
            n_wet = int((h > 0.01).sum().item())
            print(f"  t={t_s:6.1f}s  dt={dt:.4f}s  rain={rain_ms*3600*1000:6.1f}mm/hr  "
                  f"h_max={float(h.max().item()):.3f}m  wet_tri={n_wet:6d}/{T}  "
                  f"[{t_s/total_s*100:5.1f}% {time.time()-t0:.0f}s]")

        t_s += dt
        step += 1

    if not frame_times or frame_times[-1] < t_s - 1e-6:
        frames_h.append(h.to("cpu", torch.float32).numpy().copy())
        frame_times.append(t_s)

    return dict(
        h_final=h.cpu().numpy(), h_max=h_max.cpu().numpy(), frames_h=frames_h, frame_times=frame_times,
        n_steps=step, wall_s=time.time() - t0,
        mass_balance=dict(rain_vol_m3=cum_rain_vol, infil_vol_m3=cum_infil_vol,
                           outflow_vol_m3=cum_outflow_vol,
                           stored_vol_m3=float((h * area).sum().item())),
        lake_level_ts=lake_level_ts, lake_time_ts=lake_time_ts, lake_area_m2=lake_area,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=list(SITES), default="site1")
    ap.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    ap.add_argument("--peak-rain-mm-hr", type=float, default=100.0)
    ap.add_argument("--rain-duration-min", type=float, default=4.0)
    ap.add_argument("--total-min", type=float, default=8.0)
    ap.add_argument("--dt", type=float, default=0.15)
    ap.add_argument("--frame-interval-s", type=float, default=3.0)
    args = ap.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available on this machine — falling back to cpu")
        args.device = "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available on this machine — falling back to cpu")
        args.device = "cpu"

    site = get_site(args.site)
    lon_min, lat_min, lon_max, lat_max = bbox_from_center(site["lat"], site["lon"], site["radius_km"])

    print("=" * 70)
    print(f"GPU (torch/{args.device}) shallow-water solver benchmark — [{args.site}: {site['label']}]")
    print("=" * 70)

    print("\n[1/4] Building mesh (identical, unchanged, CPU — same as the NumPy version) …")
    ground = build_ground_surface(lon_min, lat_min, lon_max, lat_max)
    from build_lidar_pointcloud import load_points_in_bbox
    pts = load_points_in_bbox(lon_min, lat_min, lon_max, lat_max)
    gxmin, gymin = ground.verts[:, :2].min(axis=0)
    gxmax, gymax = ground.verts[:, :2].max(axis=0)
    buildings, building_polys = build_building_surfaces(pts, (gxmin, gymin, gxmax, gymax))
    mesh = build_combined_mesh(ground, buildings)
    print(f"  {mesh['T']:,} triangles ({mesh['Tg']:,} ground + {mesh['T']-mesh['Tg']:,} roof), "
          f"{len(mesh['edges']['i']):,} edges")

    print("\n[2/4] Impervious/lake classification (same as NumPy version) …")
    ground_xy = mesh["xy"][:mesh["Tg"]]
    ground_impervious_mask = compute_ground_impervious_mask(ground_xy)
    lake_mask = compute_lake_mask(ground_xy, site.get("pond_id3dhp"))

    print(f"\n[3/4] Running solver on {args.device} …")
    result = run_sim_torch(mesh, dt_target=args.dt, total_s=args.total_min * 60,
                            rain_duration_s=args.rain_duration_min * 60,
                            peak_mm_hr=args.peak_rain_mm_hr, frame_interval_s=args.frame_interval_s,
                            device=args.device, ground_impervious_mask=ground_impervious_mask,
                            lake_mask=lake_mask)

    mb = result["mass_balance"]
    residual = mb["rain_vol_m3"] - mb["infil_vol_m3"] - mb["outflow_vol_m3"] - mb["stored_vol_m3"]
    residual_pct = 100 * residual / mb["rain_vol_m3"] if mb["rain_vol_m3"] > 0 else 0.0
    print(f"\n[4/4] Done: {result['n_steps']} steps in {result['wall_s']:.1f}s wall time  (device={args.device})")
    print(f"  Mass balance (m^3): rain={mb['rain_vol_m3']:.3f}  infil={mb['infil_vol_m3']:.3f}  "
          f"outflow={mb['outflow_vol_m3']:.3f}  stored={mb['stored_vol_m3']:.3f}  "
          f"residual={residual:.4f} ({residual_pct:.2f}%)")
    h_max = result["h_max"]
    print(f"  Ground h_max={h_max[~mesh['is_roof']].max():.3f}m  Roof h_max={h_max[mesh['is_roof']].max():.3f}m")

    out = os.path.join(BASE_DIR, "outputs", f"bench_torch_{args.device}_{args.site}.json")
    with open(out, "w") as fh:
        json.dump({
            "device": args.device, "site": args.site, "n_triangles": mesh["T"],
            "n_edges": len(mesh["edges"]["i"]), "n_steps": result["n_steps"],
            "wall_s": result["wall_s"], "mass_balance_residual_pct": residual_pct,
            "ground_h_max_m": float(h_max[~mesh["is_roof"]].max()),
            "roof_h_max_m": float(h_max[mesh["is_roof"]].max()),
        }, fh, indent=2)
    print(f"  {out}")


if __name__ == "__main__":
    main()
