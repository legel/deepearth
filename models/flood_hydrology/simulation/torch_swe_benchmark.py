#!/usr/bin/env python3
"""
torch_swe_benchmark.py — GPU (PyTorch/MPS) real-physics rainfall-flow benchmark.

Purpose (2026-07-28 meeting direction, Lance): stop building learned surrogates;
instead run the ACTUAL shallow-water physics as pure PyTorch tensor ops on the GPU,
at TRUE physical resolution with NO downsampling, over a SMALL real domain (one house
+ yard), and measure WALL-CLOCK time per second of simulated real-world physics.

Domain: 25x25 m box centered on the ecodash / Dix.Hite digital-twin property
    17801 Champagne Dr, Winter Garden FL  (28.5217321, -81.6570725)
    https://digitaltwin.ecodash.ai/  (github.com/legel/digitaltwin, demo-site)
Terrain: real 2018 USGS QL2 LiDAR (~40 pts/m^2), ground+building returns, no downsample.
Rain: real peak-Hurricane-Ian hourly rate from this project's ASOS record, applied as a
    physical source term  h += R*dt   (R = mm_per_hr / 3.6e6  [m/s]).

Method: Bates et al. (2010) local-inertial shallow water on a structured grid — the same
    scheme this project's CPU solvers use — reimplemented as vectorized torch ops so every
    timestep is a handful of tensor slices (no python per-cell loop). Adaptive CFL dt.

Sweep: cell size dx = 0.25 -> 0.10 -> 0.05 -> 0.02 m until MPS runs out of memory.
    The OOM point is itself a measured result (like the closed-out GNN benchmark).

Output: prints a table + writes simulation/outputs/torch_swe_benchmark.json.
"""
import os, sys, json, time, argparse, math
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Site / physical constants ──────────────────────────────────────────────────
PROP_LAT, PROP_LON = 28.5217321, -81.6570725     # ecodash demo-site center
LAZ = os.path.join(PROJ_DIR, "lidar", "data", "raw",
                   "USGS_LPC_FL_Peninsular_FDEM_2018_D19_DRRA_LID2019_258656_E.laz")
FT2M = 0.3048006096012192                          # US survey foot -> metre
G       = 9.81
MANNING = 0.030                                    # mixed suburban roughness
ALPHA   = 0.7                                      # CFL safety factor
PEAK_RAIN_MM_HR = 66.8                             # real peak Hurricane Ian hourly, ASOS MCO,
                                                    # 2022-09-29 06:00 UTC (verified against the
                                                    # Ian date window directly — the previous
                                                    # value here, 77.2, was actually this same
                                                    # station's ALL-TIME peak from an unrelated
                                                    # 2023-05-18 storm, mislabeled as Ian)
SURFACE_CLASSES = (2, 6)                            # ground + building (the flow surface)
DENSITY_PTS_M2  = 40.0                              # measured LiDAR density at this site


# ── Step 1: build a full-resolution DEM crop from raw LiDAR (no downsampling) ────
def build_dem(dx_m, half_m=12.5, verbose=True):
    """Rasterize ground+building LiDAR returns to a dx_m grid over a 2*half_m box.
    Returns (Z[H,W] float32 metres, meta dict). Empty cells nearest-filled."""
    import laspy
    from pyproj import Transformer
    las = laspy.read(LAZ)
    crs = las.header.parse_crs()
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = tr.transform(PROP_LON, PROP_LAT)        # centre in ftUS
    x = np.asarray(las.x); y = np.asarray(las.y); z = np.asarray(las.z)
    cl = np.asarray(las.classification)
    half_ft = (half_m / FT2M) + 5.0 / FT2M           # +5 m margin for gap-fill
    keep = (np.abs(x - cx) < half_ft) & (np.abs(y - cy) < half_ft) & np.isin(cl, SURFACE_CLASSES)
    # local metric coords, origin at box SW corner
    xm = (x[keep] - cx) * FT2M                        # [-half..+half] m
    ym = (y[keep] - cy) * FT2M
    zm = z[keep] * FT2M
    n = int(round(2 * half_m / dx_m))
    Z = np.full((n, n), np.nan, dtype=np.float64)
    cnt = np.zeros((n, n), dtype=np.int64)
    col = ((xm + half_m) / dx_m).astype(np.int64)
    row = ((ym + half_m) / dx_m).astype(np.int64)
    ok = (col >= 0) & (col < n) & (row >= 0) & (row < n)
    # accumulate mean z per cell
    flat = row[ok] * n + col[ok]
    acc = np.bincount(flat, weights=zm[ok], minlength=n * n)
    num = np.bincount(flat, minlength=n * n)
    with np.errstate(invalid="ignore"):
        mean = acc / np.where(num > 0, num, np.nan)
    Z = mean.reshape(n, n)
    filled = int(num.reshape(n, n).astype(bool).sum())
    # nearest-neighbour fill of empty cells (thin gaps at 40 pts/m^2 are rare)
    if np.isnan(Z).any():
        from scipy.ndimage import distance_transform_edt
        ind = distance_transform_edt(np.isnan(Z), return_distances=False, return_indices=True)
        Z = Z[tuple(ind)]
    meta = dict(dx_m=dx_m, n=n, cells=n * n, filled_frac=filled / (n * n),
                z_min=float(Z.min()), z_max=float(Z.max()), relief_m=float(Z.max() - Z.min()),
                pts_used=int(keep.sum()))
    if verbose:
        print(f"    DEM dx={dx_m:>5}m  grid {n}x{n}={n*n:,} cells  "
              f"filled {meta['filled_frac']*100:.1f}%  relief {meta['relief_m']:.2f}m  "
              f"({meta['pts_used']:,} LiDAR pts)")
    return Z.astype(np.float32), meta


# ── Step 2: local-inertial shallow-water solver, vectorized torch ───────────────
def run_swe(Z_np, dx, sim_seconds, device, rain_mm_hr=PEAK_RAIN_MM_HR,
            max_steps=2_000_000, dt_max=0.1, fr_max=1.0,
            capture_n=0, capture_size=96):
    """Adaptive-CFL local-inertial SWE. dt = alpha*dx/sqrt(g*h_max), capped at dt_max
    (accuracy: resolve the rain forcing / flow, don't take multi-second jumps on a dry
    start). Froude flux limiter |q|<=fr_max*h*sqrt(g*h) keeps steep roof/slope faces
    stable — standard LISFLOOD-FP practice.

    capture_n>0 records that many evenly-spaced downsampled (capture_size^2) depth frames
    into the returned dict['frames'] (float32 metres) for animation — off by default so the
    timing benchmark path is untouched."""
    import torch
    import torch.nn.functional as Fnn
    dt_dev = torch.float32
    frames = []; vel_frames = []; frame_times = []
    cap_sched = [(i + 1) * sim_seconds / capture_n for i in range(capture_n)] if capture_n else []
    cap_ptr = 0
    Z = torch.as_tensor(Z_np, dtype=dt_dev, device=device)
    H, W = Z.shape
    h  = torch.zeros_like(Z)                          # water depth [H,W]
    qx = torch.zeros((H, W - 1), dtype=dt_dev, device=device)  # flux on E-W faces (m^2/s)
    qy = torch.zeros((H - 1, W), dtype=dt_dev, device=device)  # flux on N-S faces
    R = rain_mm_hr / 3.6e6                            # m/s source term
    n2 = MANNING * MANNING
    area = dx * dx

    rain_vol = 0.0        # cumulative m^3 in
    out_vol  = 0.0        # cumulative m^3 out (free boundary)
    t = 0.0; steps = 0
    hmax_seen = 1e-3

    while t < sim_seconds and steps < max_steps:
        if steps % 32 == 0:                          # periodic sync for adaptive dt
            hmax_seen = max(float(h.max()), 1e-3)
        dt = min(ALPHA * dx / math.sqrt(G * hmax_seen), dt_max)
        dt = min(dt, sim_seconds - t)
        if dt <= 0:
            break

        eta = Z + h                                  # water-surface elevation
        # ---- x faces (between col j and j+1) ----
        dz_x  = (eta[:, 1:] - eta[:, :-1]) / dx       # surface slope
        hf_x  = torch.clamp(torch.maximum(eta[:, 1:], eta[:, :-1])
                            - torch.maximum(Z[:, 1:], Z[:, :-1]), min=0.0)
        num_x = qx - G * hf_x * dt * dz_x
        den_x = 1.0 + G * dt * n2 * qx.abs() / torch.clamp(hf_x, min=1e-6) ** (7.0 / 3.0)
        qx = torch.where(hf_x > 1e-6, num_x / den_x, torch.zeros_like(qx))
        qcap_x = fr_max * hf_x * torch.sqrt(G * torch.clamp(hf_x, min=0.0))  # Froude limiter
        qx = torch.clamp(qx, -qcap_x, qcap_x)
        # ---- y faces ----
        dz_y  = (eta[1:, :] - eta[:-1, :]) / dx
        hf_y  = torch.clamp(torch.maximum(eta[1:, :], eta[:-1, :])
                            - torch.maximum(Z[1:, :], Z[:-1, :]), min=0.0)
        num_y = qy - G * hf_y * dt * dz_y
        den_y = 1.0 + G * dt * n2 * qy.abs() / torch.clamp(hf_y, min=1e-6) ** (7.0 / 3.0)
        qy = torch.where(hf_y > 1e-6, num_y / den_y, torch.zeros_like(qy))
        qcap_y = fr_max * hf_y * torch.sqrt(G * torch.clamp(hf_y, min=0.0))  # Froude limiter
        qy = torch.clamp(qy, -qcap_y, qcap_y)

        # ---- depth update: divergence of face fluxes ----
        div = torch.zeros_like(h)
        div[:, :-1] -= qx
        div[:, 1:]  += qx
        div[:-1, :] -= qy
        div[1:, :]  += qy
        # free-outflow boundary: flux off each open edge = velocity*depth of edge cell
        # (simple: let edge water leave proportional to local surface slope to a ghost = bed)
        # East/West/North/South edge outflow using a critical-ish free drop
        # We approximate boundary loss with the same local-inertial face vs a ghost cell = bed.
        # For benchmark timing this keeps the domain from filling unboundedly.
        # boundary handled implicitly by not adding inflow; explicit edge sink below:
        h_new = h + dt * (div / dx) + R * dt

        # explicit free-drain at the 4 borders (water surface drops to bed outside)
        # outflow rate q_edge = h*sqrt(g*h) (critical) — physically a weir/free overfall
        def edge_out(hedge):
            return torch.clamp(hedge, min=0.0) * torch.sqrt(G * torch.clamp(hedge, min=0.0))
        qb = dt / dx
        for edge in ("W", "E", "N", "S"):
            if edge == "W":
                he = h_new[:, 0];  loss = torch.minimum(edge_out(he) * qb, he)
                h_new[:, 0] = he - loss
            elif edge == "E":
                he = h_new[:, -1]; loss = torch.minimum(edge_out(he) * qb, he)
                h_new[:, -1] = he - loss
            elif edge == "N":
                he = h_new[0, :];  loss = torch.minimum(edge_out(he) * qb, he)
                h_new[0, :] = he - loss
            else:
                he = h_new[-1, :]; loss = torch.minimum(edge_out(he) * qb, he)
                h_new[-1, :] = he - loss
            out_vol += float(loss.sum()) * area

        h = torch.clamp(h_new, min=0.0)
        rain_vol += R * dt * area * (H * W)
        t += dt; steps += 1

        if cap_ptr < len(cap_sched) and t >= cap_sched[cap_ptr]:
            fr = Fnn.adaptive_max_pool2d(h[None, None], (capture_size, capture_size))[0, 0]
            frames.append(fr.cpu().numpy().astype(np.float32))
            # reconstruct cell-centered velocity from the two staggered face-flux arrays
            # (the same "accumulate flux onto adjacent cells" idea this project's sibling
            # mesh solver uses for its flow tracers — here on a structured grid instead of
            # a triangle mesh) so tracer particles can be driven by REAL solved velocity,
            # not just depth color.
            qx_pad = Fnn.pad(qx, (1, 1, 0, 0))            # (H, W+1)
            qx_cell = 0.5 * (qx_pad[:, :-1] + qx_pad[:, 1:])
            qy_pad = Fnn.pad(qy, (0, 0, 1, 1))            # (H+1, W)
            qy_cell = 0.5 * (qy_pad[:-1, :] + qy_pad[1:, :])
            hc = torch.clamp(h, min=1e-3)
            vx = (qx_cell / hc).cpu()             # avg-pool: MPS only supports divisible
            vy = (qy_cell / hc).cpu()             # input/output sizes, unlike max-pool above
            vxr = Fnn.adaptive_avg_pool2d(vx[None, None], (capture_size, capture_size))[0, 0]
            vyr = Fnn.adaptive_avg_pool2d(vy[None, None], (capture_size, capture_size))[0, 0]
            vel_frames.append(np.stack([vxr.cpu().numpy(), vyr.cpu().numpy()]).astype(np.float32))
            frame_times.append(t); cap_ptr += 1

    if device.type == "mps":
        torch.mps.synchronize()
    stored = float(h.sum()) * area
    resid = rain_vol - stored - out_vol
    resid_pct = 100.0 * resid / rain_vol if rain_vol > 0 else 0.0
    return dict(steps=steps, sim_t=t, h_max=float(h.max()),
                rain_m3=rain_vol, stored_m3=stored, out_m3=out_vol,
                mass_resid_pct=resid_pct, frames=frames, vel_frames=vel_frames,
                frame_times=frame_times)


# ── Step 3: the sweep ───────────────────────────────────────────────────────────
def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-seconds", type=float, default=10.0)
    ap.add_argument("--dx", type=float, nargs="*",
                    default=[0.25, 0.10, 0.05, 0.02])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--warmup", type=float, default=0.5,
                    help="throwaway short run per dx so kernel-compile time isn't counted")
    args = ap.parse_args()

    dev = torch.device(args.device if getattr(torch.backends, args.device,
                       None) and getattr(torch.backends, args.device).is_available()
                       else "cpu")
    print(f"device: {dev}   sim horizon: {args.sim_seconds}s of real physics   "
          f"rain: {PEAK_RAIN_MM_HR} mm/hr (peak Ian)\n")

    results = []
    for dx in args.dx:
        try:
            Z, meta = build_dem(dx)
            # warmup (compile MPS kernels, not timed)
            _ = run_swe(Z, dx, args.warmup, dev)
            if dev.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            r = run_swe(Z, dx, args.sim_seconds, dev)
            wall = time.perf_counter() - t0
            rec = {**meta, **r, "wall_s": wall,
                   "wall_per_sim_s": wall / max(r["sim_t"], 1e-9),
                   "device": dev.type, "status": "ok"}
            results.append(rec)
            print(f"    -> {r['steps']:,} steps  wall {wall:6.2f}s  "
                  f"= {rec['wall_per_sim_s']:8.2f}s wall / sim-s   "
                  f"mass resid {r['mass_resid_pct']:+.3f}%  h_max {r['h_max']*100:.1f}cm\n")
        except RuntimeError as e:
            msg = str(e)
            oom = "out of memory" in msg.lower() or "mps" in msg.lower()
            print(f"    -> dx={dx}m FAILED ({'OOM' if oom else 'RuntimeError'}): {msg[:80]}\n")
            results.append(dict(dx_m=dx, cells=meta.get("cells") if 'meta' in dir() else None,
                                status="oom" if oom else "error", error=msg[:200],
                                device=dev.type))
            if oom:
                print("    (MPS memory ceiling reached — stopping sweep, this is the boundary)")
                break

    # summary table
    native_dx = 1.0 / math.sqrt(DENSITY_PTS_M2)      # ~0.16 m at 40 pts/m^2
    print("=" * 104)
    print(f"{'dx(m)':>6} {'grid':>11} {'cells':>11} {'filled%':>8} {'ms/step':>8} "
          f"{'steps':>8} {'wall(s)':>9} {'wall/sim-s':>11} {'massres%':>9}  regime")
    print("-" * 104)
    for r in results:
        if r.get("status") == "ok":
            n = r["n"]; ms = 1000.0 * r["wall_s"] / max(r["steps"], 1)
            reg = "real-LiDAR" if r["dx_m"] >= native_dx else "interp/stress"
            print(f"{r['dx_m']:>6} {f'{n}x{n}':>11} {r['cells']:>11,} "
                  f"{r['filled_frac']*100:>7.1f}% {ms:>8.2f} {r['steps']:>8,} "
                  f"{r['wall_s']:>9.2f} {r['wall_per_sim_s']:>11.2f} "
                  f"{r['mass_resid_pct']:>+9.3f}  {reg}")
        else:
            print(f"{r['dx_m']:>6} {'—':>11} {(r.get('cells') or 0):>11,} "
                  f"{'—':>8} {'—':>8} {'—':>8} {'—':>9} {'—':>11} {'—':>9}  [{r['status'].upper()}]")
    print("=" * 104)
    print(f"native LiDAR-supported grid ~ {native_dx:.2f} m ({DENSITY_PTS_M2:.0f} pts/m^2); "
          f"finer rows are compute-stress (terrain interpolated beyond real point density).")

    outp = os.path.join(OUT_DIR, "torch_swe_benchmark.json")
    with open(outp, "w") as f:
        json.dump(dict(site=dict(lat=PROP_LAT, lon=PROP_LON, domain_m=25.0),
                       rain_mm_hr=PEAK_RAIN_MM_HR, sim_seconds=args.sim_seconds,
                       device=dev.type, results=results), f, indent=2)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
