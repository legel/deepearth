"""The storm kernel's cost per cell and sub-step on this GPU, as production runs it (float32 surface, torch.compile,
Green-Ampt with redistribution on every cell, as `parcel --cells` gives it), against the Horton kernel, over grids from
1 M to 8 M cells; one torch.profiler table of the production kernel's CUDA time by kernel. argv: out.json [sizes...]"""
import json
import sys
import time

import numpy as np
import torch

import infiltration
import solver

SIDES = [int(s) for s in sys.argv[2:]] or [1000, 2000, 2786]
DX = 0.2


def surface(n: int, gar: bool) -> solver.Surface:
    """A 0.2 m campus-like grid: a 3 % slope, curb-scale steps and noise, 1 cm of water standing everywhere, a disc."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) * DX
    z = 0.03 * x + 0.15 * (np.sin(x / 7.0) > 0.9) + 0.02 * rng.standard_normal((n, n)).astype(np.float32)
    r = np.hypot(x - n * DX / 2, y - n * DX / 2)
    z = np.where(r <= n * DX / 2, z, np.nan).astype(np.float32)
    soil = None
    if gar:
        f = lambda v: np.full((n, n), v, dtype=np.float64)  # noqa: E731
        ks = np.where(rng.random((n, n)) < 0.4, 0.0, 1e-6)          # 40 % sealed, loam elsewhere
        soil = infiltration.Soil(ks=ks, psi_f=f(0.089), theta_s=f(0.43), theta_r=f(0.027), lam=f(0.22), theta_i=f(0.2))
    return solver.Surface(z=z, f0=2e-5, fc=3e-6, k=1e-3, soil=soil, smax_m=np.full((n, n), 5e-4, dtype=np.float32),
                          initial_h=np.where(np.isfinite(z), 0.01, 0.0).astype(np.float32),
                          manning_n=np.full((n, n), 0.03, dtype=np.float32))


def rate(n: int, gar: bool, intervals: int, soil_dt_s: float = 0.0) -> dict:
    """Seconds a sub-step, from a warm run (the compile excluded) over `intervals` 5 s forcing intervals."""
    s = surface(n, gar)
    cfg = solver.SolverConfig(dx=DX, dt_s=5.0, frame_interval_min=1e9, cfl_depth="face", soil_dt_s=soil_dt_s)
    solver.simulate(s, [2e-5] * 2, cfg, verbose=False)                     # compile and warm
    torch.cuda.synchronize()
    t0 = time.time()
    r = solver.simulate(s, [2e-5] * intervals, cfg, verbose=False)        # 72 mm/h
    torch.cuda.synchronize()
    wall = time.time() - t0
    cells = n * n
    return {"side": n, "cells": cells, "valid": int(np.isfinite(s.z).sum()), "gar": gar, "soil_dt_s": soil_dt_s,
            "n_substeps": r.n_substeps, "infiltrated_m3": r.mass.infiltrated, "outflow_m3": r.mass.outflow,
            "wall_s": round(wall, 2), "ms_per_substep": round(wall / r.n_substeps * 1e3, 3),
            "s_per_cell_substep": wall / r.n_substeps / cells, "mass_residual": r.mass.residual,
            "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}


def profile(n: int) -> list:
    """CUDA time by kernel over a warm production run."""
    s = surface(n, True)
    cfg = solver.SolverConfig(dx=DX, dt_s=5.0, frame_interval_min=1e9, cfl_depth="face")
    solver.simulate(s, [2e-5] * 2, cfg, verbose=False)
    from torch.profiler import ProfilerActivity, profile as prof
    with prof(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
        solver.simulate(s, [2e-5] * 4, cfg, verbose=False)
    rows = []
    for e in p.key_averages():
        cuda = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0.0)
        if cuda:
            rows.append({"name": e.key[:120], "calls": e.count, "cuda_ms": round(cuda / 1e3, 2)})
    rows.sort(key=lambda r: -r["cuda_ms"])
    return rows[:25]


out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "dx_m": DX, "runs": []}
for n in SIDES:
    for gar, soil_dt in ((False, 0.0), (True, 0.0), (True, 0.5), (True, 1.0), (True, 2.0)):
        out["runs"].append(rate(n, gar, 12 if n > 2000 else 24, soil_dt))
        print(json.dumps(out["runs"][-1]), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
