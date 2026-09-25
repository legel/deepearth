"""Command line for the wind twin: one stage per subcommand.

    python3 cli.py fetch     --site campanile --year 2025
    python3 cli.py scene     --site campanile --dx 0.2 --bundle <dir>
    python3 cli.py simulate  --site campanile --dx 0.8 --synthetic tower --speed 8 --direction 290
    python3 cli.py series    --site campanile --day summer --dx 0.8 --synthetic tower
    python3 cli.py basis     --site campanile --year 2025 --dx 0.8 --synthetic tower
    python3 cli.py verify    --site campanile
    python3 cli.py benchmark --site campanile --nx 128 --ny 128 --nz 64 --steps 5

Run it from this directory. The modules are flat and import each other by name, so there is no
package to install and no path manipulation anywhere.
"""

import argparse
import cProfile
import json
import os
import pstats
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

import checks
import domain
import forcing
import frames
import sites
from domain import Grid, Scene
from physics import CLASSES
from solver import Model, Result, SolverConfig, solve

SYNTHETIC: Dict[str, Callable[[Grid], Scene]] = {
    "flat": lambda g: domain.flat(g),
    "cube": lambda g: domain.cube(g, 10.0),
    "tower": lambda g: domain.box(g, 10.5, 94.0),
    "canopy": lambda g: domain.porous_block(g, 12.0, CLASSES["tree_canopy"].lai,
                                            CLASSES["tree_canopy"].cd),
    "ridge": lambda g: domain.ridge(g, 10.0, 30.0),
}


def _scene(site: sites.SiteConfig, args: argparse.Namespace) -> Tuple[Scene, Dict[str, object]]:
    """A synthetic scene on the parcel's grid, or the bundle scene with its parameter receipt."""
    if args.synthetic is None:
        bundle = Path(args.bundle or site.bundle)
        return domain.from_bundle(sites.bundled(site, bundle, args.buffer), args.dx, bundle)
    n = site.cells_across(args.dx)
    grid = Grid.stretched(args.dx, n, n, site.levels(args.dx), args.dx, site.stretch)
    return SYNTHETIC[args.synthetic](grid), {"source": "synthetic", "grid_convergence_deg": 0.0}


def _grid_bearing(direction_true: float, receipt: Dict[str, object]) -> float:
    """A station's true bearing as a bearing from the scene's grid north."""
    return direction_true - float(receipt.get("grid_convergence_deg", 0.0))


def _config(args: argparse.Namespace) -> SolverConfig:
    cfg = SolverConfig(steps=args.steps, cfl=args.cfl, tol=args.tol, tol_final=args.tol_final, device=args.device,
                       dtype=torch.float32 if args.float32 else torch.float64,
                       lateral=args.lateral, verbose=True, settle_tol=getattr(args, "settle_tol", None),
                       settle_every=getattr(args, "settle_every", 20), max_steps=getattr(args, "max_steps", None),
                       scheme=getattr(args, "scheme", "fv"), tol_momentum=getattr(args, "tol_momentum", 1e-5))
    return cfg.fast() if getattr(args, "fast", False) else cfg


def _tag(args: argparse.Namespace) -> str:
    return f"{args.synthetic or 'bundle'}_{args.dx:g}m"


def _provenance(site: sites.SiteConfig, scene: Scene, receipt: Dict[str, object],
                cfg: SolverConfig, extra: Dict[str, object]) -> Dict[str, object]:
    return {"station": site.asos_station, "station_km": round(site.station_km(), 2),
            "scene": scene.summary(), "parameters": receipt,
            "solver": {"steps": cfg.steps, "cfl": cfg.cfl, "tol": cfg.tol, "tol_final": cfg.tol_final,
                       "tol_momentum": cfg.tol_momentum, "lateral": cfg.lateral,
                       "dtype": str(cfg.dtype), "device": cfg.device}, **extra}


def _field_domains(frames_seen: List[np.ndarray]) -> Dict[str, Dict[str, object]]:
    """Display domains: symmetric about zero for every signed field."""
    peak = np.max([np.abs(f).reshape(6, -1).max(axis=1) for f in frames_seen], axis=0)
    return {name: {"domain": [-float(p), float(p)], "lut": "coolwarm"}
            for (name, _), p in zip(frames.WIND_FIELDS, peak)}


def _sidecar(site: sites.SiteConfig, path: Path, t0: str, frames_seen: List[np.ndarray],
             marks: List[Dict[str, object]], provenance: Dict[str, object]) -> None:
    frames.sidecar(path, t0=t0, timezone="UTC", site=site.name, anchor_utm=site.anchor_utm,
                   fields=_field_domains(frames_seen), marks=marks, provenance=provenance)


def _publish(path: Path) -> Dict[str, object]:
    """Gzip a product and its sidecar for the wire, and report both sizes."""
    return {"frames": frames.compress(path),
            "sidecar": frames.compress(Path(str(path) + ".json"))}


def _frame(res: Result, stride: int) -> np.ndarray:
    """(6, nz, ny, nx) velocity and vorticity, horizontally strided."""
    return np.concatenate([res.velocity, res.vorticity])[:, :, ::stride, ::stride]


def _header(scene: Scene, stride: int) -> frames.Header:
    """The SIMF header for this scene's grid, strided horizontally."""
    g = scene.grid
    return frames.Header(
        n_frames=0, shape=(g.nz, g.ny // stride, g.nx // stride), fields=frames.WIND_FIELDS,
        cell_m=g.dx * stride, zf=tuple(float(z) for z in g.zf),
        origin=(scene.origin[0], scene.origin[1] + g.ny * g.dx, scene.origin[2]))


def _stats(res: Result) -> Dict[str, float]:
    return {"divergence_rel": res.divergence_rel, "divergence_max": res.divergence_max,
            "divergence_max_1_s": res.divergence_max_1_s,
            "flux_in_m3_s": res.flux_in_m3_s, "flux_out_m3_s": res.flux_out_m3_s,
            "flux_balance": res.flux_balance, "poisson_iterations": res.poisson_iterations,
            "final_change_rel": res.change[-1] if res.change else 0.0, "wall_s": res.wall_s,
            "cells": res.cells, "cells_per_s": res.cells * (res.steps + 1) / res.wall_s,
            "max_speed_m_s": float(res.speed.max()),
            "max_vorticity_1_s": float(np.abs(res.vorticity).max()),
            **({"steps": res.steps, "settled": res.settled, "settle": res.settle} if res.settled is not None else {})}


def cmd_fetch(args: argparse.Namespace) -> None:
    """Download the ASOS record for each year."""
    import fetch

    site = sites.get_site(args.site)
    print(json.dumps(fetch.all_sources(site, args.year), indent=1))


def cmd_scene(args: argparse.Namespace) -> None:
    """Voxelise the parcel and report what the solver will see."""
    site = sites.get_site(args.site)
    scene, receipt = _scene(site, args)
    report = {"scene": scene.summary(), "parameters": receipt}
    print(json.dumps(report, indent=1))
    site.out_path(f"scene_{_tag(args)}.json").write_text(json.dumps(report, indent=1))


def cmd_simulate(args: argparse.Namespace) -> None:
    """One steady field for one speed and direction."""
    site = sites.get_site(args.site)
    scene, receipt = _scene(site, args)
    cfg = _config(args)
    res = solve(scene, forcing.inflow(site, args.speed), _grid_bearing(args.direction, receipt), cfg)
    tag, frame = _tag(args), _frame(res, args.stride)
    path = frames.write_fields(site.out_path(f"wind_{tag}.bin"), _header(scene, args.stride),
                               [0.0], [frame])
    forcing_ = {"speed_m_s": args.speed, "direction_deg": args.direction}
    _sidecar(site, path, datetime.now(timezone.utc).isoformat(timespec="seconds"), [frame], [],
             _provenance(site, scene, receipt, cfg, {"forcing": forcing_}))
    summary = {"site": site.name, "scene": scene.summary(), "parameters": receipt, **forcing_,
               "steps": args.steps, "product": _publish(path), **_stats(res)}
    print(json.dumps(summary, indent=1))
    site.out_path(f"summary_{tag}.json").write_text(json.dumps(summary, indent=1))


def _view_header(scene: Scene, factor: int) -> frames.Header:
    """The viewer copy: u, v, w in float16, area-averaged over `factor` x `factor` columns."""
    g = scene.grid
    return frames.Header(
        n_frames=0, shape=(g.nz, g.ny // factor, g.nx // factor), fields=frames.WIND_FIELDS[:3],
        cell_m=g.dx * factor, zf=tuple(float(z) for z in g.zf), sample=2,
        origin=(scene.origin[0], scene.origin[1] + g.ny * g.dx, scene.origin[2]))


def _view_frame(res: Result, factor: int) -> np.ndarray:
    """(3, nz, ny / factor, nx / factor) velocity, block-averaged horizontally."""
    v = res.velocity
    ny, nx = (v.shape[2] // factor) * factor, (v.shape[3] // factor) * factor
    blocks = v[:, :, :ny, :nx].reshape(3, v.shape[1], ny // factor, factor, nx // factor, factor)
    return blocks.mean(axis=(3, 5))


def _run_set(site: sites.SiteConfig, scene: Scene, receipt: Dict[str, object], cfg: SolverConfig,
             forcings: List[Dict[str, float]], times_s: List[float], stride: int, tag: str,
             t0: str, extra: Dict[str, object], view_dx: Optional[float] = None) -> None:
    """Solve a list of {speed, direction} forcings, each warm-started from the last, streaming
    every frame to `wind_<tag>.bin` with the forcing and statistics in its sidecar, and, when
    `view_dx` is given, the viewer copy to `wind_<tag>_view.bin`."""
    path = site.out_path(f"wind_{tag}.bin")
    factor = max(1, round(view_dx / scene.grid.dx)) if view_dx else 0
    view = frames.Writer(site.out_path(f"wind_{tag}_view.bin"), _view_header(scene, factor)) if factor else None
    records, seen, previous = [], [], None
    with frames.Writer(path, _header(scene, stride)) as writer:
        for f, t in zip(forcings, times_s):
            speed = max(f["speed_m_s"], forcing.CALM_M_S)
            res = solve(scene, forcing.inflow(site, speed), _grid_bearing(f["direction_deg"], receipt),
                        cfg, initial=previous)
            previous = res.velocity
            frame = _frame(res, stride)
            writer.add(t, frame)
            if view:
                view.add(t, _view_frame(res, factor))
            seen.append(frame)
            records.append({**f, "t_s": t, "stats": _stats(res)})
            print(f"  {f['label']}: {speed:.1f} m/s from {f['direction_deg']:.0f} deg, "
                  f"{res.wall_s:.0f}s, change {res.change[-1] if res.change else 0:.1e}")
    peak = max(records, key=lambda r: r["speed_m_s"])
    marks = [{"t_s": peak["t_s"], "label": f"peak {peak['speed_m_s']:.1f} m/s, {peak['label']}"}]
    provenance = _provenance(site, scene, receipt, cfg, {**extra, "frames": records})
    _sidecar(site, path, t0, seen, marks, provenance)
    products = {path.name: _publish(path)}
    if view:
        view.close()
        _sidecar(site, view.path, t0, seen, marks,
                 {**provenance, "view": {"cell_m": scene.grid.dx * factor, "sample": "float16",
                                         "fields": [f[0] for f in frames.WIND_FIELDS[:3]],
                                         "vorticity": "computed in the viewer from u and v",
                                         "averaged_from_m": scene.grid.dx, "native": path.name}})
        products[view.path.name] = _publish(view.path)
    print(json.dumps({"frames": len(records), "products": products, **extra}, indent=1, default=str))


def cmd_series(args: argparse.Namespace) -> None:
    """Twenty-four hourly fields driven by the ASOS record of one day."""
    site, day = sites.get_site(args.site), sites.get_day(args.day)
    scene, receipt = _scene(site, args)
    hours = forcing.day_series(site, day)
    direction = 0.0
    forcings = []
    for h, (speed, drct, gust) in enumerate(hours):
        direction = drct if np.isfinite(drct) else direction
        forcings.append({"label": f"{day.date} {h:02d}:00Z", "hour_utc": h, "speed_m_s": float(speed),
                         "direction_deg": float(direction), "gust_m_s": float(gust)})
    _run_set(site, scene, receipt, _config(args), forcings, [3600.0 * h for h in range(24)],
             args.stride, f"{day.name}_{_tag(args)}", f"{day.date}T00:00:00+00:00",
             {"day": day.date})


def cmd_basis(args: argparse.Namespace) -> None:
    """One field per heading at 1 m/s, with the year's hourly table and climatology.

    The field is linear in the inflow speed (`cli.py verify`, `linearity`), so a viewer picks the
    hour's speed and direction from the table, blends the two nearest headings and scales.
    """
    site = sites.get_site(args.site)
    scene, receipt = _scene(site, args)
    headings = [360.0 * k / args.directions for k in range(args.directions)]
    forcings = [{"label": f"heading {h:6.2f}", "heading_deg": h, "speed_m_s": 1.0,
                 "direction_deg": h} for h in headings]
    docs = Path(__file__).resolve().parent / "docs"
    verified = {p.name: json.loads(p.read_text()).get("linearity")
                for p in docs.glob("verification_*.json")}
    _run_set(site, scene, receipt, _config(args), forcings, [float(k) for k in range(len(headings))],
             args.stride, f"basis{args.directions}_{args.year}_{_tag(args)}",
             f"{args.year}-01-01T00:00:00+00:00",
             {"basis": {"headings_deg": headings, "speed_m_s": 1.0, "z_ref_m": 10.0,
                        "record": "t_s is the heading index; headings are true bearings the wind "
                                  "blows from, solved at heading minus grid_convergence_deg; "
                                  "frames scale with speed"},
              "rose": forcing.rose(site, args.year), "hours": forcing.year_table(site, args.year),
              "linearity": verified},
             view_dx=args.view_dx)


def cmd_verify(args: argparse.Namespace) -> None:
    """Run every physics check and record its numbers."""
    cfg = _config(args)
    cfg.verbose = False
    report = {"config": {"steps": cfg.steps, "cfl": cfg.cfl, "tol": cfg.tol, "device": cfg.device,
                         "dtype": str(cfg.dtype)}, "vorticity": checks.vorticity_of_rotation()}
    for name, fn in checks.ALL.items():
        t0 = time.time()
        report[name] = fn(cfg)
        report[name]["check_wall_s"] = time.time() - t0
        print(f"  {name}: {json.dumps(report[name])}")
    path = Path(__file__).resolve().parent / "docs" / f"verification_{cfg.device}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
    print(f"wrote {path}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Cells per second for the projection and for a momentum step, and the profile top ten."""
    cfg = _config(args)
    cfg.verbose = False
    g = Grid.stretched(args.dx, args.nx, args.ny, args.nz, args.dx, 1.06)
    scene = domain.cube(g, 8 * args.dx * max(1, args.nx // 64), centre=(g.nx * g.dx / 3, g.ny * g.dx / 2))
    model = Model(scene, checks.profile(), checks.WEST, cfg)
    sync = torch.cuda.synchronize if cfg.device.startswith("cuda") else (lambda: None)

    faces = model.faces(model.background())
    sync()
    t0 = time.time()
    _, iterations, _ = model.project(faces)
    sync()
    t_project = time.time() - t0

    prof = cProfile.Profile()
    prof.enable()
    res = model.run()
    sync()
    prof.disable()
    t_step = (res.wall_s - t_project) / max(cfg.steps, 1)
    rows = sorted(pstats.Stats(prof).stats.items(), key=lambda kv: -kv[1][2])[:10]
    top = [{"function": f"{Path(fn).name}:{ln} {name}", "ncalls": nc,
            "tottime_s": round(tt, 4), "cumtime_s": round(ct, 4)}
           for (fn, ln, name), (_, nc, tt, ct, _) in rows]
    report = {
        "device": cfg.device, "dtype": str(cfg.dtype), "lateral": cfg.lateral,
        "grid": list(g.shape), "cells": g.cells,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if cfg.device.startswith("cuda") else None,
        "projection_s": t_project, "projection_iterations": iterations,
        "projection_cells_per_s": g.cells / t_project,
        "step_s": t_step, "step_cells_per_s": g.cells / t_step if cfg.steps else None,
        "steps": cfg.steps, "poisson_iterations": res.poisson_iterations,
        "divergence_rel": res.divergence_rel, "profile_top10_by_tottime": top,
    }
    print(json.dumps(report, indent=1))
    dtype = str(cfg.dtype).split(".")[-1]
    path = (Path(__file__).resolve().parent / "docs"
            / f"benchmark_{cfg.device.replace(':', '')}_{dtype}_{args.nx}x{args.ny}x{args.nz}.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=1))


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments and dispatch to one stage."""
    parser = argparse.ArgumentParser(prog="wind", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)

    def add(name: str, fn: Callable, help_text: str, solver: bool = False) -> argparse.ArgumentParser:
        """Register a subcommand; every stage takes --site, solver stages take the numerics."""
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--site", required=True, choices=sorted(sites.SITES))
        if solver:
            p.add_argument("--steps", type=int, default=200)
            p.add_argument("--settle-tol", type=float,
                           help="step until the near-ground speed settles to this fraction (--steps the fewest)")
            p.add_argument("--settle-every", type=int, default=20, help="steps between settle checks")
            p.add_argument("--max-steps", type=int, help="the most steps a settling run takes; then it says so")
            p.add_argument("--cfl", type=float, default=2.0)
            p.add_argument("--tol", type=float, default=1e-6)
            p.add_argument("--tol-final", type=float, default=None,
                           help="relative residual of one more projection after the last step")
            p.add_argument("--lateral", default="profile", choices=("profile", "open"))
            p.add_argument("--device", default="cpu")
            p.add_argument("--float32", action="store_true")
            p.add_argument("--scheme", default="fv", choices=("fv", "sl"),
                           help="fv: the steady finite-volume scheme, independent of the pseudo-time step; sl: semi-Lagrangian")
            p.add_argument("--fast", action="store_true",
                           help="float32 momentum and V-cycles under float64 projections (SolverConfig.fast)")
            p.add_argument("--tol-momentum", type=float, default=1e-5,
                           help="relative residual of each pseudo-time step's implicit momentum solve")
        p.set_defaults(func=fn)
        return p

    def scene_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dx", type=float, default=0.2)
        p.add_argument("--synthetic", choices=sorted(SYNTHETIC),
                       help="a synthetic scene on the parcel grid instead of the bundle")
        p.add_argument("--bundle", help="bundle directory holding semantics/ and the surface "
                                        "rasters; default data/<site>/bundle")
        p.add_argument("--flat-terrain", action="store_true",
                       help="solve over flat ground where the bundle has no surface/: a verification case, never a "
                            "customer's site; without it a bundle with no terrain fails the run")
        p.add_argument("--stride", type=int, default=1,
                       help="keep every nth column in the written frames")
        p.add_argument("--buffer", type=float, default=None,
                       help="ring beyond the disc the grid covers [m]; default the bundle's buffer_m, "
                            "0 runs over the display disc alone")

    p = add("fetch", cmd_fetch, "download the ASOS record")
    p.add_argument("--year", type=int, action="append", required=True)

    p = add("scene", cmd_scene, "voxelise the parcel")
    scene_args(p)

    p = add("simulate", cmd_simulate, "one steady field", solver=True)
    scene_args(p)
    p.add_argument("--speed", type=float, required=True, help="station speed at 10 m [m/s]")
    p.add_argument("--direction", type=float, required=True, help="blowing from [deg]")

    p = add("series", cmd_series, "hourly fields for one day", solver=True)
    scene_args(p)
    p.add_argument("--day", required=True, choices=sorted(sites.DAYS))

    p = add("basis", cmd_basis, "unit-speed fields per heading, with the year's hourly table",
            solver=True)
    scene_args(p)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--directions", type=int, default=16, choices=(16, 36))
    p.add_argument("--view-dx", type=float, default=0.5,
                   help="cell of the float16 viewer copy, area-averaged from the native solve")

    add("verify", cmd_verify, "run the physics checks and record them", solver=True)

    p = add("benchmark", cmd_benchmark, "cells per second and the profile", solver=True)
    p.add_argument("--dx", type=float, default=1.0)
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--nz", type=int, default=64)

    args = parser.parse_args(argv)
    if getattr(args, "flat_terrain", False):
        os.environ["WIND_FLAT_TERRAIN"] = "1"      # declared on this command line, read by domain.from_bundle
    args.func(args)


if __name__ == "__main__":
    main()
