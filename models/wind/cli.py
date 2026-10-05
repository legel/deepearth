"""Command line for the wind twin: one stage per subcommand.

    python3 cli.py fetch     --site campanile --year 2025
    python3 cli.py scene     --site campanile --dx 0.2 --bundle ../../../simulation/bundle
    python3 cli.py simulate  --site campanile --dx 0.8 --synthetic tower --speed 8 --direction 290
    python3 cli.py series    --site campanile --day summer --dx 0.8 --synthetic tower
    python3 cli.py basis     --site campanile --year 2025 --dx 0.8 --synthetic tower
    python3 cli.py buffer    --site campanile --dx 1 --bundle <dir> --speed 6 --direction 270 --footprint <json>
    python3 cli.py levels    --site campanile --dx 2 --bundle <dir> --forcing <json> --cache <dir> --out <dir>
    python3 cli.py refine    --site campanile --dx 2 --bundle <dir> --heading 247.5 --dz-bands 0 1 0.5 --caches <dirs>
    python3 cli.py extent    --site campanile --dx 2 --bundle <dir> --cells 256 384 512 640 --speed 7 --direction 249.06
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
from dataclasses import replace
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


def _canopy(scene: Scene, receipt: Dict[str, object], args: argparse.Namespace) -> Tuple[Scene, Dict[str, object]]:
    """The scene with the survey's canopy drag with height (`canopy.apply`) when --canopy names the sun set's files."""
    if getattr(args, "canopy", None):
        import canopy
        receipt["canopy"] = canopy.apply(scene, canopy.load(Path(args.canopy)), getattr(args, "doy", None))
        print("  canopy: " + json.dumps(receipt["canopy"]), flush=True)
    return scene, receipt


def _profile(args: argparse.Namespace):
    """The sun set's canopy profile named by --canopy, else None: the scene build's evidence of where plants stand."""
    if not getattr(args, "canopy", None):
        return None
    import canopy
    return canopy.load(Path(args.canopy))


def _scene(site: sites.SiteConfig, args: argparse.Namespace) -> Tuple[Scene, Dict[str, object]]:
    """A synthetic scene on the parcel's grid, or the bundle scene with its parameter receipt."""
    if args.synthetic is None:
        bundle = Path(args.bundle or site.bundle)
        return _canopy(*domain.from_bundle(sites.bundled(site, bundle, args.buffer), args.dx, bundle,
                                           canopy_profile=_profile(args)), args)
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
                       scheme=getattr(args, "scheme", "fv"), tol_momentum=getattr(args, "tol_momentum", 1e-5),
                       settle_rule=getattr(args, "settle_rule", "window"), inflow=getattr(args, "inflow", "log"),
                       closure=getattr(args, "closure", "mixing"), drive=getattr(args, "drive", "shear"))
    return cfg.fast() if getattr(args, "fast", False) else cfg


def _tag(args: argparse.Namespace) -> str:
    return f"{args.synthetic or 'bundle'}_{args.dx:g}m"


def _provenance(site: sites.SiteConfig, scene: Scene, receipt: Dict[str, object],
                cfg: SolverConfig, extra: Dict[str, object]) -> Dict[str, object]:
    return {"station": site.asos_station, "station_km": round(site.station_km(), 2),
            "scene": scene.summary(), "parameters": receipt,
            "solver": {"steps": cfg.steps, "cfl": cfg.cfl, "tol": cfg.tol, "tol_final": cfg.tol_final,
                       "tol_momentum": cfg.tol_momentum, "lateral": cfg.lateral,
                       "dtype": str(cfg.dtype), "march_dtype": str(cfg.march_dtype), "device": cfg.device}, **extra}


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


def cmd_buffer(args: argparse.Namespace) -> None:
    """The same bundle and forcing solved without and with the buffer, and what changed."""
    import buffer

    site, bundle, cfg, bcfg = sites.get_site(args.site), Path(args.bundle), _config(args), buffer.BufferConfig()
    profile = forcing.inflow(site, args.speed)
    arms, report = {}, {"site": site.name, "dx_m": args.dx, "speed_m_s": args.speed, "direction_deg": args.direction,
                        "inflow_profile": {"u_star_m_s": profile.u_star, "z0_m": profile.z0, "d_m": profile.d,
                                           "note": "one profile for both arms, set by the registry's fetch "
                                                   "roughness, independent of the buffer"}}
    for name, buf, data in (("before", 0.0, None), ("wider", None, site.display_radius_m), ("after", None, None)):
        arm = sites.bundled(site, bundle, buf)
        scene, receipt = domain.from_bundle(arm, args.dx, bundle, data)
        res = solve(scene, profile, _grid_bearing(args.direction, receipt), cfg)
        frame = _frame(res, 1)
        path = frames.write_fields(site.out_path(f"wind_buffer_{name}_{args.dx:g}m.bin"), _header(scene, 1),
                                   [0.0], [frame])
        _sidecar(site, path, datetime.now(timezone.utc).isoformat(timespec="seconds"), [frame], [],
                 _provenance(site, scene, receipt, cfg, {"forcing": {"speed_m_s": args.speed,
                                                                     "direction_deg": args.direction}}))
        arms[name] = (scene, res)
        report[name] = {"scene": scene.summary(), "grid": list(scene.grid.shape), "origin": list(scene.origin),
                        "parameters": receipt, **_stats(res)}
        print(f"  {name}: buffer {arm.buffer_m:g} m, {scene.grid.shape}, {res.wall_s:.0f} s, "
              f"divergence {res.divergence_max_1_s:.2e} 1/s", flush=True)
    r = site.display_radius_m
    report["inputs_differing_inside_display_disc"] = buffer.identical_inside(arms["before"][0], arms["after"][0], r)
    report["bands"] = buffer.edge_bands(arms["before"], arms["after"], r, bcfg)
    report["bands_wider_grid_only"] = buffer.edge_bands(arms["before"], arms["wider"], r, bcfg)
    report["bands_ring_data_only"] = buffer.edge_bands(arms["wider"], arms["after"], r, bcfg)
    if args.footprint:
        b = json.loads(Path(args.footprint).read_text())
        fp = buffer.footprint(arms["after"][0], np.array(b["outline_scene_m"]), bcfg.erode_m)
        report["building"] = {"name": b["name"], "osm_id": b.get("osm_id"), "erode_m": bcfg.erode_m,
                              **buffer.building(arms["before"], arms["after"], fp, r)}
    out = site.out_path(f"buffer_{args.dx:g}m_{args.direction:g}deg.json")
    out.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k not in ("before", "wider", "after")}, indent=1, default=float))


def cmd_view(args: argparse.Namespace) -> None:
    """The viewer's product: a recorded day composed from unit-speed solves per heading, on the
    viewer's grid, heights above the terrain, NaN beyond the display radius."""
    import view

    bundle, cfg = Path(args.bundle), _config(args)
    site = sites.bundled(sites.get_site(args.site), bundle, args.buffer)
    vcfg = view.ViewConfig(half_m=view.half_for(site.display_radius_m))
    scene, receipt = domain.from_bundle(site, args.dx, bundle)
    record = json.loads(Path(args.forcing).read_text())
    day = record["forcing"]
    ground = view.viewer_ground(Path(args.ground), scene, vcfg)
    basis, solves = {}, {}
    for h in view.headings(day["from_deg"], vcfg.spacing_deg):
        res = solve(scene, forcing.inflow(site, 1.0), _grid_bearing(h, receipt), cfg)
        basis[h] = view.crop(view.slices(res.velocity, scene, vcfg.heights_m, ground), scene, vcfg,
                             site.display_radius_m)
        solves[str(h)] = _stats(res)
        print(f"  heading {h:g}: {res.wall_s:.0f} s, divergence {res.divergence_max_1_s:.2e} 1/s", flush=True)
    records = [s * view.blend(basis, d, vcfg.spacing_deg) for s, d in zip(day["speed10_m_s"], day["from_deg"])]
    n = records[0].shape[-1]
    header = frames.Header(n_frames=0, shape=(len(vcfg.heights_m), n, n), fields=frames.WIND_FIELDS[:2],
                           cell_m=vcfg.cell_m, origin=(-vcfg.half_m, vcfg.half_m, 0.0), zf=tuple(vcfg.faces))
    path = frames.write_fields(site.out_path("frames.simf"), header, [60.0 * m for m in day["minute"]], records)
    speed = np.hypot(*np.stack(records)[:, :2].transpose(1, 0, 2, 3, 4))
    side = {k: record[k] for k in ("t0", "timezone", "site", "epsg", "anchor_utm", "surface", "fields", "forcing",
                                   "marks", "label", "run_s")}
    side.update(display_radius_m=site.display_radius_m, fetch_radius_m=site.fetch_radius_m, buffer_m=site.buffer_m,
                provenance={"kind": "design-forcing", "forcing_from": args.forcing,
                            "solver": "models/wind mass-consistent 3D solve, one unit-speed field per heading, "
                                      "scaled by speed and blended linearly in angle",
                            "headings_deg": sorted(basis), "spacing_deg": vcfg.spacing_deg, "steps": cfg.steps,
                            "dtype": str(cfg.dtype), "solves": solves, "scene": scene.summary(), "parameters": receipt,
                            "heights_above_ground_m": list(vcfg.heights_m), "ground": args.ground, "peak_speed_m_s": float(np.nanmax(speed)),
                            "shown": "NaN beyond display_radius_m; the buffer only moves the boundary"})
    Path(str(path) + ".json").write_text(json.dumps(side, indent=1, default=float))
    print(json.dumps({"frames": len(records), "product": _publish(path), "headings": sorted(basis)}, indent=1))


LEVEL_FIELDS = (("u", "m/s"), ("v", "m/s"), ("speed", "m/s"), ("vort_z", "1/s"))
"""Per level: horizontal velocity, its magnitude, and the vertical vorticity dv/dx - du/dy."""

GROUND_FILE, SOLID_FILE, FILLED_FILE = "ground_z_f32.bin", "solid_u8.bin", "filled_u8.bin"
"""The static files beside a levels product: bare earth per cell, and per level the solid and filled masks."""

LEVELS_VERSION = 2
"""Side-car `levels_version`: 2 stands every level on the bare earth; a product without it followed roofs."""


def _progress(j: int, n: int, cfg: SolverConfig, done: bool = False) -> None:
    """Around heading j of n: the solver prints `PROGRESS wind k/n` over the run's momentum steps while it solves
    (`solver.PROGRESS`); once the heading is done (solved, or already in the cache) its whole count is printed."""
    import solver

    solver.PROGRESS = None if done else (j, n)
    if done:
        print(f"PROGRESS wind {(j + 1) * cfg.steps}/{n * cfg.steps}", flush=True)


def _unit_field(scene: Scene, site: sites.SiteConfig, receipt: Dict[str, object], cfg: SolverConfig,
                heading: float, cache: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    """(3, nz, ny, nx) u, v, vort_z of the unit-speed solve at `heading`, solved once into `cache`."""
    path = cache / f"unit_{heading:g}.npz"
    if not path.exists():
        gpu = torch.cuda.is_available()
        if gpu:
            torch.cuda.reset_peak_memory_stats()
        res = solve(scene, forcing.inflow(site, 1.0), _grid_bearing(heading, receipt), cfg)
        st = _stats(res)
        if gpu:
            # Measured, never estimated: what a heading of this grid holds at its peak on this card, kept in the
            # sidecar's provenance. A 21 M-cell grid with buildings ran out of a 22 GiB L4 while a 23 M-cell forest
            # fitted (2026-10-03), so the bytes a cell are the site's, and the planner needs them from runs.
            dev = torch.cuda.current_device()
            st.update(cuda_peak_gib=round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 3),
                      cuda_total_gib=round(torch.cuda.get_device_properties(dev).total_memory / 2 ** 30, 3))
            print(f"  heading {heading:g}: CUDA peak {st['cuda_peak_gib']:.2f} of {st['cuda_total_gib']:.2f} GiB, "
                  f"{st['cuda_peak_gib'] * 2 ** 30 / max(st['cells'], 1):.0f} bytes a cell", flush=True)
        write_unit(path, res.velocity, res.vorticity[2], scene.grid.zf, st)
    saved = np.load(path)
    assert saved["velocity"].shape[1:] == scene.grid.shape, f"{path} was solved on another grid"
    vel = saved["velocity"][:2].astype(np.float32)
    return np.concatenate([vel, saved["vort_z"][None].astype(np.float32)]), json.loads(str(saved["stats"]))


UNIT_DTYPE = np.float16
"""A unit file's velocity and vort_z: float16, compressed. Measured on Harvard 270 against float32: 99.7 against
352 MiB, the published levels' speed within 0.10 % of its median and the survey points' within 0.07 % (largest),
vort_z within 0.13 % of its p99 (2026-10-04)."""


def write_unit(path: Path, velocity: np.ndarray, vort_z: np.ndarray, zf: np.ndarray, stats: Dict) -> None:
    """One heading's unit solve into the cache, through a partial file renamed once whole."""
    tmp = path.with_name(path.name.replace(".npz", ".partial.npz"))
    np.savez_compressed(tmp, velocity=velocity.astype(UNIT_DTYPE), vort_z=vort_z.astype(UNIT_DTYPE), zf=zf,
                        stats=json.dumps(stats, default=float))
    tmp.replace(path)


def octree_layout(scene: Scene, site: sites.SiteConfig, args: argparse.Namespace):
    """(layout, its record) of the adaptive cells for this scene under the --octree-* criterion: the leaves, the
    grid's cells, the CUDA peak a heading is expected to hold on them and on the dense grid (amr_model.octree_bytes,
    DENSE_CELL_BYTES), and which solver a heading will run on. Numpy alone: no GPU, no solve."""
    import amr
    import amr_model

    t0 = time.time()
    core_h = None if str(args.octree_core_h) == "auto" else float(args.octree_core_h)
    crit = amr.Criterion(shell=args.octree_shell, shell_out=args.octree_shell_out, band=args.octree_band,
                         margin=args.octree_margin, core_m=args.octree_core_m, core_h=core_h, disc=site.display_radius_m)
    leaf = amr.balance(amr.required(scene, crit, scene.measured), crit.lmax)
    lay = amr.layout(leaf, scene.solid, crit.lmax)
    cells = int(scene.grid.cells)
    need, dense = amr_model.octree_bytes(cells, int(lay.n)), amr_model.DENSE_CELL_BYTES * cells
    return lay, {"criterion": crit.label() + f" core {args.octree_core_m:g} m to {args.octree_core_h}",
                 "leaves": int(lay.n), "per_level": np.bincount(lay.level, minlength=crit.lmax + 1).tolist(),
                 "cells": cells, "octree_gib": round(need / 2 ** 30, 2), "dense_gib": round(dense / 2 ** 30, 2),
                 "solver": "dense" if need > dense else "octree", "build_s": round(time.time() - t0, 1)}


def _octree_units(scene: Scene, site: sites.SiteConfig, receipt: Dict[str, object], cfg: SolverConfig,
                  headings: List[float], cache: Path, args: argparse.Namespace) -> None:
    """Every heading not yet in `cache` solved on adaptive cells (amr, amr_model), `--octree-batch` headings marched
    together on the one layout, each heading's field put back on the grid's cells and written as the dense solve's
    unit file is: everything that reads the cache (levels, points, over-top) reads it unchanged."""
    import amr_model

    todo = [h for h in headings if not (cache / f"unit_{h:g}.npz").exists()]
    if not todo:
        return
    lay, layout = octree_layout(scene, site, args)
    print("OCTREE " + json.dumps(layout), flush=True)
    if layout["solver"] == "dense":     # few cells saved: the dense solve holds less, and the plan sized the card for it
        print(f"OCTREE dense: {layout['octree_gib']} GiB on {lay.n} leaves against the dense grid's "
              f"{layout['dense_gib']}", flush=True)
        return
    profile, done, total = forcing.inflow(site, 1.0), len(headings) - len(todo), len(headings)
    for g in range(0, len(todo), max(1, args.octree_batch)):
        hs = todo[g:g + max(1, args.octree_batch)]
        gpu = torch.cuda.is_available()
        if gpu:
            torch.cuda.reset_peak_memory_stats()
            held = torch.cuda.memory_allocated() / 2 ** 30
            if held > 0.05:                       # what the card already held is in every peak below
                print(f"  OCTREE held before the model: {held:.2f} GiB", flush=True)
        am = amr_model.AmrModel(scene, profile, [_grid_bearing(h, receipt) for h in hs], cfg, lay)
        out = am.run()
        if am.prof:                               # AMR_PROFILE=1: each section's CUDA peak
            print("  OCTREE sections " + json.dumps({k: round(v, 3) for k, v in am.prof.items() if k.startswith("peak")}),
                  flush=True)
        for j, h in enumerate(hs):
            many = len(hs) > 1
            u = out["u"][j] if many else out["u"]
            vel, vort = am.dense_velocity(u)
            peak = torch.cuda.max_memory_allocated() / 2 ** 30 if gpu else None   # the output's grids included
            pick = lambda v: v[j] if many else v  # noqa: E731
            st = {"divergence_max_1_s": out["divergence_max_1_s"], "flux_in_m3_s": pick(out["flux_in"]),
                  "flux_out_m3_s": pick(out["flux_out"]),
                  "flux_balance": abs(pick(out["flux_in"]) - pick(out["flux_out"])) / max(pick(out["flux_in"]), 1e-30),
                  "poisson_iterations": out["projection_iterations"], "wall_s": out["wall_s"] / len(hs),
                  "batch_wall_s": out["wall_s"], "batch_headings": hs, "cells": int(scene.grid.cells),
                  "leaves": int(lay.n), "max_speed_m_s": float(np.linalg.norm(vel, axis=0).max()),
                  "max_vorticity_1_s": float(np.abs(vort).max()), "steps": pick(out["steps"]),
                  "batch_steps": out["batch_steps"], "settled": pick(out["settled"]), "solver": "octree",
                  "layout": layout, **({"cuda_peak_gib": round(peak, 3)} if peak is not None else {})}
            write_unit(cache / f"unit_{h:g}.npz", vel, vort[2], scene.grid.zf, st)
            print(f"  heading {h:g}: octree {lay.n} leaves, {st['steps']} steps, settled {st['settled']}, "
                  f"divergence {st['divergence_max_1_s']:.2e} 1/s, {st['wall_s']:.1f} s, CUDA peak {st.get('cuda_peak_gib')} GiB",
                  flush=True)
        done += len(hs)
        print(f"PROGRESS wind {done * cfg.steps}/{total * cfg.steps}", flush=True)
        del am, out, u, vel, vort                    # nothing of this batch stays on the card into the next
        if gpu:
            torch.cuda.empty_cache()


def _split(scene: Scene, a: int, b: int, overlap_m: float) -> List[Dict[str, Tuple[int, int, int, int]]]:
    """The a x b geographic units of a scene's grid, row-major from the south-west: each unit's core
    (r0, r1, c0, c1), the grid cut evenly, and its window, the core grown by `overlap_m` a side within
    the grid (WS18 M4)."""
    ny, nx = scene.grid.ny, scene.grid.nx
    ov = int(np.ceil(overlap_m / scene.grid.dx - 1e-9))
    rs = [round(i * ny / a) for i in range(a + 1)]
    cs = [round(j * nx / b) for j in range(b + 1)]
    return [{"core": (rs[i], rs[i + 1], cs[j], cs[j + 1]),
             "window": (max(rs[i] - ov, 0), min(rs[i + 1] + ov, ny), max(cs[j] - ov, 0), min(cs[j + 1] + ov, nx))}
            for i in range(a) for j in range(b)]


def coarse_cells(scene: Scene, coarse_dx: float) -> Tuple[int, int]:
    """(nx, ny) of the coarse solve that covers exactly the fine grid's extent: the scene a unit run rebuilds to
    read it, and on a disc site the `--width-cells` its `levels --dx <coarse> --part 0/1` run takes (nx = ny)."""
    return (int(round(scene.grid.nx * scene.grid.dx / coarse_dx)), int(round(scene.grid.ny * scene.grid.dx / coarse_dx)))


def _crop(scene: Scene, window: Tuple[int, int, int, int]) -> Scene:
    """The scene over one unit's window: every column cropped, the levels and the floor kept."""
    r0, r1, c0, c1 = window
    g = scene.grid
    cut = lambda a: None if a is None else a[..., r0:r1, c0:c1]  # noqa: E731
    ox, oy, oz = scene.origin
    return replace(scene, grid=Grid(dx=g.dx, nx=c1 - c0, ny=r1 - r0, zf=g.zf), solid=cut(scene.solid),
                   sink=cut(scene.sink), z0=cut(scene.z0), origin=(ox + c0 * g.dx, oy + r0 * g.dx, oz),
                   terrain=cut(scene.terrain), top=cut(scene.top), roof=cut(scene.roof),
                   cut_open=cut(scene.cut_open), plants=cut(scene.plants), canopy_top=cut(scene.canopy_top))


def _boundary(unit: Scene, coarse: Scene, velocity: np.ndarray, window: Tuple[int, int, int, int],
              shape: Tuple[int, int]):
    """The coarse solve's velocity on a unit's side faces inside the domain (its seams), trilinear
    between the coarse cells' centres (scene metres; both grids share the site's floor). A side on the
    whole domain's edge, and the top, keep the profile, as a single solve's do."""
    from scipy.interpolate import RegularGridInterpolator

    from solver import Boundary

    axes = (coarse.grid.zc, coarse.origin[1] + coarse.grid.yc, coarse.origin[0] + coarse.grid.xc)
    fs = [RegularGridInterpolator(axes, np.asarray(velocity[k], np.float64), bounds_error=False, fill_value=None)
          for k in range(3)]
    g, (ox, oy, _) = unit.grid, unit.origin
    zc, yc, xc = g.zc, oy + g.yc, ox + g.xc
    (r0, r1, c0, c1), (ny, nx) = window, shape

    def at(z, y, x) -> np.ndarray:
        pts = np.stack(np.broadcast_arrays(z, y, x), axis=-1)
        return np.stack([f(pts) for f in fs])

    return Boundary(west=None if c0 == 0 else at(zc[:, None], yc[None, :], ox),
                    east=None if c1 == nx else at(zc[:, None], yc[None, :], ox + g.nx * g.dx),
                    south=None if r0 == 0 else at(zc[:, None], oy, xc[None, :]),
                    north=None if r1 == ny else at(zc[:, None], oy + g.ny * g.dx, xc[None, :]))


def _unit_tile(scene: Scene, coarse: Scene, coarse_cache: Path, site: sites.SiteConfig, receipt: Dict[str, object],
               cfg: SolverConfig, heading: float, cache: Path, spec: Dict, name: str,
               block: Optional[Tuple["view.ViewConfig", float]] = None, points=None) -> Dict[str, object]:
    """One unit's unit-speed solve at `heading`, its window forced on every side and above by the coarse
    solve of the whole domain; its core saved as `unit_<heading>.<name>.npz`. Its stats.

    With `block` (the published grid and the display radius), the unit is a block: it saves its core's levels on the
    published grid, sampled from its own window's solve, and no 3D field; the buffer is discarded and no join follows.
    With `points` (the indices of the survey points its core owns, and where they are read in its window,
    `points.place`), a block also saves their (u, v, w) from the window's solve before the field is dropped."""
    path = cache / f"unit_{heading:g}.{name}.npz"
    if not path.exists():
        with np.load(coarse_cache / f"unit_{heading:g}.npz") as c:
            velocity = c["velocity"].astype(np.float32)
        core, res = _unit_solve(scene, coarse, velocity, spec, forcing.inflow(site, 1.0),
                                _grid_bearing(heading, receipt), cfg)
        tmp = cache / f"unit_{heading:g}.{name}.partial.npz"
        if block is None:
            np.savez(tmp, velocity=core.astype(np.float32), core=np.asarray(spec["core"]),
                     window=np.asarray(spec["window"]), stats=json.dumps(_stats(res)))
        else:
            lv, solid, filled, rect = _block_levels(scene, spec, res, *block)
            extra = {}
            if points is not None:
                import points as P
                extra = {"points_uvw": P.sample(np.asarray(res.velocity), points[1]),
                         "points_index": np.asarray(points[0], np.int64)}
            np.savez(tmp, levels=lv.astype(np.float32), solid=solid, filled=filled, rect=np.asarray(rect),
                     core=np.asarray(spec["core"]), window=np.asarray(spec["window"]), stats=json.dumps(_stats(res)),
                     **extra)
        tmp.replace(path)
    with np.load(path) as z:
        return json.loads(str(z["stats"]))


def _view_rect(vcfg: "view.ViewConfig", scene: Scene, cells: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """(row0, row1, col0, col1) of the published grid's cells whose centres lie in the solver cells (r0, r1, c0, c1),
    rows running north as `view.ViewConfig.centres`: every published cell belongs to exactly one unit's core."""
    import math

    r0, r1, c0, c1 = cells
    g, (ox, oy, _) = scene.grid, scene.origin
    vx0, vy0 = (vcfg.box[0], vcfg.box[1]) if vcfg.box else (-vcfg.half_m, -vcfg.half_m)
    at = lambda v, lo, n: min(n, max(0, math.ceil((v - lo) / vcfg.cell_m - 0.5 - 1e-9)))  # noqa: E731
    return (at(oy + r0 * g.dx, vy0, vcfg.ny), at(oy + r1 * g.dx, vy0, vcfg.ny),
            at(ox + c0 * g.dx, vx0, vcfg.nx), at(ox + c1 * g.dx, vx0, vcfg.nx))


def _sub_view(vcfg: "view.ViewConfig", rect: Tuple[int, int, int, int]) -> "view.ViewConfig":
    """The published grid's cells in `rect` as a grid of its own, shown where the whole grid is shown."""
    j0, j1, i0, i1 = rect
    vx0, vy0 = (vcfg.box[0], vcfg.box[1]) if vcfg.box else (-vcfg.half_m, -vcfg.half_m)
    c = vcfg.cell_m
    return replace(vcfg, box=(vx0 + i0 * c, vy0 + j0 * c, vx0 + i1 * c, vy0 + j1 * c), display=vcfg.display)


def _block_levels(scene: Scene, spec: Dict, res: Result, vcfg: "view.ViewConfig", r: float):
    """A block's levels: the published grid over its window, sampled from the window's own solve (so solid pockets
    and fills see 200 m around), then cut to the cells its core owns: (levels (3, L, rows, cols), solid, filled, rect)."""
    import view

    unit = _crop(scene, spec["window"])
    field = np.concatenate([res.velocity[:2], res.vorticity[2][None]])
    win, core = _view_rect(vcfg, scene, spec["window"]), _view_rect(vcfg, scene, spec["core"])
    lv, solid, filled = view.levels(field, unit, _sub_view(vcfg, win), r)
    cut = (slice(core[0] - win[0], core[1] - win[0]), slice(core[2] - win[2], core[3] - win[2]))
    return lv[..., cut[0], cut[1]], solid[:, cut[0], cut[1]], filled[:, cut[0], cut[1]], core


def _blocks(scene: Scene, vcfg: "view.ViewConfig", heading: float, cache: Path, units: str,
            specs: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """The published levels at `heading` from every block's core (no join, no projection across a seam): (levels
    (3, L, ny, nx), solid, filled, stats), the stats the worst block's divergence and flux with every block's own."""
    L = len(vcfg.heights_m)
    lv = np.full((3, L, vcfg.ny, vcfg.nx), np.nan)
    solid, filled = np.zeros((L, vcfg.ny, vcfg.nx), bool), np.zeros((L, vcfg.ny, vcfg.nx), bool)
    owned = np.zeros((vcfg.ny, vcfg.nx), np.int32)
    unit_stats = []
    for i in range(len(specs)):
        t = cache / f"unit_{heading:g}.tile{i}of{units}.npz"
        assert t.exists(), f"heading {heading:g}: block {i} of {units} is not solved in {cache}"
        with np.load(t) as z:
            assert "levels" in z.files, f"{t} was solved for a join, not as a block"
            j0, j1, i0, i1 = (int(v) for v in z["rect"])
            lv[:, :, j0:j1, i0:i1] = z["levels"]
            solid[:, j0:j1, i0:i1], filled[:, j0:j1, i0:i1] = z["solid"], z["filled"]
            owned[j0:j1, i0:i1] += 1
            unit_stats.append(json.loads(str(z["stats"])))
    assert (owned == 1).all(), f"the blocks' cores cover the published grid {int((owned == 0).sum())} cells short, " \
                               f"{int((owned > 1).sum())} twice"
    worst = max(unit_stats, key=lambda s: abs(s["flux_in_m3_s"] - s["flux_out_m3_s"]) / max(abs(s["flux_in_m3_s"]), 1e-30))
    stats = {"divergence_max_1_s": max(s["divergence_max_1_s"] for s in unit_stats),
             "flux_in_m3_s": worst["flux_in_m3_s"], "flux_out_m3_s": worst["flux_out_m3_s"],
             "wall_s": sum(s.get("wall_s", 0.0) for s in unit_stats), "cells": sum(s.get("cells", 0) for s in unit_stats),
             "blocks": units, "unit_stats": unit_stats,
             **({"settled": all(s.get("settled") is not False for s in unit_stats),
                 "steps": max(s.get("steps", 0) for s in unit_stats)} if any("settled" in s for s in unit_stats) else {})}
    return lv, solid, filled, stats


def _unit_solve(scene: Scene, coarse: Scene, velocity: np.ndarray, spec: Dict, profile: forcing.LogProfile,
                bearing: float, cfg: SolverConfig) -> Tuple[np.ndarray, Result]:
    """One unit solved over its window, forced across its seams by the coarse solve's `velocity`: the
    velocity over its core (3, nz, rows, cols), and the unit's result."""
    unit = _crop(scene, spec["window"])
    res = solve(unit, profile, bearing, cfg,
                boundary=_boundary(unit, coarse, velocity, spec["window"], (scene.grid.ny, scene.grid.nx)))
    (r0, r1, c0, c1), (w0, _, v0, _) = spec["core"], spec["window"]
    return res.velocity[:, :, r0 - w0:r1 - w0, c0 - v0:c1 - v0], res


def _join_project(scene: Scene, profile: forcing.LogProfile, bearing: float, cfg: SolverConfig,
                  parts: List[Tuple[Dict, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """The units' cores in one field, then one projection of the whole (the solver's own Poisson, to
    `tol_final`), so it is divergence-free across the seams as a single solve is: (velocity, vort_z, stats),
    the stats with the divergence before the projection (the seams')."""
    u = np.zeros((3,) + scene.grid.shape, np.float32)
    for spec, vel in parts:
        r0, r1, c0, c1 = spec["core"]
        u[:, :, r0:r1, c0:c1] = vel
    model = Model(scene, profile, bearing, cfg)
    field = torch.as_tensor(u, device=cfg.device, dtype=cfg.dtype)
    field[:, model.solid] = 0.0
    faces = model.faces(field)
    before = model.divergence(faces)
    faces, it, rel = model.project(faces, cfg.tol_final or cfg.tol, cfg.max_iter_final)
    vel = model.cells(faces)
    div = model.divergence(faces)
    flux_in, flux_out = model.boundary_flux(faces)
    stats = {"divergence_max_1_s": float((div / model.vol).abs().max()), "flux_in_m3_s": flux_in,
             "flux_out_m3_s": flux_out, "flux_balance": abs(flux_in - flux_out) / max(flux_in, 1e-30),
             "seam_divergence_max_1_s": float((before / model.vol).abs().max()), "projection_iterations": it,
             "projection_rel": rel, "max_speed_m_s": float(vel.norm(dim=0).max())}
    return vel.cpu().numpy(), model.vorticity(vel)[2].cpu().numpy(), stats


def _joined(scene: Scene, site: sites.SiteConfig, receipt: Dict[str, object], cfg: SolverConfig, heading: float,
            cache: Path, units: str, specs: List[Dict]) -> bool:
    """`unit_<heading>.npz` from every unit's core at `heading`, then one projection of the whole joined
    field (the solver's own Poisson, to `tol_final`), so it is divergence-free across the seams as a
    single solve is; False when a unit is missing. Its stats carry the divergence before the projection
    (the seams') and every unit's own."""
    tiles = [cache / f"unit_{heading:g}.tile{i}of{units}.npz" for i in range(len(specs))]
    if not all(t.exists() for t in tiles):
        return False
    parts, unit_stats = [], []
    for t, spec in zip(tiles, specs):
        with np.load(t) as z:
            parts.append((spec, z["velocity"]))
            unit_stats.append(json.loads(str(z["stats"])))
    vel, vort_z, stats = _join_project(scene, forcing.inflow(site, 1.0), _grid_bearing(heading, receipt), cfg, parts)
    stats.update(units=units, unit_stats=unit_stats)
    tmp = cache / f"unit_{heading:g}.partial.npz"
    np.savez(tmp, velocity=vel, vort_z=vort_z, zf=scene.grid.zf, stats=json.dumps(stats))
    tmp.replace(cache / f"unit_{heading:g}.npz")
    return True


def _band(args: argparse.Namespace, dz: Optional[float] = None) -> Optional[Tuple[float, float]]:
    """The (dz, height) band of fine levels, or None for the stretched grid."""
    dz = args.dz_band if dz is None else dz
    return None if not dz else (dz, args.band)


def _ground_band(args: argparse.Namespace) -> Optional[Tuple[float, float, float]]:
    """The (dz, above, cap) band of fine levels over the terrain, or None."""
    g = getattr(args, "ground_band", None)
    return tuple(float(v) for v in g) if g else None


def _coarse_band(args: argparse.Namespace) -> Optional[Tuple[float, float, float]]:
    """The ground band the coarse solve of a unit run was made with: --ground-band with its DZ replaced by
    --coarse-band-dz when given (a 1 m solve's 2 m coarse solve in 2 m cubes), else --ground-band itself."""
    g = _ground_band(args)
    dz = getattr(args, "coarse_band_dz", None)
    return (float(dz),) + g[1:] if g and dz else g


def _pct(a: np.ndarray, q: float) -> Optional[float]:
    """The q-th percentile of the finite values, or None (JSON null) when there are none: a level no
    fluid cell resolves anywhere in the disc (W1 at 4 m, 2026-09-13, crashed here) is recorded, and
    the pipeline's gate refuses it; it never stops the blend."""
    f = a[np.isfinite(a)]
    return float(np.percentile(f, q)) if f.size else None


def view_config(args: argparse.Namespace, site: sites.SiteConfig, box: Optional[Dict]) -> "view.ViewConfig":
    """The published levels' grid: the display box (or the square about the display disc) on `--view-cell`, which
    defaults to the solver's own `--dx`, so a 1 m solve is shown on 1 m cells and not sampled onto 2 m ones."""
    import view

    cell = args.view_cell or args.dx
    if box:
        return view.ViewConfig.for_box(box["display"], cell, heights_m=tuple(args.heights))
    return view.ViewConfig(heights_m=tuple(args.heights), cell_m=cell, half_m=view.half_for(site.display_radius_m, cell))


OVER_TOP_FILE, OVER_TOP_META = "over_top_f16.bin", "over_top.json"


def _write_over_top(out: Path, ot: Dict[float, Tuple[np.ndarray, np.ndarray]], hs: List[float], vcfg, clearance: float,
                    skip: Optional[str], envelope: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """`levels --over-top-out`: each heading's unit (u, v) at `clearance` over the canopy envelope (view.envelope), float16
    [heading][u, v][ny][nx] in `over_top_f16.bin` with that point's height over the bare earth after it ([ny][nx]),
    on the levels' own grid and rows, NaN beyond the display disc; `over_top.json` says so. Nothing where a run keeps
    no whole field (blocks): the json says why."""
    out.mkdir(parents=True, exist_ok=True)
    doc = {"v": 1, "clearance_m": float(clearance), "headings": [float(h) for h in hs],
           "grid": {"nx": int(vcfg.nx), "ny": int(vcfg.ny), "cell_m": float(vcfg.cell_m),
                    "origin": [float(v) for v in vcfg.origin]},
           "rule": "each heading's unit (u, v) at clearance_m over the envelope of the measured tops (roofs and crowns, gaps "
                   "narrower than the canopy height closed, then smoothed; models/wind view.envelope, view.over_top), "
                   "trilinear over the solve's fluid centres, from the same solves as the levels; NaN inside a "
                   "structure taller than the envelope",
           **({"envelope": envelope} if envelope else {})}
    if skip or len(ot) != len(hs):
        doc["skipped"] = skip or f"{len(hs) - len(ot)} of {len(hs)} headings not read"
        (out / OVER_TOP_META).write_text(json.dumps(doc, indent=1))
        return {"skipped": doc["skipped"]}
    uv = np.stack([ot[h][0][:, ::-1, :] for h in hs]).astype("<f2")     # (H, 2, ny, nx), rows south as the frames
    height = ot[hs[0]][1][::-1].astype("<f2")
    fin = np.isfinite(height)
    doc["finite_share"] = round(float(fin.mean()), 4)
    (out / OVER_TOP_FILE).write_bytes(uv.tobytes() + height.tobytes())
    doc.update(file=OVER_TOP_FILE, dtype="float16", layout="[heading][u, v][ny][nx] then height [ny][nx]",
               rows="running south from the grid's origin y, as the levels' frames (frames.py)", height="above the bare earth [m]",
               height_p50_p90_m=[round(float(np.percentile(height[fin].astype(np.float32), q)), 2) for q in (50, 90)]
               if fin.any() else None)
    (out / OVER_TOP_META).write_text(json.dumps(doc, indent=1))
    return {k: doc[k] for k in ("clearance_m", "height_p50_p90_m", "finite_share")} | {"headings": len(hs)}


def cmd_levels(args: argparse.Namespace) -> None:
    """The viewer's day at heights above the bare earth, sampled from unit-speed solves per heading: solid
    where a structure stands taller than the level, finite in every other cell of the display disc."""
    import view

    bundle, cfg = Path(args.bundle), _config(args)
    site = sites.bundled(sites.get_site(args.site), bundle, args.buffer)
    box = sites.box_of(site, bundle)
    cells = args.width_cells
    vcfg = view_config(args, site, box)
    if box and args.extent_dx:
        # A site that follows its polygon: the product over its display box; a coarse solve (--extent-dx, the
        # fine cell) covers the fine grid's own extent, as `coarse_cells` rebuilds it in each unit run.
        fx, fy = sites.box_cells(box["fetch"], args.extent_dx)
        cells = (int(round(fx * args.extent_dx / args.dx)), int(round(fy * args.extent_dx / args.dx)))
    scene, receipt = _canopy(*domain.from_bundle(site, args.dx, bundle, band=_band(args), cells=cells,
                                        ground_band=_ground_band(args), canopy_profile=_profile(args)), args)
    if args.octree_count:               # the layout's leaves and expected peak, for planning: no forcing, no solve
        print("OCTREE " + json.dumps(octree_layout(scene, site, args)[1]), flush=True)
        return
    record = json.loads(Path(args.forcing).read_text())
    day, r = record["forcing"], site.display_radius_m
    resolution = view.resolution(scene, vcfg, r, sorted(set(args.candidates) | set(args.heights)))
    print(json.dumps(resolution, indent=1), flush=True)

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    specs = None
    if args.units:
        # Geographic units (WS18 M4): each unit solves every heading over its window, forced by the coarse
        # solve of the whole domain (`levels --dx <coarse> --part 0/1` into --coarse-cache); the run without
        # --unit joins their cores and projects the whole once per heading, then blends as always.
        a, b = (int(v) for v in args.units.lower().split("x"))
        specs = _split(scene, a, b, args.overlap_m)
        if args.unit is not None:
            # The coarse solve's square is the fine grid's own (`coarse_cells`): its sides see the profile
            # where the single solve's do. Rounded up to 128 on its own, Houston's coarse square was 1,024 m
            # against the fine 256 m, and its seams carried another domain's flow (2026-09-13).
            coarse, _ = _canopy(*domain.from_bundle(site, args.coarse_dx, bundle, band=_band(args),
                                           cells=coarse_cells(scene, args.coarse_dx), ground_band=_coarse_band(args),
                                           canopy_profile=_profile(args)), args)
            hs = view.headings(day["from_deg"], vcfg.spacing_deg)
            if args.part:                  # one share of this unit's headings: a block fleet's queue item
                hs = view.share(hs, args.part)
            unit_points = None
            if args.points and args.blocks:  # the survey points this block's core owns, read in its own window
                import points as P
                px, py, pz, ph = P.read_points(Path(args.points))
                mine = P.core_points(scene, px, py, specs[args.unit]["core"])
                zabs = pz - P.datum_offset(scene, px, py, pz, ph) if args.points_z == "datum" else None
                nrm = P.normals_arg(args.points_normals, len(px))
                unit_points = (mine, P.place(_crop(scene, specs[args.unit]["window"]), px[mine], py[mine], ph[mine],
                                             z=None if zabs is None else zabs[mine],
                                             normals=nrm[mine] if isinstance(nrm, np.ndarray) else nrm))
            for j, h in enumerate(hs):
                _progress(j, len(hs), cfg)
                s = _unit_tile(scene, coarse, Path(args.coarse_cache), site, receipt, cfg, h, cache,
                               specs[args.unit], f"tile{args.unit}of{args.units}", (vcfg, r) if args.blocks else None,
                               points=unit_points)
                _progress(j, len(hs), cfg, done=True)
                print(f"  unit {args.unit}/{args.units} heading {h:g}: divergence {s['divergence_max_1_s']:.2e} 1/s",
                      flush=True)
            print(json.dumps({"unit": f"{args.unit}/{args.units}", "core": specs[args.unit]["core"],
                              "window": specs[args.unit]["window"]}), flush=True)
            return
    if args.part:
        # One share of the unit solves into the shared cache, and nothing else: k of these on k GPUs,
        # then one `levels` without --part, which finds every heading solved and only blends.
        mine = view.share(view.headings(day["from_deg"], vcfg.spacing_deg), args.part)
        if args.octree and specs is None:
            _octree_units(scene, site, receipt, cfg, mine, cache, args)
        for j, h in enumerate(mine):
            _progress(j, len(mine), cfg)
            _, s = _unit_field(scene, site, receipt, cfg, h, cache)
            _progress(j, len(mine), cfg, done=True)
            print(f"  heading {h:g}: divergence {s['divergence_max_1_s']:.2e} 1/s", flush=True)
        print(json.dumps({"part": args.part, "headings": mine}), flush=True)
        return
    basis, solves = {}, {}
    hs = view.headings(day["from_deg"], vcfg.spacing_deg)
    if args.octree and specs is None:               # every heading on adaptive cells, batched, into the same cache
        _octree_units(scene, site, receipt, cfg, hs, cache, args)
    ot = {} if args.over_top_out else None          # each heading's unit (u, v) over the measured top, and its height
    ot_m, ot_skip = (args.over_top_m if args.over_top_m is not None else view.OVER_TOP_M), None
    if ot is not None:                    # the canopy envelope the ribbons ride, measured over the shown site
        gx, gy = np.meshgrid(scene.origin[0] + scene.grid.xc, scene.origin[1] + scene.grid.yc)
        ot_env, ot_info = view.envelope(scene, vcfg.shown(gx, gy, r))
    else:
        ot_env, ot_info = None, None
    pw = None
    if args.points:
        # Every survey point read from the 3D solve (points.py), in the same run and from the same cache as the levels.
        import points as P
        px, py, pz, ph = P.read_points(Path(args.points))
        off = P.datum_offset(scene, px, py, pz, ph)
        nrm = P.normals_arg(args.points_normals, len(px))
        placed = P.place(scene, px, py, ph, z=pz - off if args.points_z == "datum" else None, normals=nrm)
        pw = P.Writer(Path(args.points_out or args.out), hs, len(px))
        no_points = []
    for j, h in enumerate(hs):
        if specs is not None and args.blocks:
            # Blocks (WS18 M4 without the join): each core's levels as its own window solved them; no whole-site field.
            basis[h], solid, filled, solves[f"{h:g}"] = _blocks(scene, vcfg, h, cache, args.units, specs)
            if pw is not None:           # each block's own points, sampled in its window before its field was dropped
                for i in range(len(specs)):
                    with np.load(cache / f"unit_{h:g}.tile{i}of{args.units}.npz") as z:
                        if "points_uvw" in z.files:
                            pw.put(h, z["points_uvw"], z["points_index"])
                        else:
                            no_points.append(f"{h:g}/{i}")
            if ot is not None:
                ot_skip = "blocks keep no whole field to read over the top"
            _progress(j, len(hs), cfg, done=True)
            print(f"  heading {h:g}: divergence {solves[f'{h:g}']['divergence_max_1_s']:.2e} 1/s", flush=True)
            continue
        if specs is not None and not (cache / f"unit_{h:g}.npz").exists():
            assert _joined(scene, site, receipt, cfg, h, cache, args.units, specs), \
                f"heading {h:g}: not every unit of {args.units} is solved in {cache}"
        _progress(j, len(hs), cfg)
        field, solves[f"{h:g}"] = _unit_field(scene, site, receipt, cfg, h, cache)
        _progress(j, len(hs), cfg, done=True)
        basis[h], solid, filled = view.levels(field, scene, vcfg, r)
        if ot is not None:
            ot[h] = view.over_top(field[:2], scene, vcfg, r, ot_m, ot_env)
        if pw is not None:
            del field
            with np.load(cache / f"unit_{h:g}.npz") as z:
                pw.put(h, P.sample(z["velocity"].astype(np.float32), placed))
        print(f"  heading {h:g}: divergence {solves[f'{h:g}']['divergence_max_1_s']:.2e} 1/s", flush=True)
    if pw is not None:
        pmeta = pw.close(P.receipt(placed, pz, ph, scene, scene.grid.dx), placed.search_u8(),
                         {"cache": cache.name, "grid_shape": list(scene.grid.shape), "cell_m": scene.grid.dx,
                          "blocks_without_points": no_points, "placement": args.points_z,
                          "normals": args.points_normals if args.points_normals in ("geometric", "none") else "survey",
                          "datum_offset_m": round(off, 3)})
        print("POINTS " + json.dumps({k: pmeta[k] for k in ("points", "unresolved", "search", "raised_share",
                                                            "pushed_share", "dtm_difference_m", "headings_done")},
                                     default=float), flush=True)
    if ot is not None:
        print("OVER_TOP " + json.dumps(_write_over_top(Path(args.over_top_out), ot, hs, vcfg, ot_m, ot_skip, ot_info)),
              flush=True)
    records = []
    for s, d in zip(day["speed10_m_s"], day["from_deg"]):
        u, v, vort = s * view.blend(basis, d, vcfg.spacing_deg)
        records.append(np.stack([u, v, np.hypot(u, v), vort]))
    stack = np.stack(records)

    x, y = vcfg.centres()
    inside = vcfg.shown(x, y, r)
    base, top, roof = view.ground(scene, vcfg), view.surface(scene, vcfg), view.roofs(scene, vcfg)
    above = view.under(scene, vcfg, view.first_fluid(scene)[0])
    bare = top - base <= 1e-6
    per_level = []
    for j, h in enumerate(vcfg.heights_m):
        speed, vort = stack[:, 2, j][:, inside], stack[:, 3, j][:, inside]
        s, f = solid[j] & inside, filled[j] & inside
        depth = (above - (base + h - top))[f]
        gaps = np.isnan(stack[:, 2, j]).any(axis=0) & inside & ~solid[j]
        per_level.append({"height_m": h, "speed_p50_m_s": _pct(speed, 50), "speed_p99_m_s": _pct(speed, 99),
                          "vort_z_abs_p99_1_s": _pct(np.abs(vort), 99), "cells_in_disc": int(inside.sum()),
                          "solid_cells_in_disc": int(s.sum()), "fluid_cells_in_disc": int((inside & ~s).sum()),
                          "nan_fluid_cells_in_disc": int(gaps.sum()), "nan_cells_in_disc": int(gaps.sum()),
                          "filled_cells_in_disc": int(f.sum()), "filled_roof": int((f & roof).sum()),
                          "filled_ground": int((f & ~roof).sum()), "filled_open_ground": int((f & bare).sum()),
                          "fill_depth_max_m": float(depth.max()) if depth.size else 0.0})
    hi = {name: _pct(np.abs(stack[:, i][:, :, inside]), 99.9) or 0.0 for i, (name, _) in enumerate(LEVEL_FIELDS)}
    fields = {"u": {"unit": "m/s", "domain": [-np.ceil(2 * hi["u"]) / 2, np.ceil(2 * hi["u"]) / 2]},
              "v": {"unit": "m/s", "domain": [-np.ceil(2 * hi["v"]) / 2, np.ceil(2 * hi["v"]) / 2]},
              "speed": {"unit": "m/s", "domain": [0.0, np.ceil(2 * hi["speed"]) / 2], "lut": "turbo",
                        "title": "Wind speed", "note": "horizontal, hypot(u, v)"},
              "vort_z": {"unit": "1/s", "domain": [-np.ceil(20 * hi["vort_z"]) / 20, np.ceil(20 * hi["vort_z"]) / 20],
                         "lut": "diverging", "title": "Vorticity",
                         "note": "vertical component dv/dx - du/dy of the 3D solve, counterclockwise positive"}}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    header = frames.Header(n_frames=0, shape=(len(vcfg.heights_m), vcfg.ny, vcfg.nx), fields=LEVEL_FIELDS,
                           cell_m=vcfg.cell_m, origin=vcfg.origin,
                           zf=(float("nan"),) * (len(vcfg.heights_m) + 1),
                           sample=args.sample)
    path = frames.write_fields(out / args.name, header, [60.0 * m for m in day["minute"]], records)
    ground_path, solid_path, filled_path = out / GROUND_FILE, out / SOLID_FILE, out / FILLED_FILE
    ground_path.write_bytes(np.ascontiguousarray(base[::-1]).astype("<f4").tobytes())
    solid_path.write_bytes(np.ascontiguousarray(solid[:, ::-1]).astype(np.uint8).tobytes())
    filled_path.write_bytes(np.ascontiguousarray(filled[:, ::-1]).astype(np.uint8).tobytes())
    viewer = view.viewer_ground(Path(args.ground), scene, vcfg) if args.ground and not box else None
    compare = {}
    if viewer is not None:
        d = np.abs(base - view.crop(viewer[None], scene, vcfg, r)[0])[inside]
        compare = {"cells_mean_abs_m": float(np.nanmean(d)) if np.isfinite(d).any() else None,
                   "cells_p50_abs_m": _pct(d, 50), "cells_p90_abs_m": _pct(d, 90)}
    side = {k: record[k] for k in ("t0", "timezone", "site", "epsg", "anchor_utm", "forcing", "marks",
                                   "label", "run_s")}
    side.update(
        surface="ground", level_datum="bare_earth", levels_version=LEVELS_VERSION,
        levels_above_ground_m=list(vcfg.heights_m), ground_file=ground_path.name,
        # A viewer that predates levels_version 2 stands each level on `surface_file`: now the bare earth.
        levels_above_surface_m=list(vcfg.heights_m), surface_file=ground_path.name,
        solid={"file": solid_path.name, "dtype": "uint8", "shape": [len(vcfg.heights_m), vcfg.ny, vcfg.nx],
               "rows": "north first, like the frames",
               "rule": "1 where a solid column's top stands more than the level's height above the bare earth, "
                       "or in an enclosed pocket holding no 3 x 3 open square"},
        fill={"file": filled_path.name, "dtype": "uint8", "shape": [len(vcfg.heights_m), vcfg.ny, vcfg.nx],
              "rows": "north first, like the frames",
              "rule": "1 where the level lies below its column's lowest fluid cell centre; that centre's value "
                      "re-rooted down by ln(1 + z / z0) over the column's roughness"},
        fields=fields, display_radius_m=r, fetch_radius_m=site.fetch_radius_m, buffer_m=site.buffer_m,
        levels=per_level,
        ground_spec={"rows": "north first, like the frames", "shape": [vcfg.ny, vcfg.nx], "cell_m": vcfg.cell_m,
                     "dtype": "float32", "units": "scene-frame elevation [m]",
                     "definition": "bare earth under each cell, under buildings and canopy too; finite over the "
                                   "whole square", "roof_cells_in_disc": int((inside & roof).sum()),
                     "viewer_ground": compare,
                     "top_above_ground_max_m": float(np.max((top - base)[inside])) if inside.any() else 0.0},
        provenance={"kind": "design-forcing", "forcing_from": args.forcing,
                    "solver": "models/wind mass-consistent 3D solve, one unit-speed field per heading, "
                              "scaled by speed and blended linearly in angle",
                    "sampling": "at z = bare earth + height, trilinear over the fluid cell centres around the "
                                "point; solid where a structure stands taller than the level; below the column's "
                                "lowest fluid centre, that centre re-rooted by the log law (fill); NaN beyond "
                                "display_radius_m and in solid cells",
                    "legend_rule": "speed [0, p99.9] and |u|, |v| p99.9 rounded up to 0.5 m/s; vort_z +-p99.9 "
                                   "of |vort_z| rounded up to 0.05 1/s; over every frame, level and disc cell",
                    "headings_deg": sorted(basis), "spacing_deg": vcfg.spacing_deg, "steps": cfg.steps,
                    "tol_final": cfg.tol_final, "dtype": str(cfg.dtype), "solves": solves,
                    "divergence_max_1_s": max(s["divergence_max_1_s"] for s in solves.values()),
                    "scene": scene.summary(), "parameters": receipt, "resolution": resolution,
                    "band_dz_height_m": _band(args), "grid_shape": list(scene.grid.shape),
                    "grid_zf_m": [float(z) for z in scene.grid.zf]})
    if box:                                        # the display region the *_in_disc counts are over
        side["box_scene_m"] = {k: list(v) for k, v in box.items()}
    Path(str(path) + ".json").write_text(json.dumps(side, indent=1, default=float))
    summary = {"product": _publish(path), "ground": frames.compress(ground_path),
               "solid": frames.compress(solid_path), "filled": frames.compress(filled_path), "levels": per_level,
               "fields": fields, "divergence_max_1_s": side["provenance"]["divergence_max_1_s"],
               "viewer_ground": compare}
    print(json.dumps(summary, indent=1, default=float))


def cmd_refine(args: argparse.Namespace) -> None:
    """One unit-speed heading on vertical grids of shrinking cells, compared with the finest at the
    product's levels over the display disc: how much of each level is the vertical grid."""
    import view

    bundle, cfg = Path(args.bundle), _config(args)
    site = sites.bundled(sites.get_site(args.site), bundle, args.buffer)
    vcfg = view.ViewConfig(heights_m=tuple(args.heights), half_m=view.half_for(site.display_radius_m))
    x, y = vcfg.centres()
    inside = np.hypot(x, y) <= site.display_radius_m
    speeds, report = {}, {"heading_deg": args.heading, "band_m": args.band, "grids": {}}
    for dz, cache in zip(args.dz_bands, args.caches):
        scene, receipt = domain.from_bundle(site, args.dx, bundle, band=_band(args, dz), cells=args.width_cells)
        Path(cache).mkdir(parents=True, exist_ok=True)
        field, stats = _unit_field(scene, site, receipt, cfg, args.heading, Path(cache))
        level, _, filled = view.levels(field, scene, vcfg, site.display_radius_m)
        speeds[dz] = np.hypot(level[0], level[1])
        report["grids"][f"{dz:g}"] = {"levels": scene.grid.nz, "cells": scene.grid.cells, "wall_s": stats["wall_s"],
                                      "divergence_max_1_s": stats["divergence_max_1_s"],
                                      "final_change_rel": stats["final_change_rel"],
                                      "filled_in_disc": [int(u.sum()) for u in filled]}
        if args.continue_steps:
            full = np.load(Path(cache) / f"unit_{args.heading:g}.npz")["velocity"].astype(np.float64)
            more = solve(scene, forcing.inflow(site, 1.0), _grid_bearing(args.heading, receipt),
                         replace(cfg, steps=args.continue_steps), initial=full)
            later = view.levels(more.velocity[:2], scene, vcfg, site.display_radius_m)[0]
            d = np.abs(np.hypot(later[0], later[1]) - speeds[dz])
            report["grids"][f"{dz:g}"]["continued"] = {
                "steps": args.continue_steps, "final_change_rel": more.change[-1],
                "divergence_max_1_s": more.divergence_max_1_s,
                "levels": [{"height_m": h, "median_speed_m_s": float(np.nanmedian(speeds[dz][j][inside])),
                            "median_abs_change_m_s": float(np.nanmedian(d[j][inside])),
                            "p90_abs_change_m_s": float(np.nanpercentile(d[j][inside], 90))}
                           for j, h in enumerate(vcfg.heights_m)]}
        print(f"  dz {dz:g}: {scene.grid.shape}, {stats['wall_s']:.0f} s", flush=True)
    finest = args.dz_bands[-1]
    for dz in args.dz_bands[:-1]:
        rows = []
        for j, h in enumerate(vcfg.heights_m):
            both = np.isfinite(speeds[dz][j]) & np.isfinite(speeds[finest][j])
            wall = view.under(scene, vcfg, view.beside_wall(scene, h))
            row = {"height_m": h}
            for name, m in (("open", both & ~wall), ("beside_wall", both & wall)):
                d = np.abs(speeds[dz][j] - speeds[finest][j])[m]
                row[name] = {"cells": int(m.sum()), "finest_median_m_s": _pct(speeds[finest][j][m], 50),
                             "median_abs_diff_m_s": _pct(d, 50), "p90_abs_diff_m_s": _pct(d, 90)}
            rows.append(row)
        report[f"{dz:g}_against_{finest:g}"] = rows
    Path(args.out).write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))


def cmd_extent(args: argparse.Namespace) -> None:
    """One forcing over square grids of growing width around the same geometry, open ground beyond
    the fetch disc: speed at the centre and over the display disc against the sides' distance."""
    import view

    bundle, cfg = Path(args.bundle), _config(args)
    site = sites.bundled(sites.get_site(args.site), bundle, args.buffer)
    vcfg = view.ViewConfig(heights_m=tuple(args.heights), half_m=view.half_for(site.display_radius_m))
    x, y = vcfg.centres()
    inside, centre = np.hypot(x, y) <= site.display_radius_m, np.hypot(x, y) <= args.centre_m
    rows, previous = [], None
    for n in args.cells:
        scene, receipt = domain.from_bundle(site, args.dx, bundle, band=_band(args), cells=n)
        res = solve(scene, forcing.inflow(site, args.speed), _grid_bearing(args.direction, receipt), cfg)
        level = view.levels(res.velocity[:2], scene, vcfg, site.display_radius_m)[0]
        speed = np.hypot(level[0], level[1])
        stats = _stats(res)
        row = {"cells": n, "half_width_m": n * args.dx / 2, "sides_beyond_fetch_disc_m": n * args.dx / 2 - site.fetch_radius_m,
               "grid": list(scene.grid.shape), "wall_s": stats["wall_s"], "divergence_max_1_s": stats["divergence_max_1_s"],
               "final_change_rel": stats["final_change_rel"], "flux_in_m3_s": stats["flux_in_m3_s"]}
        for j, h in enumerate(vcfg.heights_m):
            s = speed[j]
            row[f"{h:g}m"] = {"centre_mean_m_s": float(np.nanmean(s[centre])), "centre_cells": int(np.isfinite(s[centre]).sum()),
                              "disc_p50_m_s": float(np.nanpercentile(s[inside], 50)),
                              "disc_p90_m_s": float(np.nanpercentile(s[inside], 90))}
            if previous is not None:
                d = np.abs(s - previous[j])[inside]
                row[f"{h:g}m"].update(median_abs_change_from_previous_m_s=float(np.nanmedian(d)),
                                      p90_abs_change_from_previous_m_s=float(np.nanpercentile(d, 90)))
        rows.append(row)
        previous = speed
        print(json.dumps(row, default=float), flush=True)
    report = {"forcing": {"speed10_m_s": args.speed, "from_deg": args.direction}, "centre_m": args.centre_m,
              "band_dz_height_m": _band(args), "extents": rows}
    Path(args.out).write_text(json.dumps(report, indent=1, default=float))


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
            p.add_argument("--settle-rule", default="window", choices=("tail", "window"),
                           help="tail: stop when the estimated change still to come is under --settle-tol")
            p.add_argument("--canopy", default=None,
                           help="the sun set's canopy files (canopy_tau.json): drag with height from the survey")
            p.add_argument("--doy", type=int, default=None, help="day of year of the canopy's leaves; default as flown")
            p.add_argument("--inflow", default="log", choices=("canopy", "log"),
                           help="canopy: the sides carry the steady column over the site's mean canopy")
            p.add_argument("--closure", default="mixing", choices=("mixing", "k-l"),
                           help="k-l: transported turbulent kinetic energy with a prescribed length (Katul et al. 2004)")
            p.add_argument("--drive", default="shear", choices=("shear", "pressure"),
                           help="pressure: a mean pressure gradient drives the column and the domain, no stress on top")
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

    p = add("buffer", cmd_buffer, "the bundle solved without and with its buffer, compared", solver=True)
    scene_args(p)
    p.add_argument("--speed", type=float, required=True, help="station speed at 10 m [m/s]")
    p.add_argument("--direction", type=float, required=True, help="blowing from [deg]")
    p.add_argument("--footprint", help="JSON with name and outline_scene_m of a building the disc cuts")

    p = add("view", cmd_view, "the viewer's day, composed from unit-speed solves per heading", solver=True)
    scene_args(p)
    p.add_argument("--forcing", required=True, help="sidecar JSON holding forcing.minute, speed10_m_s, from_deg")
    p.add_argument("--ground", required=True, help="the viewer's ground raster: float32, 1 m, rows running north")

    p = add("levels", cmd_levels, "the viewer's day at heights above the bare earth", solver=True)
    scene_args(p)
    p.add_argument("--forcing", required=True, help="sidecar JSON holding forcing.minute, speed10_m_s, from_deg")
    p.add_argument("--heights", type=float, nargs="+", default=[1.0, 5.0, 10.0, 25.0],
                   help="level heights above the bare earth [m]")
    p.add_argument("--candidates", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
                   help="heights whose cells below the lowest fluid centre are counted [m]")
    p.add_argument("--cache", required=True, help="directory of unit-speed solves per heading, solved when missing")
    p.add_argument("--out", required=True, help="directory the product is written to")
    p.add_argument("--name", default="frames_levels.simf")
    p.add_argument("--sample", type=int, default=2, choices=(0, 2), help="0 float32, 2 float16")
    p.add_argument("--ground", help="the viewer's ground raster, to compare the surface against")
    p.add_argument("--dz-band", type=float, help="cells of this height [m] from the floor to --band")
    p.add_argument("--width-cells", type=int, help="grid width in cells; default the fetch disc rounded up to 128")
    p.add_argument("--band", type=float, default=100.0, help="top of the fine band above the floor [m]")
    p.add_argument("--ground-band", type=float, nargs=3, metavar=("DZ", "ABOVE", "CAP"),
                   help="cells of DZ [m] from the floor to the highest terrain plus ABOVE, at most CAP above the floor")
    p.add_argument("--part", help="i/k: solve only every k-th heading from the i-th into --cache, then stop")
    p.add_argument("--units", help="AxB: the grid in A x B geographic units, each forced by a coarse solve (WS18 M4)")
    p.add_argument("--unit", type=int, help="solve only unit i of --units, every heading, into --cache, then stop")
    p.add_argument("--overlap-m", type=float, default=200.0, help="each unit's window beyond its core [m]")
    p.add_argument("--coarse-dx", type=float, default=4.0, help="the coarse solve's cell [m]")
    p.add_argument("--coarse-cache", help="the coarse solve's cache (`levels --dx <coarse-dx> --part 0/1`)")
    p.add_argument("--coarse-band-dz", type=float, help="the DZ of the coarse solve's --ground-band; default the unit's own")
    p.add_argument("--blocks", action="store_true",
                   help="with --units: each unit is a block, its core's levels from its own window's solve, its buffer "
                        "discarded; no join and no projection of the whole site")
    p.add_argument("--extent-dx", type=float, help="on a site following its polygon, cover the grid extent of this "
                   "(fine) cell: the coarse solve of a unit run; --width-cells is its disc-site counterpart")
    p.add_argument("--view-cell", type=float, help="the published levels' cell [m]; default the solver's --dx")
    p.add_argument("--points", help="survey points, float32 [N, 4] x, y, z, height above the bare earth (points.py): "
                                    "each read from the 3D solve, per heading, into --points-out")
    p.add_argument("--points-out", help="directory for points_basis_f16.bin and its sidecar; default --out")
    p.add_argument("--points-normals", default="geometric",
                   help="geometric (the solids' own outward normal, smoothed; points.GeometricNormals), none, or a file of "
                        "survey normals, float32 [N, 3] (NaN where none): a return near a "
                        "solid leaves it along its own normal, continuous round an edge")
    p.add_argument("--points-z", default="datum", choices=("datum", "bare_earth"),
                   help="datum: each point at its own z less the frames' median offset (points.datum_offset), its place "
                        "against the solids kept where the two DTMs differ; bare_earth: at its height over the solver's "
                        "terrain")
    p.add_argument("--over-top-out", help="directory for over_top_f16.bin and over_top.json: each heading's unit (u, v) at "
                                          "--over-top-m over each column's measured top (crown or roof, view.over_top), "
                                          "read from the same solves as the levels, and that height over the bare earth")
    p.add_argument("--over-top-m", type=float, help="the clearance over the measured top [m]; default view.OVER_TOP_M")
    p.add_argument("--octree", action="store_true",
                   help="solve each heading on adaptive cells (amr, amr_model): the grid's own cells within --octree-shell "
                        "of every surface and over the display disc's lower air, doubling outward; the field put back on "
                        "the grid's cells for every product, headings marched --octree-batch at a time")
    p.add_argument("--octree-shell", type=float, default=3.0, help="single cells within this distance of a surface [cells]")
    p.add_argument("--octree-shell-out", type=float, default=1.0, help="the same beyond the survey [cells]")
    p.add_argument("--octree-band", type=float, default=2.0, help="cells of each size before the next doubling")
    p.add_argument("--octree-margin", type=float, default=8.0, help="cells around the survey's columns counted inside it")
    p.add_argument("--octree-core-m", type=float, default=10.0,
                   help="single cells over the display disc grown by this [m], up to --octree-core-h")
    p.add_argument("--octree-core-h", default="auto",
                   help="the core's height over the bare earth [m]; auto: each column's tallest read (amr.Criterion)")
    p.add_argument("--octree-batch", type=int, default=4, help="headings marched together on one GPU")
    p.add_argument("--octree-count", action="store_true",
                   help="print the adaptive layout (leaves, cells, expected CUDA peak, the solver a heading runs on) and stop: "
                        "numpy alone, no GPU and no solve, for planning")

    p = add("refine", cmd_refine, "one heading on finer vertical grids, compared at the levels", solver=True)
    scene_args(p)
    p.add_argument("--heading", type=float, required=True, help="blowing from [deg], a solved heading")
    p.add_argument("--dz-bands", type=float, nargs="+", required=True,
                   help="fine-band cell heights [m], coarsest first; 0 is the stretched grid")
    p.add_argument("--caches", nargs="+", required=True, help="one unit-solve cache per grid")
    p.add_argument("--continue-steps", type=int, default=0,
                   help="momentum steps more from each cached field, to measure how steady each level is")
    p.add_argument("--width-cells", type=int, help="grid width in cells; default the fetch disc rounded up to 128")
    p.add_argument("--band", type=float, default=100.0, help="top of the fine band above the floor [m]")
    p.add_argument("--heights", type=float, nargs="+", default=[1.0, 5.0, 10.0, 25.0])
    p.add_argument("--out", required=True, help="report JSON")

    p = add("extent", cmd_extent, "one forcing over wider grids of open ground, compared", solver=True)
    scene_args(p)
    p.add_argument("--cells", type=int, nargs="+", required=True, help="grid widths in cells, narrowest first")
    p.add_argument("--speed", type=float, required=True, help="station speed at 10 m [m/s]")
    p.add_argument("--direction", type=float, required=True, help="blowing from [deg]")
    p.add_argument("--heights", type=float, nargs="+", default=[5.0, 10.0])
    p.add_argument("--centre-m", type=float, default=12.32, help="radius of the centre the speed is averaged over [m]")
    p.add_argument("--dz-band", type=float, help="cells of this height [m] from the floor to --band")
    p.add_argument("--band", type=float, default=100.0, help="top of the fine band above the floor [m]")
    p.add_argument("--out", required=True, help="report JSON")

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
