"""Command line for the hydro twin: one stage per subcommand.

    python3 cli.py fetch    --site site3 --storm ian
    python3 cli.py terrain  --site site3
    python3 cli.py simulate --site site3 --storm ian --cell-size 25
    python3 cli.py ensemble --site site3 --cell-size 25
    python3 cli.py validate --site site3 --storm ian
    python3 cli.py parcel   --site campanile --synthetic --dx 0.2
    python3 cli.py buffer   --site campanile --dx 0.2 --dtm <dtm> --bundle <dir> --drop-buildings
    python3 cli.py profile  --site campanile --dx 0.2

Run it from this directory. The modules are flat and import each other by name, so there is no
package to install and no path manipulation anywhere.
"""

import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import domain
import forcing
import frames
import inflow as inflow_mod
import params
import parcel
import probability
import rainseries
import sites
import surface
import terrain
import units as units_mod
import validate
from solver import FIELDS, Probes, Result, SolverConfig, Surface, simulate as run_solver

CFS_PER_CMS = 35.3146667
DOCS = Path(__file__).resolve().parent / "docs"


def _summary_path(site: sites.SiteConfig, name: str) -> Path:
    """Where a stage records what it did."""
    return site.out_path(f"{name}.json")


def _tag(storm_name: str, cell_size_m: float, surface_field: bool, infiltration: str = "horton",
         basin: bool = False, keep_depressions: bool = False, antecedent: Optional[str] = None) -> str:
    """Run identifier shared by every stage that reads or writes a run's files."""
    return (f"{storm_name}_{cell_size_m:g}m" + ("_surface" if surface_field else "")
            + ("" if infiltration == "horton" else f"_{infiltration}") + ("_basin" if basin else "")
            + ("_depressions" if keep_depressions else "") + ("_antecedent" if antecedent else ""))


def _receipt(res: Result, extra: Dict[str, object]) -> Dict[str, object]:
    """What every run records."""
    m = res.mass
    return {
        **extra,
        "grid": list(res.h_final.shape), "device": res.device,
        "peak_depth_m": float(res.h_max.max()), "n_substeps": res.n_substeps,
        "substep_cap_hits": res.substep_cap_hits, "wall_s": res.wall_s,
        "substeps_per_s": res.n_substeps / max(res.wall_s, 1e-9),
        "mass_m3": {"rain": m.rain, "initial": m.initial, "inflow": m.inflow, "created": m.created,
                    "infiltrated": m.infiltrated, "abstracted": m.abstracted,
                    "stored": m.stored, "outflow": m.outflow},
        "mass_residual": m.residual, "mass_residual_pct": m.residual_pct,
    }


def cmd_fetch(args: argparse.Namespace) -> None:
    """Download every public dataset this site needs."""
    import fetch

    site = sites.get_site(args.site)
    storms = [sites.get_storm(s) for s in (args.storm or [])]
    summary = fetch.all_sources(site, storms)
    print(json.dumps(summary, indent=1))
    if summary.get("atlas14") != "pfds":
        print("\nWARNING: Atlas 14 fell back to non-PFDS values. Design storms will refuse to "
              "run until this is re-fetched; observed-storm runs are unaffected.")


def cmd_terrain(args: argparse.Namespace) -> None:
    """Burn streams, breach depressions, route flow, build HAND and delineate."""
    site = sites.get_site(args.site)
    stats = terrain.condition(site, acc_area_m2=args.acc_area_m2)
    print(json.dumps(stats, indent=1))
    _summary_path(site, "terrain").write_text(json.dumps(stats, indent=1))


def cmd_segment(args: argparse.Namespace) -> None:
    """Segment aerial imagery into surface classes. Python 3.11 stage; see surface.segment."""
    site = sites.get_site(args.site)
    stats = surface.segment(site)
    print(json.dumps(stats, indent=1))
    _summary_path(site, "surface").write_text(json.dumps(stats, indent=1))


def _run(site: sites.SiteConfig, rain: Sequence[float], cell_size: float, dt_s: float,
         frame_min: float, with_gauge: bool = True,
         use_surface: bool = False, infiltration: str = "horton", basin: bool = False,
         keep_depressions: bool = False, antecedent: Optional[str] = None,
         rain_of: Optional[Callable] = None, manning: str = "scalar") -> Tuple[Result, dict, float]:
    """Assemble the domain and integrate one storm; `rain_of(surface, profile)`, when given, makes the rain on the
    assembled grid (a gridded record) instead of `rain`."""
    surf, profile, dx = domain.build_surface(site, cell_size, infiltration, basin, keep_depressions, antecedent)
    if rain_of is not None:
        rain = rain_of(surf, profile)
    if manning == "nlcd":               # roughness by land cover, each class's published flood-plain n
        surf.manning_n = domain.manning_nlcd(site, surf.z.shape, profile)
        v = surf.manning_n[surf.valid]
        print("  manning nlcd: " + json.dumps({f"{q}": round(float(np.percentile(v, q)), 3) for q in (5, 50, 95)}))
    if use_surface:
        fields = surface.rasterize(site, surf.z.shape, profile)
        surf.manning_n = fields["manning_n"]
        print("  surface field: " + json.dumps(surface.summary(fields)))

    probes = Probes()
    if with_gauge and site.gauge is not None:
        probes = Probes(gauge_rc=domain.snap_gauge(site, surf.z, profile, dx))
    cfg = SolverConfig(dx=dx, dt_s=dt_s, cfl_alpha=site.cfl_alpha, frame_interval_min=frame_min)
    return run_solver(surf, rain, cfg, probes), profile, dx


def cmd_simulate(args: argparse.Namespace) -> None:
    """Run one real storm and write its hydrograph, frames and summary."""
    site, storm = sites.get_site(args.site), sites.get_storm(args.storm)
    got = {}
    rain_of = None
    if args.rain in ("aorc", "aorc-hourly"):
        def rain_of(surf, profile):
            r, hourly_, weight, rec, timing = forcing.aorc_hyetograph(storm, args.dt, profile, surf.valid,
                                                                      args.extend_hours)
            surf.rain_weight = weight
            if args.rain == "aorc-hourly":         # every cell its own 1 km cell's hours, not the mean's timing
                surf.rain_hourly = timing
            got.update(hourly=hourly_, rec=rec)
            print("  AORC: " + json.dumps(rec))
            return r
        rain = None
    else:
        rain, hourly = forcing.observed_hyetograph(site, storm, args.dt, args.extend_hours)
    res, profile, dx = _run(site, rain, args.cell_size, args.dt, args.frame_interval,
                            use_surface=args.surface, infiltration=args.infiltration, basin=args.basin,
                            keep_depressions=args.keep_depressions, antecedent=args.antecedent, rain_of=rain_of,
                            manning=args.manning)
    if rain_of is not None:
        hourly = got["hourly"]
        rain = np.asarray(res.series["rain_mm_hr"])

    tag = _tag(storm.name, args.cell_size, args.surface, args.infiltration, args.basin, args.keep_depressions,
               args.antecedent) + {"aorc": "_aorc", "aorc-hourly": "_aorch"}.get(args.rain, "") \
        + ("_nlcdn" if args.manning == "nlcd" else "")
    t_h = np.arange(len(rain)) * args.dt / 3600.0
    columns = {"time_h": t_h, "rain_mm_hr": res.series["rain_mm_hr"],
               "flooded_ha": res.series["flooded_ha"],
               "outflow_total_cms": res.series["outflow_total_cms"]}
    if "gauge_cms" in res.series:
        columns["gauge_cms"] = res.series["gauge_cms"]
    np.savetxt(site.out_path(f"hydrograph_{tag}.csv"),
               np.column_stack(list(columns.values())), delimiter=",",
               header=",".join(columns), comments="")

    summary = _receipt(res, {
        "site": site.name, "storm": storm.name, "cell_size_m": dx, "dt_s": args.dt,
        "infiltration": args.infiltration, "basin": args.basin, "keep_depressions": args.keep_depressions,
        "manning": args.manning, "rain": args.rain,
        "total_rain_mm": float(hourly.sum()), **({"rain_source": "aorc", "aorc": got["rec"]} if got else {}),
        "peak_flooded_ha": float(res.series["flooded_ha"].max()),
        "peak_outflow_cfs": float(res.series["outflow_total_cms"].max() * CFS_PER_CMS)})
    print(json.dumps(summary, indent=1))
    _summary_path(site, f"summary_{tag}").write_text(json.dumps(summary, indent=1))
    frames.write(site.out_path(f"frames_{tag}.bin"), [f[0] for f in res.frames],
                 [t / 60.0 for t in res.frame_times_s])
    t0 = storm.start.replace(" ", "T") + ":00+00:00"
    frames.write_fields(site.out_path(f"fields_{tag}.bin"), FIELDS, res.frame_times_s, res.frames,
                        cell_m=dx, origin=site.scene_origin(profile["transform"]),
                        sidecar=_sidecar(site, t0, {"storm": storm.name, "summary": f"summary_{tag}.json"}))


def cmd_ensemble(args: argparse.Namespace) -> None:
    """Run the design-storm ensemble and invert it to a flood-probability surface."""
    site = sites.get_site(args.site)
    depths, stack = {}, []
    for t_yr in forcing.RETURN_PERIODS_YR:
        depth_mm = forcing.atlas14_depth_mm(site, t_yr, args.duration_hr)
        depths[t_yr] = depth_mm
        rain = forcing.design_hyetograph(depth_mm, args.duration_hr, args.dt)
        print(f"\n[T={t_yr:>3} yr] {depth_mm:.1f} mm over {args.duration_hr:g} h")
        res, _, dx = _run(site, rain, args.cell_size, args.dt, 1e9, with_gauge=False)
        stack.append(res.h_max)

    aep, fixed = probability.depth_stack_to_aep(np.stack(stack), args.threshold_m)
    summary = probability.summarize(
        aep, cell_area_m2=dx * dx, threshold_m=args.threshold_m, duration_hr=args.duration_hr,
        monotonicity_fixed_cells=fixed, r2=probability.loglinearity_r2(depths))
    print(json.dumps(summary.as_dict(), indent=1))
    _summary_path(site, f"aep_{args.cell_size:g}m").write_text(
        json.dumps(summary.as_dict(), indent=1))
    np.save(site.out_path(f"aep_{args.cell_size:g}m.npy"), aep)


def cmd_validate(args: argparse.Namespace) -> None:
    """Score the most recent simulated hydrograph against the gauge and write the receipt."""
    site, storm = sites.get_site(args.site), sites.get_storm(args.storm)
    tag = _tag(storm.name, args.cell_size, args.surface, args.infiltration, args.basin, args.keep_depressions,
               args.antecedent) + {"aorc": "_aorc", "aorc-hourly": "_aorch"}.get(getattr(args, "rain", "asos"), "") \
        + ("_nlcdn" if getattr(args, "manning", "scalar") == "nlcd" else "")
    path = site.out_path(f"hydrograph_{tag}.csv")
    assert path.exists(), f"{path} missing; run `simulate` with the same options first"

    data = np.genfromtxt(path, delimiter=",", names=True)
    summary = json.loads(_summary_path(site, f"summary_{tag}").read_text())
    score = validate.score(site, storm, data["time_h"], data["outflow_total_cms"])
    receipt = {
        "site": site.name, "storm": storm.name, "hydrograph": path.name,
        "cell_size_m": summary["cell_size_m"], "grid": summary["grid"],
        "total_rain_mm": summary["total_rain_mm"],
        "mass_residual_pct": summary["mass_residual_pct"],
        "gauge": site.gauge.site_no, "baseflow_cfs": site.gauge.baseflow_cfs,
        "score": score.as_dict(),
    }
    print(score.report())
    out = _summary_path(site, f"validation_{tag}")
    out.write_text(json.dumps(receipt, indent=1))
    print(f"wrote {out}")


def _sidecar(site: sites.SiteConfig, t0: str, provenance: Dict[str, object]) -> Dict[str, object]:
    """The frame sidecar for one run."""
    return frames.sidecar(
        t0=t0, timezone=site.timezone, site=site.name, epsg=site.epsg,
        anchor_utm=list(site.anchor_m or (0.0, 0.0, 0.0)),
        fields={"depth": {"domain": [0.0, 0.5], "lut": "turbo"},
                "u": {"domain": [-1.0, 1.0], "lut": "turbo"},
                "v": {"domain": [-1.0, 1.0], "lut": "turbo"}},
        provenance=provenance)


def _parcel_inputs(site: sites.SiteConfig, args: argparse.Namespace) -> Tuple[np.ndarray, float, object, Optional[Tuple[np.ndarray, dict]]]:
    """The DTM for a parcel run, its cell size and transform, and a class raster with its table
    when the run is not parameterised from a bundle."""
    from affine import Affine

    if args.synthetic:
        z, classes = parcel.synthetic(args.dx, radius_m=site.radius_km * 1000.0)
        half = z.shape[0] * args.dx / 2.0
        ax, ay, _ = site.anchor_m or (0.0, 0.0, 0.0)
        grid = Affine(args.dx, 0.0, ax - half, 0.0, -args.dx, ay + half)
        return z, args.dx, grid, (classes, parcel.SYNTHETIC_CLASSES)
    dtm = Path(args.dtm) if args.dtm else site.path("bundle", f"dtm_{args.dx:g}m.tif")
    z, dx, grid = parcel.read_dem(dtm)
    if args.bundle:
        return z, dx, grid, None
    cls = Path(args.classes) if args.classes else site.path("bundle", f"classes_{args.dx:g}m.tif")
    _, classes, _ = parcel.read_rasters(dtm, cls)
    table = parcel.load_class_table(Path(args.table))[0] if args.table else parcel.SYNTHETIC_CLASSES
    return z, dx, grid, (classes, table)


def _condition(z: np.ndarray, dx: float, args: argparse.Namespace,
               source: Dict[str, object]) -> np.ndarray:
    """Optionally fill the DTM's closed depressions, recording the volume raised."""
    if not args.fill_sinks:
        return z
    filled, raised, cells = parcel.fill_sinks(z)
    source.update(sinks_filled_cells=cells, sinks_filled_area_m2=cells * dx * dx,
                  sinks_filled_volume_m3=raised * dx * dx)
    return filled


def _parcel_surface(site: sites.SiteConfig, args: argparse.Namespace) -> Tuple[Surface, float, object, Dict[str, object]]:
    """Terrain and per-cell hydrology for a parcel run, from a bundle or a class table."""
    z, dx, grid, classes = _parcel_inputs(site, args)
    if classes is None:
        bundle = Path(args.bundle)
        display, buffer_m = parcel.radii(bundle)
        buffer_m = buffer_m if args.buffer is None else args.buffer
        box = parcel.box(bundle)
        if box:
            # A site that follows its ordered polygon: its fetch box, not the disc that circumscribes it.
            z, grid = parcel.window_box(z, grid, box["fetch"], dx)
            beyond = parcel.outside_box(z.shape, grid, box["fetch"])
        else:
            z, grid = parcel.window(z, grid, site.anchor_m[:2], display + buffer_m, dx)
            beyond = parcel.distance(z.shape, grid, site.anchor_m[:2]) > display + buffer_m
        z = np.where(beyond, np.nan, z) if getattr(args, "rim_inflow", False) else z
        source: Dict[str, object] = {"parameters": "bundle", "bundle": str(bundle), "display_radius_m": display,
                                     "buffer_m": buffer_m, "fetch_radius_m": display + buffer_m,
                                     "grid_cells": list(z.shape), "classes_cut_beyond_fetch_radius": int(beyond.sum())}
        if box:
            source["box_scene_m"] = {k: list(v) for k, v in box.items()}
        if args.drop_buildings:
            codes, transform = params.read_classes(
                bundle / "semantics" / params.class_raster_name(dx))
            roofs = params.building_mask(
                params.resample_nearest(codes, transform, z.shape, grid),
                bundle / "semantics" / "parameters.json") & ~beyond
            z = np.where(roofs, np.nan, z)
            source.update(buildings_dropped_cells=int(roofs.sum()),
                          buildings_dropped_area_m2=float(roofs.sum()) * dx * dx)
        z = _condition(z, dx, args, source)
        surf, receipt = params.build_surface(z, grid, bundle, dx, deficit_mm=args.deficit_mm, beyond=beyond,
                                             cells=getattr(args, "cells", None))
        return surf, dx, grid, {**source, **receipt}
    codes, table = classes
    source = {"parameters": "synthetic" if args.synthetic else "class table"}
    z = _condition(z, dx, args, source)
    surf = parcel.build_surface(z, codes, table, table[min(table)], deficit_mm=args.deficit_mm)
    return surf, dx, grid, source


def _parcel_config(dx: float, args: argparse.Namespace) -> SolverConfig:
    return SolverConfig(dx=dx, dt_s=args.dt, cfl_alpha=args.cfl_alpha,
                        frame_interval_min=args.frame_interval, cfl_depth=args.cfl_depth,
                        dtype=args.dtype, device=args.device, compile=args.compile,
                        soil_dt_s=soil_dt(dx, args.soil_dt))


SHEET_FLOW_M_S = 0.4
"""The sheet flow whose crossing of a cell is the soil's step (`soil_dt`): Emmett's (1970) overland flows run 0.01 to
0.3 m/s, so at 0.4 m/s run-on onto permeable ground is taken by the soil within the cell it reaches."""


def soil_dt(dx: float, given: Optional[str]) -> float:
    """The soil's step [s] (`SolverConfig.soil_dt_s`): "cell", the time sheet flow at SHEET_FLOW_M_S takes to cross
    one cell (0.5 s at 0.2 m); a number, that many seconds; None or 0, every sub-step."""
    if not given:
        return 0.0
    return dx / SHEET_FLOW_M_S if given == "cell" else float(given)


def _edge_inflow(z: np.ndarray, grid: object, rain: Sequence[float], args: argparse.Namespace,
                 disc: np.ndarray) -> Optional[inflow_mod.Inflow]:
    """Watershed inflow from a coarse DEM covering the parcel, or None.

    With `--rim-inflow` the domain is the fetch disc and the inflow enters across its rim
    (`inflow.rim_inflow`); otherwise across the grid's four edges.

    The window is the parcel's own footprint in coarse cells, taken from the two affine
    transforms rather than assumed centred.
    """
    if not args.coarse_dem:
        return None
    zc, dxc, tc = parcel.read_dem(Path(args.coarse_dem))
    if args.rim_inflow:
        return inflow_mod.rim_inflow(np.nan_to_num(zc, nan=float(np.nanmax(zc))), tc, disc, np.isfinite(z), grid,
                                     rain, args.dt, runoff=args.runoff, max_depth_m=args.inflow_max_depth)
    c0 = int(round((grid.c - tc.c) / dxc))
    r0 = int(round((tc.f - grid.f) / dxc))
    r1 = r0 + int(round(z.shape[0] * abs(grid.e) / dxc))
    c1 = c0 + int(round(z.shape[1] * grid.a / dxc))
    assert 0 <= r0 < r1 <= zc.shape[0] and 0 <= c0 < c1 <= zc.shape[1], (
        f"parcel window ({r0}:{r1}, {c0}:{c1}) is not inside the {zc.shape} coarse DEM")
    flow = inflow_mod.watershed_inflow(np.nan_to_num(zc, nan=float(np.nanmax(zc))), dxc,
                                       (r0, r1, c0, c1), z.shape, rain, args.dt,
                                       runoff=args.runoff)
    if args.inflow_max_depth:
        for edge in inflow_mod.EDGES:
            setattr(flow, edge, inflow_mod.cap_unit_discharge(getattr(flow, edge),
                                                              args.inflow_max_depth))
    return flow


def _units(site: sites.SiteConfig, args: argparse.Namespace, source: Dict[str, object]):
    """The run's geographic units (`units.split`) from --units AxB and --overlap-m, or None."""
    if not getattr(args, "units", None):
        return None
    a, b = (int(v) for v in args.units.lower().split("x"))
    if source.get("box_scene_m"):          # the fetch box's lines, the same at every pass's cell
        fetch = tuple(source["box_scene_m"]["fetch"])
        return units_mod.split_box(fetch, a, b, args.overlap_m), fetch
    reach = float(source.get("fetch_radius_m") or site.radius_km * 1000.0)
    return units_mod.split(tuple(site.anchor_m[:2]), reach, a, b, args.overlap_m), reach


def _core(res: Result, flow: Optional[inflow_mod.Inflow], cut: Tuple[int, int, int, int], n: int, dt_s: float,
          dx: float) -> Dict[str, object]:
    """What one unit adds to the whole run's receipt from its core alone: its peak depth, and the fetch
    disc's rim inflow into its core at each forcing interval [m^3/s], as the whole run loads it (`join`)."""
    a0, a1, b0, b1 = cut
    h = res.h_max[a0:a1, b0:b1]
    out: Dict[str, object] = {"core_peak_depth_m": float(np.nanmax(h)) if np.isfinite(h).any() else 0.0}
    if flow is not None and flow.rim_index is not None:
        rr, cc = np.divmod(np.asarray(flow.rim_index), res.h_max.shape[1])
        inside = (rr >= a0) & (rr < a1) & (cc >= b0) & (cc < b1)
        out["core_rim_inflow_cms"] = [float(flow.at(i * dt_s)["rim"][inside].sum()) * dx * dx for i in range(n)]
    return out


def cmd_parcel(args: argparse.Namespace) -> None:
    """Run a sub-metre parcel from a DTM and a class raster, with estimated edge inflow.

    With --units AxB and --record-seams (the coarse pass, at a coarse --dx) the run also records the
    discharge across every unit window's edges into seams_<tag>.npz; with --units AxB --unit i --seams
    it runs only unit i over its window, fed across its edges by the coarse pass and across the fetch
    disc's own rim as the whole run is, into fields_<tag>_unit<i>of<A>x<B>.bin; `join` then joins the
    units' cores into fields_<tag>.bin (WS18 M3).
    """
    site = sites.get_site(args.site)
    surf, dx, grid, source = _parcel_surface(site, args)
    z = surf.z
    series = None
    if args.rain_series:                        # a measured series (the tower's P): its own start and duration
        t0, start_s, depth_mm = rainseries.read(Path(args.rain_series))
        rain = list(rainseries.rates(start_s, depth_mm, args.dt))
        series = {"rain_series": Path(args.rain_series).name, "rain_series_mm": float(depth_mm.sum()),
                  "rain_series_start": t0}
        args.t0 = t0
    else:
        rain = [args.rain_mm_hr / 3.6e6] * int(round(args.duration_h * 3600.0 / args.dt))
    n = len(rain)
    box = source.get("box_scene_m")
    inflow = _edge_inflow(z, grid, rain, args, np.isfinite(z) | (
        ~parcel.outside_box(z.shape, grid, box["fetch"]) if box else
        parcel.distance(z.shape, grid, site.anchor_m[:2]) <= source.get("fetch_radius_m", np.inf)))
    tag = f"{args.dx:g}m_{args.cfl_depth}"
    split = _units(site, args, source)
    record, core_cut = None, None
    if split and args.unit is not None:
        from affine import Affine

        spec = split[0][args.unit]
        r0, r1, c0, c1 = units_mod.cells(grid, z.shape, spec["window"])
        k0, k1, j0, j1 = units_mod.cells(grid, z.shape, spec["core"])
        core_cut = (k0 - r0, k1 - r0, j0 - c0, j1 - c0)
        unit_grid = Affine(grid.a, grid.b, grid.c + c0 * grid.a, grid.d, grid.e, grid.f + r0 * grid.e)
        edges = units_mod.unit_inflow(units_mod.Seams.load(args.seams), spec["window"], unit_grid, (r1 - r0, c1 - c0))
        rim_index, rim = units_mod.crop_rim(inflow, z.shape, (r0, r1, c0, c1))
        inflow = units_mod.combine(edges, rim_index, rim, inflow.times_s if rim is not None else None)
        surf, grid = units_mod.crop_surface(surf, (r0, r1, c0, c1)), unit_grid
        z = surf.z
        source = {**source, "unit": f"{args.unit}/{args.units}", "unit_core_m": list(spec["core"]),
                  "unit_window_m": list(spec["window"]), "unit_cells": [r0, r1, c0, c1], "overlap_m": args.overlap_m}
        tag = f"{tag}_unit{args.unit}of{args.units}"
    elif split and args.record_seams:
        xs, ys = (units_mod.seam_lines_box(split[0], split[1]) if isinstance(split[1], tuple)
                  else units_mod.seam_lines(split[0], tuple(site.anchor_m[:2]), split[1]))
        record = units_mod.SeamRecorder(grid, z.shape, xs, ys)
    origin = site.scene_origin(grid)
    provenance = {**source, "dtm": args.dtm or "synthetic", "coarse_dem": args.coarse_dem,
                  "rain_mm_hr": args.rain_mm_hr, "duration_h": args.duration_h,
                  "inflow_max_depth_m": args.inflow_max_depth, "runoff": args.runoff, **(series or {})}
    with frames.Writer(site.out_path(f"fields_{tag}.bin"), z.shape, FIELDS, dx, origin,
                       _sidecar(site, args.t0, provenance)) as sink:
        if record is not None:
            record.chain = sink.append
        res = run_solver(surf, rain, _parcel_config(dx, args), inflow=inflow,
                         sink=record if record is not None else sink.append)
    if record is not None:
        record.seams().save(site.out_path(f"seams_{tag}.npz"))
    receipt = _receipt(res, {
        "site": site.name, "cell_size_m": dx, "cells": int(np.isfinite(z).sum()),
        "rain_mm_hr": args.rain_mm_hr, "duration_h": args.duration_h, "dt_s": args.dt,
        "cfl_depth": args.cfl_depth, "dtype": args.dtype, "soil_dt_s": soil_dt(dx, args.soil_dt), "inflow": inflow is not None,
        "inflow_peak_cms": float(res.series["inflow_total_cms"].max()),
        "frames": sink.n, "fields": [list(f) for f in FIELDS], "origin_scene_m": list(origin),
        **source, **(series or {}), **(_core(res, inflow, core_cut, n, args.dt, dx) if core_cut else {})})
    print(json.dumps(receipt, indent=1))
    _summary_path(site, f"parcel_{tag}").write_text(json.dumps(receipt, indent=1))


BANDS_M = (0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0)
"""Lower edges of the distance-from-edge bands of a buffer comparison [m]."""


def _bands(delta: np.ndarray, base: np.ndarray, dist: np.ndarray, limit: float) -> List[Dict[str, object]]:
    """Change against distance inside the disc edge: samples, mean and 95th percentile |change|."""
    out = []
    for lo, up in zip(BANDS_M, list(BANDS_M[1:]) + [limit]):
        m = (dist >= lo) & (dist < up)
        a = np.abs(delta[m])
        out.append({"from_edge_m": [lo, up], "samples": int(m.sum()), "mean_abs_change": float(a.mean()),
                    "p95_abs_change": float(np.percentile(a, 95)), "mean_change": float(delta[m].mean()),
                    "mean_before": float(base[m].mean())})
    return out


def _body(res: Result, grid: object, centre: Tuple[float, float], reach_m: float, deep_m: float) -> Dict[str, float]:
    """Water deeper than `deep_m` within `reach_m` of `centre` at the end of a run and at its peak."""
    near = parcel.distance(res.h_final.shape, grid, centre) <= reach_m
    cell = abs(grid.a * grid.e)
    end, peak = near & (res.h_final > deep_m), near & (res.h_max > deep_m)
    return {"area_m2_end": float(end.sum() * cell), "volume_m3_end": float(res.h_final[end].sum() * cell),
            "max_depth_m_end": float(res.h_final[near].max()), "area_m2_peak": float(peak.sum() * cell),
            "max_depth_m_peak": float(res.h_max[near].max())}


def cmd_buffer(args: argparse.Namespace) -> None:
    """The same parcel under the same rain solved without and with the bundle's buffer.

    Without it the domain is the square a bundle ending at the display disc is gridded on and the
    classes stop at the disc, as the unbuffered product is; with it both reach the fetch disc.
    Everything else is shared. Change is reported against distance inside the display disc edge,
    over the cells both arms model, and for the water body the unbuffered run held at `--probe`.
    """
    site = sites.get_site(args.site)
    rain = [args.rain_mm_hr / 3.6e6] * int(round(args.duration_h * 3600.0 / args.dt))
    arms, report = {}, {"site": site.name, "rain_mm_hr": args.rain_mm_hr, "duration_h": args.duration_h,
                        "edge_inflow": "none in either arm", "fill_sinks": args.fill_sinks,
                        "drop_buildings": args.drop_buildings}
    for name, buf in (("before", 0.0), ("after", None)):
        args.buffer = buf
        surf, dx, grid, source = _parcel_surface(site, args)
        res = run_solver(surf, rain, _parcel_config(dx, args))
        arms[name] = (surf, res, grid)
        report[name] = _receipt(res, {**source, "flow_cells": int(np.isfinite(surf.z).sum()), "cell_size_m": dx})
        print(f"  {name}: buffer {source['buffer_m']:g} m, grid {surf.z.shape}, {res.wall_s:.0f} s", flush=True)
    (sb, rb, gb), (sa, ra, ga) = arms["before"], arms["after"]
    r0, c0 = int(round((ga.f - gb.f) / dx)), int(round((gb.c - ga.c) / dx))
    win = (slice(r0, r0 + sb.z.shape[0]), slice(c0, c0 + sb.z.shape[1]))
    r = report["after"]["display_radius_m"]
    dist = r - parcel.distance(sb.z.shape, gb, site.anchor_m[:2])
    zb, za = np.asarray(sb.z), np.asarray(sa.z)[win]
    inner = dist >= 0
    both = inner & np.isfinite(zb) & np.isfinite(za)
    report["terrain_differing_inside_display_disc"] = {
        "valid_in_one_arm_only": int((np.isfinite(zb) != np.isfinite(za))[inner].sum()),
        "elevation_differs": int((zb != za)[both].sum()), "cells_compared": int(both.sum())}
    hb, ha = rb.h_max, ra.h_max[win]
    report["bands_peak_depth_m"] = _bands((ha - hb)[both], hb[both], dist[both], r)
    fb, fa = rb.h_final, ra.h_final[win]
    report["bands_final_depth_m"] = _bands((fa - fb)[both], fb[both], dist[both], r)
    report["water_in_display_disc_at_end_m3"] = {"before": float(fb[both].sum() * dx * dx),
                                                 "after": float(fa[both].sum() * dx * dx)}
    probe = (site.anchor_m[0] + args.probe[0], site.anchor_m[1] + args.probe[1])
    report["probe"] = {"scene_m": list(args.probe), "reach_m": 20.0, "deeper_than_m": 0.3,
                       "before": _body(rb, gb, probe, 20.0, 0.3), "after": _body(ra, ga, probe, 20.0, 0.3)}
    out = site.out_path(f"buffer_{dx:g}m.json")
    out.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k not in ("before", "after")}, indent=1, default=float))


def cmd_profile(args: argparse.Namespace) -> None:
    """Time every function of a parcel run started wet, and record the ten costliest."""
    site = sites.get_site(args.site)
    surf, dx, _, source = _parcel_surface(site, args)
    surf.initial_h = np.where(np.isfinite(surf.z), args.initial_depth, 0.0).astype(np.float32)
    rain = [args.rain_mm_hr / 3.6e6] * args.intervals
    cfg = _parcel_config(dx, args)

    eager = SolverConfig(**{**vars(cfg), "compile": False, "frame_interval_min": 1e9})
    prof = cProfile.Profile()
    prof.enable()
    res = run_solver(surf, rain, eager, verbose=False)
    prof.disable()
    stats = pstats.Stats(prof).sort_stats("tottime")
    top = []
    for (file, line, name), (cc, nc, tt, ct, _) in sorted(
            stats.stats.items(), key=lambda kv: kv[1][2], reverse=True)[:10]:
        top.append({"function": f"{Path(file).name}:{line}({name})", "calls": nc,
                    "tottime_s": tt, "cumtime_s": ct, "per_substep_ms": tt / res.n_substeps * 1e3})

    timed = run_solver(surf, rain, SolverConfig(**{**vars(cfg), "frame_interval_min": 1e9}),
                       verbose=False)
    receipt = {
        "site": site.name, "cell_size_m": dx, "cells": int(np.isfinite(surf.z).sum()),
        "grid": list(surf.z.shape), "dtype": args.dtype, "device": timed.device,
        "compile": timed.device.startswith("cuda") if cfg.compile is None else cfg.compile,
        "cfl_depth": args.cfl_depth, "intervals": args.intervals, "dt_s": args.dt,
        "initial_depth_m": args.initial_depth, "rain_mm_hr": args.rain_mm_hr, **source,
        "eager": {"n_substeps": res.n_substeps, "wall_s": res.wall_s,
                  "substeps_per_s": res.n_substeps / res.wall_s},
        "timed": {"n_substeps": timed.n_substeps, "wall_s": timed.wall_s,
                  "substeps_per_s": timed.n_substeps / timed.wall_s,
                  "cell_updates_per_s": timed.n_substeps / timed.wall_s * surf.z.size},
        "mass_residual": timed.mass.residual, "top10_by_tottime": top,
    }
    print(json.dumps(receipt, indent=1))
    out = Path(args.out) if args.out else DOCS / f"profile_{args.dx:g}m_{timed.device.split(':')[0]}.json"
    out.write_text(json.dumps(receipt, indent=1))
    print(f"wrote {out}")


def _unit_options(p: argparse.ArgumentParser) -> None:
    """The geographic units of a parcel run (WS18 M3)."""
    p.add_argument("--units", default=None, help="AxB: the fetch disc's square in A x B geographic units")
    p.add_argument("--overlap-m", type=float, default=100.0, help="each unit's window beyond its core [m]")


def cmd_join(args: argparse.Namespace) -> None:
    """Join a parcel run's units into its whole fields file and receipt, each cell from the one unit whose
    core holds it, frame by frame (the k-th frame of every unit, at the latest of their times).

    The receipt carries every unit's own mass balance; `mass_residual_pct` is the worst unit's, each
    unit's closing over its own window with its edge inflow and outflow.
    """
    site = sites.get_site(args.site)
    surf, dx, grid, source = _parcel_surface(site, args)
    shape = surf.z.shape
    specs, _ = _units(site, args, source)
    tag = f"{args.dx:g}m_{args.cfl_depth}"
    parts = []
    for i, spec in enumerate(specs):
        path = site.out_path(f"fields_{tag}_unit{i}of{args.units}.bin")
        header, times, data = frames.read_fields(path)
        win = units_mod.cells(grid, shape, spec["window"])
        core = units_mod.cells(grid, shape, spec["core"])
        receipt = json.loads(_summary_path(site, f"parcel_{tag}_unit{i}of{args.units}").read_text())
        parts.append((win, core, times, data, receipt))
    n = min(len(p[2]) for p in parts)
    sidecar = json.loads(Path(f"{site.out_path(f'fields_{tag}_unit0of{args.units}.bin')}.json").read_text())
    sidecar["provenance"] = {k: v for k, v in sidecar["provenance"].items() if not k.startswith("unit")}
    sidecar["provenance"]["units"] = {"split": args.units, "overlap_m": args.overlap_m, "n": len(specs)}
    seen = np.zeros(shape, np.int32)
    with frames.Writer(site.out_path(f"fields_{tag}.bin"), shape, FIELDS, dx, site.scene_origin(grid), sidecar) as w:
        for k in range(n):
            out = np.full((len(FIELDS),) + shape, np.nan, np.float32)
            for (w0, _, v0, _), (r0, r1, c0, c1), _, data, _ in parts:
                out[:, r0:r1, c0:c1] = data[k, :, 0, r0 - w0:r1 - w0, c0 - v0:c1 - v0]
                if k == 0:
                    seen[r0:r1, c0:c1] += 1
            w.append(max(float(p[2][k]) for p in parts), out)
    assert (seen == 1).all(), "the unit cores do not cover the grid once"
    worst = max(parts, key=lambda p: abs(float(p[4].get("mass_residual_pct") or 0.0)))[4]
    got = [p[4] for p in parts]
    rims = [r["core_rim_inflow_cms"] for r in got if r.get("core_rim_inflow_cms")]
    # The whole run's receipt, keys and all (the viewer's build reads the storm from it): the storm and
    # solver settings every unit shares, the whole grid's own numbers, and the work the units did between
    # them. A whole-domain mass budget has no counterpart (the windows overlap): the gate reads the worst
    # unit's residual, each unit closing over its own window, and every unit's budget is kept.
    shared = {k: v for k, v in got[0].items() if not k.startswith(("unit", "core_")) and k != "overlap_m"}
    receipt = {**shared, **{k: v for k, v in source.items() if not k.startswith("unit")},
               "site": site.name, "cell_size_m": dx, "cells": int(np.isfinite(surf.z).sum()), "grid": list(shape),
               "frames": n, "origin_scene_m": list(site.scene_origin(grid)),
               "peak_depth_m": max(float(r.get("core_peak_depth_m") or 0.0) for r in got),
               "inflow_peak_cms": float(np.sum(rims, axis=0).max()) if rims else 0.0,
               "n_substeps": sum(int(r.get("n_substeps") or 0) for r in got),
               "substep_cap_hits": sum(int(r.get("substep_cap_hits") or 0) for r in got),
               "wall_s": sum(float(r.get("wall_s") or 0.0) for r in got), "substeps_per_s": None,
               "mass_m3": None, "mass_residual": worst.get("mass_residual"),
               "mass_residual_pct": worst.get("mass_residual_pct"),
               "units": args.units, "overlap_m": args.overlap_m,
               "frame_time_spread_s": float(max(max(p[2][k] for p in parts) - min(p[2][k] for p in parts)
                                                for k in range(n))),
               "unit_receipts": got}
    print(json.dumps({k: v for k, v in receipt.items() if k != "unit_receipts"}, indent=1))
    _summary_path(site, f"parcel_{tag}").write_text(json.dumps(receipt, indent=1))


def _parcel_options(p: argparse.ArgumentParser) -> None:
    """Options shared by the parcel and profile stages."""
    p.add_argument("--dx", type=float, default=0.2)
    p.add_argument("--dt", type=float, default=60.0)
    p.add_argument("--rain-mm-hr", type=float, default=50.0)
    p.add_argument("--cfl-alpha", type=float, default=0.15)
    p.add_argument("--cfl-depth", choices=("cell", "face"), default="face")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--device", default=None)
    p.add_argument("--compile", type=lambda s: s.lower() == "true", default=None)
    p.add_argument("--soil-dt", default=None,
                   help="the soil's own step: 'cell' (a cell's crossing by sheet flow, SHEET_FLOW_M_S) or seconds; "
                        "none updates it every sub-step")
    p.add_argument("--frame-interval", type=float, default=5.0)
    p.add_argument("--bundle", default=None, help="bundle directory holding semantics/")
    p.add_argument("--dtm", default=None)
    p.add_argument("--classes", default=None)
    p.add_argument("--table", default=None, help="ontology JSON with per-class hydrology")
    p.add_argument("--deficit-mm", type=float, default=None)
    p.add_argument("--cells", default=None,
                   help="per-cell hydrology GeoTIFF (params.CELL_BANDS) overriding the class table, with --bundle")
    p.add_argument("--synthetic", action="store_true", help="use the synthetic terrain")
    p.add_argument("--fill-sinks", action="store_true",
                   help="raise closed depressions to their spill elevation, volume reported")
    p.add_argument("--drop-buildings", action="store_true",
                   help="mark bundle building classes nodata so roofs do not pond")
    p.add_argument("--buffer", type=float, default=None,
                   help="ring beyond the display disc the terrain is kept over [m]; default the "
                        "bundle's buffer_m, 0 runs over the display disc alone")


def main(argv: Optional[List[str]] = None) -> None:
    """Parse arguments and dispatch to one stage."""
    parser = argparse.ArgumentParser(prog="hydro", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="stage", required=True)

    def add(name: str, fn: Callable, help_text: str) -> argparse.ArgumentParser:
        """Register a subcommand; every stage takes --site."""
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--site", required=True, choices=sorted(sites.SITES))
        p.set_defaults(func=fn)
        return p

    p = add("fetch", cmd_fetch, "download every public dataset for a site")
    p.add_argument("--storm", action="append", choices=sorted(sites.STORMS))

    p = add("terrain", cmd_terrain, "condition the DEM and delineate")
    p.add_argument("--acc-area-m2", type=float, default=terrain.DEFAULT_ACC_AREA_M2)

    add("segment", cmd_segment, "SAM3 surface classes from aerial imagery (Python 3.11)")

    p = add("simulate", cmd_simulate, "run one observed storm")
    p.add_argument("--storm", required=True, choices=sorted(sites.STORMS))
    p.add_argument("--cell-size", type=float, default=25.0)
    p.add_argument("--dt", type=float, default=20.0)
    p.add_argument("--frame-interval", type=float, default=60.0)
    p.add_argument("--extend-hours", type=float, default=0.0,
                   help="zero-rain hours appended so the drainage tail is not truncated")
    p.add_argument("--surface", action="store_true",
                   help="use the segmentation-derived Manning field instead of the scalar")
    p.add_argument("--basin", action="store_true", help="the domain is the gauge's NLDI basin")
    p.add_argument("--keep-depressions", action="store_true", help="the DEM before depression breaching")
    p.add_argument("--rain", choices=("asos", "aorc", "aorc-hourly"), default="asos",
                   help="the storm's rain: the site's ASOS gauge; AORC's 1 km grid, each cell's total on the "
                        "mean's hours; or each cell's own hours (aorc-hourly)")
    p.add_argument("--antecedent", default=None,
                   help="raster of theta_i and deficit_mm from a continuous simulation at the storm's start (gar)")
    p.add_argument("--manning", choices=("scalar", "nlcd"), default="scalar",
                   help="roughness: the scalar, or each NLCD class's published flood-plain n (Chow 1959)")
    p.add_argument("--infiltration", choices=("horton", "gar"), default="horton",
                   help="gar: Green-Ampt with redistribution from the survey's hydraulics")

    p = add("ensemble", cmd_ensemble, "design-storm ensemble to a probability surface")
    p.add_argument("--cell-size", type=float, default=25.0)
    p.add_argument("--dt", type=float, default=20.0)
    p.add_argument("--duration-hr", type=float, default=24.0)
    p.add_argument("--threshold-m", type=float, default=0.15)

    p = add("validate", cmd_validate, "score a simulated hydrograph against the gauge")
    p.add_argument("--storm", required=True, choices=sorted(sites.STORMS))
    p.add_argument("--cell-size", type=float, default=25.0)
    p.add_argument("--surface", action="store_true",
                   help="score the segmentation-derived arm rather than the scalar baseline")
    p.add_argument("--basin", action="store_true", help="the domain is the gauge's NLDI basin")
    p.add_argument("--keep-depressions", action="store_true", help="the DEM before depression breaching")
    p.add_argument("--rain", choices=("asos", "aorc", "aorc-hourly"), default="asos",
                   help="the storm's rain: the site's ASOS gauge; AORC's 1 km grid, each cell's total on the "
                        "mean's hours; or each cell's own hours (aorc-hourly)")
    p.add_argument("--antecedent", default=None,
                   help="raster of theta_i and deficit_mm from a continuous simulation at the storm's start (gar)")
    p.add_argument("--manning", choices=("scalar", "nlcd"), default="scalar",
                   help="roughness: the scalar, or each NLCD class's published flood-plain n (Chow 1959)")
    p.add_argument("--infiltration", choices=("horton", "gar"), default="horton")

    p = add("parcel", cmd_parcel, "run a sub-metre parcel from a DTM and a class raster")
    _parcel_options(p)
    p.add_argument("--coarse-dem", default=None, help="metric DEM around the parcel for edge inflow")
    p.add_argument("--rim-inflow", action="store_true",
                   help="clip the domain to the fetch disc and deliver the inflow across its rim")
    p.add_argument("--runoff", type=float, default=1.0)
    p.add_argument("--inflow-max-depth", type=float, default=None,
                   help="cap edge inflow at the discharge this depth conveys, spreading the rest")
    p.add_argument("--duration-h", type=float, default=1.0)
    p.add_argument("--rain-series", default=None,
                   help="CSV of time,rain_mm per interval (rainseries.py), in place of --rain-mm-hr and --duration-h")
    p.add_argument("--t0", default="2026-01-15T00:00:00-08:00", help="ISO 8601 start of the storm")
    _unit_options(p)
    p.add_argument("--record-seams", action="store_true",
                   help="the coarse pass of a unit run: record the discharge across every unit window's edges")
    p.add_argument("--unit", type=int, default=None, help="run only unit i of --units, fed across its edges by --seams")
    p.add_argument("--seams", default=None, help="the coarse pass's seams_<tag>.npz, for --unit")

    p = add("join", cmd_join, "join the units of a parcel run into its whole fields and receipt (WS18 M3)")
    _parcel_options(p)
    _unit_options(p)

    p = add("buffer", cmd_buffer, "the parcel under rain without and with the bundle's buffer, compared")
    _parcel_options(p)
    p.add_argument("--duration-h", type=float, default=1.0)
    p.add_argument("--probe", type=float, nargs=2, default=(46.0, -104.0),
                   help="scene (x, y) of a water body to report in both arms [m]")

    p = add("profile", cmd_profile, "time every function of a parcel run started wet")
    _parcel_options(p)
    p.add_argument("--intervals", type=int, default=5)
    p.add_argument("--initial-depth", type=float, default=0.02, help="standing water [m] at start")
    p.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    import solver

    solver.PROGRESS_LINES = True       # every run the command line makes says how far it is (`PROGRESS hydro k/n`)
    args.func(args)


if __name__ == "__main__":
    main()
