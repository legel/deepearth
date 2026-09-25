"""Command line for the hydro twin: one stage per subcommand.

    python3 cli.py fetch    --site site3 --storm ian
    python3 cli.py terrain  --site site3
    python3 cli.py simulate --site site3 --storm ian --cell-size 25
    python3 cli.py ensemble --site site3 --cell-size 25
    python3 cli.py validate --site site3 --storm ian

Run it from this directory. The modules are flat and import each other by name, so there is no
package to install and no path manipulation anywhere.
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import domain
import forcing
import frames
import probability
import sites
import surface
import terrain
import validate
from solver import FIELDS, Probes, Result, SolverConfig, simulate as run_solver

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
    p.add_argument("--infiltration", choices=("horton", "gar"), default="horton",
                   help="gar: Green-Ampt with redistribution from the survey's hydraulics")
    p.add_argument("--basin", action="store_true", help="the domain is the gauge's NLDI basin")
    p.add_argument("--keep-depressions", action="store_true", help="the DEM before depression breaching")
    p.add_argument("--rain", choices=("asos", "aorc", "aorc-hourly"), default="asos",
                   help="the storm's rain: the site's ASOS gauge; AORC's 1 km grid, each cell's total on the "
                        "mean's hours; or each cell's own hours (aorc-hourly)")
    p.add_argument("--antecedent", default=None,
                   help="raster of theta_i and deficit_mm from a continuous simulation at the storm's start (gar)")
    p.add_argument("--manning", choices=("scalar", "nlcd"), default="scalar",
                   help="roughness: the scalar, or each NLCD class's published flood-plain n (Chow 1959)")

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
    p.add_argument("--infiltration", choices=("horton", "gar"), default="horton")
    p.add_argument("--basin", action="store_true", help="the domain is the gauge's NLDI basin")
    p.add_argument("--keep-depressions", action="store_true", help="the DEM before depression breaching")
    p.add_argument("--rain", choices=("asos", "aorc", "aorc-hourly"), default="asos",
                   help="the storm's rain: the site's ASOS gauge; AORC's 1 km grid, each cell's total on the "
                        "mean's hours; or each cell's own hours (aorc-hourly)")
    p.add_argument("--antecedent", default=None,
                   help="raster of theta_i and deficit_mm from a continuous simulation at the storm's start (gar)")
    p.add_argument("--manning", choices=("scalar", "nlcd"), default="scalar",
                   help="roughness: the scalar, or each NLCD class's published flood-plain n (Chow 1959)")

    args = parser.parse_args(argv)
    import solver

    solver.PROGRESS_LINES = True       # every run the command line makes says how far it is (`PROGRESS hydro k/n`)
    args.func(args)


if __name__ == "__main__":
    main()
