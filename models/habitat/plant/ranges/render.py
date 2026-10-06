"""Render one fitted species: 240 m maps on every US grid its calibration area reaches, the range card, Daru-style
metrics, and — where an independent reference exists — benchmarks against Daru's published map, WCVP native areas
and field plots (BLM AIM, FIA). Used by the per-species fitting worker (scripts/national_run.py)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from shapely.ops import unary_union

from . import benchmark, codec, evaluate, project, validate
from .predictors import ConusStack, region_stacks


class RenderContext:
    """Everything rendering needs, loaded once per process."""

    def __init__(self, root: Path):
        import geopandas as gpd
        self.root = Path(root)
        self.regions = region_stacks(self.root / "work")                 # conus, and alaska / hawaii where built
        self.stacks = {"worldclim": self.regions["conus"]}
        d = self.root / "daru_ref_all/rasters"
        self.daru = {p.stem for p in d.glob("*.tif")} if d.exists() else set()
        self.plots = validate.PlotTruth(self.root / "raw/validation/blm_aim_species.csv", self.root / "raw/validation/fia")
        self.wgsrpd = gpd.read_file(self.root / "raw/geo/wgsrpd_level3.geojson")
        self.ecoregions = gpd.read_file(self.root / "raw/geo/ecoregions2017/Ecoregions2017.shp")[["ECO_ID", "geometry"]]

    def stack(self, kind: str):
        if kind not in self.stacks:
            self.stacks[kind] = ConusStack(self.root / "work/conus240", self.root / "work/fine/conus240_fine.f32")
        return self.stacks[kind]


def render_species(wd: Path, label: str, ctx: RenderContext) -> dict:
    """Render the fitted species in ``wd``; writes render.json and <label>.rangecard and returns the render record."""
    s = json.loads((wd / "summary.json").read_text())
    reps = project.load_replicates(wd / "maxent/final", label, s["predictors"])
    t = time.time()
    name = s["species"]
    bench = name in ctx.daru
    kind = s.get("predictor_set", "worldclim")
    stack = ctx.stack(kind)
    outs, stats = project.compute_maps(reps, s["predictors"], s["calibration_ecoregions"], stack)
    region_maps = {"conus": outs}
    if kind == "worldclim":                      # also every other US grid the calibration area reaches
        for rg, st in ctx.regions.items():
            if rg != "conus" and st.window(s["calibration_ecoregions"]) is not None:
                region_maps[rg], rs = project.compute_maps(reps, s["predictors"], s["calibration_ecoregions"], st)
                stats[f"cells_vote_{rg}"] = rs["cells_vote"]
    if bench:
        project.write_maps(outs, stack, wd / "maps", label)
    rec = {"render": stats, "render_seconds": round(time.time() - t, 1),
           "daru_metrics": evaluate.daru_metrics(wd / "maxent/final", label, s["predictors"], wd / "maxent/background.csv")}
    if bench:
        d = ctx.root / "daru_ref_all"
        rec["calibration_vs_daru"] = benchmark.calibration_iou(s["calibration_ecoregions"], ctx.ecoregions,
                                                               d / f"raw_rasters/{name}.tif")
        rec["vs_daru"] = benchmark.compare(wd / "maps", label, d / f"rasters/{name}.tif", d / f"raw_rasters/{name}.tif")
        native = unary_union(ctx.wgsrpd[ctx.wgsrpd.LEVEL3_COD.isin(s["native_l3"])].geometry.values)
        rec["wcvp"] = {"ours": benchmark.wcvp_consistency(wd / f"maps/{label}.binary_vote.tif", (2,), native),
                       "daru": benchmark.wcvp_consistency(d / f"rasters/{name}.tif", (1,), native)}
        for src in ("aim", "fia"):
            truth = getattr(ctx.plots, src)(name)
            if truth is not None and truth.present.sum() >= 20:
                rec[src] = validate.validate(truth, wd / "maps", label, d / f"rasters/{name}.tif", d / f"raw_rasters/{name}.tif")
    (wd / "render.json").write_text(json.dumps(rec, indent=1, default=str))
    codec.encode(wd / f"{label}.rangecard", label, wd / "maxent/final", s["predictors"],
                 [r.threshold_ess for r in reps], [r.threshold_p5 for r in reps], s["calibration_ecoregions"], region_maps)
    return rec
