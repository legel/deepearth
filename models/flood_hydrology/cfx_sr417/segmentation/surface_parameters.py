"""
Per-class hydrological parameters
=================================
Attaches {material, Manning's n, surface storage, impervious fraction} to each surface class
produced by `segment_naip.py`, and writes `data/surface_parameters.json` for
`rasterize_parameters.py` to put on the solver grid.

ALL FOUR OF LANCE'S PARAMETERS ARE HERE — TWO OF THEM AS A COMPARISON
---------------------------------------------------------------------
A vision-language parameterisation of this kind is asked to supply `{material, Smax, Ks,
Manning's n}`. The first pass here declined to supply Ks and soil Smax, on the grounds that
SSURGO already measures them from an actual soil survey (28 map units at site3) and a vision
model cannot tell Immokalee sand from Basinger sand or see a water table at all.

That was a judgement call, and it has been replaced by an experiment. Both parameterisations are
now built, and both are run against the Gee Creek gauge:

  SSURGO route  `load_spatial_horton()` + `load_soil_storage_capacity()` — the measured survey
  vision route  `vision_ks_mm_hr_dry` + `vision_soil_storage_m` below — inferred from cover

**This matters well beyond settling an internal disagreement. SSURGO exists only in the United
States.** If the premise is "a coordinate anywhere becomes a working twin", the size of the gap
between a soil survey and a land-cover inference is precisely what decides whether this pipeline
is US-only or global. The comparison is the deliverable, not the winner.

Fairness is enforced in one specific place: the solver already multiplies SSURGO Ksat by an
AMC-III wet-antecedent factor of 0.07, and the same factor is applied to the vision Ks. Both are
dry-condition saturated conductivities, so correcting only one would pit a wet soil against a dry
one and let the vision arm infiltrate ~14x too much.

What imagery knows and SSURGO does not, and where segmentation is the sole source:

  Manning's n        — surface roughness. SSURGO says nothing about it, and the solver currently
                       carries a single scalar 0.040 for every non-roof cell in the domain. This
                       is the parameter segmentation actually adds, and it is the one that moves
                       conveyance.
  surface storage    — interception by canopy and litter, plus depression storage on roofs and
                       pavement. This is a DIFFERENT store from SSURGO's soil storage and is
                       additive to it: rain fills it before any water reaches the soil surface.
                       Naming it `surface_storage` rather than `Smax` keeps the two from being
                       silently conflated.
  impervious fraction— at 0.6 m. The solver's current impervious mask is OSM road/building
                       polygons (binary, only what OSM maps) graded by NLCD at 30 m. NAIP
                       resolves driveways, parking bays, sidewalks and pool decks that both miss.

So this stage supplies roughness and surface storage as genuinely new spatial fields, a sharper
impervious fraction for the existing infiltration machinery, and a second, independent route to
Ks and soil storage that is measured against the first rather than asserted over it.

WHERE THE NUMBERS COME FROM
---------------------------
Manning's n uses the land-cover roughness values conventional in 2D shallow-water practice
(HEC-RAS 2D / FEMA flood-study land-cover tables, rooted in Chow 1959 Table 5-6). That family is
the right one for this solver's regime — a 5 m grid at a documented median wet depth of 7-8 cm.

It is worth being explicit about the family NOT used. Engman (1986) overland-flow values for the
same covers are far larger (bluegrass sod 0.45, woods with light underbrush 0.40) because they
are calibrated for millimetre-deep sheet flow, where roughness elements are a large fraction of
the flow depth. Adopting them here would raise n by 10-20x, swamp every other change in the
model, and would not be defensible at 5 m resolution. The values below keep the existing scalar
0.040 inside their range, so the change is a redistribution of roughness rather than a
recalibration of its overall level.

Surface storage: canopy interception capacity for closed-canopy broadleaf/mixed forest is
~1-2 mm per storm; litter adds a few mm. Roof and pavement depression storage is ~0.5-1.5 mm.
These are small against a 392 mm storm and are included for completeness rather than effect —
`rasterize_parameters.py` reports the domain total so the size of the term is visible, not
assumed.

THE VLM STEP
------------
The table below is the parameter-assignment step, carried out by a multimodal model reasoning
over the class definitions and the measured per-class segment statistics, with each value
carrying its own justification and source. `--audit` re-runs that assignment through the
Anthropic API as an independent adversarial second opinion, reporting disagreements rather than
silently overwriting. It needs ANTHROPIC_API_KEY, which is not set in this environment today.

Usage:
    python3 segmentation/surface_parameters.py                 # write the table
    python3 segmentation/surface_parameters.py --show          # print it
    python3 segmentation/surface_parameters.py --audit         # second opinion (needs API key)
"""
import os
import sys
import json
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

TABLE_VERSION = "1.1.0"

# vision_ks_mm_hr_dry  [mm/hr] DRY saturated conductivity inferred from land cover alone — the
#                      Ks a vision-language parameterisation supplies. Kept ALONGSIDE SSURGO rather
#                      than replacing it, so the two can be run against the gauge and compared.
#                      Anchored on Rawls, Brakensiek & Saxton (1982) conductivity by USDA texture
#                      class, with the texture inferred from what the cover implies about the
#                      soil beneath it. Deliberately written down BEFORE looking at SSURGO's own
#                      numbers, so the comparison is not circular.
# vision_soil_storage_m [m] profile storage inferred as rooting depth x drainable porosity — what
#                      a vision model can reason about, since it cannot see a water table.
# manning_n            [-]  surface roughness for 2D shallow-water routing at 5 m / ~0.1 m depth
# surface_storage_m    [m]  interception + depression storage held before water reaches the soil
# impervious_fraction  [-]  fraction of the class that sheds rather than infiltrates
PARAMETERS = {
    "water": {
        "material": "open water",
        "manning_n": 0.035,
        "surface_storage_m": 0.000,
        "impervious_fraction": 1.00,
        "n_basis": "Chow 1959 — clean, straight natural channel / open water surface, 0.030-0.040.",
        "storage_basis": "A free water surface has no storage to fill; rain joins it directly.",
        "impervious_basis": "Rain on open water becomes surface water immediately; it does not infiltrate here.",
        "vision_ks_mm_hr_dry": 0.0,
        "vision_soil_storage_m": 0.0,
        "vision_ks_basis": "No infiltration through a free water surface.",
        "vision_soil_storage_basis": "No profile to fill.",
    },
    "building_roof": {
        "material": "asphalt shingle / metal roofing",
        "manning_n": 0.015,
        "surface_storage_m": 0.0005,
        "impervious_fraction": 1.00,
        "n_basis": "Smooth manufactured surface. Matches the 0.015 both mesh solvers already use for roofs, so this class does not move a value the project had already set deliberately.",
        "storage_basis": "0.5 mm depression storage on a pitched roof — shallow, since a roof is built to shed.",
        "impervious_basis": "By definition.",
        "vision_ks_mm_hr_dry": 0.0,
        "vision_soil_storage_m": 0.0,
        "vision_ks_basis": "Sealed.",
        "vision_soil_storage_basis": "Sealed.",
    },
    "road_paved": {
        "material": "asphalt concrete",
        "manning_n": 0.013,
        "surface_storage_m": 0.001,
        "impervious_fraction": 1.00,
        "n_basis": "Chow 1959 — smooth asphalt, 0.013. The smoothest surface in the domain.",
        "storage_basis": "1 mm; crowned and drained, but with real texture depth and gutter ponding.",
        "impervious_basis": "By definition; already how the solver's OSM mask treats these cells.",
        "vision_ks_mm_hr_dry": 0.0,
        "vision_soil_storage_m": 0.0,
        "vision_ks_basis": "Sealed.",
        "vision_soil_storage_basis": "Sealed.",
    },
    "impervious_other": {
        "material": "concrete / compacted gravel — driveways, parking, sidewalks, pool decks",
        "manning_n": 0.016,
        "surface_storage_m": 0.0015,
        "impervious_fraction": 0.90,
        "n_basis": "Slightly rougher than a highway-grade road: broom-finished concrete, joints, and gravel aprons. Chow's concrete range 0.012-0.018.",
        "storage_basis": "1.5 mm; flatter and less deliberately drained than a road.",
        "impervious_basis": "0.90 rather than 1.00 — this class is inferred spectrally, not from a mapped footprint, so a fraction of it is genuinely compacted bare ground rather than sealed surface.",
        "vision_ks_mm_hr_dry": 1.0,
        "vision_soil_storage_m": 0.01,
        "vision_ks_basis": "90 % sealed; the pervious remainder is compacted gravel/verge.",
        "vision_soil_storage_basis": "Thin pervious remainder only.",
    },
    "tree_canopy": {
        "material": "forest floor beneath closed canopy — litter over sandy soil",
        "manning_n": 0.120,
        "surface_storage_m": 0.0045,
        "impervious_fraction": 0.00,
        "n_basis": "HEC-RAS 2D / FEMA land-cover tables for forest, 0.10-0.12; Chow's 'heavy stand of timber, little undergrowth, flood stage below branches' is 0.10. Trunks, roots, litter and understory obstruct flow. 3x the scalar this class currently gets.",
        "storage_basis": "1.5 mm canopy interception for a closed broadleaf/mixed stand, plus ~3 mm litter storage.",
        "impervious_basis": "Fully pervious; SSURGO supplies the soil beneath.",
        "vision_ks_mm_hr_dry": 210.0,
        "vision_soil_storage_m": 0.375,
        "vision_ks_basis": "Rawls et al. (1982) sand, 210 mm/hr. Florida flatwoods forest floor is fine sand, and root macropores and litter keep it open — the highest-conductivity surface in the domain.",
        "vision_soil_storage_basis": "Rooting depth ~1.5 m x 0.25 drainable porosity. Trees root deepest and so command the largest profile.",
    },
    "shrub_scrub": {
        "material": "palmetto / scrub understory",
        "manning_n": 0.070,
        "surface_storage_m": 0.0020,
        "impervious_fraction": 0.00,
        "n_basis": "FEMA land-cover tables, shrub/scrub 0.05-0.07. Denser near the ground than forest but without the litter mat.",
        "storage_basis": "2 mm; less leaf area than closed canopy, little litter.",
        "impervious_basis": "Fully pervious.",
        "vision_ks_mm_hr_dry": 180.0,
        "vision_soil_storage_m": 0.25,
        "vision_ks_basis": "Slightly below forest: sandy, but less macropore development under palmetto scrub.",
        "vision_soil_storage_basis": "~1.0 m rooting depth x 0.25.",
    },
    "grass_turf": {
        "material": "managed turf / pasture over sandy soil",
        "manning_n": 0.040,
        "surface_storage_m": 0.0010,
        "impervious_fraction": 0.00,
        "n_basis": "FEMA grassland/pasture 0.035-0.05. Deliberately set to 0.040 — the exact scalar the solver uses domain-wide today — so that any change in the gauge metrics is attributable to the classes that MOVED, not to a quiet shift in the baseline.",
        "storage_basis": "1 mm; short vegetation intercepts little.",
        "impervious_basis": "Fully pervious.",
        "vision_ks_mm_hr_dry": 60.0,
        "vision_soil_storage_m": 0.125,
        "vision_ks_basis": "Rawls loamy sand, 61 mm/hr. Managed turf is mown, trafficked and partly compacted, so well below the forest floor on the same parent sand.",
        "vision_soil_storage_basis": "~0.5 m rooting depth x 0.25. Turf roots shallowly.",
    },
    "bare_soil": {
        "material": "bare / sparsely vegetated sand",
        "manning_n": 0.025,
        "surface_storage_m": 0.0005,
        "impervious_fraction": 0.00,
        "n_basis": "Chow 1959, bare earth 0.020-0.025. Smoother than turf, rougher than pavement.",
        "storage_basis": "0.5 mm surface roughness storage only.",
        "impervious_basis": "Pervious; SSURGO supplies the soil.",
        "vision_ks_mm_hr_dry": 40.0,
        "vision_soil_storage_m": 0.125,
        "vision_ks_basis": "Between loamy sand and sandy loam: exposed sand crusts and compacts without a root mat.",
        "vision_soil_storage_basis": "~0.5 m x 0.25; no deep root system to open the profile.",
    },
    "wetland_marsh": {
        "material": "emergent herbaceous wetland / cypress dome margin",
        "manning_n": 0.080,
        "surface_storage_m": 0.000,
        "impervious_fraction": 0.00,
        "n_basis": "FEMA emergent-wetland 0.05-0.08. Dense standing stems through the flow depth make this the second-roughest class here.",
        "storage_basis": "Zero. A wetland sits at the water table and has no unfilled surface store — the same reasoning that gives SSURGO's depressional soils zero soil storage.",
        "impervious_basis": "Pervious in principle, but saturated in practice; the SSURGO storage cap already forces these cells to shed.",
        "vision_ks_mm_hr_dry": 2.0,
        "vision_soil_storage_m": 0.0,
        "vision_ks_basis": "Organic, saturated, and sitting at the water table — effectively no capacity.",
        "vision_soil_storage_basis": "Zero — a wetland sits at the water table, the same reasoning SSURGO applies to its depressional soils.",
    },
}

PROVENANCE = {
    "table_version": TABLE_VERSION,
    "assigned_by": "multimodal reasoning over class definitions + measured per-class segment statistics",
    "roughness_family": "2D shallow-water land-cover tables (HEC-RAS 2D / FEMA), rooted in Chow (1959) Table 5-6",
    "roughness_family_rejected": ("Engman (1986) overland-flow coefficients — calibrated for "
                                  "millimetre-deep sheet flow, 10-20x larger, not defensible at "
                                  "5 m resolution and ~0.1 m depth"),
    "baseline_anchor": ("grass_turf is pinned to the solver's existing scalar MANNING_N = 0.040 "
                        "so the experiment isolates redistribution from recalibration"),
    "soil_parameters_are_now_a_measured_comparison": {
        "why": ("An earlier revision declined to supply Ks and soil Smax on the grounds that "
                "SSURGO measures them. That was a judgement call, and it has been replaced by an "
                "experiment: both parameterisations are built and both are run against the Gee "
                "Creek gauge. See run_site3_ian_segmented.py --arms vision_soil."),
        "ssurgo_route": "load_spatial_horton() + load_soil_storage_capacity(), 28 map units",
        "vision_route": "vision_ks_mm_hr_dry / vision_soil_storage_m below, from the class map",
        "why_it_matters_beyond_this_site": ("SSURGO exists only in the United States. If the "
                                            "premise is 'any coordinate -> a twin', the size of "
                                            "the gap between a soil survey and a vision estimate "
                                            "is what decides whether the pipeline is US-only or "
                                            "global."),
        "fairness": ("The same AMC-III wet-antecedent factor the solver already applies to "
                     "SSURGO Ksat is applied to the vision Ks. Both sides are dry-condition "
                     "saturated conductivities, so correcting only one would compare a wet soil "
                     "against a dry one and the vision arm would infiltrate ~14x too much."),
    },
    "references": [
        "Chow, V.T. (1959) Open-Channel Hydraulics, Table 5-6.",
        "USACE HEC-RAS 2D Modeling User's Manual — land-cover Manning's n tables.",
        "Engman, E.T. (1986) Roughness coefficients for routing surface runoff, JIDE 112(1) — the rejected family, cited so the choice is auditable.",
        "Zinke, P.J. (1967) Forest interception studies in the United States — canopy interception 1-2 mm.",
    ],
}


def write_table():
    os.makedirs(DATA_DIR, exist_ok=True)
    out = {"provenance": PROVENANCE, "classes": PARAMETERS}
    path = os.path.join(DATA_DIR, "surface_parameters.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    return path


def show():
    print("=" * 92)
    print(f"Surface parameter table v{TABLE_VERSION}")
    print("=" * 92)
    print(f"  {'class':<18} {'n':>7} {'surf.stor':>11} {'imperv':>8} {'visKs':>9} {'visSmax':>9}   material")
    print("  " + "-" * 88)
    for k, v in PARAMETERS.items():
        print(f"  {k:<18} {v['manning_n']:>7.3f} {1000*v['surface_storage_m']:>8.1f} mm "
              f"{v['impervious_fraction']:>8.2f} {v['vision_ks_mm_hr_dry']:>7.1f}mm/h "
              f"{1000*v['vision_soil_storage_m']:>6.0f}mm   {v['material']}")
    print("  " + "-" * 88)
    print(f"  solver scalar today: MANNING_N = 0.040 everywhere non-roof, 0.015 on roofs")
    print(f"  range introduced   : {min(v['manning_n'] for v in PARAMETERS.values()):.3f} - "
          f"{max(v['manning_n'] for v in PARAMETERS.values()):.3f}  "
          f"({max(v['manning_n'] for v in PARAMETERS.values()) / min(v['manning_n'] for v in PARAMETERS.values()):.1f}x)")


def audit():
    """Independent second opinion on every value, via the Anthropic API."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("--audit needs ANTHROPIC_API_KEY. Not set in this environment; the shipped "
                 "table's per-value justifications are in surface_parameters.json meanwhile.")
    try:
        import anthropic
    except ImportError:
        sys.exit("--audit needs the anthropic package: python3 -m pip install --user anthropic")

    client = anthropic.Anthropic()
    prompt = (
        "You are auditing Manning's n, surface storage and impervious fraction for a 2D "
        "local-inertial shallow-water flood model. Grid 5 m, central Florida, median wet depth "
        "7-8 cm, storm total 392 mm over 72 h. Roughness must come from the HEC-RAS 2D / FEMA "
        "land-cover family (NOT Engman overland-flow sheet-flow values).\n\n"
        "For each class below, state whether the value is defensible. If not, give a replacement "
        "and a one-line reason. Reply as JSON: "
        '{"<class>": {"verdict": "ok"|"revise", "manning_n": <float>, "reason": "<text>"}}\n\n'
        + json.dumps({k: {kk: v[kk] for kk in ("material", "manning_n", "surface_storage_m",
                                               "impervious_fraction")}
                      for k, v in PARAMETERS.items()}, indent=2)
    )
    resp = client.messages.create(model="claude-opus-5", max_tokens=4000,
                                  messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in resp.content if b.type == "text")
    path = os.path.join(DATA_DIR, "surface_parameters_audit.json")
    with open(path, "w") as fh:
        fh.write(text)
    print(text)
    print(f"\n  wrote {os.path.relpath(path, PROJ_DIR)}  (review manually; NOT auto-applied)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()

    if args.audit:
        audit()
        return
    path = write_table()
    show()
    print(f"\n  wrote {os.path.relpath(path, PROJ_DIR)}")


if __name__ == "__main__":
    main()
