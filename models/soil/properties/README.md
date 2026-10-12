# Soil properties: pH, texture and organic carbon across a site

**Question.** For planting design we need 0–30 cm pH, texture and organic carbon at a site. That means within-site
pattern, not only the regional value. Which gridded source should we trust?

**Test.** We use NEON DP1.10086.001 field plots: 544 plots at 47 sites, mineral-horizon cores ≤ 30 cm, plot-mean pH
in water and organic C, with ~20 m coordinates. These are measured by NEON's labs, and none of the products are
trained on them. We score each product two ways:
- nationally;
- within site, as the deviation from each site's mean. Sites span a few km.

Organic C is scored in log10(%). WoSIS 2023 US pedons (`benchmark.py --set wosis`) are a second set. They are mostly
NCSS pedons, which are in-sample for SOLUS100.

| | pH national R² | pH within-site R² | OC (log) national R² | OC within-site R² |
|---|---|---|---|---|
| SOLUS100 (USDA-NRCS, 100 m) | **0.81** | **0.18** | 0.23 (low ×0.6) | −0.01 |
| SoilGrids 2.0 (ISRIC, 250 m) | 0.79 | −0.01 | 0.49 | 0.02 |
| POLARIS (30 m) | 0.78 | −0.07 | −0.40 | −1.11 |
| SSURGO (map-unit rep. values) | 0.74 | −0.35 | −0.06 | −1.01 |
| SOLUS100 + SoilGrids log-mean (OC) | | | **0.59** | **0.12** |
| … + site-centred terrain ridge (served) | | | 0.59 | **0.135** |

**Findings.**
- SSURGO polygons carry no within-site information. They are worse than a constant, because component
  representative values jump at map-unit edges.
- SOLUS100 is the only source with positive within-site pH skill.
- For organic C, averaging SOLUS100 and SoilGrids in log space beats both, nationally and within site.
- A 10 m terrain correction (TPI, TWI, curvature, aspect; `terrain.py`) adds a little for organic C, with the expected
  signs: more carbon in hollows, concave ground and on north-facing slopes. It adds nothing for pH.
- A tree ensemble trained on raw targets (`fusion.py`) loses SOLUS100's spatial detail. The residual form (product
  plus a small correction) is the right structure.
- A gradient-boosted fusion trained on NEON alone does not beat SOLUS100 for pH. Within-site soil variation is the
  open problem.

**Served recipe.** `site_layers.layers(lon, lat, half_m)` returns ph / clay / sand / oc / USDA texture class on the
3DEP 1/3″ (~10 m) lattice:
- pH and texture from SOLUS100;
- organic C from the SOLUS100 / SoilGrids log-mean plus the terrain ridge in `site_model.json`;
- where SoilGrids is masked (urban), SOLUS100 plus the fitted +0.30 log10 offset.

SSURGO is kept only for drainage class and map-unit names.

**Files** (data under `$DEEPEARTH_SOIL_DATA`, default `~/.deepearth/soil`):
- `products.py`: remote point samplers for the four products.
- `terrain.py`: one definition for grid and point; the tests check equality.
- `texture.py`: the USDA triangle; the tests check every half-percent point gets exactly one class.
- `fetch_neon.py`, `benchmark.py`: the test data and the benchmark.
- `site_layers.py`: the served recipe.

Run `python -m models.soil.properties.fetch_neon`, then `python -m models.soil.properties.benchmark --set neon`. Tests: `models/soil/tests/test_properties.py`.

**Next for the water balance.** `soils.py` derives hydraulics from SSURGO / POLARIS texture. On both test sets, SOLUS100's
sand and clay are markedly better: WoSIS sand R² 0.90 vs 0.65 / 0.52, clay 0.74 vs 0.58 / 0.45. Moving the water
balance's texture input to SOLUS100 is the obvious follow-up, validated against NEON's soil-water sensors as the water
balance is today.

**ORGANIC layer.** Shown as site-relative classes (low / medium / high), not absolute %. At this skill level the
pattern is more trustworthy than the number. NPK is not modelled: it is set by fertilizer history, and no gridded
source predicts it at a site.

## Within-site soil variation: what explains it (NEON, leave-one-site-out)

| added to the product base | pH within-site R² | OC within-site R² |
|---|---|---|
| none (served base) | 0.176 | 0.121 |
| 10 m terrain, site-centred ridge | 0.175 | 0.128 |
| AOP canopy height (1 m CHM, 10-100 m rings) | 0.177 | 0.107 |
| plant community: learned species indicator values | 0.191 | 0.126 |
| plant community: genus indicator values | 0.191 | 0.125 |
| **achievable ceiling** (plot-mean noise from core repeatability) | **0.94** | **0.80** |

The site-to-site level is captured nationally, at R² 0.81 for pH. Plot-to-plot differences within a site are real:
plot means rest on about 58 pH cores per plot. Only about a fifth of that variation is explained by maps, 10 m
terrain, canopy height or the plant community.

Soil varies as much within a 20 m plot as across a site. The remaining variation must come from finer structure:
1 m micro-topography, parent material and disturbance history. That is the target for the learned soil head on
full-resolution site latents.

The training/evaluation table is `$DEEPEARTH_SOIL_DATA/benchmark/soil_head_dataset_neon.parquet`. Per plot it holds the
observations, product predictions, terrain, canopy, the base, and residuals, raw and site-centred. The served product
shows soil as the expected value, with "soil test recommended" for pH-sensitive plantings.

## Second test set: WoSIS 2023, US pedons (3,000 profiles at ~100 m, 0–30 cm)

| national R² | pH | clay | sand | OC (log) |
|---|---|---|---|---|
| SOLUS100 | **0.74** | **0.74** | **0.90** | (read failed) |
| POLARIS | 0.60 | 0.45 | 0.52 | 0.21 |
| SSURGO | 0.60 | 0.58 | 0.65 | −0.03 |

Most of these profiles are NCSS pedons, so SOLUS100 has partly seen them; its margin here is an upper bound. POLARIS
and SSURGO derive from the same soil survey yet trail SOLUS100 by 0.14–0.38 R², and this matches NEON's
independent ranking. Adding these pedons to the terrain-correction training doesn't change NEON within-site skill
(pH 0.178 vs 0.181 base).
