# Track D — Ground-truth data sourcing (Sentinel-2 precision problem)

Repo: `flood_hydrology`. Mostly investigation + small standalone analysis
scripts — does not require modifying the existing pipeline. Fully independent
of every other track; good to run in parallel with Track A.

## Why

Two independent sources converged on the same worry within two days:

**Team lead's meeting notes:**
> "Is Sentinel 2 precise enough? Like can we tell the lake changes from
> Sentinel 2 and volume difference, if in a flood event... Right now we are
> not confident with the OmniWaterMask, not because it's wrong but not
> precise enough for real-time simulation."

**Lance Legel's emails (2026-06-16 and 2026-06-17):**
> "I think the principle should be: we can see even slightly higher (+/- a
> few feet) levels of lakes and rivers... It's about having a very precise
> way to do this... we really want to know 'probability that any location
> will be inundated' [not just a binary flood event]."
>
> Second email: applied for Planet/NASA sub-meter access (pending); found a
> PlanetScope provider (3m, near-daily since 2020, ~$300/image, 3-image
> minimum, covers 100km²/image — sample images already downloaded to
> `~/Desktop/PlanetScope/` as `PlanetScope.zip`, extracted); explored
> SkyWatch (`https://explore.skywatch.com`) as a sparser-but-cheaper
> alternative; realized the team's own drone photography is RTK-precise
> (cm-scale, exact date known) and could serve as small-scale validation
> ground truth; suggested USGS gauge data
> (`https://waterdata.usgs.gov/monitoring-location/USGS-02234344/`) as
> another independent water-level source.

Qin already replied to Lance asking about drone photography access and
committed to reviewing the PlanetScope samples.

## Current state

- `~/Desktop/PlanetScope/` has 3 sample JPGs: `Rutherford_NJ_...`,
  `NYC_PlanetScope_..._II.jpg`, `NYC_PlanetScope_..._III.jpg` — **none of
  these are Johns Lake**, they're generic example shots from the provider.
- The flood_hydrology repo's authoritative ground truth today is
  OmniWaterMask (OWM) majority-vote over Sentinel-2 (10m resolution),
  documented in CLAUDE.md as F1=0.882 vs. NHD — good relative accuracy among
  the 4 segmentation methods tested, but absolute spatial resolution is the
  open question (10m pixels can't resolve foot-scale shoreline changes).
- No drone RTK imagery is in the repo yet — access was requested from Lance,
  not yet received as of this writing.
- No USGS gauge data has been pulled for this AOI yet; the URL Lance shared
  (`USGS-02234344`) has not been checked against whether it's actually in the
  Johns Lake watershed — confirm this before treating it as ground truth.
- `dem/data/lake_mask.tif` / `dem/data/lake_volume.csv` operate within the
  2x2km AOI centered on Johns Lake — not yet confirmed whether Johns Lake's
  full extent exceeds that AOI (relevant if planning to "extend coverage for
  the whole lake" per the meeting notes).

## Tasks

1. **PlanetScope sample review.** Look at the 3 sample JPGs in
   `~/Desktop/PlanetScope/` to get a feel for actual 3m-resolution image
   quality/clarity (they're not Johns Lake, but representative of what a real
   order would look like). Write up: at 3m/pixel, what shoreline-position
   precision is realistically achievable (sub-pixel edge detection can do
   better than the raw pixel size, but quantify a reasonable estimate); is
   $300/image × 3-image minimum ($900) viable for a single before/during/after
   flood triplet at Johns Lake; would need PlanetScope's actual revisit
   schedule for the AOI to know if a target event date is even achievable
   without same-day cloud cover, etc.
2. **Drone RTK imagery.** Once Lance grants access (track this as a blocker —
   check back with Qin/Lance if it hasn't arrived): document what's actually
   available (coverage extent, date range, file format). Then design (don't
   necessarily build yet, until data's in hand) a validation routine: extract
   shoreline pixels from the drone orthomosaic at the same date as an OWM
   Sentinel-2 mask, compute the actual area/position discrepancy — this turns
   "OWM might not be precise enough" from a guess into a measured number.
3. **USGS gauge cross-check.** First confirm `USGS-02234344` (or find the
   correct gauge) is actually relevant to Johns Lake's watershed — it may be
   a generic example link, not necessarily the right station. If a relevant
   gauge exists, pull its water-level time series and compare against
   `dem/data/lake_volume.csv`'s WSE assumption (28.74 m NAVD88) as an
   independent cross-check, the same way GSDR currently cross-checks Atlas14.
4. **OWM seasonality + full-lake coverage check.** Using the existing
   `sentinel2/data/water_extent_timeseries.csv` (152 rows, OWM area per date,
   2016–2026), split by wet season (Jun–Sep) vs. dry season (Oct–May) and
   check whether area estimates diverge in a way that would bias the
   "authoritative" 144.47 ha figure in `dem/data/lake_volume.csv` — if Johns
   Lake's true extent exceeds the current 2x2km AOI, that's a separate
   problem (need a wider DEM/S2 download, not just a seasonality split).
5. **Clay foundation model.** Note from the meeting ("Sentinel 2 Clay in
   play?") likely refers to the open Clay Earth-observation foundation model.
   Scope only — check if it's straightforward to add as a 5th method
   alongside MNDWI/WatNet/Prithvi/OWM in
   `sentinel2/method_comparison*.py`/`compare_methods.py`, but this is
   lower-priority than items 1–4 since it doesn't address the core
   resolution problem (Clay still runs on the same 10m Sentinel-2 input).

## Files touched

New files only — suggested home: a new `ground_truth/` directory at repo
root, or just `precipitation/` if you'd rather not add a new top-level dir.
Don't modify the existing OWM/lake_mask pipeline in this track; this is about
characterizing the precision gap, not fixing it yet.

## Verification

This track produces written findings and comparison numbers (not pass/fail
checks). Good output artifacts: a short writeup with the resolution/precision
estimates from task 1, a wet/dry season area comparison table from task 4,
and a clear yes/no on whether `USGS-02234344` is actually usable from task 3.
