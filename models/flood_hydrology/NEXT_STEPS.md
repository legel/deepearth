# Next steps

Updated 2026-08-31. Full session log for the solver work below is in
[`../PROGRESS_2026-08-29.md`](../PROGRESS_2026-08-29.md); the preceding session is in
[`../PROGRESS_2026-08-26.md`](../PROGRESS_2026-08-26.md); site detail is in each site's README.

## READ FIRST — 2026-08-29 solver state

Four defects were found in `simulation/flood_sim_ian.py`'s integration loop, all by mass balance
refusing to close. **Every magnitude number published before 2026-08-29 was produced through at
least one of them and none survive.** Timing results are far more robust and largely do.

| defect | effect | status |
|---|---|---|
| physics integrated a CFL-limited `dt` while the clock advanced `dt_s` | a "72-hour" run delivered **7-11 %** of the storm | fixed — sub-steps to each hyetograph interval |
| `cum_infil` charged the Horton *capacity* rate regardless of available rain | soil filled on paper without absorbing; infiltration 2.9 % of rain | fixed — charges `min(inf, P)·dt` |
| `CFL_ALPHA = 0.30` unstable at 5 m once water accumulates | **-517.8 %** mass residual, 8.99 m depths oscillating under zero rain | fixed — 0.15 |
| final-frame guard captured no end-of-run frame | broke any mass balance measured against `frames[-1]` | fixed |

The first two masked each other: the broken clock meant `cum_infil` accrued over a fraction of
real time and never reached its cap. The third was unreachable until the first was fixed,
because the solver never accumulated enough water to go unstable.

**A per-cell volume limiter was added and then reverted.** It was justified by a -12.6 % residual
that turned out to be a measurement artefact of the final-frame bug — outflow integrated to the
end compared against storage sampled 333 minutes earlier. Measured correctly the solver conserves
mass to **-0.001 %**, and the limiter changed results by 1 part in 10^4. Do not re-add it without
new evidence; `mesh_shallow_water.py` needs its equivalent, this solver does not.

### Current state, site3 @ 25 m, all fixes, mass residual -0.001 %

| | model | observed |
|---|---|---|
| **rising limb error** | **0.27 h** | gauge resolves 0.25 h |
| runoff coefficient | 72.1 % | **28.9 - 31.4 %** |
| infiltration | 17.3 % of rain (`cum_infil` 68 of 206 mm) | — |

**Timing is effectively solved** — flood onset reproduced to within one gauge sample, and the
gauge hydrograph now peaks, which it never did before. **Magnitude overshoots ~2.3x.**

25 m and 5 m agree exactly on these numbers, so 25 m is a valid fast proxy for the physics
(~7 min vs ~7.4 hours) — but it does NOT reproduce the 5 m instability, so any CFL or stability
question must be checked at production resolution.

### Ponded infiltration — implemented and tested, 2026-08-31

The leading candidate identified above: infiltration drew only from instantaneous rainfall
(`Pe = max(P - inf, 0)`), so once rain stopped, ponded water sitting on a half-empty soil profile
had no way to enter it. Predicted effect if fixed: infiltration 17.3 % → ~52 % of rain, runoff
72 % → ~37 %.

**Implemented in `run_sim()` as `ponded_infiltration` (default `True`, restructures the depth
update so all rainfall reaches the surface first, then infiltration draws from whatever depth is
standing — freshly fallen or already ponded — capped by remaining capacity and rate).
`--no-ponded-infiltration` reproduces the old behaviour exactly, kept for comparison.**

**Result: the fix works correctly (mass balance -0.0001 % → -0.0002 %, both clean) but the effect
is far smaller than predicted:**

| | old (rainfall-only-limited) | new (ponded infiltration) | observed |
|---|---|---|---|
| runoff coefficient | 72.08 % | **71.35 %** | 28.9-31.4 % |
| infiltrated | 17.3 % of rain | 18.2 % of rain | — |

Only a 0.7-point move, not the ~35-point jump predicted. **Why:** the prediction implicitly
assumed the "138 mm unreachable" was sitting around as a static reservoir waiting to be soaked
up. This is a routed shallow-water model — during an intense storm, water is actively flowing
downstream and exiting the domain boundary well before there is a lull for infiltration to catch
up. By the time rain stops, most of the excess has already left as outflow, not sitting on the
ground as standing depth. The fix is doing exactly what it should; the volume actually reachable
through this mechanism alone is just small. **This substantially weakens the hypothesis above
rather than confirming it.**

### Storage-cap sensitivity sweep — done, 2026-08-31, on the current solver

Direct test of the hypothesis above: scale the SSURGO storage cap alone (Manning's *n* left at
the scalar 0.040, everything else identical), site3 @ 25 m, `simulation/run_site3_ian.py
--storage-scale`.

| storage cap | mean | runoff coefficient |
|---|---|---|
| 0.5× SSURGO | 103 mm | 78.89 % |
| 1.0× SSURGO (current default) | 206 mm | 71.82 % |
| 2.0× SSURGO | 412 mm | 67.19 % |
| **uncapped** (infinite storage) | — | **43.92 %** |

Monotonic and clean — more capacity, less runoff, exactly as expected. **But it floors at 43.9 %
even with literally unlimited storage**, against an observed 28.9–31.4 %. **Conclusion: storage
capacity alone cannot close the magnitude gap, at any value.** (The above sweep used the OLD
rainfall-only-limited access rule; see the combined sweep below for the same test with ponded
infiltration also on.)

### Combined: ponded infiltration + capacity sweep — 2026-08-31

With the ponded-infiltration fix's small standalone effect and the capacity sweep's floor both
independently insufficient, the two were tested together — same capacity multipliers, ponded
infiltration on for all arms:

| capacity | mean cap | runoff coefficient |
|---|---|---|
| 0.5× SSURGO | 103 mm | 79.12 % |
| 1.0× SSURGO (survey value) | 206 mm | 71.35 % |
| 2.0× SSURGO | 412 mm | 65.32 % |
| 4.0× SSURGO | 824 mm | 59.25 % |
| **uncapped** | — | **33.57 %** |

All mass-conserving (-0.0008 % to +0.0025 %). **The two effects are synergistic, not additive** —
neither alone gets close (71.35 % / 43.9 %), but combined at the physically implausible uncapped
extreme, runoff lands within a few points of observed. Mechanism: with the old access rule, even
unlimited capacity couldn't help, because infiltration could only draw from the instantaneous
rain rate — excess ran off immediately during intense bursts no matter how much capacity existed
elsewhere. With ponded access, water that can't infiltrate instantly during a burst now sits and
keeps draining afterward, but only if there is still room to receive it.

**The bounded, physically defensible steps (0.5×-4×) show a clean ~-6 points per doubling — not
enough to reach observed at any plausible multiplier** (4× SSURGO already implies a water table
4× deeper than the survey says; even that only reaches 59.25 %). The jump to 33.57 % only happens
at literally infinite capacity, which is not real soil. **This is a genuine result, not a
calibration knob**: the soil-storage-capacity axis (however combined with infiltration access)
cannot be pushed to the observed range without an unphysical assumption.

### Second real-storm validation: Hurricane Milton (Oct 2024) — 2026-08-31

Everything above was calibrated against exactly one event (Ian) — a real methodological risk, so
a second, independent real storm was pulled from the same gauge's continuous record: real USGS
NWIS discharge (02234400) and real KSFB hourly rainfall, Oct 4-15 2024, current solver (ponded
infiltration on, SSURGO 1× cap), same scoring methodology.

| | Hurricane Ian (2022) | Hurricane Milton (2024) |
|---|---|---|
| storm total | 392 mm | 288 mm |
| rising-limb error | 0.2-0.3 h | 1.53 h |
| modeled runoff coeff | 71.35 % | 67.70 % |
| observed runoff coeff | 28.9-31.4 % | 34.4-36.6 % |
| **shortfall (model/observed)** | **~2.3×** | **~1.9×** |
| mass balance | clean | clean (-0.0009 %) |

**Two completely independent real storms — different year, different total rainfall, different
antecedent conditions — show the same ~2× overshoot.** Timing is good on both. This is the
strongest evidence yet that the magnitude gap is a structural property of the model, not an
artifact of Ian's particular intensity or antecedent soil moisture: if it were Ian-specific,
Milton should have looked meaningfully different, and it did not.

**Where this points:** the solver's numerics are not the problem — mass balance closes to
±0.002 % across every configuration tested, on both storms. What has not been validated is the
*setup*: a small, artificially-bounded rectangular domain, fed only by direct rainfall with no
external inflow and no baseflow/groundwater return, compared against a real single-channel gauge
via a domain-boundary outflow sum the project's own validation script already calls
"structurally approximate." That raised the domain-vs-real-watershed-area mismatch to the top
candidate — tested directly below.

### Domain-clip test — 2026-08-31, ruled out (in the wrong direction)

Direct test of the hypothesis above. `flood_sim_ian.py`'s `run_sim()` gained a `watershed_mask`
parameter: instead of summing outflow at the 4 outer domain-box edges, it now accumulates net
flux across every internal face where the real delineated watershed boundary sits (an aggregate
over the whole boundary, not a single noisy cell — a genuinely new diagnostic, not the unreliable
gauge-cell metric). The current delineated catchment is **3.708 km²** — smaller than a
previously-documented 11.65 km² figure, only ~11 % of the real 33.15 km² gauge area.

| | domain-box runoff coeff | watershed-boundary runoff coeff | observed |
|---|---|---|---|
| Hurricane Ian | 71.35 % | **85.48 %** | 28.9-31.4 % |
| Hurricane Milton | 67.70 % | **82.44 %** | 34.4-36.6 % |

**Restricting to the real watershed made the overshoot worse on both storms, not better.**
Mechanism, in hindsight: the 3.7 km² connected catchment isn't a random or representative slice
of the domain — it's specifically the part D8 found efficiently connected to the channel, i.e.
the flashiest, most well-drained part of the whole domain. Averaging in the other ~43 km² of
poorly-connected, ponding-prone terrain was pulling the domain-wide number *down* toward
observed, not diluting a better answer. **This meaningfully weakens the domain-area-mismatch
hypothesis** rather than confirming it.

### AMC (antecedent-moisture) factor sweep, Milton — 2026-08-31, insufficient

`AMC3_FACTOR = 0.07` is documented as calibrated specifically for Ian's pre-saturated antecedent
conditions (2 weeks of above-average rain beforehand) — it was never re-derived for Milton, whose
own pre-storm baseflow (7.87 cfs) is 5.7× lower than Ian's (45.2 cfs), strong evidence the ground
was much drier beforehand. Swept the factor for Milton alone, everything else held at the current
default:

| AMC factor | effective Ksat | runoff coefficient |
|---|---|---|
| 0.07 (Ian's wet factor, reused for Milton) | 24.9 mm/hr | 67.70 % |
| 0.20 | 71.1 mm/hr | 64.44 % |
| 0.50 | 177.7 mm/hr | 61.42 % |
| **1.00 (no wet-antecedent reduction at all)** | 355.4 mm/hr | **58.51 %** |
| observed | — | 34.4-36.6 % |

Direction is correct — drier assumption, less runoff — but even removing the correction
*entirely* (assuming completely dry soil, which is not physically true; Milton had real rain
building for several days beforehand) only closes about a third of the gap. **Insufficient alone,
same as every other mechanism tested.**

### Where this leaves the magnitude gap — 2026-08-31

Every mechanism this project's physics could plausibly explain has now been tested and found
insufficient or wrong-directioned: roughness (3 independent methods), storage-capacity size,
infiltration access, capacity+access combined at any physically plausible value,
domain-vs-watershed area, and antecedent-moisture calibration. That is six independent tests,
across two real storms, all with clean mass balance, none closing the gap and one (domain area)
actively pointing the wrong direction. **This is a strong, consistent signal that the gap is not
a single missing parameter — treating it as one to keep searching for is no longer the productive
path.**

**Current stopping point, stated plainly:** timing is genuinely solved (rising-limb error at or
near the gauge's own sampling resolution on both storms). Magnitude has a real, now
well-characterized, ~2× overshoot that survives every mechanism this model can express. Logged
here as an honest open limitation rather than continuing to sweep blindly.

**What would actually move this forward is new data, not new parameter sweeps:** independent
gridded/radar rainfall (e.g. MRMS) to check whether the single point-gauge input is
representative of what actually fell across the real watershed; or a fundamentally different
comparison method than boundary-outflow-vs-single-channel-gauge, which the project's own
validation script has called "structurally approximate" since it was written.

### HydroSHEDS v2 Upstream Accumulation Area (ACA) — reviewed 2026-09-02, not applicable

HydroSHEDS v2 released a new global Upstream Accumulation Area (flow-accumulation) dataset for
the Americas. Reviewed as a candidate input given the domain-vs-watershed-area work above; not
adopted, for three reasons.

Drainage-area/contributing-area size was already tested directly (the domain-clip test above) and
found to point the wrong direction — restricting to a smaller, well-connected delineated catchment
made the overshoot worse, not better. A different upstream-area product doesn't change that
finding. Resolution is also a mismatch: ACA is ~15 arcsec (~450 m), while this project's own D8
delineation runs on 0.875 m LiDAR conditioned via stream-burning — adopting the coarser product
would be a precision regression for a ~3.7 km² watershed. And current evidence (six ruled-out
mechanisms above) points to rainfall-input representativeness or the soil-water side as the likely
remaining driver, not delineation, so this dataset doesn't target the open question. No action
taken.

### Window-truncation test, extended to 288h — 2026-09-04, real and substantial, but NOT yet a determined asymptotic gap; a flawed stronger claim made and retracted same day

Direct follow-up to the 2026-09-03 window-truncation entry below, which found a real but
partial effect (72h->124h shrank Ian's shortfall 2.3-2.5x -> 1.6-1.8x) and flagged that neither
side had reached a fully-drained endpoint yet. Pushed both storms to the limit of available
observed data: fetched Gee Creek's NWIS record out to 2022-10-10 for Ian (new
`site3_gee_creek/infrastructure/fetch_gauge_discharge_site3.py` — no reusable fetch script for
this gauge existed before now; `gee_creek_ian_discharge_extended.csv`, 291.8h from the Ian
t0) and re-ran both storms at 25m to a matched **288h** window (`--extend-hours 216` for Ian,
`--extend-hours 168` for Milton).

| storm | window | simulated RC | observed RC (raw/baseflow-sep) | shortfall |
|---|---|---|---|---|
| Ian | 72h (original) | 71.35% | 29.0–31.5% | 2.3–2.5x |
| Ian | 124h | 72.74% | 41.3–45.7% | 1.6–1.8x |
| Ian | 288h (limit of fetched record) | 74.80% | 53.2–63.4% | 1.2–1.4x |
| Milton | 120h (original) | 64.42% | 14.2–15.2% | 4.2–4.5x |
| Milton | 168h | 66.80% | 27.9–29.3% | 2.3–2.4x |
| Milton | 288h (limit of fetched record) | 68.85% | 40.2–42.6% | 1.6–1.7x |

**A stronger claim was made from this table the same day — "the ~2x magnitude gap was mostly a
window artifact, quote 1.2–1.7x going forward" — and it does not hold up. Retracted here, not
silently corrected.** The check that broke it: at the 288h cutoff, observed discharge was
**still 2.3x pre-storm baseflow for Ian (103 cfs vs. 45.2 cfs) and 9.1x for Milton (71.8 cfs vs.
7.87 cfs)** — nowhere near returned to baseflow. Extrapolating Milton's decay rate (roughly
exponential between 240h and 288h) puts a real return to baseflow another 300+ hours out — 2+
more weeks past the end of the record actually fetched. **288h was the edge of the data we
happened to have, not a physically meaningful stopping point**, so the 1.2–1.7x figures above
are not an asymptotic answer, just where the ratio happened to be when the available record ran
out.

There is a second, more structural problem with pushing this test further as-is: **the
observed-side calculation subtracts a single fixed pre-storm baseflow value (45.2 / 7.87 cfs)
across the entire extended window.** That is a reasonable approximation over ~72-120h, but after
a storm this large the aquifer/wetland system itself recharges and baseflow rises and recedes
over a much longer timescale than the "quickflow" this metric is meant to isolate. Extending the
window further with a *fixed* baseflow cutoff will keep counting a growing share of genuine
slow, storm-associated aquifer/wetland recession as if it were direct storm runoff — inflating
the observed side for a reason that has nothing to do with correcting the simulated side's own
truncation, since this surface-only solver has no groundwater/baseflow term and its own outflow
genuinely approaches a bounded total in finite time. The two sides' apparent convergence in the
table above may therefore reflect two different, unrelated processes reaching similar numbers by
coincidence at this particular cutoff, not a real matched comparison.

**What the retraction does and does not change.** The *direction* of the original 2026-09-03
finding stands: comparing both sides over the storm's own 72h/120h rainfall duration is a real,
demonstrated, unfair truncation, and the reported shortfall shrinks substantially and
monotonically as the window lengthens (this table's own numbers show that). **What does not
stand is any specific number as the "true" residual gap** — 1.2–1.7x should not be quoted, and
neither should the original 2.3–4.5x storm-duration numbers be treated as more defensible than
they were already known to be. The seven previously ruled-out mechanisms are unaffected either
way.

**What would actually settle this**: either (a) wait for/fetch a much longer observed record —
weeks, not days — long enough to see discharge genuinely return to pre-storm baseflow on both
storms, replacing the fixed-value subtraction with a proper recession-based baseflow separation
(e.g. a standard digital filter, not a constant) once that data exists; or (b) accept that a
single finite-window "storm runoff coefficient" is not a well-defined comparison for a watershed
with a real recession this slow, and score future magnitude comparisons a different way entirely
(e.g. total event volume over a fixed, explicitly-labeled window on both sides, without implying
either side has fully drained). Neither has been done. This thread is open, not closed.

Reproducible: `simulation/run_site3_ian.py --cell-size 25 --dt 20 --extend-hours 216 --tag
_25m_288h`, `simulation/run_site3_milton.py --cell-size 25 --dt 20 --extend-hours 168 --tag
_25m_288h`.

### 5m production re-run — 2026-09-04, runoff-coefficient finding confirmed at full resolution; a real depth-jitter regression at fine resolution surfaced along the way

The canonical `hydrograph_ian_site3.csv` (what `analysis/validate_gauge_site3.py` reads by
default) was stale (last written Aug 28, before every fix below "READ FIRST"). Re-ran
`run_site3_ian.py --cell-size 5 --dt 20` at true production resolution to refresh it — **54,702s
(~15.2h) wall time**, well past the ~7.4h this file's own earlier note estimated (that estimate
predates spatial Horton, NLCD grading, the impervious mask, and the storage cap all being active
at once; the current solver is doing meaningfully more work per step than when that number was
measured).

**Runoff coefficient — the metric this whole investigation is built on — holds up: 73.99%
(5m) vs. 71.35% (25m proxy), both against the same 28.9–31.4% observed.** The magnitude-gap
finding is not a 25m-resolution artifact.

**Peak depth does NOT hold up: 5.966m (5m) vs. 3.241m (25m) — a real divergence, where the two
were previously documented to "agree exactly."** The depth trace is non-monotonic during the
zero-rain drainage tail (3.30 -> 3.88 -> 3.74 -> 3.26 -> 4.59 -> 5.97 m from t=60h to t=70h,
water level rising in several places with no rain falling) — consistent with, and now a second
real instance of, this project's own already-documented (`cfx_sr417/CLAUDE.md` future-work item
18) sub-grid pit-trapping limitation at fine resolution, previously only characterized at the
main AOI. **Do not quote a peak-depth figure from this 5m Ian run** for any external-facing
claim; the runoff-coefficient and rising-limb-timing numbers are unaffected and safe to use.

### Window-truncation test — 2026-09-03, real and substantial, but does not close the gap

Direct test of the other candidate named above: is comparing both sides over the storm's own
72h/120h rainfall duration cutting off real recession the observed record has time to show but
the simulated window doesn't run long enough to produce? New `--extend-hours` flag on
`run_site3_ian.py` (and a new sibling `run_site3_milton.py`, since no runnable Milton driver
existed on disk before this — the earlier Milton numbers in this file were produced ad hoc and
are not reproducible from any saved script) pads the hyetograph with zero rain and keeps the
solver running, so its own drainage tail isn't cut off either — a fair test needs BOTH sides
extended, not just the observed window.

**Ian: extended 72h -> 168h (observed record only goes to 123.8h, so the matched window is
0-123.8h)**

| window | simulated RC | observed RC (raw / baseflow-separated) | shortfall |
|---|---|---|---|
| 0-72h (original) | 71.35% | 29.0–31.5% | 2.3–2.5x |
| 0-123.8h (matched extended) | 72.74% | 41.3–45.7% | **1.6–1.8x** |

Simulated RC barely moves (most rain has already run off or infiltrated by 72h); observed RC
nearly doubles as the real, slower recession this project's own hydrograph-shape comparison
already noted (2026-07-27, cfx_sr417/CLAUDE.md: "real hydrograph... recedes over more than a
day") gets captured instead of cut off. **Roughly a third to a half of the previously-reported
~2.3x gap is a real window-truncation artifact, not a physics error** — a genuine, non-parametric
finding.

**Milton: extended 120h -> 168h (observed record covers the full window)**

| window | simulated RC | observed RC (raw / baseflow-separated) | shortfall |
|---|---|---|---|
| 0-120h (original) | 64.42% | 14.2–15.2% | 4.2–4.5x |
| 0-168h (matched extended) | 66.80% | 27.8–29.2% | **2.3–2.4x** |

Same direction, larger effect. **A real, honest complication surfaced computing this**: using the
identical method `analysis/validate_gauge_site3.py` already uses for Ian (documented 33.15 km²
denominator, same baseflow-separation convention — confirmed correct by reproducing Ian's own
28.9–31.4% exactly), Milton's observed RC at the original 0-120h window comes out to 14.2–15.2%,
not the 34.4–36.6% quoted earlier in this file. **That earlier figure cannot be reproduced and
should not be quoted further** — it predates any saved, re-runnable Milton script (this session's
`run_site3_milton.py` is the first), so its exact assumptions are lost. This is the same
unreproducibility problem this file already documented once for Ian's old 19.6% figure, now
found a second time, independently, for Milton — worth treating as a pattern (unrecorded ad hoc
calculations in this project have twice now produced a headline number nobody could later
reproduce) rather than two unrelated incidents.

**Where this leaves the magnitude gap**: real, still not closed, but smaller and better
characterized than before. **1.6–2.4x, not 2.3–4.5x**, once both sides are given a fair,
matched, non-truncated window. This does not reopen any of the seven ruled-out mechanisms — it
changes the target the next mechanism (if one is ever found) needs to explain, and it means any
future headline "shortfall" number should be quoted from the matched-extended window, not the
storm-duration window, since the latter is now a demonstrated artifact.

Reproducible via `simulation/run_site3_ian.py --cell-size 25 --dt 20 --extend-hours 96` and
`simulation/run_site3_milton.py --cell-size 25 --dt 20 --extend-hours 48` (both 25m, ~15-20 min
each). Neither run's flooded area had fully drained to zero even at 168h (Ian: still 560.9ha at t=166h;
Milton: still 582.7ha at t=166h, both still slowly falling) — a longer extension would likely
narrow the gap further still and has not been tried.

### MRMS gridded rainfall vs. the point gauge — tested, 2026-09-02, ruled out

Direct test of the leading untested lever named above: is the single point gauge (ASOS KSFB,
10.8km from site3) over-representing the real watershed rainfall relative to what actually fell
across Gee Creek's contributing area? New `site3_gee_creek/precipitation/fetch_mrms_site3.py`
pulls NOAA MRMS `MultiSensor_QPE_01H_Pass2` (radar+gauge multi-sensor QPE, ~1km, archived by IEM
back to 2022/2024 — NOAA's own near-real-time bucket only retains ~30 days, confirmed via a
direct bucket listing before switching sources) for every hour of both storm windows, clipped to
the real delineated watershed polygon (`dem/data/hydro/watershed.geojson`, 3.72 km²) rather than
just its bounding box.

| storm | window (UTC) | MRMS watershed-mean total | point-gauge KSFB total | ratio |
|---|---|---|---|---|
| Ian | 2022-09-28 00:00 – 2022-09-30 23:00 | 376.1 mm (71/72 hrs, 1 missing) | 391.7 mm | **0.960** |
| Milton | 2024-10-06 00:00 – 2024-10-10 23:00 | 282.1 mm (120/120 hrs) | 288.0 mm | **0.979** |

**Ruled out, and in the wrong direction to help even partially.** The independent radar-based
estimate is *slightly lower* than the point gauge on both storms (4.0% and 2.1% lower) — the
opposite of what would be needed to explain a ~2x volume overshoot via "the gauge over-reads
relative to the real watershed." If anything, this says the point-gauge rainfall input is
already a close-to-unbiased (marginally high, not low) estimate of true watershed rainfall on
both events. **This closes off the rainfall-input-representativeness hypothesis** named in "Where
this leaves the magnitude gap" above as the leading untested lever — seven mechanisms now tested
and ruled out, not six, all with clean, real, checkable numbers.

**Real, honest scope note**: this validates the total-volume input at the watershed scale, not
the *sub-hourly intensity distribution* MRMS resolves that a hourly point gauge could smear —
future-work item, not investigated here — nor whether MRMS's own radar-gauge blending has a
systematic Florida-convective-storm bias (the product's own literature-documented use case, not
independently checked against a second radar-based product in this review). Raw hourly grids
cached at `site3_gee_creek/precipitation/data/mrms_raw/` (gitignored, ~100MB, reproducible);
derived per-hour CSV and summary JSON are tracked at
`site3_gee_creek/precipitation/data/mrms_{hourly,summary}_{ian,milton}.{csv,json}`.

**Where this leaves the investigation**: with rainfall input now also ruled out, the two
remaining candidates named above are (a) the validation *methodology* itself — boundary-outflow-
vs-single-channel-gauge, already self-flagged as "structurally approximate" — and (b) an
independent, non-discharge signal (e.g. satellite/aerial flood-extent for either storm) to
cross-check against something that isn't the same gauge every other test has been measured
against. Neither has been attempted yet.

### Reviewed a published GPU/differentiable solver (Inunda) — one technique worth adopting, 2026-09-02

**Li, Z. (2026). "Inunda: A GPU-Native, Differentiable Solver for High-Resolution Flood
Inundation Modeling." arXiv:2607.09614.** Same local-inertial shallow-water family as this
project's own solvers, but implemented in PyTorch (autodiff-compatible) and calibrated via
gradient descent through the full unrolled simulation instead of hand-swept hypothesis testing.
Reviewed specifically against this project's own unresolved magnitude gap above.

**Not directly comparable on the metric that matters here**: every accuracy number the paper
reports (Hurricane Harvey high-water-mark MAE, stage NSE) is about water-surface
rise/depth, not runoff volume or discharge — it reports no mass-balance or runoff-coefficient
diagnostic at all, and its own Discussion admits the model "tends to under-spread relative to
observed inundation," consistent with a volume bias that a stage-only calibration target
wouldn't surface. This project's six-hypothesis magnitude-gap investigation above is explicitly
a volume diagnosis, a different and more relevant axis than anything this paper reports — its
Hurricane Harvey results should not be read as evidence the volume problem is solved elsewhere.

**One thing worth adopting, already independently identified as this project's own next step**:
the paper defaults to gridded/radar rainfall (NOAA MRMS QPE) rather than point-gauge input for
real events — external confirmation that pursuing independent gridded rainfall (already named
above as the leading remaining lever) is a reasonable direction, not a shot in the dark.

**Not adopted**: differentiable/gradient-based joint calibration of Manning's *n*, soil
parameters, and channel bed elevation is a real technique worth keeping in mind if this project
ever needs to search a parameter space faster than one hypothesis at a time — but any such
attempt should calibrate against a volume/discharge loss, not stage alone, given the gap this
review found in the reference paper's own methodology.

### Re-run required before any of these are quoted again

Both probability ensembles and the `--baseflow` experiment — neither has been re-run since the
2026-08-29 solver fixes. The storage-cap sweep and the surface-parameterization roughness A/B
(both spectral and SAM3 backends) **are now re-run, above and in Surface Parameterization
below** — do not treat the "even a zero storage cap only reaches 10.8 % runoff" figure as
current; it predates the fixes and is superseded by the 43.92 % uncapped-storage figure
measured above.

---

## Solver & Connectivity Fixes — what changed on 2026-08-27

This investigation was opened to test one hypothesis: that site3's burned creek channel had
*gaps*. It did
not. The vector network is continuous — 22.17 km inside the domain, both components reaching a
domain edge, stable across snap tolerances from 0.1 to 15 m. The hypothesis was wrong, and
testing it surfaced thirteen defects underneath, several of which had been silently degrading
every number this project has published.

**The dominant one: the solver destroyed its own DEM conditioning.** `flood_sim_ian.py`
bilinear-downsampled the conditioned DEM from 0.875 m to 5 m. richdem's breaching carves
drainage paths roughly one cell wide; any averaging kernel blends that trench back into the
surrounding ground, re-sealing every outlet the conditioning had opened.

| DEM | trapped depression storage |
|---|---|
| native 0.875 m conditioned | **0.000 Mm³** |
| 5 m bilinear — what the solver actually integrated | **3.710 Mm³**, 79.4 mm domain-average |

79.4 mm is 20 % of the Ian storm, and almost exactly the entire observed runoff volume. The
logged "fill adds zero volume" was true — at native resolution, which is not the surface the
solver reads. Fixed with `Resampling.min`; the conditioning now survives the downsample at both
sites. The identical bug existed in `research/build_grid_surrogate_dataset_site3_crop.py`.

**site3's stream burn had never run at all.** `HYDRO_GEO` was hardcoded to the main AOI, so
site3 loaded the main AOI's six Shingle Creek flowlines — whose EPSG:5070 extent sits ~34 km
south of site3's DEM. They rasterised to an all-zero mask and the burn became a silent no-op:
`dem_burned.tif` was bit-identical to the raw DEM. `dem_hydro.py` now resolves paths through
`site_registry` (`--site site3`) and raises on an empty burn mask.

**A constant-depth burn does not route.** Once site3 was burned correctly, D8 still fragmented:
the largest flow accumulation anywhere in the 46.78 km² domain was 0.99 km², and 191 of 399
sampled steps along the main stem ran *uphill* downstream. A uniform carve lowers a channel but
imposes no direction — on this terrain that is a flat-bottomed ditch. The burn now enforces
monotonic descent using 3DHP's own digitized flow direction, with depth scaled by stream order
and the enforced gradient measured from the DEM per order rather than assumed.

**The infiltration fix from the previous session reached only one of two sites.**
`soil/data/soil_storage.csv` existed for site3 but was never generated for the main AOI, so
every main-AOI number — including the flood-probability ensemble — was produced with the
unbounded infiltration that session diagnosed as a bug.

**`export_overlays.py` had been aborting since 2026-08-03.** It masked `hand.png` (512 px, from
`dem_hydro.PNG_SIZE`) with a waterbody mask built at its own `SIZE` (2048 px). The `IndexError`
fired partway through the export chain, so every export after it was skipped too and the hydro
overlays in `viewer/data/` were frozen for three weeks.

Also fixed: pour-point snapping took the *nearest* stream cell rather than the largest channel
(0.14 km² catchment against 33.15 documented); the stream-initiation threshold was a raw cell
count, so the same number meant 766 m² at site3 and 38,391 m² at the main AOI; `flow_dir.tif`
was documented as an output but never written; the fully-conditioned DEM D8 and HAND run on was
never saved, so those products were not reproducible from any file on disk; `src.nodata or
-9999.0` silently replaced a legitimate nodata of 0.0; `reproject` without declared nodata lets
`Resampling.min` *fill* nodata holes with surrounding terrain; and a hardcoded `area_capture_frac`
in the viewer export went stale the moment the delineation changed.

---

## Corrected baselines — do not calibrate against anything older

**Main AOI**, all three of its defects fixed and re-run end to end:

| | before | after |
|---|---|---|
| runoff coefficient | 2.93 % | **26.44 %** |
| peak boundary outflow | 35.2 cfs | 156.6 cfs |
| peak flooded area | 24.1 ha | 68.1 ha |
| any resolvable risk | 6.85 ha | **11.03 ha** |
| ≥1 %/yr | 3.98 ha | **5.97 ha** |
| ≥10 %/yr | 1.30 ha | **1.73 ha** |

26 % runoff on a 28.7 % impervious corridor is physically plausible where 2.9 % never was.
There is no gauge at this AOI (the 44x watershed mismatch), so this is a plausibility check,
not a validation.

**site3 / Gee Creek**, scored with `analysis/validate_gauge_site3.py`:

| | before | after |
|---|---|---|
| runoff coefficient (domain boundary) | 1.69 % | 8.90 % |
| rising limb vs gauge | 0.09 h* | 0.72 h |
| flooded area 24 h after rain stops | **+0.10 ha/hr, rising** | **−0.48 ha/hr, draining** |
| D8 stream cells | 1,719,908 (2.82 %) | 112,502 (0.18 %) |
| cells with HAND < 1 m | 77.2 % | 17.2 % |
| max flow accumulation in-domain | 0.99 km² | 7.12 km² |
| pour-point catchment | 0.14 km² | 3.72 km² |

`*` **below the gauge's own 0.25 h sampling interval, and therefore never claimable.** The same
resolution limit that disqualified the peak-argmax metric applies here. Only the 0.72 h figure
is resolved.

**Johns Lake is unaffected** — its solver reads the DEM at native 2.64 m with no downsampling
step. The resampling bug is confined to `cfx_sr417` and the research surrogate path.

## What is fixed, and what is still wrong

Fixed, decisively: **water no longer accumulates forever.** Flooded area was still *rising* 24 h
after rain stopped, extrapolating to ~80 days to clear. It now drains. That was the finding this
investigation was built on and it is closed.

Still wrong: **magnitude, by about 3x, and timing badly.** Runoff coefficient is 8.90 % against
an observed 28.9–31.4 %, and discharge at the gauge cell never peaks inside the 72-hour window
at all — it rises monotonically to 101.6 cfs at t=72 h, against an observed peak of 1,190 cfs at
t=37.5 h.

**The observed target itself was wrong in the docs.** The 19.6 % runoff coefficient quoted
throughout cannot be reproduced from the NWIS record under any standard baseflow or window
choice. Like-for-like over the simulated window it is 28.9–31.4 %. The model was always further
from truth than recorded. `analysis/validate_gauge_site3.py` now reports the full sensitivity
instead of a single unreproducible number; use it rather than quoting a figure.

## Next

1. **Why the gauge-cell hydrograph never peaks.** This is the sharpest open question and the
   one that most constrains everything else. Conveyance has been *ruled out* by measurement —
   along-channel bed slope on the solver grid is 1.94e-3, matching the natural 1.90e-3, and at
   the existing n=0.040 a channel depth of only 0.14–0.62 m yields 0.30–0.80 m/s, precisely the
   documented range for this stream class. Manning's *n* is not the bottleneck; do not tune it.
2. **Storage capacity, tested 2026-08-31 — ruled out at any physically plausible value.** See
   "Combined: ponded infiltration + capacity sweep" above: 0.5×-4× SSURGO capacity, with ponded
   infiltration on, tops out at 59.25 % runoff against an observed 28.9-31.4 %. Only literally
   infinite capacity gets close. Not the answer on its own.
2b. **Six mechanisms tested 2026-08-31, all insufficient or wrong-directioned — stop sweeping
    parameters.** Roughness, storage capacity, infiltration access, capacity+access combined,
    domain-vs-watershed area (wrong direction), and AMC antecedent-moisture calibration (closes
    ~1/3 of the gap at best, at a physically unrealistic extreme). See "Where this leaves the
    magnitude gap" above — logged as an honest open limitation. What would actually move this
    forward is new data (independent gridded rainfall) or a different comparison methodology,
    not another single-parameter sweep.
3. **Re-run Johns Lake's probability ensemble.** The main AOI's and site3's are both done
   (site3: 30.08 → 56.33 ha any risk, 18.53 → 33.54 at ≥1 %/yr, 8.47 → 13.68 at ≥10 %/yr, with
   peak depths falling as areas rise — the correct signature for removing fabricated depression
   storage). Johns Lake is verified free of the resampling defect but has not been re-checked
   against the burn and storage-cap fixes.
4. **Flood extent at site3 has no ground truth at all.** Peak flooded area moved 246 → 375 ha
   across this session's fixes with nothing to check it against. site3 has no PlanetScope or
   Sentinel-2 coverage; sourcing a single post-Ian scene would make extent falsifiable.

### Measured and settled — do not redo

- **The 3DHP flowlines are continuous.** Verified by node-graph connectivity, not `linemerge`,
  which splits at junctions and reports a misleading 16 components for a connected network.
- **A baseflow initial condition does not help.** Pre-filling the channel to the depth its own
  measured 45.2 cfs baseflow implies validates itself beautifully — 47.9 cfs simulated at the
  gauge cell at t=0 — and then makes the run worse: total infiltration rose 2.56 → 5.00 Mm³ and
  storm runoff reaching the gauge fell 0.189 → 0.080 Mm³. Zeroing the channel's soil storage (a
  perennial channel sits at the water table) was tried and changed nothing; only 3,504 of 8,846
  channel cells had any storage to zero. Kept behind `--baseflow`, off by default.
- **Boundary outflow vs gauge-cell discharge is not the explanation for the lag.** Measuring at
  the gauge cell was tried on the hypothesis that domain-boundary outflow unfairly charges the
  traverse to the box edge. It reports a *later* response, not an earlier one.
- Everything in the previous session's don't-redo list still stands: whitebox is not a richdem
  substitute (stream IoU 0.29), the peak-argmax metric is unresolvable, and `historical_20240212`
  is not a validation case.

### Environment

**`python3` is 3.9.6 and is the pipeline interpreter** — richdem, pysheds, rasterio, geopandas
all present. There is no `python3.9` on PATH; checking for one and concluding the environment
lacks 3.9 is a mistake worth not repeating. A separate 3.11 venv at `cfx_sr417/.venv` serves
`research/` only and carries torch.

## Surface Parameterization — segmentation-derived parameters

Built 2026-08-27 in `cfx_sr417/segmentation/`; full detail, method and limitations in
[`cfx_sr417/segmentation/README.md`](cfx_sr417/segmentation/README.md).

The pipeline is complete and runs end to end: LiDAR canopy height → 344k NAIP segments in 9
surface classes → a per-class parameter table → three rasters on the 5 m solver grid. Two
independent cross-checks pass — total impervious 21.8 % against NLCD's 18.7 %, open water 10.4 %
against 3DHP's mapped 13.6 %.

`run_sim` gained one optional argument, `manning_n`, promoting `MANNING_N` from a domain-wide
scalar to a spatial field. `manning_n=None` is the original expression, so every existing caller
is unaffected. Nothing else in the solver was touched.

**Re-run 2026-08-31 on the current (post-2026-08-29-fix) solver, site3 @ 25 m — the earlier
SUPERSEDED numbers below are replaced.** Both backends now controlled, sharing one loaded
DEM/Horton/storage/hyetograph per run, scored with `analysis/validate_gauge_site3.py`.

**Spectral backend, four arms** (`segmentation/run_site3_ian_segmented.py --cell-size 25`):

| | baseline | segmented *n* | + segm. impervious | vision Ks/Smax | observed |
|---|---|---|---|---|---|
| rising-limb error | 0.27 h | 0.30 h | 0.28 h | 0.51 h | — |
| runoff coefficient | 72.08 % | 71.45 % | 72.68 % | **60.92 %** | 28.9–31.4 % |
| peak outflow | 13,951 cfs | 11,461 cfs | 11,639 cfs | 9,909 cfs | — |

**Real SAM3 backend, three arms** (same script, `--param-tag _sam3`):

| | baseline | segmented *n* | + segm. impervious | observed |
|---|---|---|---|---|
| rising-limb error | 0.27 h | **0.22 h** | **0.20 h** | — |
| runoff coefficient | 72.08 % | 71.35 % | 72.60 % | 28.9–31.4 % |
| peak outflow | 13,951 cfs | 11,670 cfs | 11,847 cfs | — |

**Timing improves under both backends; magnitude does not, under either.** SAM3's roughness
field gives marginally better timing than the spectral one (0.20–0.22 h vs 0.28–0.30 h — both
near the gauge's 0.25 h resolution floor, so neither improvement is fully claimable), but runoff
coefficient sits at 71–73 % regardless of backend, roughness field, or impervious source — a
~2.3× overshoot against the observed 28.9–31.4 % that three independent methods now agree on:
the connectivity investigation's direct channel-slope/velocity measurement, the spectral
segmentation A/B, and now the real SAM3 segmentation A/B. **Do not tune *n*, with either
backend.**

**The one arm that moves runoff meaningfully is `vision Ks/Smax`** (60.92 %, still ~2× observed)
— not because it's a better roughness field (same segmented *n* as the other spectral arms), but
because it replaces SSURGO's Smax/Ks with the vision-derived soil route. This is a different
lever from the storage-cap sweep just above (which holds *n* and the Ks/Smax *source* fixed and
only scales SSURGO's magnitude) — the vision route changes both the infiltration rate and the
capacity's spatial pattern at once, so the two results are not directly comparable, only both
consistent with "the soil-water side of the model, not roughness, is where the remaining gap
lives."

### The transferable finding: nadir imagery puts forest roughness in the channel

Riparian canopy closes over Gee Creek, so NAIP cannot see the creek and the canopy-height model
correctly reports 12 m of tree above it. The classifier therefore called 58.9 % of mapped channel
cells `tree_canopy` and the **gauge cell itself 100 %**, putting n = 0.120 on the channel bed.
Measured cost: gauge-cell discharge collapsed 101.6 → **10.5 cfs**, a ~10× drop, while boundary
outflow fell only 23 %. A defect invisible in a domain-wide statistic dominated the validation
cell.

Fixed by extending the precedence rule the classification already used — a mapped feature
outranks a spectral inference — to the hydrography layer, forcing channel cells to n = 0.045
using the same 3DHP flowlines `dem_hydro.py` burns. Recovered gauge-cell discharge to 37.1 cfs
and runoff to 8.20 %. **A canopy roughness is not wrong for a forest; it is wrong for a channel**
(Chow's 0.10 for timber assumes flow among the trunks, "flood stage below branches"). Any
nadir-imagery parameterisation will make this mistake wherever vegetation overhangs conveyance.

### Vision-derived vs SSURGO Ks/Smax — a controlled comparison

Both soil routes built and run with the segmented Manning field held constant. Same AMC-III
factor applied to both, so dry Ks is compared against dry Ks.

| | SSURGO | vision | observed |
|---|---|---|---|
| runoff coefficient | 8.202 % | 8.249 % | 28.9–31.4 % |
| rising-limb error | 0.24 h | 0.54 h | — |
| peak flooded | 383.6 ha | 266.5 ha | — |
| **gauge-cell discharge** | **37.1 cfs** | **3.0 cfs** | 1,190 cfs |

**A tie on the water budget (0.6 % apart), 12x apart at the gauge cell** — despite fields that
share almost no spatial structure (conductivity *r* = +0.171, storage *r* = **−0.234**). At
392 mm the storm overwhelms both profiles, so basin runoff is set by what neither can hold; the
routing is where they part.

The mechanism is specific: SSURGO reports the LEAST storage where canopy is densest (140 mm at
>80 % canopy vs 234 mm below 20 %), because Florida's deep-rooted forest sits on hydric ground
with a shallow water table. The vision route infers a deep profile from deep roots — the
causality runs the other way. Since that forest is the riparian corridor, the vision route soaks
up water in the strip feeding the creek and sheds it in the suburbs draining to the box edge.

**Use SSURGO where it exists.** Where it does not — i.e. outside the US, which is the whole
"any coordinate" premise — expect a usable volume and degraded routing. Extent differs 30 %
between the two and site3 has no extent ground truth to adjudicate it.

### Canopy validated independently

`tree_canopy` is 44.4 % of the domain and drives the whole roughness change, so it was checked
against **NLCD Tree Canopy Cover 2021** (30 m, Landsat + FIA plots — no shared sensor or method
with airborne LiDAR). Domain-mean canopy fraction: **ours 34.1 %, TCC 33.9 %**; r = 0.751, bias
+1.6 pp, class agreement 78.4 % at a 50 % threshold. The aggregate is near-exact and the scatter
(MAE 17.5 pp) is about *where* canopy sits, not how much there is — TCC cannot resolve a
hedgerow, and its distribution saturates (p50 21 %, p90 95 %) where ours is continuous.

Capping tree fraction at TCC's wherever ours is higher, never raising it where ours is lower,
drops mean n from 0.0655 to 0.0543 — a strict one-sided bound putting **at most 44 %** of the
roughness change down to canopy placement error. `segmentation/validate_canopy.py`.

### Also measured, and settled

- **Surface storage is negligible here** — 1.84 mm domain mean, 0.47 % of the 392 mm storm. The
  raster is produced; wiring it would need a new solver term and is not justified at that size.
- **The 0.6 m impervious substitution changes nothing** — runoff coefficient moved −0.004 pp
  after the channel fix, −0.05 pp before it. NLCD's 30 m pixels smear road/roof imperviousness
  into cells the binary OSM mask already handles, so the coarse layer double-counts; the finer
  one is more nearly correct and immaterial.
- **The n-aggregation choice does not matter** — arithmetic and Horton/Einstein composites of the
  0.6 m classes onto 5 m cells differ by 1.4 %.
- **The LiDAR has no vegetation classes.** ASPRS 3/4/5 are absent from this acquisition (48 M
  points sampled: 61 % class 1, 33 % ground, 5 % building). The note above pointing at
  `classification_filter` for classes 3/4/5 was wrong; canopy comes from class-1 returns
  normalised against the bare-earth DEM.
- **SAM3 RAN — access approved and full scene complete (2026-08-28).** Weights are cached
  locally (3.4 GB); it runs from the 3.11 venv (transformers 5.16.1, torch 2.13 + MPS) as a
  standalone stage writing `landcover_0.6m_site3_sam3.tif`, which the 3.9 pipeline consumes
  unchanged. **121 tiles over the full 36 km2 scene in 25.4 minutes.** Two engineering notes:
  the vision encoder is run once per tile rather than once per prompt (`vision_embeds` in place
  of `pixel_values`) — **2.38x faster, verified bit-identical** — and tiles are checkpointed
  atomically, the same pattern `cache_bbox_points.py` uses.

  **Findings that do not depend on the solver and therefore stand:**
  - Object delineation is markedly better than the spectral backend — countable rooftop
    polygons, continuous curved roads, sharp shorelines vs blocky fragments. See
    `segmentation/data/backend_comparison_site3.png`.
  - **It over-detects buildings 2.4x**: 16.0 % of the domain `building_roof` against OSM's 6.8 %
    mapped footprint area and the spectral backend's 8.1 %. Total impervious 24.4 % against
    NLCD's independent 18.7 %.
  - It labels only what it detects, leaving **17.4 % of the scene unlabelled** (spectral: 0.5 %),
    and finds **no wetland at all** — that class comes from HAND, which SAM3 cannot see.
  - Agreement with the spectral map where both label a pixel: **69.8 %**. On the 5 m grid mean
    *n* = 0.0594 (spectral 0.0655, scalar 0.040) over 69.6 % classified cover.

  **Not yet done: the gauge score.** The SAM3 arm was run, but through the integration defects
  fixed on 2026-08-29, so its magnitude numbers are void along with everything else from before
  that date. It is listed in the re-run set above.

  **A correction worth keeping.** Early ponding in that arm was attributed to the impervious
  over-detection making ~20 % of cells near-frictionless. That was wrong: re-running the *scalar*
  baseline (n = 0.040, no SAM3 involved) reproduced the same behaviour, so the cause was the
  solver, not the parameter field. The over-detection is real; the consequence claimed for it
  was not.

- Original diagnosis, kept for the record: **blocked on access and Python version — not hardware.** `facebook/sam3` is
  `gated: manual` (HTTP 401 without an approved token), and `Sam3Model` ships only in
  transformers **5.x**, which requires Python >= 3.10. **There is no 4.58**; the series goes
  4.57.x -> 5.0.0, so it cannot be installed into the 3.9.6 pipeline interpreter. Run it from
  the 3.11 venv as a standalone stage writing `landcover_0.6m.tif` — the class raster is the
  contract, so the 3.9 pipeline consumes it unchanged. Weights are 3.44 GB against 17.2 GB of
  unified memory with MPS available; the MPS OOM recorded for the mesh GNN is a different
  workload and says nothing about SAM3.

### Open

- **Canopy is missing on the domain margins.** The cached LiDAR bbox is a true-north box in
  EPSG:2881 inset in a rotated EPSG:5070 domain, so 72.7 % of the domain has returns — but
  **97.4 % inside the delineated gauge watershed**, which is what the scored signal comes from.
  Re-caching at a larger radius would close it.
- **`segment_naip.py` only knows site3.** The main AOI has NAIP and LiDAR too and would need
  its own bbox cache built.

**Both surface parameterization and the connectivity investigation edit
`simulation/flood_sim_ian.py`.** They have coexisted cleanly — surface parameterization's
`manning_n` sits alongside the connectivity investigation's `gauge_rc`/`initial_h` and its
scalar branch leaves the original expression untouched — but there is no locking. Use targeted
edits, never a full-file rewrite, and re-read before editing.
