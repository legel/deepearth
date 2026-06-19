# Track E — Benchmark literature: HEC-RAS & NVIDIA HydroGraphNet

Repo: `flood_hydrology`. This track's research is already done (this file
mostly just needs to be read, not re-derived) — see "Findings" below.
Independent of every other track.

## Why

From the meeting notes:

> "Suggesting ask Claude to do a review of HEC-RAS [USACE flood] equation and
> compare where we are relative to them... and HydroGraphNet
> (github.com/NVIDIA/physicsnemo/.../hydrographnet) — does it have its own
> shallow water equation and the PyTorch setup like a loss function — review
> HEC-RAS and how much we want to incorporate (physics and gold standard in
> civil engineering); HydroGraphNet is a starting point where they have
> shallow water equation learnable by the model."

The goal: calibrate how much physics fidelity our current solver
(`simulation/flood_sim.py`) gives up relative to the civil-engineering
standard, and assess whether a learned-surrogate approach (like
HydroGraphNet) is a realistic near-term upgrade path.

## Findings

### HEC-RAS (USACE)

**Update 2026-06-18 — re-verified against the exact two URLs the team lead
gave** (`hec.usace.army.mil/software/hec-ras/2025/` and
`.../hec-ras/download.aspx`), replacing the earlier pass's generic-page +
general-domain-knowledge findings below with sourced specifics:

- **Accessibility — confirmed in the page's own words** (`download.aspx`):
  *"Use is not restricted and individuals outside of USACE may use the
  program without charge."* The only caveat: *"HEC will not provide user
  assistance or support for this software to non-USACE users."* So: free,
  open, no registration/approval gate — just no hand-holding.
- **Current stable release: HEC-RAS 7.0.1** (Windows, and a Windows+Linux
  variant), with 7.0 Example Projects; full archive back to v2.2 also hosted.
- **HEC-RAS 2025 is a separate, not-yet-production track**, confirmed via the
  `/2025/` page directly: entered **Beta in April 2026**, originally targeting
  a 1.0 release "late Fall 2025" (i.e. behind its own original schedule). The
  page's own words: *"Please don't"* use it for production work yet. Headline
  changes versus 7.0.1: redesigned UI, face-centric mesh generation (new
  quad/cartesian/triangular mesh types, replacing the old cell-centric
  approach), a new explicit solver with cloud/Linux support, 10–50× faster
  raster import/render, built-in OSM/USGS basemaps, a public API, and direct
  NLCD/NOAA/USGS data pulls. None of this changes the governing equations —
  it's a UI/performance/workflow overhaul, not a new physics model.
- **Governing equations — now confirmed against the actual docs** (HEC-RAS
  2D User's Manual via `hec.usace.army.mil/confluence/rasdocs/r2dum`, "Shallow
  Water or Diffusion Wave Equations" page), not just general domain knowledge:
  - HEC-RAS 2D offers a **toggle between two equation sets**: full
    **Saint-Venant / Shallow Water Equations (SWE)** or **Diffusion Wave**,
    both solved via an **implicit finite-volume scheme**.
  - Diffusion Wave explicitly **drops local acceleration** (∂v/∂t) **and
    convective acceleration** (v·∂v/∂x) — the manual's own words call these
    terms *"extremely important in order to model rapidly rising flood waves
    accurately."*
  - HEC-RAS's own guidance lists **8 specific cases where full SWE is
    recommended over Diffusion Wave**: highly dynamic flood waves (dam
    breach, **flash floods**), abrupt geometry contractions/expansions,
    very flat slopes (<1 ft/mi), tidal influence, wave propagation from
    structures, super-elevation around tight bends, detailed
    velocity/elevation near structures, and mixed subcritical/supercritical
    flow. **Two of our four scenario types are literally named "flash"** —
    by HEC-RAS's own published criteria, our use case is exactly the kind
    they'd steer toward full SWE, not the simpler default.
  - **Infiltration: HEC-RAS offers exactly three methods** — Deficit and
    Constant Loss, SCS Curve Number, and Green-Ampt. **Horton is not one of
    them.** Our solver's Horton model (`flood_sim.py`'s `HortonInfiltration`)
    is a legitimate, widely-used infiltration model elsewhere in hydrology
    (e.g. EPA SWMM) — this isn't a deficiency, just a genuine methodology
    difference worth being explicit about if ever directly comparing
    numbers against a HEC-RAS run.
- **Our solver vs. HEC-RAS, now more precisely placed:** `flood_sim.py`'s
  **local-inertia approximation** (Bates et al. 2010, LISFLOOD-FP) keeps the
  local acceleration term but drops convective acceleration — i.e. it sits
  *between* HEC-RAS's two options, strictly more capable than Diffusion Wave
  but short of full SWE. CLAUDE.md's existing caveat ("Peak velocities ~3.5
  m/s → Fr > 1 in channels → advection terms matter") is now backed by
  HEC-RAS's own stated rationale for why that dropped term matters, and by
  the fact that our flash-flood scenarios are explicitly one of the 8 cases
  HEC-RAS flags as needing the term we don't have.
- **Open follow-up (still not done):** actually download HEC-RAS 7.0.1 (not
  the 2025 Beta, per its own "don't use for production" notice) and run a
  full-SWE reference simulation on the Johns Lake AOI for a real
  apples-to-apples number against `flood_sim.py`'s flash scenarios
  specifically — that's now the most targeted version of this ask, since
  HEC-RAS's own guidance says flash floods are where the gap should show up
  most. Still a real time investment (DEM import, mesh generation, boundary
  conditions in HEC-RAS's GUI workflow) — keep this as Qin/Lance's explicit
  decision, not something to start opportunistically.

### NVIDIA HydroGraphNet (`NVIDIA/physicsnemo/examples/weather/flood_modeling/hydrographnet`)

Confirmed via the repo's README and file listing
(`README.md`, `train.py`, `inference.py`, `utils.py`, `conf/`,
`requirements.txt`):

- It's a **Graph Neural Network surrogate** — an autoregressive
  encoder-processor-decoder architecture operating on **unstructured mesh
  graphs**, not a differentiable numerical SWE solver.
- "Shallow water equation learnable by the model" (as the notes paraphrase
  it) is more precisely: physics enters only via a **physics-informed loss
  term** enforcing mass conservation (a volume-continuity inequality), plus a
  **"pushforward trick"** to limit autoregressive error accumulation over
  multi-step rollouts. It optionally swaps MLPs for Kolmogorov-Arnold
  Networks (KANs) for interpretability.
- Input features: node coordinates + static attributes (elevation, slope,
  roughness), dynamic time series (water depth/volume history), and global
  forcings (inflow hydrograph, precipitation).
- **Training ground truth is HEC-RAS itself** — the demonstrated case study
  (White River, 4,787 mesh nodes) was trained against **HEC-RAS-generated
  simulation output**, not observational/satellite data. Setup: configure
  `conf/config.yaml`, run `train.py`; data auto-downloads from Zenodo if
  missing; `inference.py` produces 4-panel comparison GIFs.

### Implication for our project

HydroGraphNet is **not a drop-in module** we can wire into `flood_sim.py` — 
it's a full graph-based model that needs **HEC-RAS-quality ground truth at
our specific AOI** to train meaningfully, which we don't currently have
(our solver IS the simulator; we have no independent high-fidelity reference
to train a surrogate against, short of actually running HEC-RAS ourselves —
see the HEC-RAS follow-up above). The realistic near-term value is
**conceptual, not code-reuse**: their physics-informed mass-conservation
loss is a reasonable pattern to borrow *if* we ever build a neural surrogate
for `_local_inertia_step_torch` down the line — but that's a future-state
idea, not a next sprint task.

## Recommended next step (if the team wants to act on this)

The actual decision point is: **is it worth setting up a HEC-RAS reference
run for Johns Lake** to (a) get a real fidelity-gap number against our
local-inertia solver, and (b) generate the kind of high-fidelity ground truth
that would make a HydroGraphNet-style surrogate trainable later? This is a
real time investment, not a quick task — recommend treating it as its own
scoped decision with Qin/Lance rather than starting it opportunistically.

## Files touched

None — this file is the deliverable for this track.
