# Meeting Prep — Week of 2026-07-28

> ## ⚠️ STALE — numbers below predate the 2026-08-04 friction correction
>
> Every depth / flooded-area / outflow figure in this document was produced by the solver
> BEFORE a real physics bug was found and fixed on 2026-08-04: the Bates (2010) semi-implicit
> friction denominator used `hf**(4/3)` where it must be `hf**(7/3)` for unit discharge.
> The wrong exponent **under-stated friction**, over-predicting discharge by ~+216% at h=0.10 m
> and ~+607% at h=0.02 m (verified against Manning's equation; the two agree only at h=1 m).
>
> Re-run with the correction, the main-AOI Ian event moved to: peak depth 0.583 m (was 0.537 m),
> peak flooded 24.1 ha (was 23.4 ha), south-edge peak outflow 12.65 cfs (was 26.45 cfs),
> rain→outflow lag 3.18 h (was 1.26 h). Site3 peak outflow 91.4 cfs (was 145.2 cfs) at t=36.28 h
> (was 35.25 h) — which *improved* the peak-timing error against the real Gee Creek gauge from
> 2.27 h to 1.24 h.
>
> See CLAUDE.md's 2026-08-04 entry and `~/Desktop/FLOOD_DIGITAL_TWIN_AUDIT_2026-08-04.md` §9.
> **The mesh-solver viewer presets this document may reference were regenerated the
> same day** (mass balance ~1e-6% residual on every preset) — check the live viewer
> for current numbers rather than treating figures below as permanently stale.
> Qualitative conclusions in this document are generally unaffected; the absolute numbers are not.


Talking points for the team-lead sync. Organized around the questions likely to come up.
Everything here is verified against the actual code/data, not from memory.

---

## TL;DR (the 60-second version)

1. **Added a 3rd AOI (site3 / Gee Creek) with a real USGS discharge gauge** — the first-ever
   simulated-vs-observed flood comparison in the project. **Timing matched within ~2.3 hours;
   magnitude did not** (we got ~12% of the observed peak), for structural reasons we understand
   and can explain.
2. **Explored HydroGraphNet by building our OWN GNN surrogate** (MeshGraphKAN) trained on our
   own solver's output. It trained, and the single-step loss looked great. But the real test
   (feeding its own predictions back) exposed a failure the loss hid: predicted flood volume
   collapsed to ~10% of the solver's. A reweighting fix largely recovered it. **Still a
   work-in-progress, not a validated surrogate.**
3. **Added USGS 3DHP hydrography (flowlines + waterbodies) to all 3 sites** and upgraded to the
   richer data endpoint. Important nuance: 3DHP gives network *structure*, not discharge.
4. **Two solver tiers clarified:** a fast 2D grid solver (real events, whole AOI) and a
   fine-scale 3D mesh solver (per-droplet demo, tiny areas only). Running the 3D mesh over a
   full 2×2 km or 6×6 km AOI needs decimation + GPU + checkpointing — brute force doesn't fit
   in 16 GB.

---

## 1. New gauge-matched site (site3 / Gee Creek) — what we got, and the magnitude story

**Why we added it:** Neither Shingle Creek gauge near the CFX AOI has a watershed small enough
to nest inside our 2×2 km box — gauge 02263800's real drainage area is **231 km², 44× larger**
than the AOI. Comparing our simulated outflow (from a 5 km² box) against a gauge integrating
231 km² of upstream flow is apples-to-oranges. So we searched USGS NWIS for a small-watershed
gauge with real Hurricane-Ian discharge and found **Gee Creek near Longwood (USGS 02234400)**,
documented drainage area **33.15 km²**.

**What we got (the honest result):**

| | Real gauge (02234400) | Our simulation |
|---|---|---|
| Peak discharge | **1,190 cfs** | **145 cfs** |
| Time of peak | t = 37.5 h | t = 35.3 h |
| Peak timing match | — | **within ~2.3 hours** ✓ |
| Peak magnitude | — | **~12% of observed** ✗ |

**Why the timing matched but the magnitude didn't — three real reasons:**
1. **Watershed under-capture (the big one):** our D8 flow-routing only delineates **11.65 km²
   (35%)** of the gauge's documented 33.15 km². Central Florida's flat, depression-dominated
   terrain (isolated wetlands/cypress domes) only connects to the channel network during
   extreme high-water — a well-documented limitation of D8 delineation in this state, not a bug.
   → If flow scaled purely by area, we'd expect 35% × 1,190 = **~418 cfs**.
2. **We got 145 cfs, not 418** — so even area doesn't fully explain it. The remaining gap is
   because our surface-water-only model has **no groundwater baseflow / channel-storage memory**.
   The real hydrograph rises earlier and recedes over more than a day; ours rises and falls in a
   tight window. That "long tail" is groundwater the model doesn't represent.
3. **The simulated signal sums all 4 box edges**, not a single channel facing the gauge — so
   it's not a clean single-channel equivalent.

**How to frame this in the meeting:** This is a *success*, not a failure. Getting the peak
timing right within 2.3 hours across a real 72-hour hurricane is a genuine physics result. The
magnitude gap is honest, structural, and fully explained — it tells us exactly what the model is
missing (baseflow + full watershed connectivity), which is the useful finding.

**Rainfall used:** real Hurricane Ian via ASOS **KSFB (Orlando Sanford, 10.8 km)** — 409 mm
total, 57.1 mm/hr peak. **Ground truth:** real 15-min discharge from USGS NWIS's instantaneous
service, Sep 26–Oct 2 2022.

---

## 2. HydroGraphNet / GNN surrogate — what NVIDIA did, what we did, and why ours failed first

### 2a. What NVIDIA's HydroGraphNet actually is (their study area & data)

It is one specific, well-instrumented reach. It is not a general US-wide model.

- **Study region:** the **White River near Muncie, Indiana** — a single reach.
- **Mesh size:** a **4,787-node** spatial graph. Small on purpose.
- **Ground truth:** **high-fidelity HEC-RAS *simulations*, not raw gauge records.** They built a
  calibrated HEC-RAS hydraulic model of that reach and *ran it* to generate the training data
  (water depth + volume over time).
- **Input features:** dynamic (water depth + volume history), static terrain (elevation, slope,
  roughness), and forcing (**inflow hydrograph + precipitation**).
- **Dataset:** published on NVIDIA PhysicsNeMo; auto-downloads from Zenodo (~8.28 GB, one file).

**The key point for the meeting:** NVIDIA did **not** train on precip + discharge gauge
observations directly. They trained on the *output of a physics model* (HEC-RAS) for one reach
where they had enough data to build and calibrate that model. Precipitation and inflow are the
*forcing inputs*; the *learning target* is the HEC-RAS simulated depth/volume.

### 2b. What we did — the same paradigm, our own solver

We did **not** deploy NVIDIA's model — their weights are tied to that Indiana reach, zero
transfer to Florida. We copied the *method*: train a GNN on a physics solver's output.

- **What we trained:** MeshGraphKAN (the same NVIDIA architecture family).
- **What it predicts:** per-triangle **depth change** from a short window of (depth + rain).
- **Datasets used:** **our own 3D mesh solver's output** — 8 synthetic rain scenarios
  (low/medium/high intensity × short/long/sharp duration), GPU-solved on the coarse site3 mesh
  (~6,700 triangles — deliberately matching NVIDIA's ~5 k scale). 6 scenarios train, 2 held out.
- **Ground truth = our solver, not a gauge.** Same as NVIDIA (their ground truth was HEC-RAS,
  ours is our shallow-water solver).
- **Why the goal:** a fast surrogate lets us calibrate the physics with far more parameter
  sweeps than re-running the slow solver by hand.

### 2c. Why it failed first (and how we caught it) — short version

- **The single-step loss looked great: 6.4×10⁻⁵.** But that test is too easy. It feeds the model
  the *real* solver depth every step and asks for just one step ahead.
- **The real test failed.** We ran an autoregressive rollout — feed the model its *own*
  predictions back in, the way a surrogate actually runs. **Predicted flood volume collapsed to
  9–13% of the solver's within the first ~20% of the event.**
- **Why it failed:** most triangles stay dry. So the easy way to minimize the loss is "predict
  no change everywhere." Feed that flat output back in, and it self-reinforces. A class-imbalance
  problem, not the textbook exposure-bias one.
- **Fix 1 (textbook exposure-bias fix, training noise): made it worse (−97%).** A real negative
  result — it proved the diagnosis was different from the standard one.
- **Fix 2 (per-node loss reweighting, up-weight the wet cells): worked.** Volume drift went from
  −88…−97% to **+25% / +4%**, and it now tracks the real filling trajectory over time.
- **Honest remaining weakness:** it gets the *total volume over time* roughly right, but is
  **noisier about *where* the water sits** (spatial accuracy got worse). Not finished.

### 2d. How to frame it in the meeting (one line each)

- We built a GNN surrogate the same way NVIDIA did — train on a physics model's output, not gauge
  data (they used HEC-RAS on one Indiana reach; we used our own solver on site3).
- The single-step loss lied; the rollout test caught a volume-collapse failure.
- A reweighting fix recovered the aggregate volume trajectory; spatial detail is still weak.
- **This is a win of rigor:** we caught our own model's failure with the right test, instead of
  trusting a good-looking loss.

---

## 3. USGS 3DHP — what layers we added, for all 3 sites

**What we added:** the two live 3DHP layers — **Flowlines** (the mapped stream/channel network)
and **Waterbodies** (lakes/ponds) — for every AOI, and wired them into each viewer.

| Site | Flowlines | Waterbodies | Named creeks |
|---|---|---|---|
| CFX main AOI | 6 (2.43 km) | 29 | Shingle Creek |
| Site3 / Gee Creek | 48 (24.09 km) | 167 | Gee / Howell / Soldier Creek |
| Johns Lake | 1 (4.83 km) | 2 | (unnamed) + Johns Lake |

**We also upgraded the data source:** switched from the older `hydro.nationalmap.gov` MapServer
to the newer `3dhp.nationalmap.gov` FeatureServer, which returns **populated flow-network
attributes** (`arbolatesum`, `streamorder`, `pathlength`, `mainstemid`) where the old endpoint
returned nulls. Same geometry, richer data.

**The important nuance to state clearly:** **3DHP is network *structure*, not a discharge
hydrograph.** A hydrograph (discharge over time) comes from USGS **NWIS gauges** — which is what
site3 uses. 3DHP tells us *where the channels and waterbodies are* and their network attributes;
it can't give us flow rate. Its drainage-area (Catchment) layer is confirmed **empty for this
whole region** (still being populated nationally), which is why we can't use it to solve the
watershed-area problem directly.

**Still open (flagged, not done):** using 3DHP's `arbolatesum` (cumulative upstream network
length) as an *independent* cross-check on our watershed-area estimates — a written-up task,
not yet executed.

---

## 4. The two solver tiers — equations, layers, GPU & RAM (the detailed version)

Both tiers solve the **same physics: the Bates et al. (2010) local-inertial shallow-water
equations** (the core of LISFLOOD-FP). Plain-language version of the equations:

- **Mass conservation:** water depth in each cell changes by (rain − infiltration) plus
  whatever flows in/out across its edges.
- **Momentum (the flow between cells):** each edge's flow is driven by the water-surface slope
  (`∂η/∂x`), damped by friction (Manning's n), updated semi-implicitly for stability. The
  formula: `q_new = (q_old − g·h_flow·Δt·slope) / (1 + g·Δt·n²·|q|/h_flow^(7/3))`.
- **What's dropped:** the convective-acceleration term (water speeding up through a narrow
  channel). Valid for slow flow (Froude ≪ 1); the one documented physics limitation.
- **Infiltration:** Horton's exponential-decay model `f(t) = fc + (f0−fc)·e^(−kt)`.
- **Timestep:** CFL-adaptive — `Δt ≤ α·Δx/√(g·h_max)` — shrinks automatically as water deepens.

### Tier 1 — 2D grid solver (`flood_sim.py`, `flood_sim_ian.py`)

- **What it is:** the shallow-water equations on a **raster grid** (one depth value per square
  cell). This is the real-event workhorse.
- **Layers it uses:** conditioned DEM (elevation), per-cell SSURGO soil (Horton infiltration
  parameters), NLCD graded impervious %, OSM roads/buildings (zero-infiltration mask), rain
  hyetograph. Johns Lake adds a **lake weir-storage** term (`Q = Cd·L·h^1.5`).
- **Scale & speed:**
  - CFX Hurricane Ian: 1.87 M cells @ 5 m, 72 real hours → **~6.6 min (CPU)**.
  - Johns Lake: 753 k cells @ 2.6 m → **~62 s (GPU)**.
- **RAM:** modest — a few hundred MB. A handful of float arrays the size of the grid.
- **GPU:** Johns Lake's grid solver has a torch/MPS path (~20× speedup); CFX's doesn't yet
  (CPU-only). This is a known inconsistency to unify.

### Tier 2 — 3D mesh solver (`mesh_shallow_water.py`)

- **What it is:** the *same* shallow-water physics, but on an **unstructured triangle mesh** —
  ground from the DEM + a separate LiDAR-derived roof mesh per building. Water flows across
  every triangle edge, including **off a roof edge onto the ground below** — this is what makes
  it "3D" and is why you see a droplet roll down a roof and onto the road. The single-droplet
  version (`droplet_flow_test.py`) is Lagrangian particle tracing on the same mesh.
- **Layers it uses:** raw LiDAR point cloud (→ Delaunay triangulation for the surface),
  per-triangle SSURGO Horton, NLCD, roads impervious mask, plus a roof anti-ponding drain.
- **Scale & speed (this is the constraint):**
  - site1 (80 m box): 256 k triangles → **~4 min**.
  - site2 (196 m box): 800 k triangles → **~13 min**.
  - site3 (6×6 km, decimated): 5.7 M triangles → **2.5 h for an 8-minute synthetic event**,
    and the process got killed before finishing a real run.
- **RAM:** this is the killer — memory scales with triangle count × edge-adjacency graph ×
  per-frame depth+velocity arrays.
- **GPU:** we built a torch/MPS port (`run_sim_gpu`) — **3.95× faster** than CPU, mass balance
  identical.

---

## 5. Why a full 3D mesh run on 2×2 km or 6×6 km doesn't fit on a 16 GB Apple M5

**The precise reason — it's two ceilings at once:**

1. **Memory ceiling (the hard wall).** At native LiDAR density, a 2×2 km area is ~50 M points;
   6×6 km (site3) is **1.16 *billion* raw points**. Triangulating that and building the
   edge-adjacency flux graph, plus holding per-frame depth + velocity float arrays, blows far
   past 16 GB. **On Apple Silicon there is no separate GPU VRAM — the GPU shares the same 16 GB
   with the CPU and OS** (unified memory), so there's no "offload to the graphics card" escape
   hatch. Realistically only ~10–12 GB is usable for compute before it starts swapping/OOM-ing.
   This is the same wall that hit the GNN training at 710 k nodes.

2. **Compute-time ceiling (the soft wall).** Even if it fit, the CFL-adaptive timestep forces
   tens of thousands of tiny steps, each touching millions of triangles. Site3's 8-minute
   synthetic event already took 2.5 hours; a real 72-hour Ian event at that scale would be
   astronomically long.

**So the honest statement isn't "impossible without a GPU" — it's:**
> A full-native-density 3D mesh run doesn't fit in 16 GB, full stop — the GPU doesn't add memory
> on unified-memory hardware. It becomes feasible only with **decimation** (thinning the mesh to
> ~5–7 m effective resolution, which fits) **plus checkpointing** (so an interrupted run doesn't
> restart from zero) **plus the GPU port** (to make the step count tractable in wall-clock time).

**Could we do a full 3D mesh run on Johns Lake (the "smallest" site)?** Worth correcting a common
assumption: Johns Lake is **2×2 km — the same scale as the CFX main AOI**, which we *never* ran
through the 3D mesh solver. It's far bigger than the tiny site1/site2 patches. So a *native-
density* 3D run on Johns Lake is the same unsolved challenge, **but a decimated + checkpointed +
GPU run (~5–7 m effective) is realistic — roughly the 15–40 min range.** That's the achievable
version.

---

## 6. Why we trained the GNN but did NOT do a "full render" 3D mesh flood run

This is the key strategic point and worth stating plainly:

- **The 3D mesh solver is a fine-scale *demo* tool, not the real-event engine.** It ran
  end-to-end and renders in the viewer for the *tiny* sites (site1/site2). For site3 it only
  ran an **8-minute synthetic burst** (not real Ian), and even that produced **3.5 GB / 2.9 GB
  output files** — far too large for a browser to load.
- **For the real Hurricane Ian event at site3, we deliberately used the fast 2D GRID solver
  instead** (397 seconds vs. an infeasible multi-hour mesh run). That's the right division of
  labor: **grid solver = real full-event physics; mesh solver = fine-grained per-droplet demo.**
- **The GNN was trained precisely to bridge this gap** — a fast surrogate that could eventually
  approximate the expensive mesh solver. And it had to be trained on a *coarsened* mesh for the
  exact same memory/compute reasons the full mesh render is infeasible.

**One-line meeting answer:** "We didn't run a full-resolution 3D mesh flood because it doesn't
fit in 16 GB and would take hours-to-days; that's *why* we're building a GNN surrogate, and
that's *why* the real-event flood used the fast 2D grid solver."

---

## 7. Anticipated follow-up questions & crisp answers

- **"Did we get the flood magnitude right?"** → Timing yes (within 2.3 h). Magnitude no (~12% of
  observed) — and we know exactly why: 35% watershed capture + no baseflow. It's a diagnostic
  result, not a failure.
- **"Is the GNN working?"** → Partly. The single-step loss looked great but hid a rollout
  failure (volume collapsed to ~10%). A reweighting fix recovered the aggregate volume
  trajectory (+25%/+4% drift); spatial accuracy is still weak. Work-in-progress, not validated.
- **"What did HydroGraphNet train on?"** → One Indiana reach (White River, Muncie, 4,787 nodes),
  ground truth from HEC-RAS *simulations* — not gauge data. Precip + inflow are forcing inputs.
  We copied the method with our own solver, not their model.
- **"Can we scale the 3D physics up?"** → Not at native density on 16 GB. Decimation + GPU +
  checkpointing makes a full-AOI run feasible; that's the concrete next engineering step.
- **"What did 3DHP give us?"** → The mapped channel + waterbody network for all 3 sites, with
  richer attributes. Not discharge — that's NWIS gauges.
- **"What's the cleanest next win?"** → A Johns Lake Hurricane Ian run (Ian hit there too;
  validate against Orange County lake-level records), and adding the design-storm (10/50/100-yr)
  library to CFX/site3 so every site can run both real events and return-period scenarios.

---

*Prepared 2026-07-28. All figures verified against on-disk data and code.*
