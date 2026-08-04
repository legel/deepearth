# Internship Flood Digital Twin — Project Audit

**Date:** 2026-08-03
**Scope:** `models/flood_hydrology` (Johns Lake) + `models/cfx_sr417_corridor` (CFX SR417,
including the `site1`/`site2`/`site3`/`site3_crop`/`site3_crop_coarse`/`site3_1house` registry).
**Nature of this document:** started as a read-only audit; §7's action items below were then
executed the same day (see "Status update" at the very end of this file for exactly what
changed on disk and in git as a result).

---

## 1. Git workflow — what's actually going on, and what to do about it

### What you asked
"Lance changed the upstream repo substantially... if we update git would it be local changes or
on public GitHub? ... please teach me the professional workflow... Lance moved our from main to a
models branch I think."

### What's actually true (verified via `git fetch` + `git log`, not assumed)

- The real repo root is `/Users/hqqq422/Desktop/deepearth`, and `origin` is
  `git@github.com:legel/deepearth.git` — this **is** the shared repo Lance also works in. It's
  not "public" in the sense of being open to strangers necessarily, but it's shared — anything
  you `git push` there, other collaborators can see and pull.
- Your current local branch (`feature/sr417-corridor-ecosystem`) sits at commit `15ed9d1`
  ("Johns Lake flood simulation and digital twin viewer (#22)") with a large amount of
  **uncommitted** work sitting on top of it in your working directory — every single file you've
  touched this internship, including all of `cfx_sr417_corridor`, is still just local, unsaved-
  to-git changes.
- You were right that Lance changed things substantially: `origin/main` has moved to
  `3c45b99 Merge deepcal into main: adopt the refactored DeepEarth (clean slate)` — a large
  rewrite of the core repo unrelated to your work.
- You were also right about the branch move: there is a real `origin/models` branch
  (`ab417e6 Add models/ projects on top of the merged DeepEarth core`) — this is where all 7
  `models/` sub-projects (including `flood_hydrology`) now live upstream, carried forward on top
  of the clean-slate rewrite, separately from `main`.
- **Good news, checked directly**: `git diff 15ed9d1 origin/models -- models/flood_hydrology`
  returns **nothing** — Lance's refactor didn't touch `flood_hydrology` at all; it was copied
  forward byte-for-byte. So your local uncommitted changes to `flood_hydrology` are not stale or
  conflicting — they apply cleanly as new work on top of what's current upstream.
- `models/cfx_sr417_corridor` does not exist anywhere on `origin` — it is 100% local-only,
  meaning it is currently **only as safe as this one laptop**. Nothing has been lost, but nothing
  is backed up either.

### The three concepts, explained plainly

| Action | What it touches | Reversible? | Visible to Lance? |
|---|---|---|---|
| `git add` + `git commit` | Only your local `.git` folder | Yes, easily (it's just a new local snapshot) | No |
| `git push` | Uploads your local commits to `origin` on GitHub | Harder to undo once others pull it | Yes, immediately |
| Creating/switching branches | Local unless you also push the branch | Yes | Only once pushed |

**Committing locally is always safe** — it costs nothing, creates a restore point, and does not
touch GitHub or Lance's copy of anything. Pushing is the action that actually publishes. The
standard professional pattern is: **commit early and often locally** (cheap, private, gives you
checkpoints to `git reset`/`git revert` back to if an experiment goes wrong), and **push/open a
pull request only when the work is in a state you're ready for someone else to see or merge**.
There is no "unlimited local commits, one deliberate push" downside — this is exactly how most
teams work.

### Recommended sequence, next session

1. `git add` + `git commit` locally on the current branch, right now, as a safety checkpoint —
   this does not touch `origin/models` or `origin/main` at all, it's purely local.
2. Decide the real target: since `models/flood_hydrology` is identical between your base commit
   and `origin/models`, and `models/cfx_sr417_corridor` doesn't exist upstream yet, the clean
   path is to create a new branch **off `origin/models`'s current tip**, then cherry-pick or
   re-apply your local `flood_hydrology` changes and the entire new `cfx_sr417_corridor` tree
   onto that branch — rather than trying to merge your stale `15ed9d1`-based branch into
   `origin/models` later and dealing with 188 commits of unrelated drift.
3. Only once you and Lance agree the CFX/site3 work is ready for review: `git push` the new
   branch and open a PR against `models` (not `main`) — matching how `flood_hydrology` itself
   was landed (`feature/flood-hydrology-gsdr-monthly` exists as a pushed feature branch on
   `origin` already, same pattern).
4. Before any destructive git operation (`checkout`, `reset --hard`, `clean`) on either project,
   run `git status` first — this is already standing guidance in your environment, worth
   repeating here given how much uncommitted work currently exists.

---

## 2. Per-project structure at a glance

| | `flood_hydrology` | `cfx_sr417_corridor` |
|---|---|---|
| Architecture | **Single AOI** (Johns Lake). No site registry — every script hardcodes one lat/lon. | **Multi-site registry** (`lidar/test_sites.py`): site1, site2, site3, site3_crop, site3_crop_coarse, site3_1house. |
| Disk size | 5.6 GB | 48 GB |
| Committed to git | No (branch exists upstream: `origin/feature/flood-hydrology-gsdr-monthly`, but current working copy has new untracked folders) | No, anywhere — `models/cfx_sr417_corridor` doesn't exist on any remote branch |
| Top-level data folders | `dem/`, `soil/`, `sentinel2/`, `precipitation/`, `gsdr/`, `floodplain/`†, `hydrography/`†, `infrastructure/`†, `lidar/`†, `ground_truth/`, `simulation/`, `viewer/` (†=new, untracked, mostly ported from cfx_sr417_corridor) | `dem/`, `soil/`, `precipitation/`, `hydrography/`, `floodplain/`, `imagery/`, `infrastructure/`, `lidar/`, `simulation/`, `boundary/`, `analysis/`, `planetscope/`, `viewer/`, plus the fully parallel `site3_gee_creek/` tree |
| Solver(s) | Grid-based LISFLOOD-FP (`flood_sim.py`) + a new GPU shallow-water benchmark (`torch_swe_benchmark.py`, built as a favor to inform cfx_sr417_corridor's GPU decision) | Same grid solver family (`flood_sim_ian.py`) **and** a from-scratch 3D unstructured-mesh shallow-water solver (`mesh_shallow_water.py`, 1,521 lines — the most complex script in either project) **and** a GNN surrogate (closed out with a documented negative result at full scale) |

**Cross-pollination is real and bidirectional**, confirmed via docstrings in both projects:
`infrastructure/fetch_roads_buildings.py`, `floodplain/fetch_fema_nfhl.py`, and the whole
`lidar/` point-cloud pipeline were explicitly ported **from** `cfx_sr417_corridor` **into**
`flood_hydrology`; the GPU benchmark thread in `flood_hydrology` was built explicitly **for**
`cfx_sr417_corridor`'s own GPU-vs-CPU decision. This is a sign the two projects have been
converging toward a shared toolkit organically — which is exactly the argument for formalizing
that convergence into a real shared library (see §5).

---

## 3. Dataset / layer / simulation consistency matrix

"✓ script" = a saved, reusable fetch script exists (coordinate-in reproducible). "✓ data only" =
the data exists on disk but no committed script produced it (not reproducible from scratch
today). "—" = does not exist for this site.

| Dataset | Johns Lake | CFX main AOI | site1 | site2 | site3 (Gee Creek) | site3_crop/coarse | site3_1house |
|---|---|---|---|---|---|---|---|
| DEM (USGS 3DEP) | ✓ script | ✓ script | shares main AOI DEM | shares main AOI DEM | **✓ data only, no script** | reuses site3 DEM | reuses site3 DEM |
| Raw LiDAR point cloud | ✓ script | ✓ script | own crop, script-driven | own crop, script-driven | ✓ script (`cache_bbox_points.py`, `download_laz_tiles.py` reused w/ new bbox) | reuses site3 cache | ✓ dedicated dense-cloud export |
| Precipitation | ✓ script (NOAA CDO + GSDR + Atlas 14) | ✓ script (ASOS MCO, cross-validated against ISM and rejected it) | shares main AOI | shares main AOI | **✓ data only, no script** (ASOS KSFB pulled ad hoc, no reliability cross-check like MCO got) | shares site3 | shares site3 |
| Soil/SSURGO | ✓ script | ✓ script | shares main AOI | shares main AOI | **✓ data only, no script** (only an *overlay-export* script exists, `export_ssurgo_overlay_site3.py`, which explicitly assumes a prior fetch that isn't on disk as a script) | shares site3 | shares site3 |
| NLCD impervious | ✓ script | ✓ script | shares main AOI | shares main AOI | **✓ data only, no script** | shares site3 | shares site3 |
| NAIP imagery | — (has Sentinel-2 instead, deliberate choice) | ✓ script | shares main AOI | shares main AOI | ✓ script (`fetch_naip_site3.py`) | shares site3 | shares site3 |
| Roads & buildings (OSM) | ✓ script (ported from CFX) | ✓ script (split into 2 scripts) | shares main AOI | shares main AOI | ✓ script (combined into 1 script — naming/structure drift vs. main AOI, not a gap) | shares site3 | shares site3 |
| FEMA flood zones | ✓ script (ported from CFX) | ✓ script | shares main AOI | shares main AOI | ✓ script (`fetch_fema_site3.py`) | shares site3 | shares site3 |
| Hydrography (3DHP) | ✓ script (ported from CFX) | ✓ script | shares main AOI | shares main AOI | ✓ script (`fetch_3dhp_site3.py`, newer FeatureServer) | shares site3 | shares site3 |
| USGS gauge validation | ✓ (Johns Lake outflow gauge, `ground_truth/`) | ✓ (Shingle Creek 02263800 — invalidated, 44× watershed mismatch) | — | — | ✓ (Gee Creek 02234400 — real 26.3× Ian response, chosen specifically to be gauge-matched) | — | — |
| Grid flood solver run | ✓ (`flood_sim.py`, 7 scenarios) | ✓ (`flood_sim_ian.py`, Ian) | — | — | ✓ (`run_site3_ian.py`, real Ian, compared to gauge) | — | — |
| 3D mesh shallow-water solver | — | — | ✓ (`mesh_shallow_water.py`) | ✓ (with engineered pond-outlet weir physics) | ✓ (demo-scale only — proven not viable at full 72hr/6×6km scale, see CLAUDE.md) | site3_crop/coarse: GNN training data source only, not run standalone | ✓ (single-house demo) |
| GNN surrogate | — | — | training-plumbing test only | — | — | ✓ trained + validated here (closed out: works at crop scale, fails 56× too slow at full site3 scale) | — |
| Viewer page | `flood_hydrology` `/` (Flask :5050) | `cfx_sr417_corridor` `/` (Flask :5051) | layers inside `/` | layers inside `/` | separate `/site3` page | not viewer-exposed (training-only) | layers inside `/site3` |

**Verified, not just narrated**: I directly grepped the viewer JS/HTML rather than trusting
CLAUDE.md prose. Main-AOI `index.html` layer config (`main.js`) has 11 core toggles (HAND,
Stream Network, Flow Accumulation, NAIP, SSURGO, Hydrography, FEMA, Roads & Buildings, CFX
Boundary, FEMA×HAND Risk, Ian Sim vs FEMA Extent, Ian Flood Animation) plus per-site-1/2 toggles
(dense cloud, hydrography shortcut, mesh SWE). `site3.html` has its own independent 11 toggles.
`flood_hydrology`'s `main.js` has an equivalent but smaller set (Hydrography, NAIP, SSURGO, FEMA,
Roads&Buildings, Flood Depth, Infiltration, S2 Ground Truth, LiDAR Point Cloud) — consistent with
its single-AOI, no-multi-site architecture. I also confirmed actual solver output files on disk
(`viewer/data/simulation_ian_*`, `swe_mesh_summary*.json`, `gauge_comparison_site3.json`) match
what CLAUDE.md claims — no discrepancy found between the narrative and the files.

---

## 4. Automation-completeness findings (the "coordinate in → dataset out" question)

**Core finding: the automation is real and mostly consistent for the *main* AOI of each project,
but breaks down for site3.**

1. **Site3 has no saved DEM, SSURGO, NLCD, or precipitation fetch script.** Its `dem/`, `soil/`,
   `precipitation/` folders contain real, on-disk data (a real `site3_dem.tif`, real
   `mukey_map.tif`, real `asos_hourly_SFB.csv`) but *no committed script produced them* — the
   main-AOI scripts (`dem/dem_download.py`, `soil/ssurgo_download.py`, `soil/fetch_nlcd.py`,
   `precipitation/fetch_asos_hourly.py`) were almost certainly invoked manually with a different
   `--lat/--lon/--radius_km`, and the resulting site3-specific invocation was never saved as a
   reusable script the way every other site3 fetch (NAIP, roads/buildings, FEMA, 3DHP) was. If
   site3's data were deleted today, there is no single command to regenerate it — you'd have to
   reverse-engineer the exact CLI arguments from `lidar/test_sites.py`'s recorded coordinates.
   This is the single clearest reproducibility gap in either project.
2. **`flood_hydrology` and `cfx_sr417_corridor` use structurally different automation models.**
   `flood_hydrology` has no site-registry concept at all — every script hardcodes Johns Lake's
   coordinates as a module-level default, overridable only via CLI flags per-invocation, with no
   file recording "here is what was actually run for this AOI." `cfx_sr417_corridor` solved this
   with `lidar/test_sites.py`, a single source of truth for every site's coordinates/radius/
   overridden data paths — but even there, only the LiDAR/mesh-solver code path
   (`droplet_flow_test.py`, `mesh_shallow_water.py`) actually reads that registry; every fetch
   script for DEM/soil/precip/imagery/roads still takes raw `--lat/--lon/--radius_km` flags and
   has no knowledge of the registry at all. So the registry exists but isn't the enforced entry
   point — it's easy (and apparently already happened, for site3) for someone to run a fetch
   script with hand-typed coordinates that drift from what's recorded in the registry.
3. **Site3's "overlay export" scripts implicitly assume upstream fetches exist**, without
   checking or documenting that assumption — e.g. `export_ssurgo_overlay_site3.py`'s own
   docstring says "site3 already has its own real SSURGO fetch," but there is no such fetch
   script on disk to point to. This is a documentation/traceability gap more than a functional
   one (the data is there and works), but it means a future contributor reading that docstring
   would go looking for a script that doesn't exist.
4. **Everything else checked out as genuinely consistent.** NAIP, roads/buildings, FEMA, and
   3DHP hydrography all have real, working, reproducible per-site fetch scripts for every site
   that has that dataset at all — these are not gaps, just naming-convention drift (site3 splits
   things differently than the main AOI in a few places — combined vs. split roads/buildings
   fetch, for instance — which is a style inconsistency, not a functionality gap).

---

## 5. Reorganization proposal (design only — not applied this session)

The current layout grew organically (Johns Lake → CFX pivot → site1/2 → site3 gauge search →
site3_crop for GNN training → site3_1house demo), and it shows: two independent per-category-
folder conventions (`<category>/data/` at the main-AOI level, `<category>/data/` again inside
`site3_gee_creek/`, with suffix-based filenames like `_site2`/`_site3`/`_site3_1house` doing the
disambiguation instead of directory structure), plus real cross-project duplication of fetch
logic between `flood_hydrology` and `cfx_sr417_corridor` that's currently handled by copy-paste-
and-adapt rather than a shared library.

**Proposed target shape** (for your approval before any of it is executed):

```
deepearth/models/flood_digital_twin/          # or keep as 2 separate projects, see note below
├── common/                                    # shared, tested-once library
│   ├── fetch_dem.py            (USGS 3DEP, takes lat/lon/radius_km)
│   ├── fetch_ssurgo.py
│   ├── fetch_nlcd.py
│   ├── fetch_naip.py
│   ├── fetch_precip_asos.py
│   ├── fetch_roads_buildings.py
│   ├── fetch_fema_nfhl.py
│   ├── fetch_3dhp.py
│   ├── fetch_lidar_laz.py
│   └── site_registry.py        # ONE registry format, ONE loader, used by every fetch/solver script
├── sites/
│   ├── johns_lake/
│   │   ├── site.yaml           # lat, lon, radius_km, gauge info, solver presets
│   │   └── data/                (dem/, soil/, precip/, ... — same sub-structure every site uses)
│   ├── cfx_main_aoi/
│   │   ├── site.yaml
│   │   └── data/
│   ├── cfx_site1/ ... cfx_site3_1house/
│   │   ├── site.yaml
│   │   └── data/
├── simulation/                                 # solver code, imports from common/, takes a site.yaml
├── viewer/                                      # one Flask app, one set of JS layer modules,
│                                                 # reads whichever site.yaml the user picks
└── docs/
    ├── CLAUDE.md            (one, not two 250KB+ narrative logs — see note below)
    └── audits/               (this file, and future ones, dated)
```

Key ideas, each independently adoptable (you don't have to take the whole thing at once):
- **One fetch library, parameterized by `(lat, lon, radius_km, out_dir)`**, used by every site in
  both projects instead of copy-pasted-and-adapted per project. This directly closes the site3
  gap in §4 — a fetch that's a real function call from a registry entry can't silently skip being
  saved.
- **One site-registry format** (a `site.yaml` per site, or one central file) that every fetch
  script *and* every solver *and* the viewer all read, instead of `test_sites.py` being read only
  by the LiDAR code path while other scripts take raw CLI coordinates.
- **Per-site data folders instead of suffix-based filenames.** Right now "which file belongs to
  which site" is encoded in filenames (`swe_mesh_frames_site2_high.bin`, `dem_site3.bin`) — a
  real, working convention, but it means every new site multiplies filenames across every shared
  `data/` folder. Isolating each site's data under its own folder scales more cleanly and makes
  "what does this site have" a single `ls`, not a grep across a dozen folders.
- **Whether to actually merge the two projects into one is a separate, bigger decision** —
  `flood_hydrology`'s single-AOI simplicity is arguably a feature for that specific site (Johns
  Lake never needed a multi-site registry), and merging risks over-engineering it. A lighter-
  weight version of this proposal: keep the two projects separate, but factor out the shared
  fetch scripts they've already been organically copy-pasting between each other into one
  importable package both projects depend on. This gets most of the reproducibility benefit
  without the larger repo-restructuring risk.
- **Split the two 200KB+ append-only CLAUDE.md files.** Both are genuinely valuable — they're a
  detailed, honest lab notebook with real negative results (the GNN closeout, the watershed-
  mismatch caveats) preserved rather than hidden, which is exactly the right instinct for
  scientific reproducibility. But at their current length they're hard to use as a map of
  "what's true right now." A lighter split — a short STATUS.md with current state + links, and
  the existing CLAUDE.md files kept as dated history/changelog — would preserve the audit trail
  while making "what's actually running today" fast to find.

---

## 6. Cleanup targets (sizes verified directly, nothing deleted)

| Item | Size | Where | Verified how |
|---|---|---|---|
| Duplicated LiDAR point-cloud binary | **1.5 GB** reclaimable | `flood_hydrology/lidar/data/lidar_pointcloud.bin` vs `flood_hydrology/viewer/data/lidar_pointcloud.bin` | MD5-checked: **identical**, `c0fcf7d9...` both files |
| Orphaned NAIP temp tiles | **1.9 GB** | `cfx_sr417_corridor/site3_gee_creek/imagery/data/_naip_tiles_tmp/` | `du -sh` confirmed 1.9G; confirmed the main-AOI equivalent (`imagery/data/_naip_tiles_tmp/`) is correctly empty (0B), so this is leftover cruft specific to site3, not a normal working directory |
| 3DHP Flow Network Derivatives archive | 880 MB (not 2.4GB as the whole `hydrography/` folder measures — corrected after direct verification) | `cfx_sr417_corridor/hydrography/data/fnd/3dhp_all_csv_FY26_FlowNetwork.zip` | `du -sh` on the actual zip file; per `TASK_3DHP_FLOW_NETWORK_DERIVATIVES.md`, this task was never executed — the zip has been sitting unused |
| Raw LAZ tile cache | 11 GB | `cfx_sr417_corridor/lidar/data/raw/*.laz` | Already gitignored; re-downloadable via `download_laz_tiles.py` if ever needed again — candidate for deletion once you're confident no further mesh rebuilds are coming |
| Site3 bbox point cache | 7.6 GB | `cfx_sr417_corridor/site3_gee_creek/lidar/data/bbox_cache/*.npz` | Same category as above — derived/regeneratable cache |
| Empty stray directory | 0 bytes, but confusing | `flood_hydrology/outputs/` | Confirmed empty; likely superseded by `simulation/outputs/` |
| `.DS_Store` / `__pycache__` clutter | negligible size | scattered across both projects (18 `.DS_Store` in cfx, 11 in flood_hydrology; several `__pycache__/`) | Already gitignored everywhere checked — cosmetic only, zero risk to remove |

**Total easy, low-risk reclaim if all of the above were addressed: ~3.4 GB immediately** (the
duplicate binary + orphaned temp tiles + unused zip), **plus ~18.6 GB of re-derivable cache**
(raw LAZ + bbox cache) if you're willing to accept re-downloading/rebuilding if ever needed again.

---

## 7. Recommended next-session sequence

1. `git add` + `git commit` locally on the current branch — a pure safety checkpoint, touches
   nothing on GitHub (see §1). Do this before anything else, including any cleanup.
2. Decide which pieces of §5's reorganization proposal you actually want (all, some, or "factor
   out shared fetch library only" as the lighter option) — this is a real design decision, not
   something to default into.
3. Fill the concrete automation gap from §4: write `fetch_dem_site3.py` /
   `fetch_ssurgo_site3.py` / `fetch_nlcd_site3.py` / a saved precipitation-fetch invocation for
   site3, following the exact same pattern already used for `fetch_naip_site3.py` /
   `fetch_fema_site3.py` — this is a small, low-risk, high-value fix on its own regardless of
   whether the bigger reorg happens.
4. Execute the cleanup items in §6 you're comfortable with (the duplicate binary and orphaned
   temp tiles are essentially risk-free; the raw LAZ/bbox caches are a judgment call about
   whether you'll need to rebuild meshes again).
5. Branch off `origin/models`'s current tip (not the stale local `15ed9d1` base) once you're
   ready to bring this work back into the shared repo, and push only when it's in a
   review-ready state.

---

## Status update (2026-08-03, same day) — what actually got done

Items 1, 3, and 4 above were executed the same session. Item 2 (reorg scope) and the "push"
half of item 5 were deliberately **not** done unattended — see notes below.

- **§7.1 (local commit)**: done. Two commits on `feature/sr417-corridor-ecosystem`: `c1802b9`
  (full checkpoint of both projects) and `b5331bb` (the automation-gap fixes below). Still
  100% local — nothing pushed.
- **§7.3 (site3 automation gap)**: done. Wrote `fetch_dem_site3.py`, `fetch_soil_site3.py`
  (SSURGO + NLCD), `fetch_precip_site3.py`, all reading coordinates from `test_sites.py`
  instead of hardcoding them a second time. Found and fixed two real bugs while writing these
  (not worked around): `ssurgo_download.py`'s `main()` was silently rasterizing every caller's
  soil map against the hardcoded MAIN-AOI DEM regardless of which site called it — now takes an
  optional `dem_path` param (default behavior unchanged for existing callers); `fetch_naip.py`
  never cleaned up its own `_naip_tiles_tmp/` scratch directory (root cause of the 1.9GB orphan
  in §6) — now removed after the final mosaic is written.
  **Verified, not just written**: ran `fetch_precip_site3.py` for real against a live backup —
  reproduced `asos_hourly_SFB.csv` **byte-for-byte identical** to the file already on disk
  (409.0mm total, 57.1mm/hr peak — matches CLAUDE.md's own figures exactly).
  **`fetch_dem_site3.py` verification surfaced a real operational risk, handled safely**: ran it
  against a pre-made backup of the real DEM; USGS 3DEP's 1m elevation service was down at the
  moment of the call (`Service is currently not available` on both the 1m and 3m attempts), so
  `dem_download.py`'s own pre-existing resolution fallback ladder (not new code — inherited,
  intentional behavior) silently produced a much coarser 10m/712x716 DEM and overwrote the real
  7810x7819 1m production file. Caught immediately (checked the output shape against the known-
  good meta.json) and restored from the backup — confirmed byte-identical via MD5 afterward.
  **`fetch_soil_site3.py` was deliberately NOT run** as a result: it rasterizes soil polygons
  directly onto whatever DEM is currently on disk, so running it during the same live 3DEP
  outage risked compounding the problem rather than verifying anything. Re-attempt DEM/soil
  verification once 3DEP's 1m service is confirmed back up — this is a live-service reliability
  finding, not a defect in the new script's logic (its structure was independently confirmed
  correct beforehand via the import-chain check in the prior commit).
- **§7.4 (risk-free cleanup)**: done. Deleted `flood_hydrology/lidar/data/lidar_pointcloud.bin`
  after confirming via MD5 it was byte-identical to `viewer/data/lidar_pointcloud.bin` (the copy
  `server.py` actually serves) — reclaimed 1.5GB. Deleted the 1.9GB orphaned
  `site3_gee_creek/imagery/data/_naip_tiles_tmp/`. **Total reclaimed: 3.4GB.**
- **§7.5 (branch off `origin/models`)**: half done, deliberately. Created a new local branch
  `cfx-models-base` directly off `origin/models`'s real current tip (confirmed via `git fetch`:
  `ab417e6 Add models/ projects on top of the merged DeepEarth core`), then cherry-picked both
  local commits onto it — both applied cleanly with zero conflicts (confirmed
  `models/cfx_sr417_corridor` doesn't exist anywhere upstream, so there was nothing to
  conflict with). This branch is ready to push whenever you decide it's PR-ready — **not pushed
  yet**, since pushing publishes to the same shared repo Lance works in, and that's a
  publish/visibility decision that should be a deliberate choice, not something done as a side
  effect of an audit follow-up.
- **§7.2 (reorg scope decision)**: intentionally not resolved unattended. Both repos are now
  committed locally, which lowers the risk of a full restructure somewhat, but actually moving
  ~54GB of production data across two live, actively-referenced folder trees (with hundreds of
  hardcoded relative-path references throughout the scripts) is still a large, multi-hour,
  failure-prone undertaking that deserves an explicit go/no-go rather than a default. The
  lighter option from §5 (factor out a shared fetch-library package both projects import,
  without moving any existing data folders) remains the lower-risk path if/when you want to
  proceed with either version of this.
