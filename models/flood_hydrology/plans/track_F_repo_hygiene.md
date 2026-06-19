# Track F — Repo hygiene

**Status (verified 2026-06-18): partially done, started by someone without
updating this file.** Confirmed via `git status`/`git diff`:
- ✅ Task 3 done — `sentinel2/cloud_summary.csv` stray duplicate is gone.
- 🟡 Task 4 partially done — `.gitignore` now has `sentinel2/data/s2_*.tif`
  and a `cache/` pattern (plus a `.env` secrets exclusion that wasn't even
  asked for, a good catch) — but `OWM_cache/` is still untracked/not
  ignored, and `../../cache/` is outside this repo entirely so it can't be
  gitignored from here regardless (that part of the original task was based
  on a misread — it's not "this repo's problem").
- ✅ Task 5 done — `CLAUDE.md` has the `plans/00_INDEX.md` pointer.
- ❌ Task 1 not done — `README.md` is unchanged, still GSDR-only.
- ❌ Task 2 not done — no new commits; still the same 5 commits as session
  start, all the WIP is still uncommitted.

Repo: `flood_hydrology`. Touches `README.md`, `.gitignore`, and git commits
only — no source/logic changes. Lowest priority of all tracks; do anytime,
doesn't block anything else.

## Why

`git status` currently shows 50+ modified files and a dozen+ untracked files
sitting uncommitted on `feature/flood-hydrology-gsdr-monthly` (9 commits
ahead of `origin/main`), a stray duplicate CSV already flagged in CLAUDE.md,
and a top-level `README.md` that documents only the GSDR precipitation
dataset — not the full DEM/Sentinel-2/soil/simulation/viewer pipeline the
project has grown into.

## Current state (confirmed by reading `README.md` and `git status`)

- `README.md` at repo root is a well-written doc, but it's scoped entirely to
  the GSDR Global Sub-Daily Rainfall dataset (setup, data structure, query
  scripts) — it does not mention DEM processing, Sentinel-2 water mapping,
  SSURGO soils, the flood simulation, the viewer, or `verify_all.py`'s 53
  checks (all of which are extensively documented in `CLAUDE.md` instead).
- `verify_all.py` / `verify_summary.txt`: 53/53 checks passing as of last
  run — DEM (5), Sentinel-2 (21), soil/precip (9), flood simulation (18).
- `sentinel2/cloud_summary.csv` (top-level, untracked) is a stray duplicate
  of `sentinel2/data/cloud_summary.csv` with an old/different schema —
  already flagged as item 4 in CLAUDE.md's OPEN list.
- New untracked dirs from recent work: `../../cache/`, `OWM_cache/` — these
  look like cache directories that shouldn't be committed at all.
- `.gitignore` currently doesn't exclude `sentinel2/data/s2_*.tif` (large
  raster files) — already flagged as item 10 in CLAUDE.md's OPEN list.

## Tasks

1. **Refresh `README.md`.** Either expand it to briefly cover the full
   pipeline (DEM → Sentinel-2 → soil → simulation → viewer) with a pointer to
   `CLAUDE.md` for full detail, or keep it GSDR-focused but add a clear
   top-of-file note + link: "this covers the GSDR precipitation module only —
   see CLAUDE.md for the full flood_hydrology pipeline." Either is fine; just
   don't leave it implying GSDR is the whole project.
2. **Commit the WIP in logical chunks**, not one giant commit — group by
   directory/feature (e.g. one commit for `dem/` changes, one for
   `sentinel2/` changes, one for the new `precipitation/fetch_*_gauge.py`
   scripts, etc.). Check `git diff --stat` per directory first to see what's
   actually changed before grouping.
3. **Remove the stray duplicate**: `rm sentinel2/cloud_summary.csv` (the
   top-level one — keep `sentinel2/data/cloud_summary.csv`).
4. **Update `.gitignore`**: ~~add `sentinel2/data/s2_*.tif`~~ done; still need
   `OWM_cache/`. `../../cache/` is outside this repo's tree — not fixable
   from `.gitignore` here, drop it from scope.
5. **Add the cross-reference**: a single line in `CLAUDE.md`'s "OPEN — next
   steps" section pointing to `plans/00_INDEX.md` for the active backlog
   (this was already added as part of generating this plan set — confirm
   it's there, don't duplicate it).

## Files touched

`README.md`, `.gitignore`, git commit history (no source file content
changes beyond the one `cloud_summary.csv` removal).

## Verification

- `git status` after commits shows a clean or near-clean tree (only
  genuinely new WIP remaining, not the current 50+ file backlog).
- `README.md` no longer reads as if GSDR is the entire project.
- `git log --oneline -10` shows sensible, scoped commit messages rather than
  one undifferentiated dump.
