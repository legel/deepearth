# Parallel-track backlog — index

Generated 2026-06-18 from team-lead meeting notes + Lance Legel's two emails
(2026-06-16/17) on Sentinel-2 precision concerns. See each track file for full
context — they're written to be read standalone in a fresh terminal session.

| Track | File | Domain | Depends on | Status |
|---|---|---|---|---|
| A | `track_A_sim_physics.md` | `simulation/flood_sim.py` — localized rain, infiltration toggle/viz | none | **Done 2026-06-18, independently re-verified** — code + sanity-check re-run confirmed (water flows downhill from hilltop, infiltration reduces runoff); a data-loss gotcha from single-scenario runs was found and repaired, see file |
| B | `track_B_viewer_ux.md` | `viewer/static/js/*` — mesh-projected textures, speed control, rain preview | loosely on A (can start now) | **Done 2026-06-18, spot-verified in code** — found 2 sim-physics issues during verification, written up as a Track A follow-up in `track_A_sim_physics.md` |
| C | `track_C_extreme_scenarios.md` | extreme-storm + drought scenarios, lake extent comparison | A (infiltration toggle) recommended first | **Done 2026-06-18, spot-verified in code/data** — true GSDR extreme is 1945-09-16 (245.6mm/24hr), not the 1960 event; see `CLAUDE.md` Track C |
| D | `track_D_ground_truth_data.md` | PlanetScope / drone RTK / USGS gauges / OWM seasonality / Clay FM | none | **Done 2026-06-18, independently re-verified** — see `ground_truth/track_D_findings.md` |
| E | `track_E_benchmark_literature.md` | HEC-RAS + HydroGraphNet writeup | none | **Done 2026-06-18, deepened 2026-06-18 PM** — re-fetched the exact 2 HEC-RAS URLs the team lead gave (not just generic pages); now has sourced equation/infiltration-method specifics and confirmed free-download status, see file |
| F | `track_F_repo_hygiene.md` | README refresh, commit backlog, .gitignore | none | **Partially done (verified 2026-06-18)** — stray CSV removed + .gitignore partially updated by someone, but README refresh and WIP commit still not done; see file for exact split |
| G | `track_G_aoi_expansion.md` | Expand 2x2km AOI to cover full Johns Lake (found by Track D — currently clipped on 3 of 4 edges) | none, but cascades through nearly the whole pipeline | **Validation passed 2026-06-18** — new 7.37×3.87km box has zero edge-clipping (re-confirmed by re-running the script); expensive part (steps 3+) still blocked on team-lead go/no-go on timing *and* data source |

## How to pick a track for a new terminal session

1. Open the track file directly — each one has its own Context/Why, current
   state (with file:line references), tasks, and verification steps.
2. Check the table above for dependencies before starting C (it benefits from
   A landing first, though it isn't strictly blocked).
3. A and D are the best two to run truly in parallel right now — different
   domains (solver code vs. data sourcing), zero file overlap.
4. Update this index (status column, if useful) as tracks complete — keep it
   short, this is a router, not a status report.

## Source material

- Team lead's meeting notes (verbatim, messy) and Lance Legel's two emails are
  not reproduced here — they live in the conversation that produced this
  backlog. Each track file extracts only the parts relevant to it.
