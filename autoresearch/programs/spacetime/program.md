# Spacetime encoder — autoresearch loop

## Goal
Make Earth4D **induct** biology across space-time and **forecast forward** — not memorize coordinates.
**Maximize one scalar: `st_gain`** = mean of the `*_spacetime_gain` deltas (capability WITH Earth4D −
WITHOUT, via `_ablate_spacetime`). **Done when** `st_gain > +0.02`, forecasting (B25/B31) is active and
non-trivial, and no spacetime capability regresses.

> **Bootstrap (S0): `st_gain` does not exist yet.** The first task builds the instrument that *creates* the
> objective — the `_ablate_spacetime` flag + the `*_spacetime_gain` deltas. Until S0 lands the loop has no
> selection signal.

## Requirements (science.md — this encoder must satisfy)
| rule | requirement | status |
|---|---|---|
| 1 | **causal auto-regressive** model trained to **forecast future states from past states** | ✗ missing |
| 2a | absolute encoder (NeRF-like GIS) over GPS + timestamp | ✓ |
| 2b | relative encoder = **physics-inspired 4D-LSTM** over a context window going back in time | ✗ offset-hash only |
| 3 | positional encoding fused with **every** token — a unifying fabric | ✓ |
| 4 | fast / compute-optimized (CUDA kernels) | ✓ |
| 5 | large-scale capacity (≥100M params) | ✓ |
| 6 | parallelizable over subsets of geography and time concurrently | ~ |
| 24 | model the **dense 4D field** — infer every variable at every space-time point, sampling between sparse observations, **forward in time** | ~ decode-only, untrained |

`st_gain ≈ 0` (once measured) will mean the ✗ rows are unmet: Earth4D is a static positional lookup, not a
causal forecaster (1), the relative path has no temporal recurrence (2b), and the dense field is never
trained (24). The backlog closes those rows.

## Loop
```
   ┌──────────────────────────────  maximize st_gain  ───────────────────────────────┐
   │                                                                                  │
 ① READ ──► ② PICK ──► ③ RUN ──────► ④ MEASURE ──────► ⑤ DECIDE ──► ⑥ WRITE ──┐    │
   Ensue      next       A/B: 1 toggle    score.py         beyond noise      Ensue │    │
  (tag=st)   hypothesis  vs champion,    → st_gain +       & floor held?     trace │    │
  open + dead  from ⑤'s  fixed budget    floor + BOTTLENECK  keep : diagnose (tag=st)    │
             bottleneck                                                          │    │
   └──────────────────────────────────────────────────────────────────────◄─────┘    │
   └──────────────────────────────────────────────────────────────────────────────────┘
```

## ② Pick — preferences
Rank the backlog by the last trace's **bottleneck**, not by the scalar. Config toggles before code changes;
cheapest-highest-leverage first. One variable per A/B. Target the unmet science.md rows first (1, 2b, 24).

## ③ Run — one variable, fixed budget
`VARIANT` = champion.yaml with exactly one Levers-table change set. `TAG` = `st_<lever#>`.
```
rm -f data/deepcal/prepared_*.pt                                                             # cache round-trip is lossy — rm before every run
python -m deepearth.autoresearch.programs.run_experiment VARIANT --st-gain --cache_dir data/deepcal --tag TAG > TAG.log 2>&1
```
`--st-gain` builds `st_gain` (a second eval under `ablate_spacetime`, the S0 instrument); budget = the
champion.yaml `time_budget_s` (rule 20). CONTROL = the same on champion.yaml → `CTRL.log`. Forecast levers
(S1) additionally set `data.holdout: temporal` in VARIANT (`time_axis` already true).

## ④ Measure — one command
```
python -m deepearth.autoresearch.programs.score --log TAG.log --encoder spacetime --champion CTRL.log --ensue-tag spacetime
```
Emits `st_gain` + Δ vs control · capability floor · per-benchmark Δ · trace→Ensue.
**Bottleneck to read** = the per-benchmark `*_spacetime_gain` deltas (WITH − WITHOUT Earth4D) — they show
which capabilities Earth4D does/doesn't carry. Isolation = `ablate_spacetime` (Earth4D ON vs OFF).

## ⑤ Decide
Keep if `st_gain` rises beyond the single-seed noise floor **and** the capability floor (B1, B5, B6, B8,
B23, B29, B39, B40, B34, B42, B50, B51, B26, B27, B28) does not regress. Else: read the bottleneck, set the
next hypothesis.

## Levers (backlog — each closes an unmet science.md row)
| # | rule | bottleneck it targets | change | expect |
|---|---|---|---|---|
| S0 | — | no isolation metric exists | build `_ablate_spacetime` flag + `*_spacetime_gain` deltas | `st_gain` measurable (≈0 today) |
| S1 | 1 | no forecast objective → B25/B31 inactive | `data.holdout: temporal` + forecast reconstruction loss on the future split | B25/B31 active (0→>0), B23 ↑ |
| S2 | 24 | dense field never trained | wire `query_field` into the loss (`field_decode_weight`, default 0) | B29/B39/B40 dist-skill ↑ |
| S3 | 5 | absolute capacity hardcoded | expose `abs_spatial_levels`/`abs_temporal_levels`/`abs_log2_hashmap` (default 18/18/20) | B29/B40 fine-scale ↑ if starved |
| S4 | 2b | relative path is an offset hash, not recurrence | temporal-context recurrence in the relative encoder (new toggle, isolated) | B5/B8/B25 ↑ |

## Ensue (steps ① and ⑥, tag `spacetime`)
- **① READ** before picking: pull open hypotheses + logged dead-ends for `spacetime`; skip anything tried.
- **⑥ WRITE** after measuring: push `trace.json` (scalar, per-benchmark deltas, bottleneck) with a one-line
  verdict (kept / dead-end + reason). `score.py … --ensue-tag spacetime` does this.
