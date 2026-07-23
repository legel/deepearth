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

## ③ Run
`train.py champion.yaml <one toggle> --tag st_<hyp>` at the fixed budget (`time_budget_s`, rule 20),
`profile: true` on. Matched control = champion.yaml, same seed. Forecast hypotheses set
`data.holdout: temporal` (`time_axis` already true).

## ④ Measure — tools
- **Objective:** `python -m deepearth.autoresearch.programs.score --log <run.log> --encoder spacetime --champion <champion_run.log> --json trace.json`
  → prints `st_gain`, capability floor, per-benchmark deltas, and the bottleneck block.
- **Bottleneck (read this to choose ②):** per-benchmark ablation-sensitivity (Δpred when Earth4D zeroed),
  learned `freq_log_scale` (fine vs coarse resolution actually used), forecast loss on the temporal
  holdout. Emitted as `[profile] key=value` when `profile: true`.
- **Isolation:** the gains use `_ablate_spacetime` (Earth4D ON vs OFF) — wired by S0.

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
