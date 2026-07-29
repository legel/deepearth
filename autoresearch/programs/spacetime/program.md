# Spacetime encoder — autoresearch loop

## Scientific evidence gate (binding before scale)

The fast probes below are **discovery instruments**, not confirmation.  They may rank hypotheses, but a probe
record must not be described as Earth4D science or promoted into a scaled run until it passes this gate:

1. **Measured state, not a proxy made from the query.** Forecast LFMC, weather/vegetation anomalies, or another
   independently observed state. GBIF collection day and occurrence count measure sampling time/effort and cannot
   establish ecological dynamics.
2. **Three-way split.** Search on train, select once on validation, and evaluate once after freezing on test.
   A future+new-place test uses `train = past & seen_place`, `test = future & held_place`; the other two quadrants
   are embargoed, never folded into `~test`. Fit coordinate ranges, normalizers, aggregates, effort models, and
   imputers on train only.
3. **Actual autoregression.** A causal-state claim must consume observed past state and roll predictions forward.
   Reading a positional table at `t-lag`, or directly classifying a timestamped coordinate, is a delayed/static
   basis—not memory and not autoregression.
4. **Fair controls and nulls.** Compare persistence, seasonal climatology, raw coordinates, RFF/SIREN or a
   matched-capacity MLP, and the same propagator without Earth4D. Include shuffled-history, time-reversal, and
   future-sentinel controls. All paired arms get identical data, initialization seeds, wall time, and tuning budget.
5. **Predeclared statistics.** Primary endpoint is 90-day LFMC MAE under the joint rolling
   held-site-with-history design; 30-day MAE is the powered secondary endpoint. Report RMSE and R² at both.
   Report 180-day temporal results, but treat joint 180-day results as lower-power secondary evidence (only
   53 test sites). Use at least five matched seeds and a site×year block bootstrap. Pass only if the lower
   95% confidence bound for improvement over the strongest fair baseline is above zero and the point estimate
   is at least 5%, with no registered capability regression.
6. **Fixed budget and immutable record.** Every selection run is 600 seconds (science.md rule 20). Record code,
   data SHA-256, split membership, seeds, config, signed per-arm outcomes, and all attempted variants; never select
   the maximum of repeated identical runs. Freeze a commit/config before opening test and confirm on a second
   region or later year before a scientific headline.

Start the real-data split and baseline audit on the official Globe-LFMC table with:

```bash
python3 -m autoresearch.programs.spacetime.science_gate \
  --download --json-out data/lfmc/earth4d_science_gate_dev.json
python3 -m unittest discover -s tests -p 'test_earth4d_*.py'
```

The audit reports random (interpolation diagnostic), spatial transfer, temporal future-time, and joint
spatiotemporal future-time partitions. These are necessary split checks, not sufficient proof of
autoregression. Random-split performance is never evidence for rule 1.

The gate collapses repeated site×species×date rows, then pairs each target with the closest strictly earlier
same-series observation at 30/90/180 days within a preregistered ±7-day tolerance. Validation/test are evaluated
in date order: reveal a measured LFMC value only after forecasts targeting that timestamp are scored, then allow
it as later origin state without updating weights. This is **rolling held-site forecasting with local history**,
not zero-shot site transfer.

Test metrics require both `--open-test` and a non-empty `--frozen-id`; omit both throughout model development.
The artifact records the data hash, gate-code hash, split/pair membership hashes, and frozen identifier. This is
a procedural lock: the final model ledger still belongs in append-only external storage.

**Pinned one-time baseline audit (2026-07-29; test opened; not an Earth4D result):** source commit `65e84f4`, CSV SHA-256
`b44bf99d…ce3a6`, 88,744 valid rows. The temporal gate has 57,359/15,785/15,600 train/validation/test rows;
its strongest train-only baseline is site×species×month (test MAE 17.35). The joint gate has
41,302/1,934/2,494 rows with strict chronology and zero site/coordinate overlap; its strongest
transfer-compatible baseline is species×month (MAE 20.48, RMSE 29.93, R² 0.468). The official random test
reuses 972 of its 974 sites in training and is interpolation only. After collapsing 88,744 rows to 81,039
visits, the joint rolling gate contains 25,721/1,262/1,553 train/validation/test pairs at 30 days,
14,964/812/938 at 90 days, and 7,091/532/602 at 180 days. Test persistence MAE is 19.50/33.56/35.62
respectively; those are causal baseline figures, not Earth4D performance.

**Current status:** data, split, rolling-pair, and causal-baseline gates pass; `earth4d_evaluated=false`.
The next experiment must give a recurrent transition only LFMC states observed by each origin, recursively
feed its own predictions at 30-day steps, and compare raw-coordinate, generic-PE, Earth4D, and no-history/null
arms under matched seeds and 600-second budgets. The current `causal_lags` branch is only a backward positional
basis; it can be an ablation inside that model, but it is not the autoregressive mechanism.

## Discovery goal
Make Earth4D **induct** biology across space-time and **forecast forward** — not memorize coordinates.
**Maximize one scalar: `st_gain`** = mean of the `*_spacetime_gain` deltas (capability WITH Earth4D −
WITHOUT, via `_ablate_spacetime`). A candidate is eligible for confirmation when `st_gain > +0.02`,
forecasting (B25/B31) is active and non-trivial, and no spacetime capability regresses. It is not done until
the scientific evidence gate passes.

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

## ② Pick — architecture, not knobs
One structural change per round that satisfies a science.md rule this encoder fails. Reject anything that leaves
the mechanism unchanged. Filters: upholds science.md · fair controls (untouched baseline + mechanism ablation) ·
beats a predeclared confidence interval. A fixed ±0.008 threshold is not a substitute for repeated paired runs.

## ③ Run — one variable, fixed budget
`VARIANT` = the champion path with your one structural change applied. `TAG` = `st_<short-name>`.
```
rm -f data/deepcal/prepared_*.pt                                                             # cache round-trip is lossy — rm before every run
python -m deepearth.autoresearch.programs.run_experiment VARIANT --st-gain --time_budget 600 --cache_dir data/deepcal --tag TAG > TAG.log 2>&1
```
`--st-gain` builds `st_gain` (a second eval under `ablate_spacetime`, the S0 instrument); `--time_budget 600`
enforces rule 20 even while the committed champion config still contains a conflicting 4,000-second value.
CONTROL = the identical 600-second command on champion.yaml → `CTRL.log`. Forecast levers
(S1) additionally set `data.holdout: temporal` in VARIANT (`time_axis` already true).

## ④ Measure — one command
```
python -m deepearth.autoresearch.programs.score --log TAG.log --encoder spacetime --champion CTRL.log --ensue-tag spacetime
```
Emits `st_gain` + Δ vs control · capability floor · per-benchmark Δ · trace→Ensue.
**Bottleneck to read** = the per-benchmark `*_spacetime_gain` deltas (WITH − WITHOUT Earth4D) — they show
which capabilities Earth4D does/doesn't carry. Isolation = `ablate_spacetime` (Earth4D ON vs OFF).

## ⑤ Decide
Keep as an exploratory hypothesis if `st_gain` rises on validation **and** the capability floor (B1, B5, B6, B8,
B23, B29, B39, B40, B34, B42, B50, B51, B26, B27, B28) does not regress. Call it confirmed only after the
scientific evidence gate above passes. Else: read the bottleneck, set the next hypothesis.

## Search space (axes, non-exhaustive — invert or invent past them)
| axis | rule | structural move |
|---|---|---|
| Objective | 1 | make Earth4D causal autoregressive — forecast future state from past (`data.holdout: temporal` + future-reconstruction loss) |
| Dense field | 24 | train the field decoder to infer every variable at every space-time point between sparse obs, forward in time |
| Relative path | 2b | replace the offset-hash with a physics-inspired 4D recurrence (4D-LSTM / GNN rollout) over a temporal window |
| Encoding fabric | 3–5 | position code fused with every token — but only in service of the above; static capacity alone tested neutral/harmful |

## Ensue (steps ① and ⑥, tag `spacetime`)
- **① READ** before picking: pull open hypotheses + logged dead-ends for `spacetime`; skip anything tried.
- **⑥ WRITE** after measuring: push `trace.json` (scalar, per-benchmark deltas, bottleneck) with a one-line
  verdict (kept / dead-end + reason). `score.py … --ensue-tag spacetime` does this.
