# Spacetime encoder — evidence gate and split provenance

> **This is not an agent program.** The single agent and loop definition for this surface is
> **`autoresearch/spacetime/program/program.md`** — objective, lever families, diagnosis, experiment tracking. This file
> holds only the preregistered LFMC evidence gate and the pinned split/baseline provenance that the
> agent program's evidence standard refers to.
>
> A competing loop used to live here: a separate objective (`st_gain`), a separate instrument
> (`run_experiment … --st-gain` against `champion.yaml`, which trains the full fusion model) and a
> separate Ensue tag. It contradicted the agent program, which forbids training the fusion model and
> selects on capability records. It has been removed; two definitions of the same surface is how an
> agent ends up optimizing a scalar nobody is scoring.
>
> `autoresearch/main/program/autoresearch.md` is a DIFFERENT surface: the full-model loop over the whole B1..B60
> suite. Do not mix the two.

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
python3 -m autoresearch.spacetime.editable_files.lib.science_gate \
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

## Where each thing lives

| question | file |
|---|---|
| what do I run, and how do I pick a capability? | `autoresearch/spacetime/program/program.md` |
| what is the board, and what is excluded? | `autoresearch/spacetime/program/scorecard.md` |
| which probe modes move my capability, and where do I edit? | `probe_registry.py` (`--capability X`) |
| what must a result satisfy before it is science? | `autoresearch/spacetime/program/program.md`, "Evidence standard" |
| the LFMC gate's pinned data/split provenance | this file, above |
| the full-model B1..B60 loop (different surface) | `autoresearch/main/program/autoresearch.md` |

