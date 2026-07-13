# autoresearch

Autonomous research on DeepEarth: improve the model, train for a fixed budget, score the full benchmark suite, keep
gains, repeat indefinitely. DeepCal is the first instance (California plant ecology).

**Objective:** maximize the **harmonic mean of the CAPABILITY benchmarks** (`net_score` in `evaluate.py`) — the suite
is B1..B60 and climbing, each natively in [0,1] (accuracy/F1/cosine/recall/AUC/skill/calibration). Report the harmonic
mean (headline: no metric may be sacrificed — lift the weakest) AND the arithmetic mean. **No individual metric may
regress** to raise it.

**Scoring integrity (metrics are gaming-proof, audited 2026-07-13):** every benchmark's metric is chosen so a
no-information baseline scores ~0 and there is no artificial ceiling. (a) Community/species-distribution benchmarks use
a KL SKILL score `1 - KL(p‖q)/KL(p‖uniform)` (raw exp(-KL) had a 0.83-0.95 uniform floor — a broad guess won). (b)
Reconstruction benchmarks use CENTERED (anomaly) cosine — the modality's train-mean is subtracted so a constant
mean-prediction scores ~0 (raw cosine floors were 0.35-0.95 because embeddings share a mean direction). (c) The
ablation-delta / information-gain benchmarks (`*_gain`: B24, B56-B60) are DERIVED differences that isolate a mechanism's
contribution; they are reported as DIAGNOSTICS but EXCLUDED from the net (`is_diagnostic()`) — folding a small
difference into a harmonic mean double-counts its constituents and pins the north star near 0. Characterize any new
metric's floor on real data before trusting it.

**Conditional Bayesian queries (core capability):** DeepEarth must answer masked queries at any constraint level and
return calibrated posteriors that tighten as evidence is added. Canonical case = plant-pollinator (B41,B51-B54): given
a plant with everything masked → the marginal pollinator distribution across all spacetime; add geography → pollinators
at that location across all time; add time → only pollinators present then (out-of-season migrants vanish); given plant
+ pollinator + location, query TIME → a (start,stop) interaction-phenology interval (mixture-of-Gaussians, multi-modal
per year). Priors/posteriors must map to data. This rides the mask-anything `infer(given,targets)` PerceiverIO core;
distribution heads (species/pollinator distributions, temporal-interval posteriors) are the refinements. Modifying
core architecture (fusion, encoders, Perceiver core, heads, posteriors) to achieve this is fair game and expected.

**Benchmark heads (paradigm, 2026-07-11):** a benchmark may have a small supervised **head** trained JOINTLY with the
self-supervised masked-reconstruction objective on the TRAIN split. Head losses carry a **small total weight** so SSL
stays dominant (with dozens of benchmarks each head is tiny — no domination). At **test time there is NO fine-tuning**:
heads run immediate inference on the model's (real or *imagined*) embeddings. Capabilities still emerge organically —
SSL shapes the representation; the head only reads it, and imagined-vs-real tests (e.g. flowering B27) measure whether
the representation itself carries the signal. Keep head grad light (small weight, optionally stop-grad) so it never
hard-codes a supervised shortcut into the shared representation.

**Fair game (edit + commit):** `core/fusion.py` (main target), `encoders/spacetime/earth4d.py`,
`encoders/biological/phylogenomic.py`, `deepcal.yaml`, `evaluate.py` (benchmark definitions + heads),
`data.py`/`prepare.py` (integrate new datasets). Architecture, encoders, fusion, operators, Bayesian priors/posteriors,
GNN autoregressive rollout (GraphCast/GenCast), diffusion/recurrent decoding, optimizer, schedule, hyperparameters, and
the DATA itself are all in scope. Never tune a scoring definition to inflate a result — improve the model.

**Rules of the science:** every change respects `science.md`. Realize 100% of its principles; shortcomings are the
focus, opportunities the inspiration.

## The loop
1. Read `results.tsv` + the per-benchmark profile; form a hypothesis grounded in `science.md`.
2. Edit a model/config/data file; `git commit` (report every benchmark score in order + the means).
3. Train (`python -m deepearth.autoresearch.train`); budget is set in `deepcal.yaml` (`time_budget_s`).
4. Read `grep "^net_score:" run.log`; empty ⇒ crash (`tail -50`), fix at the root or revert.
5. Append to `results.tsv`. Keep the commit if the mean rose with no metric regressed; else revert.

**Simplicity wins** (science.md rule 19 — DeepSeek parsimony: terse code, minimal comments). **Never stop**: think
harder, read the papers, combine near-misses, try radical architecture. Launch dataset downloads/preprocessing when
they lift induction. New datasets must be >= US-national in extent (DeepCal then crops to California).
