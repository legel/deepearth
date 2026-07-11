# autoresearch

Autonomous research on DeepEarth: improve the model, train for a fixed budget, score the full benchmark suite, keep
gains, repeat indefinitely. DeepCal is the first instance (California plant ecology).

**Objective:** maximize the **mean of all benchmarks** (`net_score` in `evaluate.py`) — currently ~50 metrics (B1..),
each natively in [0,1] (accuracy/F1/cosine/recall/AUC/skill/calibration). Report the harmonic mean (headline: no metric
may be sacrificed — lift the weakest) AND the arithmetic mean. **No individual metric may regress** to raise the mean.

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
