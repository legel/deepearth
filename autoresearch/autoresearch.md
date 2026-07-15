# autoresearch

Autonomous research on DeepEarth: improve the model, train for a fixed budget, score the full benchmark suite, keep
gains, repeat indefinitely. DeepCal is the first instance (California plant ecology).

## Branch & commit rules (Ensue collaboration)

All Ensue autoresearch work lives on the **`deepcal-ensue-autoresearch`** branch of `github.com/legel/deepearth`
(branched from Lance's `deepcal`). Never commit to `deepcal`/`main` directly; propose changes via PR from this branch.

- **Commit the current best, not scratch.** `autoresearch/deepcal.yaml` on this branch mirrors the reigning champion
  config (the settings that produced the best arithmetic mean). Bump it when a new champion is promoted, and put the
  score + Ensue result key (`results/deepcal-loop/...`) in the commit message.
- **Never commit secrets or bulk artifacts.** `.env` / API keys / Ensue tokens are gitignored — keep it that way.
  Do NOT commit data (`data/`, `*.npz`, `*.pt`, prepared caches), the `deepearth` self-symlink, or `__pycache__`.
  Code + config + docs only; the swarm reproduces data from recipes, not from git.
- **Data-channel recipes** (`autoresearch/recipes/`): the extra environmental channels (phenology, rsveg, ...)
  are reproduced from committed recipes, never from git-tracked data. `build_variables` skips any config
  variable whose channel is absent from the cache, so the champion config stays runnable on the standard data
  download — a recipe is only needed to *enable* a channel. Run `recipes/build_phenology.py` to reproduce the
  champion exactly.
- **Code levers stay default-safe** (see "Architecture search surface"): any toggle a box adds defaults to current
  behaviour, is committed here, and synced byte-identical to every box.
- **Commit granularity:** one logical change per commit (a data channel, a toggle, a champion bump); the message
  states what changed + the measured arithmetic-mean delta. Push to `origin deepcal-ensue-autoresearch`.

**Objective:** maximize the **harmonic mean of the CAPABILITY benchmarks** (`net_score` in `evaluate.py`) — the suite
is B1..B60 and climbing, each natively in [0,1] (accuracy/F1/cosine/recall/AUC/skill/calibration). Report the harmonic
mean (headline: no metric may be sacrificed — lift the weakest) AND the arithmetic mean. **No individual metric may
regress** to raise it.

**Selection signal (operator-authorized 2026-07-15): you MAY optimize and select on the ARITHMETIC mean.** The
harmonic net is dominated by a few near-zero benchmarks (community recall B20-B22) whose single-seed variance
(0.003<->0.016) swings it 0.02<->0.13, so it is noise for single-seed A/B. The arithmetic mean is the stable
discovery signal: keep/promote candidates on arithmetic mean. Still report BOTH means, and no individual metric
may regress to raise it.

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

## Architecture search surface — go the whole nine yards (swarm addendum, permanent)

### ARCHITECTURE-FIRST MANDATE (operator directive 2026-07-15 — this reorders the whole loop)
The loop's PRIMARY job is to find the best **reasoning STRUCTURE**, via LARGE structural variation across every
layer of the model stack — NOT hyperparameter tuning. Config-knob sweeps (`sdist_weight`, `capacity`,
`contrastive_*`, `hide_prob`, `poll_weight`, head weights, data-channel on/off) are **diminishing-returns MANUAL
work done AFTER an architecture is confirmed** — not the loop's main effort, and must not dominate the candidate
queue. Knob-tweaking plateaued deepcal at ~0.5718 precisely because it never changed how the model reasons. Bias
hard toward BIG structural bets over small deltas.

**Vary each layer with genuinely-different implementations (not width/depth knobs):**
1. **Tokenization** — bare Linear vs MLP vs patch/set/perceiver tokenizer; per-modality frequency features;
   shared vs modality-specific.
2. **Position / space-time encoding** — Earth4D hash vs RFF vs spherical-harmonic / SatCLIP / sinusoidal; how
   absolute/relative/neighbor geometry is combined and gated.
3. **The cross-attention READ (`self.read`)** — Perceiver latents←tokens: softmax MHA vs gated/linear variants;
   iterative vs single; latent count/structure; routing.
4. **The latent BLOCK internals (`self.blocks`)** — replace `nn.TransformerEncoderLayer` with configurable
   attention (softmax / rotary / gated), FFN (MLP / SwiGLU / GeGLU), norm (LayerNorm / RMSNorm), norm placement;
   or non-attention blocks (state-space / MoE).
5. **Fusion topology & iterative refinement** — state/context/neighbor interaction; write-back / revise / rounds
   / deep supervision; diffusion / recurrent / autoregressive decoders.
6. **Decode** — attention-pooling vs cross-attention decode vs dense field; per-variable vs shared heads.
7. **Scaling** — d_model / n_latents / n_layers / n_heads AS SCALING LAWS under the step budget, not one-off knobs.

**Mechanism (how the loop searches architectures):** every architectural choice is a PLUGGABLE component selected
by a `deepcal.yaml` key that DEFAULTS to current behaviour (pulling the code is a no-op until a config sets it).
The candidate generators sweep these ARCHITECTURE keys as their TOP tier. A new architecture needing new code is
(a) implemented default-safe, (b) validated in an ISOLATED git worktree that it REPRODUCES the champion (~0.5718)
at its default config — proving the refactor caused no regression — THEN (c) merged to the shared loop and swept.
Register every architecture key in the table below. Confirm a winning ARCHITECTURE first; only then hand its
hyperparameters to a knob sweep.

Do not settle for tuning mechanism knobs. Search the ENTIRE architecture — tokenizers, embeddings, encodings,
fusion, operators, heads — to find the absolutely best model. The surface, grouped (config = a key in
`deepcal.yaml` `model:`; code = edit `core/fusion.py` / `encoders/*`):

- **Embeddings & latent capacity** (config): `d_model` (token/latent width), `n_latents` (Perceiver latent bank
  size), `manifolds` (phylo/biological subspace width), `decoder_hidden`. Wider is not free under a fixed step
  budget (fewer steps) — sweep, don't assume.
- **Positional / space-time encodings**: config — `capacity` (Earth4D relative hash levels), `relative_window`
  (4D neighbor half-window N/E/up/days), `time_km`, `time_axis`. Code (not yet config) — absolute Earth4D
  `spatial_levels` / `temporal_levels` / `log2_hashmap_size`, relative `finest` resolution, `smooth_geo`
  `per_scale` width + sigma bank. This is the raw "encoding" surface.
- **Modality tokenizers / encoders** (code, `fusion.py` `self.encoders`): each variable is currently a bare
  `nn.Linear(dim, d_model)` (continuous) or `nn.Embedding(num_classes, d_model)` (categorical). Deeper/normalized
  encoders (MLP depth, LayerNorm, per-modality frequency features, learned/patch/set tokenization for the
  vision & NAIP embeddings) are wide-open architecture — a primary lever, not yet exposed as config.
- **Fusion / processor depth & operators**: config — `n_layers` (processor depth), `heads`, `rounds` +
  `revise`/`round_loss`/`write_back` (iterative refinement), `species_graph` operator/`layers`/`heads`,
  `flex_attention`. Code — cross-attention topology, latent routing/allocation, decoder families
  (diffusion/recurrent), autoregressive GNN rollout.
- **Mechanisms & heads** (config): `sdist_weight`, `comm_attached`, `poll_weight`, `contrastive_*`,
  `smooth_geo*`, `feedback_detach`, `learned_mask`, `loss_weights`, per-benchmark head weights.

### Reproducibility across the swarm (MANDATORY — this is the forever rule)
Every box shares one leaderboard, so a result is only trustworthy if any box can reproduce it. If you try
something the other boxes do NOT have natively (i.e. it needs a code change, not just a config value):
1. **Make it a config toggle that DEFAULTS to current behaviour** — `m.get("<key>", <current_default>)` — so
   pulling the code changes nothing until a config sets the key. Never a bare hardcoded edit.
2. **Commit the toggle** to `core/fusion.py` / `encoders/*` and **sync the changed files to every box** (all
   boxes must run byte-identical code — verify with `md5sum`).
3. **Document the new lever in the registry below** (key, file, default, effect) so any box can drive it from
   config alone and the candidate generators can sweep it.

Never run a code-level experiment that exists only in one box's working tree — it is unreproducible and pollutes
the shared leaderboard. Un-toggled code patches still under development run in an ISOLATED repo copy / git
worktree, not the shared loop repo, so concurrent candidate runs are never contaminated.

### Registry of added config toggles (append one line per new lever)
| key | file | default | effect |
|---|---|---|---|
| `comm_attached` | `core/fusion.py` | `false` | let the community-head loss shape the backbone (un-detached) |
| `mod_encoder` | `core/fusion.py` | `"linear"` | modality tokenizer for continuous vars: `"linear"` (bare Linear) / `"mlp2"` (2-layer MLP) / `"mlp2ln"` (+LayerNorm) |
| `block_ffn` | `core/fusion.py` | `"torch"` | latent self-attn block: `"torch"` (nn.TransformerEncoderLayer, champion-identical) / `"mlp"` / `"swiglu"` / `"geglu"` (configurable pre-norm LatentBlock) |
| `block_norm` | `core/fusion.py` | `"ln"` | LatentBlock normalizer (only when block_ffn≠torch): `"ln"` (LayerNorm) / `"rms"` (RMSNorm) |
| `read_depth` | `core/fusion.py` | `1` | fusion depth: re-read the context between latent blocks (`read_depth-1` extra cross-attentions); `1` = single up-front read (champion-identical) |
| `poll_phylo_weight` | `core/fusion.py` | `0` | pollinator phylo self-distillation: predict a species' pollinators from phylo-relatives (masked seed) and distill toward full-info -> trains interaction transfer (B55) |
| _(add new tokenizer/embedding/encoding/attention toggles here as they are introduced)_ | | | |
