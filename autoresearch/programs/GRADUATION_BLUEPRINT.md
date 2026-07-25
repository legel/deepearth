# DeepEarth Graduation Blueprint (overnight 2026-07-25)

All findings below are **probe-level** (encoder-isolation), leak-guarded, multi-seed. The champion
benchmark (arith ~0.6153) is **unchanged** — nothing here is graduated. This document is the reviewable
spec for the operator PRs that would turn these into actual bench movement. Every probe is additive and
flag-gated; champion default paths are byte-identical.

## The unifying architectural finding
In **both** encoders the fancy learned operator is redundant-to-harmful; the real engine is the
**objective/mechanism** on a good base representation. Graduation should ship the objectives, demote the operators.

| encoder | operator (demote) | engine (ship) | base representation |
|---|---|---|---|
| spacetime | Earth4D hash → spatial index | causal seasonal-persistence propagator | raw coords / small coord-MLP |
| biological | phylo-graph → supervised-only minor refine | supervised masked phylo-imputation | BioCLIP text seed (0.90 family-coherent) |

## 1. Spacetime forecast head (marquee) — the interpretable dynamical model
- **Within-year**: LSTM propagator over K causal-nearest past neighbours (own past DOY + spatial offset).
  Mechanism = seasonal auto-persistence, recency kernel τ = 0.741 ± 0.000 d ("freshest local obs"),
  modulated by Hopkins latitude cline +2.31 d/°lat and elevation cline +2.8–3.2 d/100m (ecology-validated).
  Skill +51–73d MAE-gain over static floor; 80% recovered by explicit physics.
- **Across-year**: condition on year-specific spring-temp anomaly (Daymet). Spring-guild plants shift
  −5.9 d/°C (R²0.67, OOS +0.17); woody/late do not (guild-specific). Fit β per guild.
- **Unified head** = climatology + within-year-persistence (primary) + β·spring_anom (sparse-data backstop).
  Beats persistence by +0.64d [95% +0.46,+0.83]. Climate term dominant where neighbour data is sparse.
- Earth4D-hash: keep only as a static spatial index for SDM; drop the time-conditioned branch (harmful).
- Probe: `autoresearch/programs/spacetime/earth4d_engine.py`.

## 2. Biological head — seed + supervised imputation
- BioCLIP text seed by default; **trait-supervised masked-imputation objective is MANDATORY** (without it the
  graph operator is net-negative — it destroys seed structure).
- Route fused text⊕vision seed on conserved categorical axes (seasonality); text-only on continuous/interaction (lep).
- Defensible axes (held-out impute, multi-seed): num_lep_support 0.126→0.793 (strong); seasonality +0.114;
  rarity/sun +0.05–0.07 (modest). Seed-saturation caps the operator: corr(seed-saturation, graph-gain) = −0.52.
- Probe: `autoresearch/programs/biological/traitprobe.py` (`--trait_supervised`, seed routing, `--blanket_k`).

## 3. Env-encoder niche capabilities (routing law: niche → env)
- **Rarity** = range-size (n occupied cells): Spearman 0.741 ± 0.048.
- **Co-occurrence** = env (AlphaEarth): micro-AP 0.871 ± 0.007, +0.433 gain.
- **SDM presence** = env: +0.219 (660-cell block-CV).
- **Pollinator syndrome** = env/climate (latitude-dominated): bee-vs-lep AUC 0.733. (phylo two-tree is dead.)
- Probe: `autoresearch/programs/spacetime/probe.py` (`--env_construct`, `--cooccur`, `--sdm_hard`).

## 4. Calibration (rule 17 — cheapest first bench win)
- **Temperature scaling: mandatory, near-free on every eval head** — fit on held-out spatial-block cal split,
  ~10 LOC, zero accuracy cost, cuts ECE −90 to −98%. Raw ECE: Earth4D-B53 0.74, Env-B53 0.50, Env-B23 0.19.
- **Deep-ensemble k=3 only on the low-data B53 pollinator branch** for honest uncertainty (softmax confidence is
  anti-correlated with correctness; ensemble variance ranks it, AUROC 0.55–0.61; use variance as abstention).
- Always report conf-AUROC beside ECE. Skip isotonic (costs acc) and conformal (huge set-size).
- Probe: `autoresearch/programs/spacetime/calib_probe.py`.

## Genuine dead-ends (definitively characterized, not just unexplored)
- Earth4D-hash unique contribution (redundant with coords/env, even under causal end-to-end training).
- Phylo two-tree pollinator induction (rule 27) — sparse interaction signal, dead on real GloBI.
- Across-year *persistence* forecasting (inter-annual noise) — but climate-anomaly model IS a GO (see §1).
- ease_of_care beyond nature-ceiling (~0.33) — residual is irreducibly human judgment.

## Recommended graduation order (confidence × cheapness)
1. Temperature-scaling calibration — cheapest, near-free, first real bench win (calibration 0.143 is worst).
2. Env niche capabilities (rarity, co-occurrence) — clean, strong, env-encoder additive heads.
3. Spacetime forecast head — activates dormant B25/B31; the marquee science.
4. Biological supervised-imputation objective — narrow but real on lep/seasonality.
