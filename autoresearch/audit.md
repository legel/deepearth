# DeepEarth / DeepCal — Full System Audit

**Scope.** A two-dimension audit: (1) completeness of every architectural/scientific deliverable in `science.md`, and (2) the correctness and true performance of every benchmark in `evaluate.py`, reviewed ground-truth-vs-output against the trained champion.

**Champion inspected.** `ckpt_merged5.pt` — `deepcal.yaml`, 797.6M params, 343,035 observations (291,493 train / 51,542 spatially held-out), spatial holdout. **Net (harmonic) = 0.108**, **arithmetic capability mean = 0.502**, 61/63 benchmarks active. Aggregate scores reproduced identically by two independent eval runs (both 0.1082); ground-truth samples captured on 1,280 held-out rows. Verified against champion-path code (`fusion.py`, `phylogenomic.py`, `earth4d.py`, `data.py`), not the prose.

---

## Executive summary — four conclusions

1. **The model is genuinely strong on most capabilities; the headline harmonic number understates it.** The arithmetic capability mean is **0.502**, with flagship species-ID at **0.83 top-1** from a photo, traits at **0.85–0.93** F1, family-from-phylogeny at **0.89**, honest environmental reconstruction at 0.55–0.71. The harmonic net (0.108) is pinned near-zero by **exactly three species-distribution-skill benchmarks** (B39=0.003, B40=0.016, B29=0.071), where the SDM head emits distributions nearly indistinguishable from uniform. That family is the single highest-leverage target on the north-star metric — a real model gap, not a scoring artifact.

2. **Architecture: 20 of 28 science deliverables are fully built, 7 partial, 1 missing.** The crown jewels are real and validated — the **exact O(N) tree-GP phylogenetic refinement** (patristic fidelity <1e-3), BioCLIP-2.5 seeding, Earth4D space-time, iterative joint decoding. The honest gaps: the **"causal forecaster" headline is barely materialized**, **long-term memory is a dead hook**, and **interim scientific stand-ins persist** (the pollinator graph still uses an embedding-shadow distance rule 29 forbids as champion).

3. **The benchmark suite is mostly honest, with four fixable defects to disclose.** The anomaly-cosine safeguards were verified to actually work (predictions are not the modality mean; not silently raw cosine). But **B12 is a trivial lookup** (leaks `identity` into a per-species trait table — the suite's highest score is its least meaningful), **B14/B45 are leak-inflated** (they leave the sister embedding of the *same photo* in the input), **B24 is mislabeled** (top-1 minus top-10), and **B54/B27 use gameable metrics** not normalized like their siblings.

4. **Two clear gaps with different owners.** Mycorrhiza prediction (B42/B63) is a genuine **model** failure — a 2-of-5-class collapse well below its own 0.584 baseline. Live-fuel-moisture (B34) is a **data** gap — ~6% labeled with impossible values (up to 3,581% where the physical max is ~300%).

---

## Dimension 1 — Science.md deliverables (SCI-1…SCI-28)

Verified against champion-path code. Champion = `latent-clade` operator, spatial holdout.

| SCI | Deliverable | Status | Evidence / note |
|---|---|---|---|
| **Earth4D space-time** ||||
| SCI-1 | Absolute encoder (NeRF-like GPS+time memory) | Implemented | `Earth4D enable_absolute`, hash log2=20 |
| SCI-2 | Relative offset encoder (local window) | Implemented | `SpaceTimeField.encode_relative` |
| SCI-3 | Positional encoding fused into every token | Implemented | `encode: tok_norm+pos_norm` |
| SCI-4 | Causal autoregressive forecasting | **Partial** | Only a future-holdout convention, no AR mechanism; off in champion (spatial) → B25/B31 inactive |
| SCI-5 | Production CUDA kernels, stay fast | Implemented | `hashencoder.cu`, sparse-Adam path |
| SCI-6 | Large capacity (≥100M) | Implemented | ~798M params |
| SCI-7 | Parallelizable geo/time tiling | **Partial** | Data-parallel only; no spatial sharding |
| **Phylogenomic species GNN** ||||
| SCI-8 | Per-species embeddings shared by ancestry | Implemented | `SpeciesGraph` |
| SCI-9 | Self-supervised on a dated scientific tree | Implemented | real `ca_subtree.dated.nwk` |
| SCI-10 | Out-of-tree / novel-species projection | Implemented | `LatentCladeAttention` soft-attach |
| SCI-11 | Batch on species A updates neighbors B,C… | Implemented | unit-tested extended backprop |
| SCI-12 | Extremely fast CUDA gather/update | **Partial** | O(N) two-pass but Triton/PyTorch, no bespoke kernel |
| SCI-13 | **Exact O(N) tree-GP refinement (rule 29)** | Implemented | `LatentCladeAttention` — strongest deliverable, champion operator, fidelity <1e-3 |
| SCI-14 | Frozen BioCLIP-2.5 text seed + small probe | Implemented | 1024-d, seed-once (data.py comment stale, code correct) |
| SCI-15 | Phylo embedding maskable/reconstructable | Implemented | `phylo_mask_weight: 0.1` self-distill |
| **Multi-modal fusion / MADE** ||||
| SCI-16 | Token = space-time · phylo · modality · type | Implemented | `_variable_token`, `type_emb` |
| SCI-17a | Mask-as-query, attend-all-context | Implemented | Perceiver latent bottleneck |
| SCI-17b | **Long-term memory bank** | **Missing** | `experience/_memory_key` are dead hooks |
| SCI-18 | Joint distribution of all variables (DBM) | Implemented | latent self-attention |
| SCI-19 | MADE Bayesian posterior sharpening | **Partial** | masked-recon conditioning, not literal MADE ordering |
| SCI-20 | Ingest every modality (~14) | Implemented | "each must lift" enforced by guard, not code |
| SCI-21 | Iterative K-round joint decoding + write-back | Implemented | `rounds: 2` |
| SCI-22 | Per-variable channels, O(N·L) highway | Implemented | + marginal-fidelity monitor |
| SCI-23 | Dense 4D "measure-everywhere" field | **Partial** | decode capability exists; trained only at G=1 single-query |
| **Foundations & interactions** ||||
| SCI-24 | DINOv3 / CLAY / NAIP / SAT foundation encoders | Implemented | frozen embeddings ingested |
| SCI-25 | No fuzzy science — SOTA models on real data | **Partial** | interim stand-ins remain (pollinator OU-shadow, text-shadow distances) |
| SCI-26 | Plant↔pollinator bilinear across two trees | Implemented | but pollinator side is ou-attention, not latent-clade |
| **Learning protocol** ||||
| SCI-27 | Detached heads; universal recon inviolable | **Partial** | detach done; trait-subspace / reliability-weighting escape valve not built |
| SCI-28 | Score + optimize 100% of suite (harmonic) | Implemented | renormalized deltas |

**Tally: 20 Implemented · 7 Partial · 1 Missing.** Earth4D and the phylogenomic GNN are the most complete subsystems; the MADE dense-field ambition and temporal forecasting the least.

**Five headline gaps.** (1) The "causal forecaster" framing (rule 1) is the least-built thing — no AR mechanism, and the champion doesn't run the future split. (2) Long-term memory (rule 15) does not exist. (3) Interim scientific stand-ins persist (rules 28/29): the pollinator graph uses embedding-shadow OU-attention, and non-tree taxa fall back to a BioCLIP text-shadow distance. (4) The detached-head discipline is half-built — the sanctioned isolated trait-subspace for a learning head isn't implemented, capping every niche capability. (5) The dense 4D field is a decode capability, not a trained objective.

---

## Dimension 2 — Per-benchmark ground-truth audit

Score = champion value on held-out data. **Verdict key:** Healthy (measures what it should, score real) · Model-limited (real test, model underperforms) · Defect (leak / trivial / mislabeled / gameable metric) · Data-gap · Inactive · Diagnostic.

### A. Species & family identification — *SCI-14/16/17/18/19*
| B | Measures | Score | Verdict |
|---|---|---|---|
| B1 | Species from environment alone (SDM), top-10 | 0.253 | Model-limited — env alone weak; not degenerate (195 unique preds) |
| B2 | **Flagship**: species from env+photo, top-1 | **0.830** | Healthy — near-perfect on sample; system's strongest result |
| B3 | Species from env+photo, top-5 | 0.949 | Healthy |
| B4 | Species from photo only, top-1 | 0.819 | Healthy — photo alone ≈ photo+env |
| B5 | Species from bare space-time, top-10 | 0.231 | Model-limited — regional prior only |
| B6 | Family from environment | 0.093 | Model-limited — env→family genuinely weak |
| B7 | Family from phylo embedding | 0.893 | Healthy (near-lookup: phylo vector ≈ species fingerprint) |
| B8 | Family from bare space-time | 0.082 | Model-limited |
| B23 | Env→species posterior calibration (MRR) | 0.120 | Model-limited — median rank 32 |
| B24 | Info gain of photo over env | 0.578 | Diagnostic / **Defect** — real gain but mislabeled (top-1 − top-10) |

### B. Phylogeny & traits — *SCI-8/13/15/19*
| B | Measures | Score | Verdict |
|---|---|---|---|
| B9 | Evolutionary embedding from env+photo (cosine) | 0.784 | Healthy — discriminative, not mean-collapsed |
| B10 / B11 | Categorical traits from photo+env / photo, F1 | 0.879 / 0.878 | Healthy |
| B12 | Trait from all-other-variables, F1 | 0.954 | **Defect (floor-trivial)** — leaks `identity` → per-species trait lookup, not inference |
| B30/B32/B33/B35/B36/B38/B49 | Seasonality / plant-type / growth-rate / sun / ease-of-care / water-soil / form traits, F1 | 0.83–0.93 | Healthy — all classes covered |

### C. Environmental modality reconstruction ("measure-everything") — *SCI-20/22/23*
| B | Measures | Score | Verdict |
|---|---|---|---|
| B13 | Imagine ground-vision (DINO) from non-vision U | 0.541 | Healthy — honest anomaly-cosine |
| B14 / B45 | Vision leave-one-out (DINO / BioCLIP) | 0.674 / 0.701 | **Defect (leak-inflated)** — sister embedding of the same photo left in input |
| B15 | Ground-vision from aerial (NAIP) | 0.194 | Model-limited — resolution bridge weak |
| B16/B17/B18/B19/B37/B43/B44/B46/B47 | Reconstruct clay / soil / climate / aerial / bio-vision / hydro / topo / chm / NAIP-IR | 0.49–0.71 | Healthy — centering verified (B16 raw 0.41 → anomaly 0.60) |

### D. Community & species distribution (SDM) — *SCI-18/19/23* — the north-star bottleneck
| B | Measures | Score | Verdict |
|---|---|---|---|
| B20/B21/B22 | Local community / companions recall@10 | 0.205 / 0.221 / 0.205 | Model-limited — modest |
| B29 | Species-distribution skill @30m | 0.071 | Model-limited — ≈ no skill vs uniform |
| B39 | Species-distribution skill @3km | **0.003** | Model-limited — **pins the harmonic net**; distribution ≈ uniform |
| B40 | Species-distribution skill @300m | 0.016 | Model-limited — same |

### E. Phenology — *SCI-4/15*
| B | Measures | Score | Verdict |
|---|---|---|---|
| B26 | Flowering probability, ROC-AUC | 0.743 | Healthy — real AUC, balanced labels (605+/457−) |
| B27 | Imagined-vs-real flowering agreement | 0.742 | Healthy score but **metric rewards ignoring the photo** |
| B28 | Peak-bloom month from 12-month time sweep, MRR | 0.381 | Healthy — month-sweep genuinely varies |

### F. Ecophysiology & symbiosis — *SCI-25 (no fuzzy science) / 15*
| B | Measures | Score | Verdict |
|---|---|---|---|
| B34 | Live fuel moisture from env | 0.349 | **Data-gap** — 6% labeled, dirty values to 3,581%; model predictions physically sane |
| B42 | Mycorrhizal type from env, F1 (5-class) | 0.182 | Model-limited — **real 2-of-5-class collapse** (79% AM majority) |
| B63 | Mycorrhiza given species (pure phylo imputation) | 0.177 | Model-limited — far below own 0.584 baseline |

### G. Interactions / pollinators — *SCI-26 (two-tree bilinear)*
| B | Measures | Score | Verdict |
|---|---|---|---|
| B48 / B52 | Pollinators from photo(+env), recall@10 | 0.423 | Healthy — photo lifts recall ~2.6× |
| B41/B50/B51 | Pollinators from species / space-time / env | ~0.16 | Model-limited — modest |
| B53/B54/B55 | Pollinator calibration / distribution-KL / phylo-transfer | 0.26 / 0.070 / 0.061 | Model-limited — cross-tree induction weak; B54 metric non-skill-normalized |

### H. Forecasting — *SCI-4*  &  I. Phylo-graph-gain diagnostics — *SCI-11/13*
| B | Measures | Score | Verdict |
|---|---|---|---|
| B25 / B31 | Future climate / appearance (temporal holdout) | INACTIVE | Correctly inactive (spatial run) |
| B56 | Family-from-phylo gain from graph refinement | 0.124 | Diagnostic / Healthy — graph genuinely helps where phylo is given |
| B57–B62 | Flowering / lfmc / pollinator / community / trait / myco graph-gains | ~0.000 | Diagnostic / Model-limited — real ≈0: env-only path → diffuse posterior the graph can't move (known mechanism, not a bug) |

**Classification tally (61 active):** ~30 Healthy · ~22 Model-limited · 4 benchmark defects (B12; B14/B45; B24; B27/B54 metrics) · 1 Data-gap (B34) · 2 Inactive (B25/B31).

**Verified safeguards (checked, sound).** Anomaly (mean-centered) cosine is genuinely applied (cos(pred, train-mean) ≈ 0.0–0.14), so reconstruction benchmarks are not gameable by a mean prediction. Community/pollinator targets are used only as targets and test neighbors come from a train-only tree — leak-safe. The eval-only `_seed()` regression is fixed (returns a plain tensor). The chunked `_train_mean` gather avoids the eval OOM.

---

## What the report should say

- **Where DeepEarth is state-of-the-art:** photo→species/traits/phylogeny, the exact tree-GP phylogenetic engine, and honest dense environmental reconstruction — real, verified, defensible.
- **The one number to fix:** species-distribution skill (B29/39/40) — it alone collapses the harmonic net from a respectable ~0.5 arithmetic to 0.108. Highest-leverage research target.
- **Two credibility items to own before presenting:** clean the four benchmark defects (B12, B14/B45, B24, B54/B27 metrics), and either build genuine temporal forecasting or reframe the "causal forecaster" claim.
- **Two concrete gaps with owners:** mycorrhiza class-collapse (model) and LFMC label quality (data).

*Every figure is from the actual champion on held-out data, cross-verified by two independent eval runs (both 0.1082).*
