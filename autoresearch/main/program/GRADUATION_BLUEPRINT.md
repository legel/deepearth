# DeepEarth Graduation Blueprint (v2 — CORRECTED on repaired data foundation, 2026-07-25)

**READ THIS FIRST.** v1 of this blueprint (commit 6f9bcc7) contained numbers measured on a BROKEN data
foundation and are now corrected below. A data-integrity audit found 2 mislabeled arrays + 5 missing
files; all 6 are repaired (`data/deepcal/derived/*_rebuilt.*`, alignment-verified, non-destructive).
Several v1 conclusions changed when re-tested on correct data. All findings remain probe-level; champion
bench 0.6153 is unchanged (nothing graduated). Numbers below are on the CORRECTED foundation.

## 0. DATA FOUNDATION — repair first, then graduate (6/6 fixed)
Activating a rebuilt file = copy `derived/<name>_rebuilt.*` onto its live name (a champion-pipeline change).
| file | was | now | re-enables |
|---|---|---|---|
| gbif_species_dist | MISSING | rebuilt from 621k occ | B21/B29/B39/B40 SDM supervision (was silently OFF) |
| gbif_plant_dist | MISSING | real dated patristic (all 2141 tips) | real tree (was embedding shadow) |
| pollinator_distance | dim-mismatch | full 8157² | real pollinator tree (was BioCLIP shadow) |
| bioclip_taxon_text_emb | MISSING | BioCLIP-2.5 regenerated | rule-26 species-text seed (was never loaded) |
| gbif_mycorrhiza | STALE 4628 (mislabeled) | FungalRoot, 77% cov | B42 correct labels |
| gbif_lfmc | STALE (corrupt) | Globe-LFMC, 19% cov | B34 correct labels |

## 1. Recoverable signal — CORRECTED (wrong routing/framing, not "capped")
| benchmark | champion | best route | corrected lift | fix |
|---|---|---|---|---|
| **B42 myco** | 0.656 macro-F1 (E1-embed, leak-guarded) | **phylo-kNN patristic 0.81** | **+0.15 over champ route, +0.64 over floor** | route myco through the real phylo distance; predicts rare EcM/ErM classes (not AM-majority) |
| **B16 infer_clay** | 0.426 | ridge **0.92** | **+0.50** | higher-capacity env-block→env-block decode head |
| B43 hydro / B17 soil / B18 climate | 0.72/0.64/0.88 | 0.96/0.85/0.98 | +0.24/+0.21/+0.11 | same decode head (~+1.0 summed w/ clay) |
| **B1/B6 SDM argmax** | species 0.32 / family 0.10 | ArcFace metric head **2.3×** (species top1 0.035→0.080; family→0.099) | real | NOT "env-capped" — it was a metric artifact; env-niche AUC is 0.91. Add a cosine-prototype/ArcFace head + direct env→family aux head |
| **B23 calibration** | 0.143 | temp-scaling −83 to −98% ECE | large, free | temperature scaling on every eval head (~10 LOC, 0 acc cost); k=3 ensembles on low-data B53 for honest uncertainty |

## 2. Spacetime forecast head (unchanged — was on clean data)
LSTM propagator (encoder-agnostic; Earth4D hash demotes to index) + interpretable surrogate:
seasonal persistence τ=0.74d + Hopkins lat/elev clines (+2.3 d/°lat, +2.8-3.2 d/100m, ecology-validated);
cross-year climate-anomaly term (spring guild −6 d/°C, OOS +0.17). Activates dormant B25/B31.

## 3. Biological head — CORRECTED (the "graph is redundant" story was shadow-contaminated)
- Seed: the rule-26 BioCLIP-2.5 text seed (now rebuilt; the champion ran on the E1 shadow).
- The trait-supervised masked-imputation objective is the dominant lever (mandatory).
- **Phylo graph (latent-clade topology) is SELECTIVELY additive on deeply-conserved biology:**
  num_lep_support +0.046, family +0.031, myco +0.038 (5/5 seeds; via topology, not branch-length distance).
  7 of 11 trait axes stay flat — so it's selective, not a broad unlock, but NOT "redundant" as v1 claimed.
- rule-27 two-tree: DEAD confirmed on the real pollinator distance (interaction signal genuinely sparse).

## 4. Honest ceilings (levers exhausted) & genuine gaps
- B55 pollinator-transfer (0.037), B21 community (0.289), B28 peak-month (0.451): near leak-safe ceilings.
- ease_of_care: nature-ceiling ~0.33 (rest is human judgment — don't chase via labels).
- B29/B39/B40 dist-skill were INACTIVE (missing file, now rebuilt — re-score after activation).

## Recommended graduation order
1. **Activate the 6 rebuilt data files + re-run champion_report** (before→after) — this alone re-enables SDM
   supervision, the real trees/distances, and the rule-26 seed, and gives the first real corrected baseline.
2. **Temperature-scaling calibration** — cheapest, near-free, first bench win (B23 0.143 is worst).
3. **Env-recon decode head** (~+1.0 across B16/17/18/43).
4. **Myco phylo-route + SDM metric head.**
5. **Spacetime forecast head + biological rule-26 seed / topology-selective trait routing.**

Every number here is probe-level and on the corrected foundation; graduation (core PRs + data activation)
is required to move the bench. v1's inflated/miscredited claims (myco +0.61 accuracy, "env-capped",
graph "seasonality+myco") are superseded by the corrected values above.
