# Earth4D Scorecard

**Base:** `origin/deepcal-ensue-autoresearch @ 0a643fc` (contains `e016af6`) · champion **arith 0.6153** · source `autoresearch/champion_scores.json`
**Purpose:** the requirements Earth4D must EARN from environment / spacetime coordinates alone. Every ≥0.90 row elsewhere in the suite is **borrowed frozen vision** (photo→X); these rows are the actual DeepEarth innovation and are where it currently fails.
**How to use:** `Record` starts = `Baseline`. Update `Record`/`Status` **only** on a genuine improvement measured by the native `autoresearch/evaluate.py` (no reimplemented metrics), gated by `champion_report` before→after. `Target` is a proposed bar — adjust freely.
**Status key:** ❌ <0.45 · ⚠️ 0.45–0.70 · ✅ ≥0.70

---

## 🔬 Encoder-probe records (LIVE — the overnight loop fills these)

Same capabilities, **scoped to the fast Earth4D encoder probe** (`trace.py` → `probe.py`, minutes, encoder-only). `fair_gain` = Earth4D vs a generic *trained* PE (RFF/MLP) — the honest "does the encoder earn it." These are encoder-probe numbers, distinct from the full-model baselines below.

| Capability | Probe record | fair_gain | Best lever | Read |
|---|---|---|---|---|
| family_from_vision | **0.945** (acc) | **+0.835** vs coord-PE | fam_vision_both(dino+bio) | DATA LEVER: family signal is in the PLANT IMAGE, not env (0.125→0.945). BORROWED vision (env=where, vision=which), not an Earth4D win |
| community_from_env | **0.887** (micro-AP) | **+0.460** | cooccur_both | EARNING — strongest signal |
| species_from_env | **0.634** (micro-AP) | **+0.407** | sdm_hard | EARNING — 7.5× over prevalence |
| family_from_spacetime | **0.182** | **+0.132** vs RFF | th8_ff1024_hh512 | EARNING — genuine Earth4D-arch win; temporal-harmonic + wider head 0.165→0.171→0.182 (th32/recurrence HURT) |
| family_from_env | **0.144** | **+0.034** vs best-coord-PE | famenv_alphaearth (real 64d AlphaEarth) | DATA LEVER, plumbing bug fixed: `load_env` was HARD-WIRED to 19wc+9soil+1elev and ignored `--env_channels`, so all 53 prior "channel swaps" fed the IDENTICAL 29 columns — AlphaEarth had never reached this path. Joining it for real: 0.125→0.144, gain −0.006→+0.034 (all=93d gives 0.141, pure AlphaEarth 64d wins). Satellite-embedding channel (env=where), not plant-photo vision |
| flowering_peak_month | 0.0674 | — | pheno_none_env | env-conditioning nudge; MODIS phenology channel NEGATIVE (landscape greenness ≠ species flowering timing) |
| calibration | 0.629 | — | cal_earth4d | conf→correct AUROC (0.5=useless); raw Earth4D overconfident (ECE 0.078→0.027 temp-scaled) |
| species_from_spacetime · lfmc · mycorrhiza · pollinator · flowering_auc/fidelity · infer_* | — | — | — | not Earth4D-probeable (non-encoder heads) |

**ARCHITECTURAL PROBE-WIN (earth4d.py), NOT a graduation candidate:** a gated **spatial-only random-Fourier-features branch** (default off, champion byte-identical) fixes the *bare probe's* weakness — a raw Earth4D hash grid loses to a generic RFF PE on smooth/static tasks (0.069→~0.08–0.10 static; forecast 0.153→0.165, +0.096 vs RFF). **BUT: the champion already carries this exact prior** — `core/fusion.py:311` wires `SmoothGeoField` (an RFF geo prior added to the hash position; `champion.yaml smooth_geo: true`). So the probe-win is the probe catching up to what the champion has, NOT a new champion lever. **Do NOT graduate** — verified redundant (Ensue `earth4d_FOURIER_redundant_with_smooth_geo_NO_graduation_2026_07_28`). Keep the branch default-off for probe-fairness only. Single-seed, noisy.

**Earning directions:** (1) env→biology SDM/co-occurrence, (2) **temporal state/propagation INSIDE the encoder** — the genuine open champion gap: Earth4D has no causal temporal state (static hash + RFF geo prior); the forecast probe shows Earth4D+propagators earn. NOT the Fourier branch (redundant with the champion's `smooth_geo`). Records auto-tracked in `records.json`; net card printed after every run.

---

## A. Env → identity (SDM) — the core failures

| # | Requirement | Bench | Metric | Baseline | Record | Target | Status |
|---|---|---|---|---|---|---|---|
| 1 | species from environment | B1 | top-10 acc | 0.323 | 0.323 | 0.90 | ❌ |
| 2 | species from spacetime | B5 | top-10 acc | 0.399 | 0.399 | 0.70 | ❌ |
| 3 | family from environment | B6 | acc | 0.103 | 0.103 | 0.90 | ❌ |
| 4 | family from spacetime | B8 | acc | 0.127 | 0.127 | 0.70 | ❌ |
| 5 | community from environment | B20 | recall@10 | 0.309 | 0.309 | 0.70 | ❌ |

## B. Env → ecology

| # | Requirement | Bench | Metric | Baseline | Record | Target | Status |
|---|---|---|---|---|---|---|---|
| 6 | live fuel moisture from env | B34 | Pearson r | 0.433 | 0.433 | 0.70 | ❌ |
| 7 | mycorrhiza type from env | B42 | macro-F1 | 0.268 | 0.268 | 0.70 | ❌ |
| 8 | pollinators from env | B51 | recall@10 | 0.174 | 0.174 | 0.70 | ❌ |

## C. Calibration — worst in suite

| # | Requirement | Bench | Metric | Baseline | Record | Target | Status |
|---|---|---|---|---|---|---|---|
| 9 | species posterior calibration | B23 | MRR | 0.143 | 0.143 | 0.70 | ❌ |

## D. Phenology

| # | Requirement | Bench | Metric | Baseline | Record | Target | Status |
|---|---|---|---|---|---|---|---|
| 10 | flowering presence | B26 | ROC-AUC | 0.740 | 0.740 | 0.85 | ✅ |
| 11 | flowering fidelity (env vs env+photo) | B27 | 1−MAD | 0.702 | 0.702 | 0.85 | ✅ |
| 12 | flowering peak month | B28 | MRR | 0.451 | 0.451 | 0.85 | ⚠️ |

## E. Env → env reconstruction

| # | Requirement | Bench | Metric | Baseline | Record | Target | Status |
|---|---|---|---|---|---|---|---|
| 13 | infer clay (held-out) | B16 | cosine | 0.426 | 0.426 | 0.85 | ❌ |
| 14 | infer soil (held-out) | B17 | cosine | 0.643 | 0.643 | 0.85 | ⚠️ |
| 15 | infer hydro (held-out) | B43 | cosine | 0.720 | 0.720 | 0.85 | ✅ |
| 16 | infer climate (held-out) | B18 | cosine | 0.875 | 0.875 | 0.90 | ✅ |

---

**Snapshot @ 0a643fc:** 9 ❌ · 2 ⚠️ · 5 ✅ (of 16). Mean baseline over these 16 = 0.409.
**Priority order (worst first):** B6 0.103 · B8 0.127 · B23 0.143 · B51 0.174 · B42 0.268 · B20 0.309 · B1 0.323 · B5 0.399 · B16 0.426 · B34 0.433 · B28 0.451.
