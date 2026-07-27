# Earth4D Scorecard

**Base:** `origin/deepcal-ensue-autoresearch @ 0a643fc` (contains `e016af6`) · champion **arith 0.6153** · source `autoresearch/champion_scores.json`
**Purpose:** the requirements Earth4D must EARN from environment / spacetime coordinates alone. Every ≥0.90 row elsewhere in the suite is **borrowed frozen vision** (photo→X); these rows are the actual DeepEarth innovation and are where it currently fails.
**How to use:** `Record` starts = `Baseline`. Update `Record`/`Status` **only** on a genuine improvement measured by the native `autoresearch/evaluate.py` (no reimplemented metrics), gated by `champion_report` before→after. `Target` is a proposed bar — adjust freely.
**Status key:** ❌ <0.45 · ⚠️ 0.45–0.70 · ✅ ≥0.70

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
