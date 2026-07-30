# Earth4D Scorecard

The agent picks **one capability from Layer 1, with intention**, declares it as `--metric`, and is then
free to change anything — data channel, probe mode, encoder internals, objective. What the harness
enforces is not *which edits are allowed*, it is that the run still measures **the same thing**:

```
   measurement identity = capability · mode · split · n_shards · protocol · code hash
   two runs are comparable  ⇔  their identities match
```

A different mode or shard count is a different target, not a better score.

| source of truth | holds |
|---|---|
| `autoresearch/main/state/champion_scores.json` | Layer 1 champion scores (full model) |
| `autoresearch/spacetime/state/records.json` | Layer 2 probe records (gitignored, lives on the box) |
| `autoresearch/main/program/BENCHMARKS.md` | committed reference baseline, reproduction command |
| this file | the canonical registry — capability ⇄ bench ⇄ probe mode ⇄ probeability |

**Base:** champion `arithmetic 0.6153` · `harmonic 0.348744` · label `latent_diffusion:true`.
**Never compare Layer 1 to Layer 2.** They are different instruments on different metrics: B20
`community_from_env` is recall@10 from the full 799M fusion model; the probe's `community_from_env`
0.8845 is micro-AP from a light head on frozen encoder features. Neither bounds the other.

---

## Layer 1 — the board (what Earth4D must EARN from coordinates + environment)

Every ≥0.90 row elsewhere in the suite is **borrowed frozen vision** (photo→X). These 16 rows are the
actual DeepEarth innovation and where it currently fails. Scores are champion, full-model.
**Status:** ❌ <0.45 · ⚠️ 0.45–0.70 · ✅ ≥0.70

### A. Env → identity (SDM)
| # | requirement | bench | metric | champion | target | status |
|---|---|---|---|---|---|---|
| 1 | species from environment | B1 | top-10 acc | 0.323 | 0.90 | ❌ |
| 2 | species from spacetime | B5 | top-10 acc | 0.399 | 0.70 | ❌ |
| 3 | family from environment | B6 | acc | 0.103 | 0.90 | ❌ |
| 4 | family from spacetime | B8 | acc | 0.127 | 0.70 | ❌ |
| 5 | community from environment | B20 | recall@10 | 0.309 | 0.70 | ❌ |

### B. Env → ecology
| # | requirement | bench | metric | champion | target | status |
|---|---|---|---|---|---|---|
| 6 | live fuel moisture from env | B34 | Pearson r | 0.433 | 0.70 | ❌ |
| 7 | mycorrhiza type from env | B42 | macro-F1 | 0.268 | 0.70 | ❌ |
| 8 | pollinators from env | B51 | recall@10 | 0.174 | 0.70 | ❌ |

### C. Calibration
| # | requirement | bench | metric | champion | target | status |
|---|---|---|---|---|---|---|
| 9 | species posterior calibration | B23 | MRR | 0.143 | 0.70 | ❌ |

### D. Phenology
| # | requirement | bench | metric | champion | target | status |
|---|---|---|---|---|---|---|
| 10 | flowering presence | B26 | ROC-AUC | 0.740 | 0.85 | ✅ |
| 11 | flowering fidelity (env vs env+photo) | B27 | 1−MAD | 0.702 | 0.85 | ✅ |
| 12 | flowering peak month | B28 | MRR | 0.451 | 0.85 | ⚠️ |

### E. Env → env reconstruction
| # | requirement | bench | metric | champion | target | status |
|---|---|---|---|---|---|---|
| 13 | infer clay (held-out) | B16 | cosine | 0.426 | 0.85 | ❌ |
| 14 | infer soil (held-out) | B17 | cosine | 0.643 | 0.85 | ⚠️ |
| 15 | infer hydro (held-out) | B43 | cosine | 0.720 | 0.85 | ✅ |
| 16 | infer climate (held-out) | B18 | cosine | 0.875 | 0.90 | ✅ |

**Snapshot:** 10 ❌ · 2 ⚠️ · 4 ✅ (of 16) · **mean = 0.4272**
**Priority (worst first):** B6 0.103 · B8 0.127 · B23 0.143 · B51 0.174 · B42 0.268 · B20 0.309 ·
B1 0.323 · B5 0.399 · B16 0.426 · B34 0.433 · B28 0.451

---

## Layer 2 — encoder-probe records (exploratory)

Fast frozen-encoder probes. `fair_gain` = Earth4D − the strongest fair baseline on the same probe.
**Discovery instruments: a probe record is never science** — it must clear the evidence standard in
`program.md` first. Records below are the live board; a record may only be beaten like-for-like.

| capability | record | metric | fair_gain | vs baseline | mode | shards | protocol |
|---|---|---|---|---|---|---|---|
| community_from_env | **0.8845** | micro-AP | **+0.4570** | GAIN | COOCCUR-ROUTING | 12 | v2-leakfix |
| species_from_env | **0.6275** | micro-AP | **+0.4000** | GAIN | SDM-PRESENCE | 16 | v2-leakfix |
| calibration | **0.5910** | AUROC conf→correct | — | *none reported* | *none* | 8 | v2-leakfix |
| family_from_spacetime | **0.1769** | acc | **+0.0772** | RFF | FORECAST(past→future) | 12 | v2-leakfix |
| family_from_env | **0.1423** | acc | **+0.0411** | best-coord-PE | ENV | 12 | v2-leakfix |
| flowering_peak_month | **0.0521** | within-tol acc | **+0.0087** | RFF | PHENOLOGY-FUTURE | 12 | v2-leakfix |
| species_from_spacetime | **0.0512** | acc | **+0.0432** | RFF | FORECAST | 12 | v2-leakfix |

Reads:
- **calibration** reports no fair baseline, so its bottleneck is undiagnosable and 0.5910 is barely
  above the 0.5 useless floor. Its stored probe uses `--feature/--ensemble`, which belong to
  `calib_probe.py`, not `probe.py` — the record cannot currently be reproduced through the harness.
- **family_from_spacetime** was **invalidated on 2026-07-30** and restored to 0.1769. A second agent tree
  on the box (`/workspace/codex-earth4d-native-fb35a7f`, its own harness and `--causal_clock_*` flags)
  walked the record 0.1769 → 0.19143524765968323 in seven accepted single-seed steps
  (+0.0007 / +0.0008 / +0.0112 / +0.0005 / +0.0002 / +0.0006 / +0.0006), each stacking another
  `--causal_*` flag on the same held-out split. Every delta is inside single-seed noise and the walk is a
  maximum selected over repeated runs. It is also unreproducible here: nothing in this tree emits its
  claimed `trained_rff` baseline, and replaying the stored command gives `st_gain(vs RFF) +0.0481`.
  Prior state kept at `records.pre-invalidation-20260730.json`; the reason is in the ledger dead-ends.
  **The board is shared and singular** — see `BOARD_FROZEN.md`.
- **flowering_peak_month**'s stored probe passes `--pheno_env`, which is **silently ignored**: in
  `probe.py` the `if a.phenology:` block returns before the `--pheno_env/--pheno_disttarget/
  --pheno_taxon` and `--pheno_densefield` blocks can run. Verified — `--phenology --forecast` with
  `--pheno_env`, with `--pheno_taxon family`, and with `--pheno_densefield` all produce a byte-identical
  `PHENOLOGY-FUTURE` header, the same 98,304 obs / 19,662 queries, and the same +0.0087. So this record
  measures the plain temporal path, not an env-channel phenology probe, and ~120 lines of pheno modes
  are unreachable whenever `--phenology` is set.
- **community_from_env / species_from_env** carry large fair-gains on *fused env channels*, not on the
  coordinate encoder alone. Label them as such.

## Layer 3 — excluded, with reason

| capability | why excluded |
|---|---|
| family_from_vision (0.9445) | **Borrowed frozen DINO/BioCLIP**, no trace provenance or shard identity. Not an Earth4D probe record and not a legal `--metric`. Kept visible only so nobody re-publishes it as a win. |
| lfmc_from_env · mycorrhiza_from_env · pollinator_from_env | non-encoder heads: the capability lives in a downstream head, not in the positional encoding |
| flowering_auc · flowering_fidelity | same — measured on the fusion model's flowering head |
| infer_clay · infer_soil · infer_climate · infer_hydro | env→env reconstruction runs through the field decoder, not the encoder probe |

These 9 are **not** legal `--metric` values. They remain on Layer 1 because the full model is still
scored on them; they are simply not reachable by the encoder probe.

## Architectural note — a probe-win that must NOT graduate

A gated **spatial-only random-Fourier-features branch** (default off) fixes the bare probe's weakness:
a raw Earth4D hash grid loses to a generic RFF PE on smooth/static tasks (0.069→~0.08–0.10 static;
forecast 0.153→0.165, +0.096 vs RFF). **But the champion already carries this exact prior** —
`core/fusion.py:311` wires `SmoothGeoField` (an RFF geo prior added to the hash position;
`champion.yaml smooth_geo: true`). The probe-win is the probe catching up to the champion, not a new
lever. **Do not graduate** (Ensue `earth4d_FOURIER_redundant_with_smooth_geo_NO_graduation_2026_07_28`).
Keep the branch default-off for probe fairness only. Single-seed, noisy.
