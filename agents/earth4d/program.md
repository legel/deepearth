# Earth4D Agent — program

**Objective:** break records on the scorecard capabilities (species/family from env & spacetime, phenology, env-decode, calibration) **scoped to the encoder** and measured by the fast Earth4D **probe**, not full-model training. There are **two co-equal lever families**: **DATA** (what signal feeds the encoder) and **ARCHITECTURE** (how the encoder represents it). **Surface area** = `encoders/spacetime/earth4d.py` + the probes in `autoresearch/programs/spacetime/` **+ the data channels those probes feed** (`--env_channels`, `--sdm_channels`, `--vision`, `--pheno_channel`, densification). A probe run is minutes, so iterate fast and broad.

**Box & ops (read first):** the box connection, repo paths, reboot recovery, GPU health check, launch + self-heal commands, Ensue token location, and commit identity are in **`agents/earth4d/box-operations.md`** (gitignored — it holds box connection details).

## 1. Loop
1. **Pick the objective** — the worst / highest-leverage scorecard capability. Declare it as `--metric`.
2. **Diagnose the bottleneck from the trace's fair-gain** (encoder vs a generic trained PE/RFF), then pick the matching lever:
   - **fair-gain ≈ 0 or negative → the INPUT is the bottleneck, not the encoder → DATA lever.** The coordinate/current channel lacks the signal; feed a different/richer channel. (e.g. `family_from_env` sat at 0.125 with gain ≈ 0 for *every* env channel → swapping to per-obs **vision** gave 0.945.)
   - **fair-gain positive but score low → the ENCODER is the bottleneck → ARCHITECTURE lever.**
3. **Run through the harness:** `python -m deepearth.agents.earth4d.trace --metric <capability> --probe "<flags = the lever>" --tag <id> --device cuda:N`. The probe trains a light head on **frozen** encoder features in **minutes** (the encoder is constructed fresh and read under `no_grad`, so its hash table stays RANDOM — fair-gain therefore compares architectural PRIORS as fixed feature maps). `--train_encoder` trains the encoder end-to-end (warmup + coarse-to-fine level unmasking + its own low LR); measured on family_from_spacetime it changes the score by <0.004 even with the table moving 41%, so the frozen protocol is a fair default — but say which one a number came from and reports the encoder-isolated score + fair-gain.
4. **Read the trace:** capability score + fair-gain + delta vs the probe baseline.
5. **Sweep in breadth** — both GPUs saturated; cross DATA × ARCHITECTURE (e.g. each env channel × each encoder variant).
6. **Gate + record + PUBLISH (taxonomy):** every run through `trace.py --ensue` upserts one key per capability, **`LOOP-earth4d-<capability>`**, holding the running best + record-history + this run's outcome + deduped dead-ends *with their bottleneck reason* — win OR dead-end, per the main PROGRAM.md ("publish every outcome, never lose a run's result, no noise-chasing as a win"). A record-break also updates `scorecard.md`. The full per-run ledger lives in `records.json`. The loop (`agents/earth4d/loop.sh`, launched by `start.sh`) passes `--ensue` on every iteration, so this is automatic — never run the loop without it.

## 2. Preferences
- **Surface = the Earth4D encoder + its probes + the data channels they feed.** Never the full fusion model.
- **DATA lever — first-class, NOT new.** The macro phase moved arith 0.446→0.6153 heavily on data (terrain, AlphaEarth, occurrence densification 207k→621k); data was always viable. For the probes it means: **which channel/modality feeds the encoder+head.**
  - env channels: `--env` / `--env_channels {worldclim, alphaearth, all}`, `--env_extra` (soil+elev), `--sdm_channels`.
  - vision: `--vision --vision_feats {dino, bio, both}` (per-obs DINO/BioCLIP) — carries morphology/family where env can't.
  - phenology / remote-sensing: `--pheno_channel` (MODIS NDVI/EVI).
  - occurrence densification, channel fusion (env+vision), per-species aggregation.
  - When a fair-gain is flat across an input type, that input is signal-limited — **change the channel, don't just swing the architecture.**
- **ARCHITECTURE lever — also first-class. YOU CAN AND SHOULD EDIT `encoders/spacetime/earth4d.py` ITSELF** — the encoder's `__init__`, forward pass, and training objective are all in scope, not just probe flags. Add new internal structure to the encoder (done already: the gated temporal-harmonic path `6a02ef0`, the spatial-Fourier branch `7d44651`). Levers: propagation / forecasting (`--recurrence`, `--gnn`, `--forecast`, internal temporal-harmonic path), field-decode (`--env_decode`, `--field_decode`), **new encoding architecture inside `earth4d.py`** (learned Fourier, SIREN, attention-over-neighbours, causal temporal state — demanding rewrites), new training objectives (cline-aware DOY, contrastive env alignment). Capacity knobs (`spatial_levels`, `log2_hashmap`, `head_hidden`, `time_harmonics`) tune a *winning* config, not the main move.
  - **Safe encoder-edit workflow:** back up `earth4d.py` → make the edit **gated + default-off** (new constructor arg defaulting to 0/False so the champion path is byte-identical) → `py_compile` → wire a matching probe flag in `probe.py` → `scp` to newbox → sweep via `trace.py`. Only a positive probe result graduates; the edit stays reversible.
- **Diagnose before you swing:** the trace's fair-gain tells you whether the problem is input-limited (→ DATA) or encoder-limited (→ ARCHITECTURE). Don't default to one family.
- **Measurement = the encoder probe, fast, native metrics** (never reimplemented). Single-seed; noise negligible. `rm -f`/rebuild the prepared cache (`--fresh-data`) whenever the DATA lever changes what's loaded — the cache is lossy across data changes.

## 3. Don'ts
- **Don't train the full 799M fusion model** — confounded and slow. Encoder + probe only.
- **Don't default to architecture-only.** DATA is co-equal; pick the lever the fair-gain points to. (This program used to say "architecture only, big swings" — that was wrong; the data lever was always live.)
- No experiment without a declared `--metric`; no multi-seed / re-verification (GPU → breadth).
- Native probe metrics only — no reimplemented scoring.
- **Attribute borrowed signal honestly** — a vision-channel win is *borrowed* DINO/BioCLIP, not a coordinate-encoder gain; label it (env=where, vision=which). Don't launder it as an Earth4D win, and don't chase the aggregate arith.
- Don't declare a capability a ceiling / "done" / "exhausted" — change the lever family and keep going.
