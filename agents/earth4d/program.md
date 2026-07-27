# Earth4D Agent — program

**Objective:** break records on the scorecard capabilities (the SAME capabilities the science/`evaluate.py` measures — species/family from env & spacetime, phenology, env-decode, calibration) **but scoped to the encoder** and measured by the fast Earth4D **probe**, not full-model training. **Surface area = ONLY the spacetime encoder** (`encoders/spacetime/earth4d.py` + the probes in `autoresearch/programs/spacetime/`). Because the scope is bounded to the encoder, **take big architectural swings** — demanding redesigns are cheap here (a probe run is minutes) and that is the whole point.

## 1. Loop
1. **Pick the objective** — the loop decides: the worst / highest-leverage scorecard capability (measured via its probe mode). Declare it as `--metric`.
2. **Make ONE big architectural change to the encoder** (Preferences) — a demanding jump, *not* a config tweak: a new propagator/forecaster, a field decoder, a new positional-encoding architecture, a new training objective. Edit `earth4d.py` or add a probe variant — encoder scope only.
3. **Run through the harness:** `python -m deepearth.agents.earth4d.trace --metric <capability> --probe "<probe flags = the lever>" --tag <id> --device cuda:N --ensue`. The probe trains ONLY the encoder + a light head on ~65k obs in **minutes** and reports the encoder-isolated score — `st_gain` = Earth4D vs the fair coordinate / RFF / MLP baseline.
4. **Read the trace:** the capability's probe score + `st_gain` (the encoder's marginal) + delta vs the probe baseline. `st_gain ≈ 0` → the architecture isn't earning it → **swing bigger**.
5. **Sweep in breadth** — both GPUs saturated with many architectural variants at once (each run is minutes, so run lots).
6. **Gate + record:** a real gain on the objective → update `scorecard.md` (probe-scoped Record/Status) + Ensue. Dead-ends Ensue'd too (they steer the next swing). Keep swinging.

## 2. Preferences
- **Surface = the Earth4D encoder ONLY** (`encoders/spacetime/earth4d.py` + `autoresearch/programs/spacetime/*`). Never the full fusion model.
- **Big architectural levers — take risks. NOT config knobs.** The general levers:
  - **Propagation / forecasting:** Earth4D is a static hash lookup today. Give it temporal state — a causal auto-regressive forecaster / 4D-LSTM (`recurrence.py`, `--recurrence`), a GNN message-passing propagator (`gnn.py`, `--gnn`), causal future split (`--forecast`).
  - **Field decode:** train the encoder end-to-end to decode the dense env / biology field (`env_field.py`, `--env_decode`, `--field_decode`) — represent the environment, don't index coordinates.
  - **New encoding architecture:** replace/augment the hash grid — learned Fourier features, SIREN, attention-over-neighbours, a coordinate transformer. Demanding rewrites of `earth4d.py`.
  - **New training objectives on the encoder:** cline-aware DOY loss, forecast reconstruction, contrastive env alignment, dense-field interpolation.
  - Capacity knobs (`spatial_levels`, `log2_hashmap`, `head_hidden`) are the SMALL end — only to tune a *winning* architecture, never the main move.
- **Measurement = the encoder probe, fast, native metrics** (never reimplemented). Same capabilities as the scorecard/science: env→species/family = `--sdm_presence`; phenology/timing = `--phenology`; env-vs-coord & field = `--env` / `--env_decode`; calibration = `calib_probe.py`. `st_gain` is the encoder's isolated marginal. Every run goes through `trace.py --ensue` (auto-logs to Ensue; token in `/workspace/.env`, never committed). Single-seed; noise negligible.

## 3. Don'ts
- **Don't train the full 799M fusion model** — confounded and slow. The surface is the encoder, measured by the probe.
- **Don't make marginal config tweaks the main move** (hash levels, loss weights) — take big architectural swings; don't be timid.
- No experiment without a declared `--metric`; no multi-seed / re-verification (GPU → breadth).
- Native probe metrics only — no reimplemented scoring.
- Don't game a capability via non-encoder signal; don't chase aggregate arith.
- Don't declare a capability a ceiling / "done" / "exhausted" — swing bigger.
- Don't leave the lossy prepared cache stale across data changes (`--fresh-data` when the probe rebuilds it).
