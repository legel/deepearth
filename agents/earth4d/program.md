# Earth4D Agent — program

**Objective:** break records on the 16 `scorecard.md` rows — the capabilities Earth4D must EARN from environment + spacetime coordinates. Every experiment runs through the fixed harness `trace.py` with a **required `--metric`** (the one row it optimizes). Levers are the *means*; the declared metric is the objective.

## 1. Loop
1. **Select the objective** — pick ONE of the 16 scorecard metrics (the worst row, or an operator-named target).
2. **Choose one lever** (Preferences) — write a full experiment yaml = `champion.yaml` + that single change, save under `agents/earth4d/exp/<name>.yaml`.
3. **Run through the fixed harness:** `python -m deepearth.agents.earth4d.trace --config agents/earth4d/exp/<name>.yaml --metric <Bxx> --device cuda:N --budget 4000 --ensue [--fresh-data]` (`--fresh-data` REQUIRED for any DATA-lever change). Every run produces the same trace **and `--ensue` auto-POSTs it to Ensue** (see §Ensue).
4. **Assess the bottleneck from the trace:** read the objective verdict + the per-row `spacetime_gain` — *earth4d-limited* (gain≈0 → encoder/data is the ceiling → reach for DATA/CAPACITY) vs *earth4d-contributing* / *supervision-limited* (score low but gain present, or no probe → reach for SCIENCE/LOSS). Pick the next lever from that read.
5. **Sweep in breadth** — keep both GPUs saturated with concurrent lever runs; bias toward the worst rows but cover widely.
6. **Gate (loose):** any real improvement on the objective row with no `>0.02` regression across the 16 → update that row's `Record`/`Status` in `scorecard.md` + `champion_report.py --save`. Every run (win OR dead-end) is logged to Ensue by the harness's `--ensue` (dead-ends steer the next sweep). Keep going broad.

### Ensue coordination
Run the harness with **`--ensue`** — it POSTs each experiment's full trace (objective verdict, 16-row deltas, bottleneck read, high-signal metrics) to Ensue as a `create_memory` record, so every result is coordinated automatically. The token is read from `ENSUE_API_TOKEN` (env) or `/workspace/.env` and is **never committed to the repo**. If no token is present the harness prints a skip notice and the run still completes. (Endpoint: `https://api.ensue-network.ai/`, JSON-RPC `tools/call` → `create_memory`.)

## 2. Preferences
- **GPU utilization + breadth first.** Single-seed (`seed 1337`, fixed per A/B). Noise is negligible — don't re-verify; don't stop at diminishing returns.
- **Levers (means), structural under-ask first.** The champion under-asks Earth4D: env `loss_weights` are `1.0` while `phylo/vision` are `2.0`; `condition_on` always hands the model vision; `worldclim`+`alphaearth` are loaded but OFF; `holdout` is spatial. Attack that first.
  - **A. DATA** (`data`/`variables`/`condition_on`): activate dormant `worldclim`+`alphaearth` (add to `variables`); `condition_on: []` (drop vision so env earns SDM); `holdout: temporal` + `time_axis` (unlocks forecasting); `time_km`, `n_neighbors`. *Any DATA change → `--fresh-data`.*
  - **B. SCIENCE/LOSS** (`model`): raise env `loss_weights` (climate/soil/topo/chm/hydro/clay → 2.0) — the primary under-ask; `sdist_weight ↑` (→ B29/B39/B40 dist-skill, the tractable env→where target); `flower_/lfmc_/myco_/poll_weight` per row (B26-28/B34/B42/B51); `contrastive_vars` +env; `smooth_geo_sigmas` (the lat→phenology cline).
  - **C. CAPACITY / Earth4D:** `capacity`, `freq_lr`, `relative_window`, absolute-encoder levels (`core/fusion.py:302`), `alphaearth_geo`. Training-objective levers (need code, FLAG them): cline-aware DOY loss on `pos_s`; forecast reconstruction loss on the temporal split.
  - **D. INFERENCE (no retrain):** temperature-scaling + distance-gated abstention → B23; cross-clade suitability rescue (vision→niche-centroid→env-match) → env-suitability rows.
- **Measurement discipline:** every run goes through `trace.py`; scores are `evaluate.py`'s, read from the run log — never reimplemented. `champion_report.py` gives the before→after record.

## 3. Don'ts
- **No experiment without a declared objective `--metric`** — the harness enforces it.
- **No multi-seed / re-verification runs** — noise is negligible; that GPU goes to breadth.
- **No reimplemented metrics** — native `evaluate.py` only (a reimplemented metric once gave a false +0.41 "clay win").
- Don't game the score via the borrowed-vision rows (photo→X ≥0.90) or chase the aggregate arith — earn the 16 env/spacetime rows.
- Don't over-gate on noise or stop at diminishing returns — keep sweeping broad.
- Don't destroy other rows (keep the `>0.02` regression guard).
- Don't declare a row a ceiling / "done" / "exhausted".
- **No unattended core edits** (`evaluate.py`/`fusion.py`/`train.py`) — training-objective levers get flagged for review, not silently patched.
- Don't leave the lossy prepared cache stale — `--fresh-data` on every data-lever change.
