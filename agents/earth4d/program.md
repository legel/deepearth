# Earth4D Agent — program

**Mandate:** raise the 16 rows in `scorecard.md` (env→identity, ecology, calibration, phenology, env-recon). *Earn* them from environment + spacetime coordinates — never lean on borrowed vision to inflate a row. Update `Record`/`Status` ONLY on a verified gain.

## Non-negotiable rules
1. Score with the native `autoresearch/evaluate.py` only. **No reimplemented metrics** — a reimplemented metric already produced a false +0.41 "clay win."
2. Promote a change only if `autoresearch/champion_report.py` before→after shows the target row up **and** no regression on the other 15 **and** no full-suite regression.
3. One lever at a time, matched A/B at fixed budget. Single-seed exploratory; verify by shipping the next step, not by re-seeding.
4. Prefer additive config/inference levers first; training-objective changes are the big levers — flag them, they need a full run.
5. Every A/B (win or dead-end) → Ensue + update `scorecard.md`.

## Lever backlog (worst-first, from the mechanism map)
- **B23 calibration 0.143** → temperature-scaling + distance-gated abstention. Near-free; halves cross-clade ECE in probes. *Cheapest first win.*
- **B6 0.103 / B8 0.127 family, B28 0.451 phenology** → cline-aware training objective: pos_s encodes latitude (R² 0.96) but *discards* lat→phenology (R² ≈ 0). Add an auxiliary DOY / seasonal-timing loss on the spatial channel.
- **B1 0.323 / B5 0.399 species-identity** → structurally bounded (env → ~182 co-suitable species; identity is the image's job). Target env→**suitability** (dist-skill B29/B39/B40), not top-1 identity. Cross-clade suitability rescue (vision→niche-centroid→env-match) is +0.45 AUC but *suitability-only*.
- **B29/B39/B40 dist-skill (inactive)** → train the dense-field decode (`query_field`, currently inference-only).
- **B25 forecast (inactive)** → forecast reconstruction loss on the temporal-holdout split; also lifts B23.
- **Capacity** → absolute-encoder levels hardcoded 18/18/20 in `core/fusion.py`; expose if fine-scale-starved (watch throughput).

## Loop
pick worst-scoring row → hypothesis + one lever → run on the synced box → native-score the 16 via `evaluate.py` → `champion_report` gate → update `scorecard.md` Record/Status → Ensue log.

## State
Baseline @ `0a643fc` · 9 ❌ · 2 ⚠️ · 5 ✅ · mean 0.409. **Box `newbox` must `git pull` to 0a643fc before running (currently 6 behind).** Prior `autoresearch/programs/spacetime/` is prior art, not a base — this agent is a clean restart.
