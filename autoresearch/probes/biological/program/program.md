# Biological encoder — autoresearch loop

## Goal
Make the phylogenetic species-graph impute biology for species from their relatives.
**Maximize one scalar: `bio_gain`** = mean of the seven graph-refinement gains (B56–B62 = capability WITH
the graph − WITHOUT). **Done when** every gain > +0.02 and no biological capability regresses.

## Requirements (science.md — this encoder must satisfy)
| rule | requirement | status |
|---|---|---|
| 7 | one embedding per species, shared along the evolutionary-tree topology | ✓ |
| 8 | self-supervised on a scientifically-derived dated tree only | ✓ |
| 9 | project species NOT in the tree into the same embedding space | ~ |
| 10–11 | every batch, an observation of species A updates its in-context neighbours B, C, … | ✓ |
| 12 | fast to gather/update (CUDA) | ✓ |
| 25 | phylo embedding is **maskable/reconstructable** — withhold a fraction per batch, reconstruct from relatives | ✗ starved |
| 26 | seed each species from a **frozen BioCLIP-2.5 ViT-H 1024-d text prior + small probe**, once per species/batch; unseen species use the same text→probe path | ✗ wrong seed |
| 27 | induce interactions **bidirectionally across two trees** (plant↔pollinator bilinear on two phylo-refined reps) | ✗ off |
| 29 | refine by the **exact O(N) two-pass OU-GP** (internal clade nodes = Markov blanket); out-of-tree species soft-attach; this exact op is the champion, not a dense/top-k kernel | ✓ |

`bio_gain ≈ 0` means the ✗ rows are unmet: the graph is built (29) but nothing forces it to reconstruct
masked species (25), the seed is wrong (26), and interactions don't flow across trees (27). The backlog
below closes those rows.

## Loop
```
   ┌──────────────────────────────  maximize bio_gain  ──────────────────────────────┐
   │                                                                                  │
 ① READ ──► ② PICK ──► ③ RUN ──────► ④ MEASURE ──────► ⑤ DECIDE ──► ⑥ WRITE ──┐    │
   Ensue      next       A/B: 1 toggle    score.py         beyond noise      Ensue │    │
  (tag=bio)  hypothesis  vs champion,    → bio_gain +      & floor held?     trace │    │
  open + dead  from ⑤'s  fixed budget    floor + BOTTLENECK  keep : diagnose (tag=bio)   │
             bottleneck                                                          │    │
   └──────────────────────────────────────────────────────────────────────◄─────┘    │
   └──────────────────────────────────────────────────────────────────────────────────┘
```

## ② Pick — architecture, not knobs
One structural change per round that satisfies a science.md rule this encoder fails. Reject anything that leaves
the mechanism unchanged. Filters: upholds science.md · fair controls (untouched baseline + mechanism ablation) ·
beats the ±0.008 noise floor.

## ③ Run — one variable, fixed budget
`VARIANT` = the champion path with your one bold hypothesis applied (it may bundle several coordinated changes
under a single thesis). `TAG` = `bio_<short-name>`.
```
rm -f data/deepcal/prepared_*.pt                                                  # cache round-trip is lossy — rm before every run
python -m deepearth.autoresearch.main.editable_files.harness.run_experiment VARIANT --cache_dir data/deepcal --tag TAG > TAG.log 2>&1
```
`run_experiment` installs the feedback instrument (auto-emits `[profile] refined_seed_norm`); budget = the
champion.yaml `time_budget_s` (rule 20). CONTROL = the same command on champion.yaml, run once → `CTRL.log`.

## ④ Measure — one command
```
python -m deepearth.autoresearch.main.editable_files.harness.score --log TAG.log --encoder biological --champion CTRL.log --ensue-tag biological
```
Emits `bio_gain` + Δ vs control · capability floor · per-benchmark Δ · the bottleneck · trace→Ensue.
**Bottleneck to read** (`[profile]`): `refined_seed_norm` (≈0 ⟹ graph moves nothing), `ou_rate_*`
(tree engaging?). Isolation (`_ablate_species`, graph ON vs OFF) is already inside the B56–B62 gains.

## ⑤ Decide
Keep if `bio_gain` rises beyond the single-seed noise floor **and** the capability floor (B7, B21, B41,
B53, B54, B55, B63) does not regress. Else: read the bottleneck, set the next hypothesis.

## Search space (axes, non-exhaustive — invert or invent past them)
| axis | rule | structural move |
|---|---|---|
| Seed | 26, 9 | reseed orthogonal to tree topology (BioCLIP trait/appearance prior) so the graph is additive, not redundant |
| Objective | 25, 27 | off family onto axes the seed lacks; loss = reconstruct a masked species from relatives, never identity |
| Operator | 29, 27, 9 | bidirectional plant↔pollinator two-tree message passing; internal-clade Markov blanket; out-of-tree soft-attach |
| Readout | 10–11 | autoregressive rollout — an observation of A updates neighbours B, C — not a detached head |

## Ensue (steps ① and ⑥, tag `biological`)
- **① READ** before picking: pull open hypotheses + logged dead-ends for `biological`; skip anything tried.
- **⑥ WRITE** after measuring: push `trace.json` (scalar, per-benchmark deltas, bottleneck) with a one-line
  verdict (kept / dead-end + reason). `score.py … --ensue-tag biological` does this.
