# Biological encoder — autoresearch loop

## Goal
Make the phylogenetic species-graph impute biology for species from their relatives.

This loop has two deliberately different levels of evidence:

1. **Standalone research screen:** improve one biological capability's masked-imputation score against
   the strongest matched null tree. This establishes that the *real phylogeny*, rather than merely an
   extra graph module, carries the gain.
2. **Full-model integration:** periodically measure `bio_gain`, the mean of B56–B62 (capability WITH
   the graph − WITHOUT), while holding every biological capability floor. This establishes whether
   accumulated encoder progress is being used by the production model.

The null-tree screen chooses and validates hypotheses, and an exact two-seed confirmation decides
**encoder promotion**. The seven-term full-model score is the integration test; it is reported honestly
but does not gate pushing a confirmed encoder-specific breakthrough.
**Done when** every B56–B62 gain is > +0.02, every applicable standalone fair gain is positive, and no
biological capability regresses.

## Authority and scope

Scientific experiments edit only `editable_files/phylogenomic.py` or its local `editable_files/lib/**`
(`seeds.py` = the rule-26 lever, `training.py` = the rule-9/25/10-11 levers). The fixed evaluators
(`harness/probe.py`, `harness/traitprobe.py`), the board (`harness/board.py`), the fair control
(`harness/nulltree.py`) and the pollitree stages own validation data, splits, metrics, controls and
reporting, and are never edited to win a run. Fusion consumes the same public `SpeciesGraph` entrypoint
as the fixed probe and must not import biological probe code directly.

## The fair control: a null tree, not the seed

`bio_gain` compares the graph against **its own seed**. That is an ablation, not a control — it
confounds the tree with the parameter count, the training loop and the objective, and it is a hard bar
only because E1 already scores ~0.89 family-NN unaided. The probe's recordable gain is therefore
`vs null-tree`: the SAME operator with the SAME parameters and budget, run on a tree that is not the
phylogeny (five tip-label permutations of the identical buffers, plus a seed-built dendrogram; the
strongest is the baseline). `vs seed` is still reported and can never set a record. See
[`scorecard.md`](scorecard.md) and `harness/nulltree.py`.

Expect `vs null-tree` to be ≤ 0 at first. That is this program's own redundancy diagnosis becoming
measurable rather than being asserted — and it points at the Seed row of the search space below.

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

## ③ Run — standalone screen and confirmation; periodic full-model integration

`TAG` = `bio_<short-name>`. First run the fixed standalone evaluator for the capability. For example,
the family-imputation row is:

```
python -m deepearth.autoresearch.probes.biological.harness.probe \
  --cache_dir autoresearch/data/deepcal --result-json /tmp/$TAG.json
python -m deepearth.autoresearch.probes.biological.harness.board \
  --capability family_from_phylo --result-json /tmp/$TAG.json --tag $TAG
```

The other recordable modes and their evaluator flags are listed in `program/scorecard.md`. A screen is
evidence only when `vs null-tree` clears the noise barrier; `--no_control` is diagnostic.
For confirmation, run the evaluator once per matched seed and pass both files to one board command:

```
python -m deepearth.autoresearch.probes.biological.harness.board \
  --capability family_from_phylo --result-json /tmp/$TAG.seed0.json /tmp/$TAG.seed1.json --tag $TAG
```

Only a screen-clearing mechanism earns the exact two-seed confirmation. A confirmed encoder breakthrough
may be promoted immediately under the encoder gate below. Periodically—after a promotion or a small batch
of accumulated encoder promotions—run full-model integration. `VARIANT` is the champion path with those
encoder changes applied:

```
rm -f autoresearch/data/deepcal/prepared_*.pt                                                  # cache round-trip is lossy — rm before every run
python -m deepearth.autoresearch.main.harness.run_experiment VARIANT --cache_dir autoresearch/data/deepcal --tag TAG > TAG.log 2>&1
```
`run_experiment` installs the feedback instrument (auto-emits `[profile] refined_seed_norm`); budget = the
champion.yaml `time_budget_s` (rule 20). CONTROL = the same command on champion.yaml, run once → `CTRL.log`.

## ④ Measure — one command
```
python -m deepearth.autoresearch.main.harness.score --log TAG.log --encoder biological --champion CTRL.log --ensue-tag biological
```
Emits `bio_gain` + Δ vs control · capability floor · per-benchmark Δ · the bottleneck · trace→Ensue.
**Bottleneck to read** (`[profile]`): `refined_seed_norm` (≈0 ⟹ graph moves nothing), `ou_rate_*`
(tree engaging?). Isolation (`_ablate_species`, graph ON vs OFF) is already inside the B56–B62 gains.

## ⑤ Decide

Keep a standalone candidate only if its masked-imputation fair gain against the strongest null tree
rises beyond the single-seed noise floor. Promote it as an **encoder breakthrough** only when the exact
two matched seeds confirm the declared margin, each seed beats its strongest paired null, the result is
clean and attributable, and no registered standalone encoder capability regresses. Push that confirmed
encoder breakthrough to `deepcal-ensue-autoresearch` with the before→after encoder score in the commit.

Run full-model `bio_gain` periodically as an integration scorecard. A fusion regression is real follow-up
work, but it does not erase or block a confirmed encoder-specific result: the standalone score attributes
the encoder mechanism, while fusion measures whether the current product consumes it effectively.

**`bio_gain` can be negative, and a negative is information.** Until recently it could not be: B57, B58
and B62 were rectified at zero in `evaluate.py` while B56, B59, B60 and B61 were not, and
`scoring/definitions.py::normalized` then clipped all thirteen `_gain` keys to `[0,1]` regardless. The
objective was therefore a mean of seven terms where three could not represent a loss, and a mechanism
that hurt flowering, LFMC or mycorrhiza scored exactly like one that did nothing. Both clamps are gone.
Any `bio_gain` in a log or trace written before that is not comparable to one written after — it was
measured on an instrument that could only round up.

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
