# Earth4D Agent — program

**This loop owns one probe: Earth4D over space-time coordinates and environmental channels.** Its job is
to *recover a real signal* on one capability at a time and prove the signal is the encoder's — not
borrowed from a frozen pretrained embedding, and not an artefact of the split. A signal that survives
this loop's validation is what later earns a place in the fusion layer (`autoresearch/main/`); the full
model runs last, after the science in `science.md` is filled out. Raising an aggregate is not this loop's
job.

Two co-equal lever families: **DATA** (what signal feeds the encoder) · **ARCHITECTURE** (how it
represents it).

| | |
|---|---|
| **Surface** | `editable_files/earth4d.py`, `editable_files/probe.py` (the `CONFIG` block), `editable_files/lib/`. Never the fusion model. |
| **The judge** | `probes/spacetime/harness.py` + `autoresearch/scoring/`. **Not editable to win a run.** |
| **Shared state** | **Ensue** (swarm-wide) ⇄ `records/records.json` (per-run ledger) ⇄ `program/scorecard.txt` (the board) |
| **Ops** | box, GPUs, token, commit identity → `program/box-operations.md` |

## What is measured — protocol v5-encoder-only

**The probe measures Earth4D as `fusion.py` instantiates it, and nothing else.**

```
   encoder   Earth4D(spatial_levels=18, temporal_levels=18, log2_hashmap=20,
                     freq_log_scale_init=-2.5)          = 36 spatial + 108 tri-plane = 144 dims
   control   RFF at FAIR_CONTROL_DIM = 144              matched width, train-extent, bandwidth-selected
   protocol  train_encoder=True, EARTH4D_DETERMINISTIC=1, 800 steps per arm
   gain      "vs RFF"  — every capability, same quantity
```

Why this is stated so precisely: until v5 it was not true. At the v4 champion the head received **20,663
features and the hash grid was 36 of them — 0.17%**. CMAC tile coding was 18,432 (89.2%), a fixed RFF
another 2,048 (9.9%), and `drop_spatiotemporal` had deleted the tri-planes. Every `fair_gain` on this
board scored a tile coder with a hash-shaped residue attached. "Dropping the inert tri-planes" looked
free because it removed 108 dims out of 20,663.

The bolt-on bases (`fourier`, `time_harmonics`, `spatial_cline`, `nystrom`, `tile*`) all still exist and
all default OFF. They are legitimate experiments. They are not the encoder — **a run that turns one on is
measuring the encoder PLUS that basis and its record must say so.**

**`EARTH4D_DETERMINISTIC=1` on every run.** The trained path used to be nondeterministic at fixed seed
(five seed-0 runs: 0.1873 / 0.1925 / 0.1867 / 0.1872 / 0.1952, sd 0.0038 — as large as the entire
across-seed spread), because the hash-grid backward accumulated colliding gradients with float
`atomicAdd`. That is why every pre-v5 record was frozen-random-encoder: an irreproducible number cannot
set a record. `utils.cuh::atomicAddFixed` replaces those with order-independent int64 accumulation —
verified bit-identical on all four encoders, and 4.5% *faster*. Check it any time with
`harness.py --determinism`.

## Loop

```
        ┌───────────────────────── bottleneck reason ─────────────────────────┐
        │                                                                     │
        ▼                                                                     │
   ① READ ──► ② PICK ──► ③ DIAGNOSE ──► ④ RUN ──► ⑤ MEASURE ──► ⑥ DECIDE ────┤
        │                                                                     │
   Ensue +   worst /      fair-gain     harness      score        beyond       │
   records   highest-     → lever       1 variable  fair-gain    noise?        │
   dead-ends leverage     family        fixed       Δ baseline   no regress?   │
        ▲    row                        budget                                 │
        │    --metric                                                          ▼
        └───────────────── ⑦ WRITE ──► Ensue + records.json ◄────────── keep ──┘
                                                                     else ─► ③
```

| step | do | rule |
|---|---|---|
| ① READ | `harness.py --insights [--metric <cap>]` — 2,531 prior runs and 123 dead-ends with their reasons, plus the Ensue key | **mandatory before picking.** Never reason from a cached board; skip a lever whose recorded reason still applies |
| ② PICK | one capability from `scorecard.md` Layer 1, **with intention** | no run without a declared `--metric` |
| ③ DIAGNOSE | fair-gain + signal-capture → lever family | diagnose before you swing |
| ④ RUN | `EARTH4D_DETERMINISTIC=1 python -m deepearth.autoresearch.probes.spacetime.harness --metric <cap> --tag <id> --device cuda:N --ensue` | **change whatever the hypothesis needs**; one variable per run; SCREEN at one seed and go broad |
| ⑤ MEASURE | score · fair-gain · signal-captured · Δ vs baseline | native probe metrics only |
| ⑥ DECIDE | keep if beyond the noise barrier with no registered regression | probe = discovery; science needs the gate |
| ⑦ WRITE | `--ensue` on **every** run — win or dead-end | a run that isn't published didn't happen |

### ① READ — the old board is void, its reasons are not

v5 voids every stored SCORE: they measured a feature vector that was 0.17% encoder, against a fair-gain
column that mixed encoder / env-channel / class-prior gains. It does **not** void the hypotheses. 123
dead-ends carry the reason they stopped, and re-buying them under the new regime is pure waste.

```bash
python -m deepearth.autoresearch.probes.spacetime.harness --insights --metric <capability>
```

Read the reason, not the number, and sort each into one of two piles:

| the lever failed because... | status under v5 |
|---|---|
| its own mechanics — `extent_fit -0.0199`, the 17-row `fc_hh*_ff*_th*` capacity sweep, `--gnn -0.0261` | **settled.** Don't re-run it. |
| it was drowned in 18,432 tile-code dims, or judged by a `share` that moved with output width, or scored against the class prior | **never measured.** It is a fresh hypothesis on the encoder. |

That second pile is the alpha the clean regime unlocks — most of the architecture arms in `earth4d.py`
were rejected while contributing a fraction of a percent of the signal.

**Where do I edit?** `python -m deepearth.autoresearch.scoring.definitions --capability <cap>` names the
files. It is the one routing table, it lives in the harness, and an experiment cannot widen its own scope
by editing it.

### What constrains a run — identity, not permitted edits

You may change anything the hypothesis needs: data channel, encoder internals, objective. Nothing is
gated behind a menu of approved flags. What the harness enforces is that the run still measures the
capability you declared:

```
   measurement identity = capability · mode · split · n_shards · protocol · config_digest
   comparable  ⇔  identities match      different mode/shards = a DIFFERENT TARGET, not a better score
```

A run whose identity differs from the record's is recorded as a re-baseline or withheld — never as a win.

## ③ Diagnose

Every run reports two independent reads. Use both.

### A. Does the encoder beat a matched encoder?

```
        fair-gain = Earth4D − RFF at the SAME width, same data, same split, same head
                                  │
             ┌────────────────────┴────────────────────┐
        fair-gain ≤ 0                          fair-gain > 0
             ▼                                          ▼
       INPUT-limited                            the mechanism carries signal
       DATA lever — change the channel          push the ARCHITECTURE further
```

`share = fair-gain / score` is reported but **do not act on it across arms of different width.** Measured:
padding the encoder output with columns of literal zeros — adding no information — moved
`family_from_spacetime`'s share from 20.7% (dim 2592) to 27.2% (dim 3024) to 15.1% (dim 3744). Under v5
every capability runs at 144 dims so share is comparable *between capabilities*; it stops being
comparable the moment an arm turns on a bolt-on basis.

### B. How much of the available signal did it capture?

```
   floor    predict the train marginal — the score with ZERO coordinate information
   ceiling  empirical p(family | spatial cell), finest cell with >=3 train points, backing off
            = the Bayes-optimal predictor GIVEN POSITION. No function of the coordinate beats it.
   captured (encoder − floor) / (ceiling − floor)
```

| reading | meaning | action |
|---|---|---|
| `captured` → 1.0 | the coordinates are exhausted | **stop tuning architecture — add a channel** |
| `captured` low, `ceiling` high | the signal is there and the encoder cannot represent it | **this is when architecture work is justified** |
| `ceiling` ≈ `floor` | position does not carry this target on this split | no encoder fixes it — wrong target or wrong split |

`fair-gain` alone cannot distinguish the last two: both look like "small gain". This is the read that
tells you when to stop, and it is the only one that does.

Flat fair-gain across a whole input type = signal-limited. Change the channel, don't swing the
architecture. A high share does not mean a capability is finished, and a low absolute score is not
evidence of a ceiling — where the ceiling is, is the thing being discovered.

### C. The axes every run carries

`axis_R5_params_M` (science.md rule 5 floor is 100M — the v4 champion silently ran ~37.7M),
`axis_R21_fwd_bwd_ms_per_1k_coords` (rule 21 makes speed a score lever; the budget is still
`CONFIG["steps"]=800`, so a speedup is *visible* here but cannot yet move the primary — `budgeted()` and
`CONFIG["time_budget_s"]` are wired and switched OFF, waiting on one run to measure what 800 steps
costs, because flipping the budget in the same change that reshaped WHAT is measured would confound the
v5 re-baseline), and
`axis_signal_*` from B above. Reported on all seven declare sites, computed once per run.

## Levers

An experiment is a **diff of the `CONFIG` block in `probe.py`**, or of `earth4d.py`. There are no flags —
the CLI carries only what decides WHICH measurement is being made (capability, seed, device, result path).

| DATA | ARCHITECTURE |
|---|---|
| `env_channels`, `env_extra`, `sdm_channels`, `cooccur_channels` | `earth4d.py`: `__init__`, forward, objective |
| `vision`, `vision_feats` | `lib/recurrence.py`: `run_recurrence`, `run_field_decode`, propagators |
| `pheno_channel`, densification, per-entity aggregation | `lib/phenology.py`, `lib/dyntargets.py`: targets and their coord encoders |
| channel fusion | new structure: learned Fourier, SIREN, attention over neighbours, causal temporal state |
| | new objectives |

Capacity knobs (`spatial_levels`, `log2_hashmap`, `head_hidden`) tune a winner — they are not the move.
Neither is a sweep over them: 17 `fc_hh*_ff*_th*` dead-ends on this board bought nothing.

### An experiment is an EDIT on a BRANCH — not a new file, not a new flag

```
   ① branch          git worktree add ../e4d-<tag> -b exp/<tag>
   ② EDIT in place   change CONFIG in probe.py and earth4d.py directly. No copy, no new module,
                     no gated flag. The branch is the isolation.
   ③ sweep           run it; the diff IS the experiment
   ④ dies            delete the branch. The edit vanishes with it; Ensue keeps the reason.
   ⑤ graduates       only then gate it default-off so the champion path stays byte-identical
```

**Gate at graduation, not at conception.** This rule used to read "gate DEFAULT-OFF → wire a probe flag"
as the *first* step, and with no branch discipline that was the only isolation available. So every idea
became permanent surface: **113 flags**, 19 `if a.<flag>:` branches, a 1,552-line `main()`, and when a
flag felt too invasive, a copied file. Nothing ever said *remove the flag when the idea dies*, so nothing
ever shrank. The flags are gone; the same accretion then regrew inside `earth4d.py` as a 60-parameter
constructor, most of whose arms have a recorded dead-end and are still there.

- **A dead flag is a bug.** If an experiment ends, its lever and its branch go with it.
- **Never add a file for one idea.** A new module is justified only when a mode is permanent.
- Before graduating an architectural win, confirm the champion doesn't already carry that prior.

## Evidence standard — binding before any claim or scale-up

```
   probe record ──► ranks hypotheses                      ✗ not science
                     │
                     └─ passes ALL SIX ──► confirmable claim ──► scale
```

| # | requirement |
|---|---|
| 1 | **Measured state, not a proxy of the query.** A target derived from the sampling process measures the observer, not the system. |
| 2 | **Three-way split.** Search on train, select once on validation, evaluate once after freezing. Future+new-place test = `train: past & seen`, `test: future & held`; other quadrants embargoed. Fit every range, normalizer, aggregate, imputer on train alone. |
| 3 | **Real autoregression for a causal claim.** Consume observed past state; roll your own predictions forward. A positional lookup at `t-lag` is a delayed basis, not memory. **No mode implements this today** — see `scoring.definitions --coverage`. |
| 4 | **Fair controls.** Persistence · climatology · raw coordinates · RFF/SIREN · matched-capacity MLP · the same propagator without Earth4D · shuffled-history · time-reversal · future-sentinel. Paired arms get identical data, seeds, wall time, tuning budget, and matched width. |
| 5 | **Predeclared statistics.** Endpoints declared before running. ≥5 matched seeds, block bootstrap over the relevant unit. Pass only if the lower 95% bound of improvement over the strongest fair baseline > 0 and the point estimate clears the declared margin, with no regression. |
| 6 | **Fixed budget, immutable record.** Equal declared wall-clock per arm. Append-only ledger: code/data/split/config/seed hashes, per-arm outcomes, every attempted variant. Freeze before opening test; replicate on a second region or later period before a headline. |

### Two tiers: screen at one seed, spend seeds only on a candidate

```
   MANY hypotheses ──► 1 seed each ──► barrier (2% of record, floor 0.002) ──┬─ clears → PROBE RECORD
                                                                             └─ flat   → dead-end, publish
   PROBE RECORD ──► 5 matched seeds ──► requirement 5 ──► CLAIM
```

**Screening is one seed.** The barrier is the filter; that is what it is for. Measuring one idea five
times costs the same as screening five ideas once, and only the second discovers anything. A flat arm at
one seed is a complete answer — publish it and move on.

**A probe record is not a claim.** It ranks hypotheses. Promoting one to a claim, or to graduation,
requires the five matched seeds of requirement 5. `graduation.py` refuses any record still marked
`provisional` (<5 seeds) or `code.dirty`, precisely so this tier boundary cannot be crossed by accident.

**Quarantine:** any lever whose neighbour state or target window can cross the forecast origin is
unusable until future-sentinel, horizon-purge and right-censoring tests pass. Log it against the lever in
Ensue so the whole swarm inherits the block.

## Ensue — the swarm's shared memory

Every agent reads the same keys before picking and writes back after measuring. That is the only thing
keeping parallel workers from re-running each other's dead-ends.

```
        LOOP-earth4d-<capability>   ← ONE key per capability, upserted, never appended blindly
        ├─ BEST  score (fair-gain, fair baseline)   ← the running record
        ├─ records[]   last 20 {tag, score, gain, protocol, rebaseline_from}
        └─ deadends{}  last 40 {tag: {score, gain, why}}   ← deduped BY TAG, each with its reason
```

| | |
|---|---|
| **Transport** | `harness.py --ensue` → POST `https://api.ensue-network.ai/` |
| **Auth** | `ENSUE_API_TOKEN` from env or `autoresearch/.env`. **Never commit it.** |
| **Local mirror** | `records/records.json` holds the same ledger plus the full per-run trace |

- Pass `--ensue` on **every** run. A lever that failed is information; publish it with its bottleneck
  reason or the swarm pays to rediscover it.
- **Read before you pick, re-read after you write.** Another agent may have taken your capability.
- A record is only beaten **like-for-like** — same mode, shard count, protocol, config digest.
- Never publish a max-of-reruns as an estimate. Seven consecutive +0.0006 steps were each accepted as a
  new best on `family_from_spacetime` before that walk was invalidated; the noise barrier exists because
  of it.

## Experiment tracking

One experiment = one branch. Many agents share this box and **one** board, so never run in the shared
checkout — another agent is reading it.

```
   ① git worktree add ../e4d-<tag> -b exp/<tag>      one experiment, one branch, own directory
   ② commit EVERY run on it — failures too           a .pre-X copy is a commit you didn't make
   ③ scratch/ for logs, one-offs, dumps              gitignored, disposable, never beside source
   ④ dead end  → leave it on the branch              the Ensue dead-end entry is the durable record
   ⑤ BREAKTHROUGH → rebase, then the OPERATOR pushes nothing else reaches the remote. Ever.
```

- A record from an unpushed or **dirty** commit is discovery-only — nobody can reproduce it, and
  `graduation.py` will refuse it. Commit before you record.
- An agent never pushes on its own judgement. Commit everything; the operator decides what leaves.
- Only the checkout that owns `records.json` writes records; everyone else measures with
  `EARTH4D_ALLOW_UNRECORDED=1`.
- No config variants as files. An override lives in the `CONFIG` diff, committed *with* its result.

## Don'ts

- Don't train the full fusion model — confounded and slow, and not this loop's job.
- Don't default to architecture. DATA is co-equal; follow the fair-gain and the signal-capture read.
- Don't turn on a bolt-on basis and call the result an encoder number.
- No reimplemented metrics. Scoring lives in `autoresearch/scoring/` and is not editable to win a run.
- Attribute borrowed signal: a vision win is frozen DINO/BioCLIP (env = *where*, vision = *which*), not a
  coordinate-encoder gain. Don't launder it; don't chase an aggregate mean.
- Never call a capability done or exhausted — switch lever family and continue.
- Keep specific experiments, datasets and campaign directives out of this file. They belong in Ensue and
  `records.json`, where the swarm can update them.
