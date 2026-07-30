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
| **Surface** | `autoresearch/probes/spacetime/editable_files/earth4d.py` + `autoresearch/probes/spacetime/` probes + the channels they feed. Never the fusion model. |
| **Shared state** | **Ensue** (swarm-wide) ⇄ `records.json` (per-run ledger) ⇄ `scorecard.md` (the board) |
| **Ops** | box, GPUs, token, commit identity → `autoresearch/probes/spacetime/program/box-operations.md` |

## Loop

```
        ┌───────────────────────── bottleneck reason ─────────────────────────┐
        │                                                                     │
        ▼                                                                     │
   ① READ ──► ② PICK ──► ③ DIAGNOSE ──► ④ RUN ──► ⑤ MEASURE ──► ⑥ DECIDE ────┤
        │                                                                     │
   Ensue +   worst /      fair-gain     trace.py    score        beyond        │
   records   highest-     → lever       1 variable  fair-gain    noise?        │
   dead-ends leverage     family        fixed       Δ baseline   no regress?   │
        ▲    row                        budget                                 │
        │    --metric                                                          ▼
        └───────────────── ⑦ WRITE ──► Ensue + records.json ◄────────── keep ──┘
                                                                     else ─► ③
```

| step | do | rule |
|---|---|---|
| ① READ | pull Ensue keys + `records.json` from disk | never reason from a cached board; skip logged dead-ends |
| ② PICK | one capability from `scorecard.md` Layer 1, **with intention** | no run without a declared `--metric` |
| ③ DIAGNOSE | fair-gain → lever family | diagnose before you swing |
| ④ RUN | `trace.py --metric <cap> --probe "<flags>" --tag <id> --device cuda:N --ensue` | **change whatever the hypothesis needs**; one variable per run, fixed budget; sweep across both GPUs |
| ⑤ MEASURE | score · fair-gain · Δ vs baseline | native probe metrics only |
| ⑥ DECIDE | keep if beyond noise with no registered regression | probe = discovery; science needs the gate |
| ⑦ WRITE | `--ensue` on **every** run — win or dead-end | a run that isn't published didn't happen |

Default protocol trains a head on **frozen random-hash** features — fair-gain compares priors as fixed
feature maps, not learned Earth4D. `--train_encoder` trains end-to-end. State the protocol behind every
number.

### What constrains a run — identity, not permitted edits

You may change anything the hypothesis needs: data channel, probe mode, encoder internals, objective.
Nothing is gated behind a menu of approved flags. What the harness enforces is that the run still
measures the capability you declared:

```
   measurement identity = capability · mode · split · n_shards · protocol · code hash
   comparable  ⇔  identities match        different mode/shards = a DIFFERENT TARGET, not a better score
```

A run whose identity differs from the record's is recorded as a re-baseline or withheld — never as a
win. That is the whole guardrail; within it, swing as hard as the hypothesis demands.

## Ensue — the swarm's shared memory

Every agent in the swarm reads the same keys before picking and writes back after measuring. That is the
only thing keeping parallel workers from re-running each other's dead-ends.

```
   agent A ─┐                                    ┌─► agent A picks next lever
   agent B ─┼──► ① READ ── Ensue ── ⑦ WRITE ──┼─► agent B skips A's dead-end
   agent C ─┘        keys      (upsert)         └─► agent C sees the new record
                       │
        LOOP-earth4d-<capability>   ← ONE key per capability, upserted, never appended blindly
        ├─ BEST  score (fair-gain, fair baseline)   ← the running record
        ├─ records[]   last 20 {tag, score, gain, protocol, rebaseline_from}
        └─ deadends{}  last 40 {tag: {score, gain, why}}   ← deduped BY TAG, each with its reason
```

| | |
|---|---|
| **Transport** | `trace.py --ensue` → POST `https://api.ensue-network.ai/`, `{"items":[{"key_name": ...}]}` |
| **Auth** | `ENSUE_API_TOKEN` from env or `/workspace/.env`; `trace.py` finds it. **Never commit it.** |
| **Key taxonomy** | `LOOP-earth4d-<capability>` — one per capability, so the board is readable at a glance |
| **Local mirror** | `records.json` holds the same ledger plus the full per-run trace; `<tag>.trace.json` per run |

Rules:
- Pass `--ensue` on **every** run. A lever that failed is information; publish it with its bottleneck
  reason or the swarm pays to rediscover it.
- **Read before you pick, re-read after you write.** Another agent may have taken your capability or
  killed your lever mid-cycle.
- A record is only beaten **like-for-like** — same probe mode, same shard count, same protocol. A
  different mode is a different target; it is withheld and flagged, not published as a win.
- Never publish a max-of-reruns as an estimate.

## ③ Diagnose

```
        fair-gain = Earth4D − strongest fair baseline (trained PE / RFF / MLP)
        share     = fair-gain / score      ← how much of the score the ENCODER contributes
                                  │
             ┌────────────────────┴────────────────────┐
        fair-gain ≤ 0                          fair-gain > 0
             │                          ┌────────────┴────────────┐
             │                    share < 25%              share ≥ 25%
             ▼                          ▼                         ▼
       INPUT-limited              ENCODER-limited             EARNING
       DATA lever                 ARCHITECTURE lever          push the mechanism further
       change the channel         change the mechanism
```

**The read is a fraction, not an absolute cutoff.** It used to be `fair-gain > 0 and score < 0.20 →
ENCODER-LIMITED`, which applied one constant to every target regardless of difficulty:
`species_from_spacetime` (~2,009 classes, chance ≈0.0005) at 0.0512 and `family_from_spacetime` (166
classes, chance ≈0.006) at 0.1769 both tripped it. Acting on that sent four consecutive mechanism
changes at a capability whose encoder was already contributing 84% of its score — `--recurrence`
−0.0180, `--gnn` −0.0261, `--train_encoder` +0.0037, a tri-plane conjunction edit −0.0001.

A share is comparable across targets of any difficulty and answers the question the lever choice turns
on: is the encoder doing the work, or barely beating a generic positional encoding?

Flat fair-gain across a whole input type = signal-limited. Change the channel, don't swing the
architecture. And note what the read does **not** say: a high share does not mean a capability is
finished, and a low absolute score is not evidence of a ceiling — where the ceiling is, is the thing
being discovered.

## Levers

| DATA | ARCHITECTURE |
|---|---|
| `--env_channels {worldclim, alphaearth, all}` · `--env_extra` · `--sdm_channels` | edit `earth4d.py`: `__init__`, forward, objective |
| `--vision --vision_feats {dino, bio, both}` | propagation / forecasting: `--recurrence` `--gnn` `--forecast` |
| `--pheno_channel` | field decode: `--env_decode` `--field_decode` |
| densification · channel fusion · per-entity aggregation | new structure: learned Fourier, SIREN, attention-over-neighbours, causal temporal state |
| | new objectives |

Capacity knobs (`spatial_levels`, `log2_hashmap`, `head_hidden`, `time_harmonics`) tune a winner — they
are not the move.

### An experiment is an EDIT on a BRANCH — not a new file, not a new flag

```
   ① branch          git worktree add ../e4d-<tag> -b exp/<tag>
   ② EDIT in place   change probe/ and autoresearch/probes/spacetime/editable_files/earth4d.py directly. No copy, no new
                     module, no gated flag. The branch is the isolation.
   ③ sweep           run it; the diff IS the experiment
   ④ dies            delete the branch. The edit vanishes with it; Ensue keeps the reason.
   ⑤ graduates       only then gate it default-off so the champion path stays byte-identical,
                     and only then does a flag exist
```

**Gate at graduation, not at conception.** This rule used to read "back up → gate DEFAULT-OFF → wire a
probe flag" as the *first* step, and with no branch discipline that was the only isolation available. So
every idea became permanent surface: **113 flags**, 19 `if a.<flag>:` branches, a 1,552-line `main()`,
and when a flag felt too invasive, a copied file — `b42.py`, `diag2…diag8`, `mm_envrecon.py`,
`*_sota.py`, 21 `champion_*.yaml`. Nothing ever said *remove the flag when the idea dies*, so nothing
ever shrank, and every agent after paid the reading cost.

The cost is not theoretical: `--phenology` silently shadows `--pheno_env`/`--pheno_taxon`/
`--pheno_densefield` (≈120 unreachable lines, and a live record standing on an inert flag), and eight
modes require `--forecast` through bare `assert`s buried mid-function.

- **Never add a file to `probe/` for one idea.** Those files are the instrument; an experiment edits
  them. A new module there is only justified when a mode is permanent and registered.
- **A dead flag is a bug.** If an experiment ends, its flag and branch go with it.
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
| 2 | **Three-way split.** Search on train, select once on validation, evaluate once after freezing. Future+new-place test = `train: past & seen`, `test: future & held`; other quadrants embargoed, never folded into `~test`. Fit every range, normalizer, aggregate, imputer on train alone. |
| 3 | **Real autoregression for a causal claim.** Consume observed past state; roll your own predictions forward. A positional lookup at `t-lag` is a delayed basis, not memory. |
| 4 | **Fair controls.** Persistence · climatology · raw coordinates · RFF/SIREN · matched-capacity MLP · the same propagator without Earth4D · shuffled-history · time-reversal · future-sentinel. Paired arms get identical data, seeds, wall time, tuning budget, and asserted-matched head params. |
| 5 | **Predeclared statistics.** Endpoints declared before running. ≥5 matched seeds, block bootstrap over the relevant unit. Pass only if the lower 95% bound of improvement over the **strongest fair baseline** > 0 and the point estimate clears the declared margin, with no regression. |
| 6 | **Fixed budget, immutable record.** Equal declared wall-clock per arm. Append-only hash-chained ledger: code/data/split/config/seed hashes, signed per-arm outcomes, every attempted variant. Freeze before opening test; replicate on a second region or later period before a headline. |

**Quarantine:** any lever whose neighbour state or target window can cross the forecast origin is
unusable until future-sentinel, horizon-purge, and right-censoring tests pass. Log it against the lever
in Ensue so the whole swarm inherits the block.

## Experiment tracking

One experiment = one branch. Many agents share this box and **one** board, so never run in the
shared checkout — another agent is reading it.

```
   ① git worktree add ../e4d-<tag> -b exp/<tag>      one experiment, one branch, own directory
   ② commit EVERY run on it — failures too           a .pre-X copy is a commit you didn't make
   ③ scratch/ for logs, one-offs, dumps              gitignored, disposable, never beside source
   ④ dead end  → leave it on the branch              the Ensue dead-end entry is the durable record
   ⑤ BREAKTHROUGH → rebase, then the OPERATOR pushes    nothing else reaches the remote. Ever.
```

**A breakthrough is a result that cleared the evidence standard above** — multi-seed, no regression,
identity matched, reproducible from a committed tree. Not a probe record. Not a single seed. Not a
score that beat the board by a hair.

Two exceptions may go to main without a breakthrough: **harness/contract/test changes** the whole
swarm depends on, and **provenance fixes** to the board. Both are infrastructure, not findings.

- A record from an unpushed commit is **discovery-only** — nobody else can reproduce it. That is a
  statement about how much such a record can CLAIM, not a licence to push: an agent never pushes on its
  own judgement. This is a shared repository, and an experiment branch — above all a failed one — stays
  local. Commit everything; let the operator decide what leaves the machine.
- Only the checkout that owns `records.json` writes records; everyone else measures with
  `EARTH4D_ALLOW_UNRECORDED=1`.
- No config variants as files (21 `champion_*.yaml` = 21 undocumented experiments). Pass overrides as
  flags; if one wins, commit it *with* its result.
- Tests are always tracked.

## Don'ts

- Don't train the full fusion model — confounded and slow.
- Don't default to architecture. DATA is co-equal; follow the fair-gain.
- No reimplemented metrics. No publication without multi-seed re-verification.
- Attribute borrowed signal: a vision win is frozen DINO/BioCLIP (env = *where*, vision = *which*), not
  a coordinate-encoder gain. Don't launder it; don't chase an aggregate mean.
- Never call a capability done or exhausted — switch lever family and continue.
- Keep specific experiments, datasets, and campaign directives out of this file. They belong in Ensue
  and `records.json`, where the swarm can update them.
