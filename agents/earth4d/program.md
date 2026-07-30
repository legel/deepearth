# Earth4D Agent — program

Discover, then confirm, what Earth4D earns from space-time coordinates and environmental channels.
Two co-equal lever families: **DATA** (what signal feeds the encoder) · **ARCHITECTURE** (how it
represents it).

| | |
|---|---|
| **Surface** | `encoders/spacetime/earth4d.py` + `autoresearch/programs/spacetime/` probes + the channels they feed. Never the fusion model. |
| **Shared state** | **Ensue** (swarm-wide) ⇄ `records.json` (per-run ledger) ⇄ `scorecard.md` (the board) |
| **Ops** | box, GPUs, token, commit identity → `agents/earth4d/box-operations.md` |

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
              fair-gain  =  Earth4D  −  generic trained PE / RFF
                                  │
             ┌────────────────────┴────────────────────┐
        ≈ 0 or negative                        positive but score low
             │                                          │
      INPUT-limited                             ENCODER-limited
             ▼                                          ▼
        DATA lever                             ARCHITECTURE lever
      change the channel                       change the mechanism
```

Flat fair-gain across a whole input type = signal-limited. Change the channel; don't swing the
architecture.

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

```
   encoder edit:  back up ─► gate DEFAULT-OFF (champion byte-identical) ─► py_compile
                          ─► wire the probe flag ─► scp ─► sweep ─► only a positive probe graduates
```
Before graduating an architectural win, confirm the champion doesn't already carry that prior.

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

## Don'ts

- Don't train the full fusion model — confounded and slow.
- Don't default to architecture. DATA is co-equal; follow the fair-gain.
- No reimplemented metrics. No publication without multi-seed re-verification.
- Attribute borrowed signal: a vision win is frozen DINO/BioCLIP (env = *where*, vision = *which*), not
  a coordinate-encoder gain. Don't launder it; don't chase an aggregate mean.
- Never call a capability done or exhausted — switch lever family and continue.
- Keep specific experiments, datasets, and campaign directives out of this file. They belong in Ensue
  and `records.json`, where the swarm can update them.
