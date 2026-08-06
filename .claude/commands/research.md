---
description: Run the autonomous DeepEarth research loop against val_bpb.
---

# DeepEarth autoresearch

`autoresearch/main/program/program.md` is the binding contract. Read it first and follow it literally.
This command is an entry point, not an override. `autoresearch/scorecard.md` defines how science is
measured; `autoresearch/science.md` defines what the model must be.

## Fixed boundary

- Edit only `autoresearch/main/editable_files/**` — the model, the encoders, the training loop, the configs.
- Treat as read-only: `main/harness/**` (evaluate, ablation hooks), `scoring/objective.py`, `tests/**`,
  prepared data, and every document under `main/program/**`.
- Never tune a metric, a split, or a baseline to make a candidate pass. Improve the model.
- If fixed infrastructure looks wrong, report it as a blocker and pick another legal hypothesis. Do not
  repair the instruments inside the loop.

## Scope

$ARGUMENTS

With no argument, target the weakest variable in the current `val_bpb` decomposition.

## The loop

1. **Read the state.** The last run's `val_bpb`, its per-variable decomposition, and the recorded dead
   ends. Do not re-run a hypothesis that is already published as a dead end.
2. **Predeclare one causal hypothesis**, naming which variables should lose bits and through which
   subsystem. One hypothesis per run — a change that touches three things cannot be attributed.
3. **Make it a config toggle defaulting to current behaviour**, so the default path stays byte-identical.
4. **Commit the diff before running.** A dirty tree is unreproducible and cannot set a record.
5. **Run the pair at fixed steps** on the screen scale (~24M): candidate and control, same seed, same
   cache, same prepared data. Both GPUs, one arm each.
6. **Get the noise floor at that scale** — two matched seeds of the control, full spread — unless a
   current one is already recorded.
7. **Judge.** Keep only if `val_bpb` drops by more than that floor, the drop lands in the variables the
   hypothesis named, and an ablation of the subsystem accounts for it. A gain that lands elsewhere is an
   initialization re-roll: adding parameters shifts the RNG stream and re-initializes the model.
8. **Confirm at a second scale** before promoting. A result that holds only at one size is an artifact.
9. **Record it** — `val_bpb` and its decomposition, with both benchmark means alongside. Publish dead
   ends with their reason; an unrecorded negative gets re-run by the next agent.
10. **Repeat.** Never stop on a single negative.

## Operational

- Warm inductor cache makes startup ~1s. Never delete `/tmp/torchinductor_root`, and reuse one worktree
  per config since the cache keys on source.
- Run over SSH in the foreground; backgrounding a run detaches and kills it.
- Never compare a mirror run to a public-main run — the evaluators differ and the same config scores
  ~0.279 on the mirror against 0.332464 on public main.
- Delivery to the public repository is a separate, explicitly authorized step: `ship-deepearth-improvement`.
