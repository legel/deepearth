# records — written by the harness. NEVER hand-edited.

The board and its ledgers live here: the current record per metric, the history of how it moved, and the
dead-ends with their reasons. The harness writes it; nothing else may.

**Hand-editing a score forges a result.** To correct a record, write the correction *and its reason*
through the ledger, so the next agent sees why — see how the invalidated `family_from_spacetime` walk is
recorded in its dead-ends rather than deleted.

Single owner: only the checkout that owns this board writes it. Everyone else measures with

`records.json` is now **tracked**. It had lived only on a vast.ai container, and in one session it was
deleted outright once and lost to a full disk once — the campaign's entire memory, 116 dead-ends
included, surviving on luck. Git is the durable store.

Tracking it does not relax the single-owner rule: that governs who may WRITE, and it still holds. Only

If two writers ever do produce a conflict here, resolve it by **union of the ledgers** — never by
picking one side's scores, which would silently delete the other's dead-ends.

`../program/scorecard.txt` remains the skimmable view, regenerated from this board after every run.

## What actually protects this board

`EARTH4D_ALLOW_UNRECORDED=1` was documented here as the way to measure without recording. It appears in
no `.py` file in this repository and never did anything. Two mechanisms are real:

- **The dirty-tree gate** (`harness.py`): a run from a tree with uncommitted changes still publishes its
  dead-end and still posts to Ensue, but cannot take the record. The experiment IS the CONFIG/earth4d.py
  diff, so a record nobody can reconstruct is not a record.
- **One worktree at a time.** `records.json` is git-tracked, so every worktree carries its own physical
  copy — two trees do not contend for a lock, they diverge silently. Merge boards back by UNION of
  `ledger.deadends` keyed by tag; picking one side deletes the other's dead-ends, which is how 74 were
  lost once already.
