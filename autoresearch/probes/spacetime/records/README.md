# records — written by the harness. NEVER hand-edited.

The board and its ledgers live here: the current record per metric, the history of how it moved, and the
dead-ends with their reasons. The harness writes it; nothing else may.

**Hand-editing a score forges a result.** To correct a record, write the correction *and its reason*
through the ledger, so the next agent sees why — see how the invalidated `family_from_spacetime` walk is
recorded in its dead-ends rather than deleted.

Single owner: only the checkout that owns this board writes it. Everyone else measures with
`EARTH4D_ALLOW_UNRECORDED=1`, which produces no record by design.

`records.json` is now **tracked**. It had lived only on a vast.ai container, and in one session it was
deleted outright once and lost to a full disk once — the campaign's entire memory, 116 dead-ends
included, surviving on luck. Git is the durable store.

Tracking it does not relax the single-owner rule: that governs who may WRITE, and it still holds. Only
the checkout that owns this board writes it; everyone else measures with `EARTH4D_ALLOW_UNRECORDED=1`.
If two writers ever do produce a conflict here, resolve it by **union of the ledgers** — never by
picking one side's scores, which would silently delete the other's dead-ends.

`../program/scorecard.txt` remains the skimmable view, regenerated from this board after every run.
