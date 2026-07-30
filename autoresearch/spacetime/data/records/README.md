# data/records — HARNESS-WRITTEN. Never hand-edit.

`records.json` is the board: one record per capability plus the ledger of record history and dead-ends
with their reasons. `traces/` holds per-run logs, `.trace.json` and `.result.json`.

Hand-editing a score here forges a result. To correct a record, write the correction **and its reason**
through the ledger so the next agent sees why — see how the invalidated `family_from_spacetime` walk is
recorded in its dead-ends.

Single owner: only the checkout that owns this board writes it. Everyone else measures with
`EARTH4D_ALLOW_UNRECORDED=1`.
