# main/data

Two kinds of file live here, with opposite rules.

**Harness-written files are never hand-edited** — for `main` that is the scores and ledgers the
harness produces. Editing them forges a result; corrections go through the ledger with a reason.

**Input data here is editable, and changing it is a legitimate experiment** when the bottleneck is a
lack of signal rather than a lack of capacity. Rebuild the prepared cache after any change, and
attribute borrowed pretrained signal honestly.

The shared dataset stays at the repo root `data/deepcal` — all three loops read it, so it belongs to no
single loop.
