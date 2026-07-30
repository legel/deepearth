# spacetime/data

Two kinds of file live here, with opposite rules.

**`records.json` and `traces/` are written by the harness — never hand-edit them.** `records.json` is
the board: one record per capability plus the ledger of record history and dead-ends with their reasons.
Editing a score by hand forges a result; corrections go through the ledger with the reason attached.
Single owner — everyone else measures with `EARTH4D_ALLOW_UNRECORDED=1`.

**Everything else here is input data, and editing it IS a legitimate experiment.** When the diagnosis
reads INPUT-LIMITED — the encoder does not beat a generic PE, so the current channel lacks the signal —
changing what feeds the model is the correct move, not a bigger architecture swing. `lfmc/` holds the
Globe-LFMC table and the gate artifacts for the evidence program.

Two rules for any input change:

- **Rebuild the prepared cache** (`rm -f data/deepcal/prepared_*.pt`, or `--fresh-data`). It is lossy
  across data changes and will silently serve the old signal.
- **Attribute borrowed signal.** A win from a frozen pretrained embedding (DINO, BioCLIP) is that
  model's signal, not the encoder's: env = *where*, vision = *which*.

The shared dataset stays at the repo root `data/deepcal` — all three loops read it and the box's
`/workspace/data` symlink points there, so it belongs to no single loop.
