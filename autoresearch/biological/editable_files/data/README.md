# biological/editable_files/data — the DATA lever

**Data sources are meant to be added, moved and removed based on the signal they provide.** When the
bottleneck is missing signal rather than missing capacity, changing what feeds the model is the correct
move. One source per directory, named for what it is; its loader lives in `../lib/`.

```
   ADD ──► probe against the same target ──► raises the metric beyond noise?  KEEP
                                        └─► flat?  REMOVE — an inert channel is cost every run pays
```

Rules: change one source per run so the gain is attributable · rebuild the prepared cache after any
change (it is lossy and will serve the old signal) · attribute borrowed pretrained signal honestly ·
publish a removal as a dead-end with its reason so nobody re-adds it.

`../../records/` is harness-written — never hand-edited. The shared dataset stays at the repo root
`data/deepcal`, since all three loops read it.
