# editable_files/data — the DATA lever

**Data sources are meant to be added, moved and removed based on the signal they provide.** That is not
a maintenance chore, it is one of the two ways this loop makes progress. When the diagnosis reads
INPUT-LIMITED — the encoder does not beat a generic PE, so the current channel lacks the signal —
changing what feeds the model is the *correct* move, not a bigger architecture swing.

Both of this loop's coordinate capabilities read INPUT-LIMITED right now, so this is where their next
move lives.

## The cycle

```
   ADD a source ──► probe it against the SAME capability, same mode, same shards
                         │
                         ├─ raises fair-gain beyond noise  ──► KEEP it, and record what it contributed
                         ├─ flat                            ──► REMOVE it. A channel that adds nothing
                         │                                      is cost every future run pays.
                         └─ raises the score but not the    ──► the source is doing the work, not the
                            encoder-vs-PE gain                  encoder. KEEP but LABEL it borrowed.
```

One source per directory, named for what it is (`lfmc/`, `worldclim/`, `alphaearth/`). A source's
directory holds its raw or derived artifacts; the loader that reads it lives in `../lib/`.

## Four rules that make a data change trustworthy

1. **Change one source per run.** Two new channels at once and you cannot attribute the gain.
2. **Rebuild the prepared cache** (`rm -f data/deepcal/prepared_*.pt`, or `--fresh-data`). The cache is
   lossy across data changes and will silently serve the old signal — a "gain" that is really the cache.
3. **Attribute borrowed signal.** A win from a frozen pretrained embedding (DINO, BioCLIP, AlphaEarth)
   is that model's signal, not Earth4D's: env = *where*, vision = *which*. `family_from_vision` 0.9445
   is on the board's excluded layer for exactly this reason.
4. **Removal is a result too.** Publish it as a dead-end with its reason, so the next agent does not
   re-add the channel you just ruled out.

## Not here

`../../records/` holds the board, traces and ledgers — written by the harness, never hand-edited.
Editing a score by hand forges a result; corrections go through the ledger with their reason.

The shared dataset stays at the repo root `data/deepcal`: all three loops read it and the box's
`/workspace/data` symlink points there, so it belongs to no single loop.
