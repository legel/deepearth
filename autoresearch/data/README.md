# autoresearch/data — the corpus, one source per directory

Every loop reads from here; nothing writes results here (a loop's results go to its own `records/`).
Gitignored: this is ~53GB on the box, and the box's `/workspace/data` symlink points at it.

    deepcal/     the audited DeepCal corpus every probe reads (built by main's prepare.py)
    lfmc/        Globe-LFMC table + gate artifacts for the spacetime evidence program

## This is the DATA lever

**Sources are added, moved and removed based on the signal they provide.** When a probe's diagnosis
reads INPUT-LIMITED — the encoder does not beat a generic PE, so the current channel lacks the signal —
changing what feeds the model is the correct move, not a bigger architecture swing.

```
   ADD a source ──► probe it against the SAME capability, mode and shard count
                     ├─ fair-gain rises beyond noise → KEEP, and record what it contributed
                     ├─ flat                         → REMOVE; an inert channel is cost every run pays
                     └─ raises score, not the gain   → the SOURCE is doing the work, not the encoder:
                                                       keep it, and LABEL it borrowed
```

One source per directory, named for what it is. The loader that reads it belongs to the loop that uses
it, in that loop's `editable_files/lib/`.

Four rules that make a data change trustworthy:

1. **One source per run**, or the gain is not attributable.
2. **Rebuild the prepared cache** (`rm -f deepcal/prepared_*.pt`, or `--fresh-data`). It is lossy across
   data changes and will silently serve the old signal — a "gain" that is really the cache.
3. **Attribute borrowed signal.** A win from a frozen pretrained embedding (DINO, BioCLIP, AlphaEarth) is
   that model's signal, not the encoder's: env = *where*, vision = *which*.
4. **A removal is a result.** Publish it as a dead-end with its reason so nobody re-adds what you ruled
   out.
