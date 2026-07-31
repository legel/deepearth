# spacetime — Earth4D encoder loop

```
  program/                 the contract. Read first, change only when doctrine changes.
  editable_files/          THE ONLY CODE AN EXPERIMENT MAY EDIT
      harness/               the loop: probe.py, modes, trace.py, contract, registry
      lib/                   auxiliary: propagators, phenology, targets, channels, gate
  data/             EDITABLE — the DATA lever: channels, densification, new sources
  data/            records.json, traces/, ledgers. Never hand-edited.
```

| path | contains | notes |
|---|---|---|
| `program/program.md` | the loop: pick → diagnose → run → measure → decide → write | read every cycle |
| `program/scorecard.md` | the board: 7 probeable capabilities, and what is excluded and why | pick your metric here |
| `program/box-operations.md` | box, GPUs, token location, commit identity | gitignored |
| `editable_files/harness/probe.py` | the probe: shared loading, dispatch, the recording modes | edit in place, on a branch |
| `editable_files/harness/probe_modes_tables.py` | the four encoder-free modes (env → identity from tables) | DATA lever only |
| `editable_files/harness/trace.py` | declares the metric, runs the probe, applies the record gate, writes the ledger and Ensue | changing what a number MEANS is its own commit, with a test |
| `editable_files/harness/probe_contract.py` | `ProbeResult`: identity, validation, fair-gain, rendering | the probe declares; nothing parses stdout |
| `editable_files/harness/probe_emit.py` | `declare()` — the one path by which a number becomes recordable | |
| `editable_files/harness/probe_registry.py` | capability → modes → what each requires → where to edit | `--capability X` |
| `editable_files/lib/recurrence.py` | 4D-LSTM propagator (science.md rule 2b), time normalization, guards | imported by most of the loop |
| `editable_files/lib/gnn.py` | message-passing propagator — the alternative mechanism to compare against | |
| `editable_files/lib/phenology.py` | phenology runners | |
| `editable_files/lib/dyntargets.py` | target builders for cooccur / SDM / pheno modes | |
| `editable_files/lib/env_field.py` | env-field decode | |
| `editable_files/lib/calib_probe.py` | the calibration capability (own CLI; not yet on the contract) | |
| `data/` | the channels that feed the encoder | **editable — the DATA lever.** `family_from_env` and `family_from_spacetime` both read INPUT-LIMITED right now, so this is where their next move lives |
| `data/records.json` | the board: one record per capability + ledger of history and dead-ends | single owner, gitignored |
| `data/traces/` | per-run log + `.trace.json` + `.result.json` | |

## An experiment is an edit on a branch

Not a new file, not a new flag. `git worktree add ../e4d-<tag> -b exp/<tag>`, edit
`editable_files/**` and `autoresearch/probes/spacetime/editable_files/earth4d.py` in place, sweep, and let the branch hold the
isolation. A flag is what you add when something **graduates**; a dead flag is a bug. Gating at
conception instead is what produced 113 flags, a 1,552-line `main()`, 21 `champion_*.yaml` variants and
a pile of `diag*.py` copies — and cost every later agent the reading.

## Running

```bash
# what can move my capability, and where do I edit?
python -m deepearth.autoresearch.probes.spacetime.editable_files.harness \
    --capability family_from_spacetime

# one experiment, recorded
python -m deepearth.autoresearch.probes.spacetime.editable_files.harness \
    --metric family_from_spacetime --probe "--forecast --n_shards 12" \
    --tag my_swing --device cuda:0 --ensue

# measure WITHOUT recording (parity checks, smoke tests)
EARTH4D_ALLOW_UNRECORDED=1 python -m deepearth.autoresearch.probes.spacetime.editable_files.probe \
    --forecast --n_shards 12 --device cuda:0 --result-json /tmp/r.json
```
