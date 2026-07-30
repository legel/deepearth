# spacetime — Earth4D encoder loop

Four directories, one rule each. The split exists so an agent never has to guess whether a file is
fair game.

```
  program/       READ FIRST, edit deliberately     the contract: what to pick, what counts as evidence
  experiments/   EDIT FREELY                       the science: probe modes, propagators, data channels
  harness/       DO NOT EDIT to win a run          the judge: what gets measured and what gets recorded
  state/         NEVER hand-edit                   generated: records.json, traces, ledgers
```

| directory | contains | policy |
|---|---|---|
| `program/` | `program.md` (the loop), `scorecard.md` (the board), `lfmc_gate.md` (pinned gate provenance), `box-operations.md` | Read before every cycle. Change it when the *doctrine* changes — never to accommodate a result. |
| `experiments/` | `probe.py`, `probe_modes_*.py`, `recurrence.py`, `gnn.py`, `phenology.py`, `dyntargets.py`, `env_field.py`, `calib_probe.py`, `lfmc_recurrent.py` | **This is where an experiment lives.** Add modes, change mechanisms, swap channels, rewrite a propagator. Also edit `encoders/spacetime/earth4d.py` for architecture work. |
| `harness/` | `trace.py`, `probe_contract.py`, `probe_emit.py`, `probe_registry.py`, `science_gate.py` | The measurement and recording layer. **Editing this to make a run look better is the one move that invalidates everything.** Change it only as deliberate infrastructure work, with tests, never inside an experiment. |
| `state/` | `records.json`, `traces/`, `events.jsonl`, backups | Written by the harness. Hand-editing it forges a result. To correct a record, write the correction *and its reason* through the ledger. |

## Why `harness/` is fenced off

The harness decides what a number means: which capability a run measured, whether two runs are
comparable, whether a score becomes a record. Every bad record in this project's history came from that
layer being wrong or bypassed — a different target scraped into a capability's record, a control's
accuracy stored as the encoder's, a research directive compiled into the gate so nothing could run, a
dead-end ledger evicting history by first letter. An agent that edits the judge to pass its own run
produces a number nobody can trust, including itself.

If the harness is genuinely wrong, fixing it is welcome — as its own change, with a test that fails
before and passes after, separate from any experiment.

## Running

```bash
# what can move my capability, and where do I edit?
python -m deepearth.autoresearch.spacetime.harness.probe_registry --capability family_from_spacetime

# one experiment, recorded
python -m deepearth.autoresearch.spacetime.harness.trace \
    --metric family_from_spacetime --probe "--forecast --n_shards 12" \
    --tag my_swing --device cuda:0 --ensue

# measure without recording (parity checks, smoke tests)
EARTH4D_ALLOW_UNRECORDED=1 python -m deepearth.autoresearch.spacetime.experiments.probe \
    --forecast --n_shards 12 --device cuda:0 --result-json /tmp/r.json
```
