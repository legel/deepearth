# probes/biological — the phylogenomic species-graph loop

One probe, its own data, its own metric, its own evals. It recovers one signal: **can a species'
biology be imputed from its relatives, along a real dated phylogeny?**

```
  harness/          THE JUDGE      what a number means, and whether it may become a record
    board.py          capabilities, protocol, board paths, FAIR_ORDER, the CLI
    nulltree.py       the fair control — a tree that is not the phylogeny
    probe.py          fixed evaluator: family, interaction (two-tree)
    traitprobe.py     fixed evaluator: traits, community, symbiosis, guilds
    stage*.py         the pollitree build (real pollinator phylogeny)
  editable_files/   THE SCIENCE    the only surface an experiment may edit
    phylogenomic.py   the PUBLIC entrypoint — SpeciesGraph, imported directly by main/fusion
    lib/seeds.py      what a species embedding starts as        (rule 26)
    lib/training.py   how the graph is fitted and read          (rules 9, 25, 10-11)
  program/          the contract: program.md, scorecard.md, scorecard.txt
  records/          the board — written by the harness, never by hand
```

Dependencies point one way only, and the audit enforces it:

```
harness/  ->  editable_files/lib/{seeds,training}.py  ->  editable_files/phylogenomic.py
```

The judge imports the science it measures; the science never imports the judge. `traitprobe.py` used to
live under `editable_files/lib/` and import `harness.probe` for its loaders and metrics — an
experiment's own surface reaching back into the thing that grades it.

## How a breakthrough reaches the model

`main/editable_files/fusion/fusion.py:17` imports `SpeciesGraph` from `phylogenomic.py` directly, as do
`train.py` and `lib/data.py`. That import is the whole coupling: an improvement to the encoder is built
on top by the fusion model without anything being copied. Probe *scores* never cross — a probe record
becomes a prediction that `graduation.py` tests against the champion, and the result is appended to the
graduation ledger under `main/records/`.

Nothing here may edit another loop's directory, and nothing here may edit `harness/`.

## Running it

```bash
# one capability, with the fair control on (the default)
python -m deepearth.autoresearch.probes.biological.harness.probe \
    --cache_dir autoresearch/data/deepcal --result-json /tmp/r.json

# gate the result onto the board and regenerate scorecard.txt
python -m deepearth.autoresearch.probes.biological.harness.board \
    --capability family_from_phylo --result-json /tmp/r.json --tag bio_<hypothesis>

python -m deepearth.autoresearch.probes.biological.harness.board --list-capabilities
python -m deepearth.autoresearch.probes.biological.harness.board --scorecard
```

Full-model experiments (the `bio_gain` objective over B56–B62) go through
`main/harness/run_experiment.py`, which installs the `[profile] refined_seed_norm` instrument.

Read [`program/program.md`](program/program.md) for the loop, and
[`program/scorecard.md`](program/scorecard.md) for what the numbers mean — in particular why the fair
control is a null tree and not the seed.
