# DeepEarth scoring

This directory is the fixed public measurement surface:

- `evaluate.py` defines the human-capability suite and harmonic/arithmetic aggregates.
- `champion_report.py` compares a run with the registered record.
- `champion_scores.json` stores the exact registered record.
- `BENCHMARKS.md` renders the readable scorecard.
- `science.md` defines the public scientific contract.

Architecture, optimization, and data loading live in `core/`. Run the production model with:

```bash
python -m deepearth.core.train --cache /path/to/deepcal --device cuda --steps 2291 --seed 1337
```

Repeat with seed 1338 before promoting a record. Scoring code never imports a second model implementation.
