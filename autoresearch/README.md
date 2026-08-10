# DeepEarth autoresearch

A self-contained environment for autonomously researching and improving **DeepEarth**.

1. Clone `github.com/legel/deepearth` (branch `deepcal`).
2. `pip install -r requirements.txt`, then build the Earth4D CUDA hash encoder against your torch: `cd encoders/spacetime && bash install.sh` (the shipped .so is ABI-specific — you MUST rebuild it).
3. `cd deepearth/autoresearch`; read `autoresearch.md` + `science.md` (binding research rules).
4. `python -m deepearth.autoresearch.prepare` — auto-downloads + extracts the audited dataset (deepcal_data.zip) from NERSC into `data/deepcal/`.
5. `python -m deepearth.autoresearch.train autoresearch/deepcal.yaml --steps 8000 --device cuda:0` (batch 512 needs ~27GB; on a 24GB card set `batch: 256` + `pollinator_top_k: 32`). Score vs the committed baseline in `BENCHMARKS.md`, edit, repeat.

## Evaluation contract

Screen ideas for 10 minutes with `--time_budget 600`. Evaluate promotion at **8,000 steps** with two matched seeds:

```bash
python -m deepearth.autoresearch.train autoresearch/deepcal.yaml --steps 8000 --seed 1337 --device cuda:0
python -m deepearth.autoresearch.train autoresearch/deepcal.yaml --steps 8000 --seed 1338 --device cuda:0
```

Both models therefore see the same number of batches. A candidate passes when its mean harmonic score improves beyond
the control's two-seed spread and its mean arithmetic score does not regress beyond that spread. Report the complete
public scorecard from both seeds.
