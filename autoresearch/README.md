# DeepEarth autoresearch

A self-contained environment for autonomously researching and improving **DeepEarth**.

1. Clone `github.com/legel/deepearth` (branch `deepcal`).
2. `pip install -r requirements.txt`, then build the Earth4D CUDA hash encoder against your torch: `cd encoders/spacetime && bash install.sh` (the shipped .so is ABI-specific — you MUST rebuild it).
3. `cd deepearth/autoresearch`; read `autoresearch.md` + `science.md` (binding research rules).
4. `python -m deepearth.autoresearch.prepare` — auto-downloads + extracts the audited dataset (deepcal_data.zip) from NERSC into `data/deepcal/`.
5. `python -m deepearth.autoresearch.train autoresearch/deepcal.yaml --steps 8000 --device cuda:0` (batch 512 needs ~27GB; on a 24GB card set `batch: 256` + `pollinator_top_k: 32`). Score vs the committed baseline in `BENCHMARKS.md`, edit, repeat.

## Experiment budget: 10 minutes (hard cap)

Every run trains for at most **10 minutes** of wall-clock (`time_budget_s: 600`, measured from step 10 so startup and
compilation are excluded), then is scored by `evaluate.py` (science.md rule 20). This is a hard cap, not a target:
never raise it, never report benchmarks from a longer run. Comparing experiments only at the equal 10-minute budget is
what makes a gain reflect real efficiency (throughput, architecture) rather than just more steps. Kill any run that
exceeds the budget and rerun at 600s.
