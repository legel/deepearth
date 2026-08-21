# DeepEarth autoresearch

A self-contained environment for autonomously researching and improving **DeepEarth**.

1. Clone `github.com/legel/deepearth` (branch `main`).
2. `pip install -r requirements.txt`, then build the Earth4D CUDA hash encoder against your torch: `cd encoders/spacetime && bash install.sh` (the shipped .so is ABI-specific — you MUST rebuild it).
3. `cd deepearth/autoresearch`; read `autoresearch.md` + `science.md` (binding research rules).
4. `python -m deepearth.autoresearch.prepare` — auto-downloads + extracts the audited dataset (deepcal_data.zip) from NERSC into `data/deepcal/`.
5. `python -m deepearth.autoresearch.train autoresearch/deepcal.yaml --steps 2291 --device cuda:0` (batch 512 needs ~20GB; on a smaller card reduce `batch` and `pollinator_top_k`). Score vs the committed baseline in `BENCHMARKS.md`, edit, repeat.

## Experiment budget: 2,291 steps

Every comparable run completes exactly **2,291 optimizer steps**, then is scored by `evaluate.py` (science.md rule 20).
Wall time, parameters, and VRAM are reported separately; faster hardware or code never buys a candidate extra updates.
Run both seeds 1337 and 1338 and compare each against its seed-matched control.
