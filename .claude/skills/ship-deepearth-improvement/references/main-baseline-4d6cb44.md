# Public main baseline — `4d6cb44`

Use this baseline only for candidates whose public base resolves to commit `4d6cb447086f553757fdf601518c63b9533cdf5e` and whose evaluator is identical to that commit.

## Provenance

- Repository: `legel/deepearth`
- Branch: `main`
- Commit: `4d6cb447086f553757fdf601518c63b9533cdf5e`
- Config: `autoresearch/deepcal.yaml` at the baseline commit
- Evaluator: `autoresearch/evaluate.py` at the baseline commit
- Training budget: `600` seconds, excluding initial compilation (~148 s)
- Observations: `621,558`; train `591,890`; held-out regions `29,668`
- Steps: ~`5,400`; final printed loss `0.419`
- Active suite: `58/63`
- Harmonic net: `0.318693`
- Arithmetic mean: `0.5707`
- Peak VRAM: `37,003.6 MB`
- Hardware: 1x NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Prepared cache: `lance-main-shared-cache-c69ee8c`
- Run tag: `main_4d6cb44_baseline`

Command shape:

```bash
python -m deepearth.autoresearch.train autoresearch/deepcal.yaml \
  --cache_dir DATA_CACHE \
  --device cuda:0 \
  --time_budget 600 \
  --tag main_4d6cb44_baseline
```

## Reconciliation with the open PRs

Neither open PR's stated baseline for this SHA reproduces:

| Source | Harmonic | Arithmetic | Active | Note |
|---|---:|---:|---:|---|
| This measurement | 0.318693 | 0.5707 | 58/63 | trained to completion, loss monotone to step 5,000 |
| PR #31 | 0.328898 | 0.5722 | 68/73 | a 73-benchmark suite does not exist in this evaluator |
| PR #32 | 0.008263 | 0.0998 | 40/63 | claims persistent non-finite loss from step 4,500; did not reproduce |

PR #32's stabilization may still be correct on its own stress evidence, but its motivating failure is not a property of `4d6cb44` under this config and cache, so its `+0.307` harmonic delta cannot be read as a gain over public `main`.

## Known noise

The harness is not run-to-run deterministic: the same checkout reported `797.1M` and `796.2M` parameters on consecutive runs. At a 120 s budget, repeated base runs returned net `0.293180` and `0.273141`. Single-seed deltas on this suite are not resolvable below roughly a point of arithmetic mean without matched-seed repetition.
