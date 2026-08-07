# Public main baseline — `3c45b99`

Use this baseline only for candidates whose public base resolves to commit `3c45b99d5c5860875f12639e808743832c680faf` and whose evaluator is identical to that commit.

## Provenance

- Repository: `legel/deepearth`
- Branch: `main`
- Commit: `3c45b99d5c5860875f12639e808743832c680faf`
- Git tree: `5ec7d64c3612748a481ae178ded8cd85ccf73f21`
- Config: `autoresearch/deepcal.yaml` at the baseline commit
- Evaluator: `autoresearch/evaluate.py` at the baseline commit
- Seed: `1337`
- Training budget: `600` seconds, excluding initial compilation
- Training: `6,001` steps; final printed loss `0.425`
- Held-out rows: `29,668`
- Active suite: `58/63`
- Harmonic net: `0.331022`
- Arithmetic mean: `0.5806`
- Peak VRAM: `35,630.7 MB`
- Run tag: `lance_main_3c45b99_baseline`
- Raw log SHA-256: `e27cca67022def4aa6e6561cd729cdb089ab75771fcbd6b7f6f404dc0561a30d`

Command shape:

```bash
python -m deepearth.autoresearch.train autoresearch/deepcal.yaml \
  --cache_dir DATA_CACHE \
  --device cuda:0 \
  --time_budget 600 \
  --tag lance_main_3c45b99_baseline
```

## Complete active scorecard

| Benchmark | Score |
|---|---:|
| B55 pollinator phylo transfer recall | 0.038 |
| B8 family from spacetime | 0.081 |
| B6 family from environment | 0.082 |
| B23 species calibration MRR | 0.089 |
| B51 pollinator from environment recall | 0.151 |
| B50 pollinator from spacetime recall | 0.153 |
| B42 mycorrhiza from environment | 0.184 |
| B1 species from environment top-10 | 0.194 |
| B15 vision from aerial cosine | 0.227 |
| B5 species from spacetime top-10 | 0.269 |
| B20 community from environment recall | 0.296 |
| B22 companions recall | 0.307 |
| B21 community from species recall | 0.326 |
| B47 infer NAIP-IR cosine | 0.329 |
| B34 LFMC from environment | 0.345 |
| B19 infer aerial cosine | 0.426 |
| B44 infer topography cosine | 0.432 |
| B28 flowering peak-month MRR | 0.436 |
| B48 pollinator from photo-only recall | 0.459 |
| B52 pollinator from photo recall | 0.462 |
| B17 infer soil cosine | 0.509 |
| B41 pollinator from species recall | 0.526 |
| B43 infer hydro cosine | 0.536 |
| B37 imagine BioCLIP vision cosine | 0.542 |
| B45 BioCLIP leave-one-out cosine | 0.572 |
| B16 infer clay cosine | 0.600 |
| B18 infer climate cosine | 0.646 |
| B13 imagine vision cosine | 0.665 |
| B54 pollinator distribution KL score | 0.676 |
| B46 infer CHM cosine | 0.688 |
| B26 flowering AUC | 0.739 |
| B53 pollinator calibration MRR | 0.746 |
| B27 flowering fidelity | 0.755 |
| B14 vision leave-one-out cosine | 0.756 |
| B2 species from photo top-1 | 0.801 |
| B4 species from photo-only top-1 | 0.821 |
| B49 form trait F1 | 0.887 |
| B38 water/soil regime F1 | 0.903 |
| B9 phylo from photo cosine | 0.904 |
| B35 sun trait F1 | 0.918 |
| B10 traits from photo and environment F1 | 0.923 |
| B11 traits from photo F1 | 0.931 |
| B32 plant-type trait F1 | 0.931 |
| B30 seasonality trait F1 | 0.945 |
| B3 species from photo top-5 | 0.948 |
| B36 ease-of-care trait F1 | 0.949 |
| B33 growth-rate trait F1 | 0.950 |
| B7 family from phylo | 0.977 |
| B63 myco from species F1 | 0.996 |
| B12 traits leave-one-out F1 | 0.999 |

## Ablation and information-gain contributions

| Benchmark | Raw delta | Net contribution |
|---|---:|---:|
| B24 geo information gain | +0.607 | 0.998 |
| B60 community phylo-graph gain | +0.055 | 0.634 |
| B56 family phylo-graph gain | +0.022 | 0.556 |
| B61 trait phylo-graph gain | +0.021 | 0.553 |
| B58 LFMC phylo-graph gain | +0.004 | 0.511 |
| B57 flowering phylo-graph gain | +0.001 | 0.502 |
| B59 pollinator phylo-graph gain | +0.001 | 0.502 |
| B62 mycorrhiza phylo-graph gain | 0.000 | 0.500 |

## Inactive benchmarks

| Benchmark | Reason |
|---|---|
| B25 forecast climate cosine | Requires temporal holdout |
| B31 forecast vision cosine | Requires temporal holdout |
| B29 species distribution 30 m skill | Required inputs or labels absent |
| B39 species distribution 3 km skill | Required inputs or labels absent |
| B40 species distribution 300 m skill | Required inputs or labels absent |

## Held-out reconstruction summary

| Variable | Score |
|---|---:|
| identity | 0.821 |
| phylo | 0.897 |
| plant_type | 0.950 |
| growth_rate | 0.812 |
| seasonality | 0.784 |
| sun | 0.849 |
| water | 0.750 |
| soil_drainage | 0.942 |
| ease_of_care | 0.566 |
| form | 0.524 |
| climate | 0.611 |
| soil | 0.454 |
| naip_rgb | 0.182 |
| naip_ir | 0.084 |
| clay | 0.365 |
| topo | 0.216 |
| chm | 0.456 |
| hydro | 0.126 |
| phenology | 0.682 |
