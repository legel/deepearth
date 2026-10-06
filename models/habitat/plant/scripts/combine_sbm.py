"""Combine per-family SBM fits from the two dated trees into the dispersal table the pipeline reads:
mean km/yr per family across trees, excluding diverged fits (> 100 km/yr); an 'ALL' row = median family
rate, the fallback for families with < 10 located tips or no stable fit."""
import sys

import pandas as pd

frames = [pd.read_csv(p).assign(tree=p) for p in sys.argv[1:-1]]
d = pd.concat(frames)
unstable = d[(d.clade != "ALL") & (d.km_per_year > 100)]
if len(unstable):
    print("diverged fits excluded:", ", ".join(f"{c} ({t.split('_')[-1][:4]})" for c, t in zip(unstable.clade, unstable.tree)))
d = d[~d.index.isin(unstable.index) | (d.clade == "ALL")]
d = d[d.km_per_year <= 100]
fam = d[d.clade != "ALL"].groupby("clade", as_index=False).agg(
    n_tips=("n_tips", "max"), diffusivity_km2_per_Myr=("diffusivity_km2_per_Myr", "mean"),
    km_per_year=("km_per_year", "mean"), km_per_kyr=("km_per_kyr", "mean"), n_trees=("tree", "nunique"))
all_row = pd.DataFrame([{"clade": "ALL", "n_tips": int(fam.n_tips.sum()),
                         "diffusivity_km2_per_Myr": fam.diffusivity_km2_per_Myr.median(),
                         "km_per_year": fam.km_per_year.median(), "km_per_kyr": fam.km_per_kyr.median(), "n_trees": 0}])
out = pd.concat([fam, all_row], ignore_index=True)
out.to_csv(sys.argv[-1], index=False)
print(f"families {len(fam)}; km/yr median {fam.km_per_year.median():.2f}, range {fam.km_per_year.min():.2f}-{fam.km_per_year.max():.2f}")
