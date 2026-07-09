# Fire Ecology

We are directing DeepEarth toward **per-species fire risk evaluation** — modeling not just where and when fires will occur, but how each plant species contributes to ignition risk, fuel behaviour, and ember production based on its physiology and condition.

Preliminary result (live fuel moisture content prediction): [`encoders/spacetime/benchmarks/lfmc`](../../encoders/spacetime/benchmarks/lfmc).

**Earth4D + species embeddings** currently lead the Globe-LFMC 2.0 benchmark (Yebra et al. 2024) on the Allen Institute for AI split, achieving **R² 0.78 / MAE 11.7 pp / RMSE 18.7 pp** for live fuel moisture across 180+ plant species and the continental US — surpassing AI2's pre-trained Vision Transformer foundation model (R² 0.72) without satellite imagery, weather data, or topography. The species are learned as randomly-initialized embeddings — no prior knowledge — and the (_x_, _y_, _z_, _t_) signal alone carries enough structure for state-of-the-art generalization.

We plan to extend this from moisture alone to a per-species fire-risk surface:

- **Oils & terpenes** — many California natives (eucalyptus, chaparral shrubs, conifers) emit volatile organic compounds that change ignition energy. We will learn species-specific oil/terpene profiles as auxiliary outputs from the same Earth4D backbone, conditioned on phenophase.
- **Ember production & spotting distance** — bark texture, leaf morphology, and post-fire architecture determine how far each species throws embers downwind. We will encode this as a per-species emission profile, joined to fire-spread physics.
- **Fuel structure** — canopy bulk density, surface fuel load, and ladder-fuel connectivity vary across species and phenophase. We will couple species embeddings to vertical-fuel-profile predictions.
- **Phenology coupling** — leaf-on/leaf-off and reproductive state modulate all of the above. The flowering and pollination models in this `models/` tree provide the phenology layer; the fire-ecology model consumes it.

The goal: a continuous, species-resolved fire-risk field across California and beyond, refreshed daily from in-situ sensors + Earth observation, ready to inform fuel-treatment planning, evacuation timing, and landscape-architecture decisions.
