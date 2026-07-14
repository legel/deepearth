# DeepCal data pipeline — build, publish, prepare

The dataset ships as **pre-computed embeddings only** (DINOv3 ground vision, NAIP DINOv3-SAT `rgb_pool`/`ir_pool`,
Daymet, SSURGO soil, 3DEP topo/hydro, NAIP-CHM, Clay/Sentinel-2 tokens + the dated phylogenies, vocab, splits,
benchmark labels). Raw imagery is a transient build intermediate — never packaged. Three engineered steps, each
runnable on any machine; no step requires SSH access to a private box.

## 1. Build the zip  (maintainer, where the cache lives)
    python -m deepearth.data.deepcal.package_dataset --out <zip>       # explicit 216-file manifest, ~6.6 GB
    python -m deepearth.data.deepcal.package_dataset --dry-run         # validate the manifest + REQUIRED gate

## 2. Publish to NERSC  (maintainer) — served at https://portal.nersc.gov/cfs/m5239/deepcal/deepcal_data.zip
Bulk transfer via **Globus** (the directed path for large files; ~2 min for 6.6 GB on Globus infrastructure):
    globus transfer --sync-level checksum <SRC_GCP>:<zip> <NERSC>:/global/cfs/cdirs/m5239/www/deepcal/deepcal_data.zip
    # NERSC Perlmutter collection = 6bdc7956-fc0f-4ad2-989c-7aa5ee643a79  (maps /global/cfs)
Alternative: `python -m deepearth.data.deepcal.nersc_upload` (SFAPI, 150 MB resumable chunks; needs iris.nersc.gov creds).

## 3. Prepare + run  (any collaborator, any machine)
    python -m deepearth.autoresearch.prepare        # downloads the zip from the portal (HTTPS, no auth),
                                                    # extracts, compiles the Earth4D CUDA kernel (install.sh
                                                    # auto-installs ninja), builds the prepared cache + test I/O
    python -m deepearth.autoresearch.train autoresearch/deepcal.yaml --device cuda:1
Override the source with `DEEPCAL_DATA_URL`. Run from the parent of the `deepearth/` package dir.

## Status (2026-07-14)
- **Plant modalities**: complete + audited (see `data_provenance.yaml`). Vocab 4628 (2141 tree + recovered out-of-tree).
- **NAIP embeddings**: ~57% at last zip build (the M2M/DINOv3-SAT run is still embedding); the zip is **rebuilt +
  republished when NAIP completes**. All other modalities are 100%/near-100%.
- **Pollinator observations** (38,029, GBIF 0035626): vision + 8157 vocab + dated dist + CPU env fan-out
  (soil/topo/hydro/daymet/chm) done; NAIP+Clay for these obs pending GPU; **not yet loaded by `data.py` as
  first-class training examples** (the pollinator-SDM integration — tracked).

## Reproducibility (mandate: no cache we cannot regenerate)
Committed, audited builders regenerate most modalities (cosine/byte-checked vs the cache — see `data_provenance.yaml`):
env priors (`env_priors/build_*.py`), NAIP (`build_naip_m2m2024.py`), Clay (`build_clay.py`), mycorrhiza
(`build_mycorrhiza.py`), species-dist (`build_species_dist.py`), dated phylogenies (`plant_dated_*`, `pollinator_dated_*`).

Also: `build_patristic_ref.py` regenerates `derived/patristic_ref.npy` (3487² tree cophenetic; audited max err 3e-5).
`build_lfmc.py` regenerates `gbif_lfmc.npz` from `fire/lfmc_data_conus.csv` (per-species median CA LFMC — a recompute
that also fixed an unphysical shipped outlier, Artemisia californica 3581%→123%).

**Open reproducibility gaps** (files ship in the zip but have no committed builder yet — TODO):
- `bioclip_taxon_text_emb.npy` — BioCLIP-2.5 taxon-string prior (encode logic in `add_species.py`; needs a standalone
  full-vocab builder using `observations_meta.parquet` for K/P/C + species_index for O/F/G).
- `gbif_lfmc.npz` — Live Fuel Moisture (B34, 37 species); reconciled via a one-off vocab remap — needs a committed
  builder from the `lfmc_data_conus.csv` source.
