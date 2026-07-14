"""Package the audited DeepCal cache into deepcal_data.zip for NERSC hosting (upload via 4tb/nersc_upload.py; consumed
by autoresearch/prepare.py). EXPLICIT manifest — never a blind `zip -r` — so we ship exactly what the /autoresearch
pipeline reads (provenance-pinned) and omit build intermediates (env_priors checkpoints, the 313 MB CHM index, raw
photos). `--dry-run` lists the manifest + sizes and checks prepare.py's readiness gate without writing the zip.

    python -m deepearth.data.deepcal.package_dataset --dry-run          # validate the manifest
    python -m deepearth.data.deepcal.package_dataset --out /home/photon/4tb/deepcal_data.zip   # build (post-recompute)
"""
import argparse, os, zipfile, glob
from pathlib import Path

HERE = Path(__file__).resolve().parent                      # .../data/deepcal

# What data.py / train.py / evaluate.py load. Single-file caches + the phylogeny + the shipped provenance.
ROOT = ["gbif_vocab.npz", "ca_subtree.dated.nwk", "data_provenance.yaml",
        "gbif_elev.npz", "gbif_eventtime.npz",
        "gbif_soil_tokens.npz", "gbif_topo_tokens.npz", "gbif_chm_tokens.npz", "gbif_hydro_tokens.npz",
        "gbif_clay_tokens.npz",
        "gbif_species_dist.npz", "gbif_pollinator_dist.npz", "gbif_flower_all.npz", "gbif_lfmc.npz", "gbif_mycorrhiza.npz",
        "gbif_plant_dist.npz",
        "bioclip_taxon_text_emb.npy",                       # BioCLIP-2.5 taxon-string prior (the primary; supersedes the legacy BioCLIP-2 bioclip_text_emb.npy)
        "pollinator_taxon_text_emb.npy", "pollinator_distance.npy", "pollinator_animal_mask.npy"]
# Globbed multi-shard modalities (ground vision, aerial, climate, pollinator obs) + the derived tables.
DIRS = ["derived", "gbif_tokens", "gbif_daymet_tokens", "gbif_naip_tokens", "gbif_pollinator_obs"]
# Included if present (reproducible re-runs / legacy readers), skipped without error otherwise.
OPTIONAL = ["observations_meta.parquet", "gbif_pollinator_tokens.npz", "gbif_flower.npz", "env_priors/obs_coords.npz"]
# prepare.py's readiness gate — the zip is invalid if any is missing.
REQUIRED = ["gbif_vocab.npz", "ca_subtree.dated.nwk", "derived/species_index.csv", "derived/patristic_ref.npy"]


def manifest():
    """Resolve the manifest to a list of (absolute_path, arcname) pairs, in a stable order."""
    items, missing_root = [], []
    for r in ROOT:
        p = HERE / r
        (items if p.exists() else missing_root).append((p, r) if p.exists() else r)
    for o in OPTIONAL:
        p = HERE / o
        if p.exists():
            items.append((p, o))
    for d in DIRS:
        base = HERE / d
        for p in sorted(base.rglob("*")):
            if p.is_file():
                items.append((p, str(p.relative_to(HERE))))
    return items, missing_root


def main():
    ap = argparse.ArgumentParser(description="Package the audited DeepCal cache into a zip for NERSC.")
    ap.add_argument("--out", default="/home/photon/4tb/deepcal_data.zip")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    items, missing_root = manifest()
    total = sum(p.stat().st_size for p, _ in items)
    have_arc = {arc for _, arc in items}
    missing_req = [r for r in REQUIRED if r not in have_arc]

    # per-top-level summary
    from collections import defaultdict
    by_top, cnt_top = defaultdict(int), defaultdict(int)
    for p, arc in items:
        top = arc.split("/")[0]
        by_top[top] += p.stat().st_size; cnt_top[top] += 1
    print(f"manifest: {len(items)} files, {total/1e9:.2f} GB")
    for top in sorted(by_top, key=lambda t: -by_top[t]):
        print(f"  {top:<28} {cnt_top[top]:>6} files  {by_top[top]/1e9:>7.2f} GB")
    if missing_root:
        print(f"WARNING missing ROOT files (not yet built?): {missing_root}")
    if missing_req:
        print(f"ERROR missing REQUIRED (prepare.py gate) — zip would be INVALID: {missing_req}");
        if not a.dry_run: raise SystemExit(1)
    else:
        print("REQUIRED gate: OK (all present)")

    if a.dry_run:
        print("DRY-RUN: no zip written."); return
    out = Path(a.out); tmp = out.with_suffix(".zip.tmp")
    print(f"writing {out} ...")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:     # STORED: .npz/.npy are already compressed; STORED is fast + avoids double-compression
        for p, arc in items:
            z.write(p, arc)
    tmp.replace(out)
    print(f"wrote {out} ({out.stat().st_size/1e9:.2f} GB). Next: python /home/photon/4tb/nersc_upload.py")


if __name__ == "__main__":
    main()
