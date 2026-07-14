"""Normalize the DeepCal modality caches into a DEDUPLICATED, unified-indexed structure for NERSC.

(a) NO DUPLICATION. Cell-based modalities (elev, soil, topo, hydro, daymet, clay, chm) give obs that share a cell a
    BYTE-IDENTICAL feature vector -- currently stored once PER OBS (thousands of copies for a dense cell). We store each
    UNIQUE vector once in <modality>_values.npz and map every obs to its row. Per-obs modalities (ground vision, NAIP
    embeddings) are genuinely unique per obs and stay per-obs. NAIP raw IMAGERY dedups to whole tiles + a per-obs
    pixel-window (handled in the NAIP builder / naip_index, not here -- imagery is not a fixed-width vector).

(b) FAST PER-OBS INDEX. One manifest.parquet keyed by gbifID: lat/lon/species/eventDate + <modality>_idx (int32 row
    into <modality>_values.npz, -1 = obs lacks the modality) + NAIP tile+window + provenance ids. Assembling EVERY
    modality for any observation = a single manifest row lookup + O(1) gathers. No shard scanning, no float matching.

Reproducible + lossless: values[obs] == <modality>_values.values[manifest.<modality>_idx[obs]] exactly (asserted).

    python -m deepearth.data.deepcal.normalize_cache --cache <dir> --out <dir>/norm [--report]
"""
import os, glob, json, argparse, hashlib
import numpy as np


# modality -> (path glob relative to cache, value key, provenance key or None, valid-mask key or None)
MODALITIES = {
    "vision_dino": ("gbif_tokens/*.npz", "dino", None, None),          # per-obs (unique)
    "vision_bio":  ("gbif_tokens/*.npz", "bio", None, None),           # per-obs (unique)
    "naip_rgb":    ("gbif_naip_tokens/*.npz", "rgb_pool", "naip_scene", None),  # per-obs embed + tile provenance
    "naip_ir":     ("gbif_naip_tokens/*.npz", "ir_pool", "naip_scene", None),
    "daymet":      ("gbif_daymet_tokens/*.npz", "daymet", None, "has_daymet"),  # cell-based -> dedups
    "clay":        ("gbif_clay_tokens.npz", "clay", "clay_year", "has_clay"),
    "elev":        ("gbif_elev.npz", "elev", "src", None),
    "soil":        ("gbif_soil_tokens.npz", "soil", None, "has_soil"),
    "topo":        ("gbif_topo_tokens.npz", "topo", "topo_scene", "has_topo"),    # scene = 3DEP 1m tile filename
    "hydro":       ("gbif_hydro_tokens.npz", "hydro", "hydro_scene", "has_hydro"),
    "chm":         ("gbif_chm_tokens.npz", "chm", None, "has_chm"),
}


def _load_modality(cache, glob_rel, vkey, pkey, mkey):
    """Concatenate a modality's shards -> (gbifID[N], values[N,D], provenance[N] or None, valid[N])."""
    files = sorted(glob.glob(os.path.join(cache, glob_rel)))
    gids, vals, provs, valids = [], [], [], []
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
        except Exception as e:                                          # skip a truncated/mid-build shard, don't crash
            print(f"    skip unreadable {os.path.basename(f)}: {type(e).__name__}"); continue
        if vkey not in z:
            continue
        gids.append(z["gbifID"].astype(np.int64))
        v = z[vkey]; vals.append(v.reshape(len(v), -1))                # flatten any [N,...] to [N,D]
        provs.append(z[pkey] if (pkey and pkey in z) else None)
        valids.append(z[mkey].astype(bool) if (mkey and mkey in z) else np.ones(len(z["gbifID"]), bool))
    if not gids:
        return None
    gid = np.concatenate(gids); val = np.concatenate(vals); valid = np.concatenate(valids)
    prov = np.concatenate([p for p in provs]) if provs[0] is not None else None
    # a gbifID can recur across shards (resumable rebuilds) -> keep first occurrence
    _, keep = np.unique(gid, return_index=True)
    gid, val, valid = gid[keep], val[keep], valid[keep]
    if prov is not None: prov = prov[keep]
    return gid, val, prov, valid


def _dedup(values):
    """Row-dedup by exact bytes -> (unique[K,D], inverse[N], first[K]). Byte-identical cell copies collapse to one row;
    first[k] = index of the first obs mapping to unique row k (for carrying one provenance id per unique value)."""
    seen, uniq, first, inv = {}, [], [], np.empty(len(values), np.int32)
    for i, row in enumerate(values):
        h = row.tobytes()
        j = seen.get(h)
        if j is None:
            j = len(uniq); seen[h] = j; uniq.append(row); first.append(i)
        inv[i] = j
    return (np.stack(uniq) if uniq else values[:0]), inv, np.array(first, np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", action="store_true", help="only print dedup ratios, write nothing")
    a = ap.parse_args()
    out = a.out or os.path.join(a.cache, "norm")
    if not a.report:
        os.makedirs(out, exist_ok=True)

    # base index = every observation we have coordinates for (env_priors/obs_coords.npz), gbifID-ordered
    z = np.load(os.path.join(a.cache, "env_priors", "obs_coords.npz"))
    base_gid = z["gbifID"].astype(np.int64)
    order = np.argsort(base_gid); base_gid = base_gid[order]
    lat = z["lat"][order]; lon = z["lon"][order]
    pos = {int(g): i for i, g in enumerate(base_gid)}                  # gbifID -> manifest row
    N = len(base_gid)
    manifest = {"gbifID": base_gid, "lat": lat, "lon": lon}
    total_before = total_after = 0

    for name, (glob_rel, vkey, pkey, mkey) in MODALITIES.items():
        loaded = _load_modality(a.cache, glob_rel, vkey, pkey, mkey)
        col = np.full(N, -1, np.int32)
        if loaded is None:
            manifest[f"{name}_idx"] = col
            print(f"  {name:12} absent"); continue
        gid, val, prov, valid = loaded
        gid, val = gid[valid], val[valid]                             # only real values get a row
        prov = prov[valid] if prov is not None else None
        uniq, inv, first = _dedup(val)                                # first[k] = index of the 1st obs mapping to unique row k
        rows = np.array([pos.get(int(g), -1) for g in gid])
        ok = rows >= 0
        col[rows[ok]] = inv[ok]
        manifest[f"{name}_idx"] = col
        before = val.nbytes; after = uniq.nbytes
        total_before += before; total_after += after
        print(f"  {name:12} obs={len(gid):>7} unique={len(uniq):>7} ({len(uniq)/max(len(gid),1)*100:4.1f}%)  "
              f"{before/1e6:7.1f}MB -> {after/1e6:6.1f}MB  ({before/max(after,1):4.1f}x)")
        if not a.report:
            save = {"values": uniq.astype(np.float32)}
            if prov is not None:                                      # one provenance id per unique row (its first obs)
                save["provenance"] = prov[first]
            np.savez_compressed(os.path.join(out, f"{name}_values.npz"), **save)

    print(f"\nTOTAL value bytes: {total_before/1e9:.2f} GB -> {total_after/1e9:.2f} GB  "
          f"({total_before/max(total_after,1):.1f}x dedup)")
    if not a.report:
        import pandas as pd
        pd.DataFrame(manifest).to_parquet(os.path.join(out, "manifest.parquet"))
        print(f"wrote {out}/manifest.parquet ({N} obs) + {len([m for m in MODALITIES])} <modality>_values.npz")


if __name__ == "__main__":
    main()
