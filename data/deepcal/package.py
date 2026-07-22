"""DeepCal packaging / cache scripts, consolidated into one dispatcher.

Each subcommand is an original standalone run-once script, preserved verbatim. Invoke as:
    python -m deepearth.data.deepcal.package <cmd> [args]
  cmds: package_dataset, nersc_upload, normalize_cache

See each cmd_*() docstring for the original script's purpose and usage.
"""
import os, io, glob, json, argparse, zipfile, hashlib, subprocess, time
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent                      # .../data/deepcal


# ============================================================================ package_dataset
"""Package the audited DeepCal cache into deepcal_data.zip for NERSC hosting. EXPLICIT manifest — never a blind
`zip -r`. `--dry-run` lists the manifest + sizes and checks prepare.py's readiness gate without writing the zip.

    python -m deepearth.data.deepcal.package package_dataset --dry-run          # validate the manifest
    python -m deepearth.data.deepcal.package package_dataset --out /home/photon/4tb/deepcal_data.zip   # build
"""
# What data.py / train.py / evaluate.py load. Single-file caches + the phylogeny + the shipped provenance.
_PKG_ROOT = ["gbif_vocab.npz", "ca_subtree.dated.nwk", "data_provenance.yaml",
        "gbif_elev.npz", "gbif_eventtime.npz",
        "gbif_soil_tokens.npz", "gbif_topo_tokens.npz", "gbif_chm_tokens.npz", "gbif_hydro_tokens.npz",
        "gbif_clay_tokens.npz",
        "gbif_species_dist.npz", "gbif_pollinator_dist.npz", "gbif_flower_all.npz", "gbif_lfmc.npz", "gbif_mycorrhiza.npz",
        "gbif_plant_dist.npz",
        "bioclip_taxon_text_emb.npy",                       # BioCLIP-2.5 taxon-string prior (supersedes legacy BioCLIP-2)
        "pollinator_taxon_text_emb.npy", "pollinator_distance.npy", "pollinator_animal_mask.npy"]
# Globbed multi-shard modalities (ground vision, aerial, climate, pollinator obs) + the derived tables.
_PKG_DIRS = ["derived", "gbif_tokens", "gbif_daymet_tokens", "gbif_naip_tokens", "gbif_pollinator_obs"]
# Included if present (reproducible re-runs / legacy readers), skipped without error otherwise.
_PKG_OPTIONAL = ["observations_meta.parquet", "gbif_pollinator_tokens.npz", "gbif_flower.npz", "env_priors/obs_coords.npz"]
# prepare.py's readiness gate — the zip is invalid if any is missing.
_PKG_REQUIRED = ["gbif_vocab.npz", "ca_subtree.dated.nwk", "derived/species_index.csv", "derived/patristic_ref.npy"]

def _pkg_manifest():
    """Resolve the manifest to a list of (absolute_path, arcname) pairs, in a stable order."""
    items, missing_root = [], []
    for r in _PKG_ROOT:
        p = HERE / r
        (items if p.exists() else missing_root).append((p, r) if p.exists() else r)
    for o in _PKG_OPTIONAL:
        p = HERE / o
        if p.exists():
            items.append((p, o))
    for d in _PKG_DIRS:
        base = HERE / d
        for p in sorted(base.rglob("*")):
            if p.is_file():
                items.append((p, str(p.relative_to(HERE))))
    return items, missing_root

def cmd_package_dataset(argv):
    ap = argparse.ArgumentParser(prog="package package_dataset",
                                 description="Package the audited DeepCal cache into a zip for NERSC.")
    ap.add_argument("--out", default="/home/photon/4tb/deepcal_data.zip")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    items, missing_root = _pkg_manifest()
    total = sum(p.stat().st_size for p, _ in items)
    have_arc = {arc for _, arc in items}
    missing_req = [r for r in _PKG_REQUIRED if r not in have_arc]

    # per-top-level summary
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
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:     # STORED: .npz/.npy already compressed; STORED is fast
        for p, arc in items:
            z.write(p, arc)
    tmp.replace(out)
    print(f"wrote {out} ({out.stat().st_size/1e9:.2f} GB). Next: python -m deepearth.data.deepcal.package nersc_upload")


# ============================================================================ nersc_upload
"""Publish deepcal_data.zip to NERSC CFS via SFAPI, in resumable 150 MB chunks. Build the zip first with
package_dataset.

  python -m deepearth.data.deepcal.package package_dataset --out $DEEPCAL_ZIP   # build (post-recompute)
  python -m deepearth.data.deepcal.package nersc_upload                         # publish

Credentials (SFAPI, from iris.nersc.gov) are read from $SUPERFACILITY_DIR (default ~/.superfacility).
"""

def cmd_nersc_upload(argv):
    argparse.ArgumentParser(prog="package nersc_upload").parse_args(argv)
    from sfapi_client import Client
    SF = Path(os.environ.get("SUPERFACILITY_DIR", str(Path.home() / ".superfacility")))
    ZIP = Path(os.environ.get("DEEPCAL_ZIP", "/home/photon/4tb/deepcal_data.zip"))
    PARTS = ZIP.parent / "parts"
    CHUNK = 150 * 1024 * 1024                       # 150 MB parts (under the SFAPI per-file limit)
    REMOTE = os.environ.get("DEEPCAL_REMOTE", "/global/cfs/cdirs/m5239/www/deepcal")
    KEY = os.environ.get("SFAPI_KEY", "legel.pem")

    if not ZIP.exists():
        raise SystemExit(f"{ZIP} not found — build it first: python -m deepearth.data.deepcal.package package_dataset --out {ZIP}")
    c = Client((SF / "clientid.txt").read_text().strip(), (SF / KEY).read_text())
    pm = c.compute("perlmutter")
    pm.run(f"mkdir -p {REMOTE}/parts")
    PARTS.mkdir(exist_ok=True)
    if not any(PARTS.glob("part_*")):
        print("splitting zip..."); subprocess.run(["split", "-b", str(CHUNK), str(ZIP), str(PARTS / "part_")], check=True)
    parts = sorted(PARTS.glob("part_*"))
    print(f"{len(parts)} parts, {ZIP.stat().st_size / 1e9:.2f} GB total")
    existing = {l.split()[-1] for l in pm.run(f"ls {REMOTE}/parts 2>/dev/null").splitlines() if l}
    [d] = pm.ls(f"{REMOTE}/parts", directory=True)
    for i, p in enumerate(parts):
        if p.name in existing:
            print(f"[{i+1}/{len(parts)}] {p.name} already present"); continue
        for attempt in range(4):                                       # resumable: skips uploaded parts, retries transient
            try:
                b = io.BytesIO(p.read_bytes()); b.filename = p.name
                d.upload(b); print(f"[{i+1}/{len(parts)}] {p.name} uploaded"); break
            except Exception as e:
                print(f"[{i+1}/{len(parts)}] {p.name} attempt {attempt+1} failed: {repr(e)[:120]}"); time.sleep(3)
        else:
            print(f"ABORT: {p.name} failed 4x"); return
    print("reassembling on NERSC...")
    print(pm.run(f"cd {REMOTE} && cat parts/part_* > deepcal_data.zip && rm -rf parts && ls -la deepcal_data.zip"))
    local = ZIP.stat().st_size
    remote = int(pm.run(f"stat -c%s {REMOTE}/deepcal_data.zip").strip())
    print(f"local {local} vs remote {remote} -> {'MATCH' if local == remote else 'MISMATCH'}")
    subprocess.run(["rm", "-rf", str(PARTS)])
    print("URL: https://portal.nersc.gov/cfs/m5239/deepcal/deepcal_data.zip")


# ============================================================================ normalize_cache
"""Normalize the DeepCal modality caches into a DEDUPLICATED, unified-indexed structure for NERSC: one
manifest.parquet keyed by gbifID + per-modality <modality>_values.npz (unique cell vectors stored once).

    python -m deepearth.data.deepcal.package normalize_cache --cache <dir> --out <dir>/norm [--report]
"""
# modality -> (path glob relative to cache, value key, provenance key or None, valid-mask key or None)
_NC_MODALITIES = {
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

def _nc_load_modality(cache, glob_rel, vkey, pkey, mkey):
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

def _nc_dedup(values):
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

def cmd_normalize_cache(argv):
    ap = argparse.ArgumentParser(prog="package normalize_cache")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", action="store_true", help="only print dedup ratios, write nothing")
    a = ap.parse_args(argv)
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

    for name, (glob_rel, vkey, pkey, mkey) in _NC_MODALITIES.items():
        loaded = _nc_load_modality(a.cache, glob_rel, vkey, pkey, mkey)
        col = np.full(N, -1, np.int32)
        if loaded is None:
            manifest[f"{name}_idx"] = col
            print(f"  {name:12} absent"); continue
        gid, val, prov, valid = loaded
        gid, val = gid[valid], val[valid]                             # only real values get a row
        prov = prov[valid] if prov is not None else None
        uniq, inv, first = _nc_dedup(val)                             # first[k] = index of the 1st obs mapping to unique row k
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
        print(f"wrote {out}/manifest.parquet ({N} obs) + {len([m for m in _NC_MODALITIES])} <modality>_values.npz")


if __name__ == "__main__":
    import sys
    cmds = {
        "package_dataset": cmd_package_dataset,
        "nersc_upload": cmd_nersc_upload,
        "normalize_cache": cmd_normalize_cache,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print("usage: python -m deepearth.data.deepcal.package <cmd> [args]\n  cmds: " + ", ".join(cmds))
        sys.exit(1)
    cmds[sys.argv[1]](sys.argv[2:])
