"""DeepCal PLANT-modality build scripts, consolidated into one dispatcher.

Each subcommand is an original standalone run-once builder, preserved verbatim. Invoke as:
    python -m deepearth.data.deepcal.build_plant <cmd> [args]
  cmds: add_species, add_observation, build_species_dist, build_lfmc, build_mycorrhiza,
        build_flower_gap, build_patristic_ref, build_bioclip_text, plant_dated_distance, fetch_photos_oot

See each cmd_*() docstring for the original script's purpose and usage.
"""
import os, io, sys, csv, json, glob, time, argparse, subprocess, re, zipfile
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

# open_clip / model deps (via wandb) reference np.float_ etc., removed in numpy 2.0; restore aliases before import.
for _a, _t in [("float_", np.float64), ("int_", np.int64), ("uint", np.uint64),
               ("unicode_", np.str_), ("complex_", np.complex128), ("bool_", np.bool_)]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)

HERE = os.path.dirname(os.path.abspath(__file__))


# shared: normalized binomial (byte-identical in add_species + build_lfmc originals)
def _norm(s):
    p = str(s).split(); return (p[0] + " " + p[1]).lower() if len(p) >= 2 else str(s).strip().lower()


# ============================================================================ add_species
"""Production species registration: add a species to the DeepCal vocabulary so the model can train/infer on it,
whether or not it already has a position in the phylogeny. Idempotent.

Usage:
  python -m deepearth.data.deepcal.build_plant add_species --species "Clarkia williamsonii" --taxonomy "Plantae,Tracheophyta,Magnoliopsida,Myrtales,Onagraceae,Clarkia"
  python -m deepearth.data.deepcal.build_plant add_species --pending pending_species.json     # from add_observation
"""

def add_species_bioclip25_text(strings, dev=None):
    """Encode taxonomy strings with the frozen BioCLIP-2.5 text tower (rule 26), unit-normalized [N,1024]."""
    import torch, open_clip, torch.nn.functional as F
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    m, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2.5-vith14"); m = m.eval().to(dev)
    tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2.5-vith14")
    out = []
    with torch.no_grad():
        for i in range(0, len(strings), 128):
            t = m.encode_text(tok([f"a photo of {s}." for s in strings[i:i + 128]]).to(dev))
            out.append(F.normalize(t, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

def cmd_add_species(argv):
    ap = argparse.ArgumentParser(prog="build_plant add_species")
    ap.add_argument("--cache", default=HERE)
    ap.add_argument("--species", default=None); ap.add_argument("--taxonomy", default=None,
                    help="Kingdom,Phylum,Class,Order,Family,Genus for --species")
    ap.add_argument("--pending", default=None, help="JSON list of observation records (from add_observation)")
    a = ap.parse_args(argv); cache = a.cache

    vocab = np.load(os.path.join(cache, "gbif_vocab.npz"), allow_pickle=True)
    E1 = vocab["E1"].astype(np.float32); binos = [str(b) for b in vocab["binomial"]]; gidx = list(vocab["global_idx"])
    have = {_norm(b): i for i, b in enumerate(binos)}
    genus_of = np.array([b.split()[0] for b in binos])
    text = np.load(os.path.join(cache, "bioclip_taxon_text_emb.npy")).astype(np.float32)  # [n,1024], vocab order
    sidx = list(csv.DictReader(open(os.path.join(cache, "derived", "species_index.csv"))))
    next_global = max(int(r["idx"]) for r in sidx) + 1

    # ---- gather requested species with full taxonomy ----
    reqs = {}                                            # norm binomial -> (binomial, [K,P,C,O,F,G])
    if a.species:
        tx = (a.taxonomy or "").split(",")
        reqs[_norm(a.species)] = (a.species, (tx + [""] * 6)[:6])
    if a.pending and os.path.exists(a.pending):
        for r in json.load(open(a.pending)):
            reqs.setdefault(_norm(r["species"]), (r["species"], [r.get("kingdom", ""), r.get("phylum", ""),
                            r.get("klass", ""), r.get("order", ""), r.get("family", ""), r.get("genus", "")]))
    new = {k: v for k, v in reqs.items() if k not in have}
    print(f"{len(reqs)} requested, {len(new)} not yet in the vocab", flush=True)
    if not new:
        return

    keys = list(new); strings = [f"{tx[0]} {tx[1]} {tx[2]} {tx[3]} {tx[4]} {tx[5]} {b}".replace("  ", " ").strip()
                                 for b, tx in (new[k] for k in keys)]
    txt = add_species_bioclip25_text(strings)            # BioCLIP-2.5 seed per new species

    # Per-genus inductive position for genera ABSENT from the existing vocab: place ALL the batch's congeners of a
    # novel genus at ONE shared E1 (the BioCLIP-2.5-nearest known species to the genus's MEAN text seed), so congeners
    # cluster (cos~1) instead of scattering to independent nearest species -- we have no within-genus resolution for a
    # novel genus, so genus-level placement is the honest, principled seed (relatives near relatives).
    gmembers = defaultdict(list)
    for j, k in enumerate(keys):
        bino, tx = new[k]; gmembers[tx[5] or bino.split()[0]].append(j)
    novel_genus_e = {}
    for genus, members in gmembers.items():
        if not len(np.where(genus_of == genus)[0]):      # novel genus -> one shared position from the genus text centroid
            novel_genus_e[genus] = E1[int((text @ (txt[members].mean(0))).argmax())]
    add_E1, add_text, add_rows = [], [], []
    for j, k in enumerate(keys):
        bino, tx = new[k]; genus = tx[5] or bino.split()[0]
        cong = np.where(genus_of == genus)[0]            # congeners already in the vocab -> mean E1 (best placement)
        e = E1[cong].mean(0) if len(cong) else novel_genus_e[genus]   # else the shared novel-genus inductive position
        add_E1.append(e); add_text.append(txt[j])
        add_rows.append({"idx": next_global + j, "tip_label": f"{bino.replace(' ', '_')}__{next_global + j}",
                         "binomial": bino, "genus": genus, "family": tx[4], "order": tx[3]})

    # ---- persist: extend every per-species array consistently ----
    E1_new = np.concatenate([E1, np.stack(add_E1)]); text_new = np.concatenate([text, np.stack(add_text)])
    gidx_new = gidx + [r["idx"] for r in add_rows]; bino_new = binos + [new[k][0] for k in keys]
    np.savez(os.path.join(cache, "gbif_vocab.npz"), global_idx=np.array(gidx_new),
             binomial=np.array(bino_new, object), E1=E1_new)
    np.save(os.path.join(cache, "bioclip_taxon_text_emb.npy"), text_new)
    with open(os.path.join(cache, "derived", "species_index.csv"), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sidx[0].keys()))
        for r in add_rows:
            w.writerow({c: r.get(c, "") for c in sidx[0].keys()})
    # traits: append an all-unknown (0) row per new species so the trait heads treat them as unobserved
    tp = os.path.join(cache, "derived", "traits_syn.npz")
    if os.path.exists(tp):
        z = dict(np.load(tp, allow_pickle=True))
        need = next_global + len(keys)                    # traits are indexed by global_idx (species_index rows), not vocab
        for kk in list(z):
            if kk.startswith("cat_") and z[kk].ndim >= 1 and z[kk].shape[0] < need:
                z[kk] = np.concatenate([z[kk], np.zeros((need - z[kk].shape[0],) + z[kk].shape[1:], z[kk].dtype)])
        np.savez(tp, **z)
    print(f"registered {len(keys)} species -> vocab now {len(gidx_new)} (global_idx {next_global}..{next_global+len(keys)-1})", flush=True)
    print("delete prepared_*.pt so the dataset rebuilds with the new species.", flush=True)


# ============================================================================ add_observation
"""Production ingestion: add iNaturalist/GBIF observations to the DeepCal dataset with the FULL co-occurring-modality
fan-out (nothing left out). Idempotent + resumable.

Usage:
  python -m deepearth.data.deepcal.build_plant add_observation --kingdom Plantae  --limit 5000
  python -m deepearth.data.deepcal.build_plant add_observation --kingdom Animalia --taxon 797,1470 ...
  python -m deepearth.data.deepcal.build_plant add_observation --gbif-zip <path>
"""
_ADDOBS_PHOTOS = "/home/photon/4tb/deepearth_gbif/photos"   # 39 GB of already-downloaded iNat images, reused when present
_ADDOBS_INAT_RG = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"    # GBIF datasetKey for iNaturalist Research-grade Observations
_ADDOBS_FILTERS = dict(datasetKey=_ADDOBS_INAT_RG, country="US", stateProvince="California", year="2025",
                       hasCoordinate="true", mediaType="StillImage")   # the base-dataset download predicate (2025-locked)
_ADDOBS_MAX_COORD_UNC = 10.0                                # metres; high-GPS-precision only
_ADDOBS_KINGDOM_KEY = {"Plantae": 6, "Animalia": 1, "Fungi": 5}    # GBIF backbone taxonKeys — filters by KEY, not name
_ADDOBS_ENV_BUILDERS = [                                    # full co-occurring-modality fan-out, keyed off obs_coords, resumable
    "env_priors/build_daymet.py",                           # climate (Daymet) — REST, CPU
    "env_priors/build_soil.py",                             # soil (SSURGO) — REST, CPU
    "env_priors/build_topo.py",                             # 3DEP microtopography
    "env_priors/build_chm.py",                              # NAIP-CHM canopy structure
    "env_priors/build_hydrowind_torch.py",                  # drainage + wind (GPU physics)
    "env_priors/build_naip.py",                             # NAIP aerial DINOv3-SAT (GPU) — records naip_scene
    "env_priors/build_clay.py",                             # Clay / Sentinel-2 (GPU) — records clay_scene
]

def _addobs_dev():
    import torch; return "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ fetch
def _addobs_fetch_gbif(kingdom="Plantae", taxon_keys=None, limit=5000, page=300):
    """Stream GBIF occurrences matching the base predicate. ``taxon_keys`` (list) restricts to specific taxa (e.g.
    pollinator families/genera). Returns dicts with coords, time, taxonomy, and the still-image URL."""
    import requests
    out, offset = [], 0
    base = dict(_ADDOBS_FILTERS, limit=page)
    while len(out) < limit:
        params = dict(base, offset=offset)
        params["taxonKey"] = taxon_keys if taxon_keys else _ADDOBS_KINGDOM_KEY.get(kingdom, 6)   # filter by KEY
        r = requests.get("https://api.gbif.org/v1/occurrence/search", params=params, timeout=60)
        r.raise_for_status(); js = r.json()
        for o in js["results"]:
            unc = o.get("coordinateUncertaintyInMeters")
            if unc is not None and unc > _ADDOBS_MAX_COORD_UNC:    # enforce the high-precision filter client-side
                continue
            media = [m.get("identifier") for m in o.get("media", []) if m.get("type") == "StillImage"]
            if not (media and o.get("decimalLatitude") and o.get("species")):
                continue
            out.append(dict(gbifID=int(o["key"]), lat=float(o["decimalLatitude"]), lon=float(o["decimalLongitude"]),
                            eventDate=o.get("eventDate", ""), species=o["species"], taxonKey=o.get("speciesKey"),
                            kingdom=o.get("kingdom", ""), phylum=o.get("phylum", ""), klass=o.get("class", ""),
                            order=o.get("order", ""), family=o.get("family", ""), genus=o.get("genus", ""),
                            media=media[:6]))
        if js.get("endOfRecords") or not js["results"]:
            break
        offset += page
    return out[:limit]

# ------------------------------------------------------------------ bulk fetch via the GBIF Download API (all qualifying)
def _addobs_fetch_gbif_download(kingdom, taxon_keys, cache, email="lance@3co.ai", resume_key=None):
    """Submit a GBIF predicate download (the SAME mechanism the base plant dataset used) and return ALL qualifying
    records — no pagination cap. Async: submit -> poll -> fetch DwCA zip -> parse occurrence + multimedia. Robust to
    transient SSL/network errors; ``resume_key`` re-attaches to an in-flight download instead of resubmitting."""
    import requests, time as _t
    auth = (os.environ["GBIF_USER"], os.environ["GBIF_PWD"])
    def _rget(url, **kw):                                  # retry transient SSL/connection errors ~indefinitely
        for i in range(240):
            try: return requests.get(url, timeout=90, **kw)
            except Exception as e: print(f"  (retry {i}: {type(e).__name__})", flush=True); _t.sleep(min(15 + i, 60))
        return requests.get(url, timeout=90, **kw)
    if resume_key:
        dkey = resume_key; print(f"resuming GBIF download {dkey}...", flush=True)
    else:
        tk = [str(k) for k in (taxon_keys if taxon_keys else [_ADDOBS_KINGDOM_KEY.get(kingdom, 6)])]
        pred = {"type": "and", "predicates": [
            {"type": "equals", "key": "DATASET_KEY", "value": "50c9509d-22c7-4a22-a47d-8c48425ef4a7"},
            {"type": "equals", "key": "COUNTRY", "value": "US"},
            {"type": "equals", "key": "STATE_PROVINCE", "value": "California"},
            {"type": "equals", "key": "YEAR", "value": "2025"},
            {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
            {"type": "equals", "key": "MEDIA_TYPE", "value": "StillImage"},
            {"type": "lessThanOrEquals", "key": "COORDINATE_UNCERTAINTY_IN_METERS", "value": str(int(_ADDOBS_MAX_COORD_UNC))},
            {"type": "in", "key": "TAXON_KEY", "values": tk}]}
        body = {"creator": os.environ["GBIF_USER"], "notificationAddresses": [email],
                "sendNotification": False, "format": "DWCA", "predicate": pred}
        for _ in range(10):
            try:
                r = requests.post("https://api.gbif.org/v1/occurrence/download/request", json=body, auth=auth, timeout=90)
                r.raise_for_status(); break
            except Exception as e: print(f"  (submit retry: {type(e).__name__})", flush=True); _t.sleep(20)
        dkey = r.text.strip()
        print(f"GBIF download requested: {dkey} ({kingdom}, {len(tk)} taxa). polling...", flush=True)
    while True:
        meta = _rget(f"https://api.gbif.org/v1/occurrence/download/{dkey}").json()
        st = meta["status"]
        if st in ("SUCCEEDED", "KILLED", "CANCELLED", "FAILED"): break
        _t.sleep(30)
    if st != "SUCCEEDED":
        raise RuntimeError(f"GBIF download {dkey} ended {st}")
    dl = os.path.join(cache, "gbif_dl"); os.makedirs(dl, exist_ok=True)
    zpath = os.path.join(dl, f"{dkey}.zip")
    with _rget(meta["downloadLink"], stream=True) as resp:
        with open(zpath, "wb") as fh:
            for chunk in resp.iter_content(1 << 20): fh.write(chunk)
    print(f"downloaded {zpath} ({os.path.getsize(zpath)//1024//1024} MB); parsing DwCA...", flush=True)
    zf = zipfile.ZipFile(zpath)
    media = {}                                             # gbifID -> [image urls]
    if "multimedia.txt" in zf.namelist():
        for row in csv.DictReader(io.TextIOWrapper(zf.open("multimedia.txt"), "utf-8"), delimiter="\t"):
            if (row.get("type") == "StillImage" or "image" in (row.get("format") or "")) and row.get("identifier"):
                media.setdefault(row["gbifID"], []).append(row["identifier"])
    out = []
    for o in csv.DictReader(io.TextIOWrapper(zf.open("occurrence.txt"), "utf-8"), delimiter="\t"):
        gid = o.get("gbifID"); m = media.get(gid, [])
        if not (gid and m and o.get("decimalLatitude") and o.get("species")):
            continue
        out.append(dict(gbifID=int(gid), lat=float(o["decimalLatitude"]), lon=float(o["decimalLongitude"]),
                        eventDate=o.get("eventDate", ""), species=o["species"], taxonKey=o.get("speciesKey"),
                        kingdom=o.get("kingdom", ""), phylum=o.get("phylum", ""), klass=o.get("class", ""),
                        order=o.get("order", ""), family=o.get("family", ""), genus=o.get("genus", ""), media=m[:6]))
    print(f"parsed {len(out)} qualifying observations from the DwCA", flush=True)
    return out

# ------------------------------------------------------------------ dedup
def _addobs_existing_ids(cache, *extra_dirs):
    ids = set()
    for d in ("gbif_tokens", *extra_dirs):
        for f in glob.glob(os.path.join(cache, d, "*.npz")):
            ids |= set(np.load(f)["gbifID"].tolist())
    return ids

# ------------------------------------------------------------------ vision embed
def _addobs_open_image(rec):
    """ALL of an observation's photos (mean-pooled downstream, matching the original vision pipeline): cached
    `<gbifID>_*.jpg` under the species dir if present, else download every media URL. Up to 6 (as the source did)."""
    from PIL import Image
    imgs = []
    sp = rec["species"].replace(" ", "_"); d = os.path.join(_ADDOBS_PHOTOS, sp)
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.startswith(str(rec["gbifID"]) + "_"):
                try: imgs.append(Image.open(os.path.join(d, fn)).convert("RGB"))
                except Exception: pass
    if not imgs:
        import requests
        for url in rec["media"]:
            try: imgs.append(Image.open(io.BytesIO(requests.get(url, timeout=30).content)).convert("RGB"))
            except Exception: continue
    return imgs[:6]

def _addobs_binomial_to_local(cache):
    """Map normalized binomial -> vocab class index, so an added obs of a KNOWN species is immediately trainable."""
    v = np.load(os.path.join(cache, "gbif_vocab.npz"), allow_pickle=True)
    return {str(b).split()[0].lower() + " " + " ".join(str(b).split()[1:2]).lower(): i
            for i, b in enumerate(v["binomial"])}

def _addobs_embed_vision(records, cache, dev, outdir, batch=64, workers=24, known_only=False, shard_n=2000):
    """DINOv3-ViT-L/16 CLS (1024) per observation -> a new token shard. Images are downloaded in parallel (network-bound)
    and embedded in GPU batches (compute-bound) so the GPU stays busy. Known species get their vocab class index in
    ``species_local``; unknown species get -1 (add_species assigns their index later)."""
    import torch, concurrent.futures as cf
    from transformers import AutoImageProcessor, AutoModel
    import open_clip
    b2l = _addobs_binomial_to_local(cache)
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
    dino = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m").eval().to(dev)
    bmodel, _, bpre = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")   # BioCLIP-2 image tower (768)
    bmodel = bmodel.eval().to(dev)
    acc = {k: [] for k in ("gbifID", "species_local", "lat", "lon", "ord", "dino", "bio", "eventDate", "species")}
    os.makedirs(os.path.join(cache, outdir), exist_ok=True)
    written_ids, sidx = [], [0]
    def flush():                                                      # persist a shard every shard_n obs -> crash-safe + resumable
        if not acc["gbifID"]: return
        shard = os.path.join(cache, outdir, f"add_{int(time.time())}_{sidx[0]}.npz")
        np.savez_compressed(shard, **{k: np.array(v, object if k in ("eventDate", "species") else
                                                  (np.float32 if k in ("lat", "lon", "dino", "bio") else np.int64))
                                      for k, v in acc.items()})
        written_ids.extend(acc["gbifID"]); sidx[0] += 1
        print(f"  wrote {os.path.basename(shard)} (+{len(acc['gbifID'])}, total {len(written_ids)})", flush=True)
        for k in acc: acc[k].clear()
    def fetch(rec):
        try: return rec, _addobs_open_image(rec)
        except Exception: return rec, []
    for i in range(0, len(records), batch):
        with cf.ThreadPoolExecutor(workers) as ex:                    # parallel image download
            pairs = list(ex.map(fetch, records[i:i + batch]))
        flat, owner, recs = [], [], []                                # flat = all photos; owner[k] = obs index
        for rec, ims in pairs:
            if not ims: continue
            if known_only and b2l.get(" ".join(rec["species"].lower().split()[:2]), -1) < 0:
                continue                                              # skip obs of species not in the vocab
            r = len(recs); recs.append(rec)
            for im in ims: flat.append(im); owner.append(r)           # ALL photos of the obs (mean-pooled below)
        if not flat: continue
        de, be = [], []
        with torch.no_grad():                                         # embed every photo (sub-batched) with BOTH towers
            for j in range(0, len(flat), 128):
                px = dproc(images=flat[j:j + 128], return_tensors="pt")["pixel_values"].to(dev)
                de.append(dino(px).last_hidden_state[:, 0].float().cpu().numpy())
                bpx = torch.stack([bpre(im) for im in flat[j:j + 128]]).to(dev)
                be.append(bmodel.encode_image(bpx).float().cpu().numpy())
        de = np.concatenate(de); be = np.concatenate(be); owner = np.array(owner)
        for r, rec in enumerate(recs):
            emb = de[owner == r].mean(0)                              # DINOv3 CLS, mean over this obs's photos
            bemb = be[owner == r].mean(0)                             # BioCLIP-2 image, mean over photos -> vision_bio
            sl = b2l.get(" ".join(rec["species"].lower().split()[:2]), -1)
            for k, v in (("gbifID", rec["gbifID"]), ("species_local", sl), ("lat", rec["lat"]), ("lon", rec["lon"]),
                         ("ord", 0), ("dino", emb), ("bio", bemb), ("eventDate", rec["eventDate"]), ("species", rec["species"])):
                acc[k].append(v)
        print(f"  embedded {len(written_ids) + len(acc['gbifID'])}/{min(i + batch, len(records))}", flush=True)
        if len(acc["gbifID"]) >= shard_n: flush()          # incremental persistence
    flush()
    return written_ids                                     # gbifIDs written to shards; species_local filled in later

# ------------------------------------------------------------------ coords + env
def _addobs_append_coords(cache, records):
    """Append gbifID/lat/lon to the shared obs_coords.npz that every environmental builder reads."""
    p = os.path.join(cache, "env_priors", "obs_coords.npz"); z = np.load(p)
    gid = np.concatenate([z["gbifID"], np.array([r["gbifID"] for r in records], z["gbifID"].dtype)])
    lat = np.concatenate([z["lat"], np.array([r["lat"] for r in records], np.float32)])
    lon = np.concatenate([z["lon"], np.array([r["lon"] for r in records], np.float32)])
    np.savez(p, gbifID=gid, lat=lat, lon=lon)

def _addobs_fanout_existing(cache, outdir, dry_run=False):
    """Backfill env for obs embedded earlier with --no-env: read their coords from the --outdir token shards, append
    the ones missing from obs_coords.npz, then run the full env fan-out. Idempotent (builders are resumable)."""
    p = os.path.join(cache, "env_priors", "obs_coords.npz"); z = np.load(p)
    have = set(int(g) for g in z["gbifID"]); ng, nlat, nlon = [], [], []
    for f in sorted(glob.glob(os.path.join(cache, outdir, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        for i in range(len(d["gbifID"])):
            g = int(d["gbifID"][i])
            if g not in have:
                have.add(g); ng.append(g); nlat.append(float(d["lat"][i])); nlon.append(float(d["lon"][i]))
    print(f"[env-only] {outdir}: {len(ng)} new coords to append -> obs_coords.npz ({len(have)} total after)", flush=True)
    if dry_run:
        print("[env-only] DRY-RUN: not appending, not running builders", flush=True); return
    if ng:
        np.savez(p, gbifID=np.concatenate([z["gbifID"], np.array(ng, z["gbifID"].dtype)]),
                 lat=np.concatenate([z["lat"], np.array(nlat, np.float32)]),
                 lon=np.concatenate([z["lon"], np.array(nlon, np.float32)]))
    _addobs_run_env_builders(cache)

def _addobs_run_env_builders(cache):
    """Every environmental builder is resumable and keyed by gbifID off obs_coords.npz, so it only processes the new
    rows and appends to its own token cache. Adding a modality = adding its builder to ENV_BUILDERS (no other change)."""
    for b in _ADDOBS_ENV_BUILDERS:
        print(f"[env] {b}", flush=True)
        subprocess.run([sys.executable, os.path.join(cache, b)], cwd=cache, check=False)   # sys.executable keeps conda env

# ------------------------------------------------------------------ orchestrate
def cmd_add_observation(argv):
    ap = argparse.ArgumentParser(prog="build_plant add_observation")
    ap.add_argument("--cache", default=HERE)
    ap.add_argument("--kingdom", default="Plantae")
    ap.add_argument("--taxon", default=None, help="comma-separated GBIF taxonKeys to restrict to (e.g. pollinator clades)")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--outdir", default="gbif_tokens", help="token-shard subdir (use a separate track for non-plant taxa)")
    ap.add_argument("--shard_n", type=int, default=2000, help="write a token shard every N embedded obs (crash-safe/resumable)")
    ap.add_argument("--no-env", action="store_true", help="skip the env builders (vision embed + coords only)")
    ap.add_argument("--env-only", action="store_true", help="no fetch/embed: backfill coords from --outdir shards into obs_coords.npz, then run the full env fan-out (for obs embedded earlier with --no-env)")
    ap.add_argument("--download", action="store_true", help="use the GBIF Download API (ALL qualifying, no cap) — the plant-build path")
    ap.add_argument("--gbif-key", default=None, help="resume an in-flight GBIF download key instead of resubmitting")
    ap.add_argument("--known-only", action="store_true", help="embed only observations whose species is already in the vocab")
    ap.add_argument("--dry-run", action="store_true", help="fetch + dedup only; report counts, write nothing")
    a = ap.parse_args(argv)
    if a.env_only:                                          # backfill env on already-embedded (--no-env) obs, then stop
        _addobs_fanout_existing(a.cache, a.outdir, a.dry_run); return
    taxa = [int(t) for t in a.taxon.split(",")] if a.taxon else None
    if a.download or a.gbif_key:
        recs = _addobs_fetch_gbif_download(a.kingdom, taxa, a.cache, resume_key=a.gbif_key)   # bulk: all qualifying records
    else:
        print(f"fetching up to {a.limit} {a.kingdom} obs (2025, research-grade, CA, <=10 m, has image)...", flush=True)
        recs = _addobs_fetch_gbif(a.kingdom, taxa, a.limit)
    seen = _addobs_existing_ids(a.cache, a.outdir)
    new = [r for r in recs if r["gbifID"] not in seen]
    print(f"fetched {len(recs)}, {len(new)} new (not already in the dataset)", flush=True)
    if a.dry_run or not new:
        return
    dev = _addobs_dev()
    print(f"embedding {len(new)} observations on {dev}...", flush=True)
    written = _addobs_embed_vision(new, a.cache, dev, a.outdir, known_only=a.known_only, shard_n=a.shard_n)
    print(f"wrote {len(written)} vision tokens -> {a.outdir}/ (incremental shards)", flush=True)
    if not a.no_env:
        wset = set(written)
        _addobs_append_coords(a.cache, [r for r in new if r["gbifID"] in wset])
        _addobs_run_env_builders(a.cache)
    # new species -> add_species (vocab + BioCLIP-2.5 seed + tree placement)
    new_species = sorted({r["species"] for r in new})
    with open(os.path.join(a.cache, "pending_species.json"), "w") as f:
        json.dump([r for r in new if r["species"] in set(new_species)], f)
    print(f"{len(new_species)} distinct species touched; run add_species to register any not in the vocab.", flush=True)


# ============================================================================ build_species_dist
"""One-off: local species-abundance distributions around every observation at 3 scales (3km/300m/30m).
Ground truth for the distribution-prediction benchmarks (KL vs held-out). Cache -> gbif_species_dist.npz."""

def cmd_build_species_dist(argv):
    argparse.ArgumentParser(prog="build_plant build_species_dist").parse_args(argv)   # no args; validate none passed
    gid=[]; sp=[]; lat=[]; lon=[]
    for f in sorted(glob.glob("gbif_tokens/*.npz")):
        z=np.load(f); gid.append(z["gbifID"]); sp.append(z["species_local"]); lat.append(z["lat"]); lon.append(z["lon"])
    gid=np.concatenate(gid); sp=np.concatenate(sp).astype(np.int32); lat=np.concatenate(lat); lon=np.concatenate(lon)
    N=len(gid); print(f"{N} obs, {len(np.unique(sp))} species")
    SCALES={"3km":0.027, "300m":0.0027, "30m":0.00027}   # deg ~ meters at CA latitude
    TOPK=30
    out={"gbifID":gid}
    for name,d in SCALES.items():
        cell=(np.floor(lat/d).astype(np.int64)*4000000 + np.floor(lon/d).astype(np.int64))
        # per-cell species multiset
        cell_sp=defaultdict(lambda: defaultdict(int))
        for i in range(N): cell_sp[cell[i]][int(sp[i])]+=1
        # per-obs: top-K species idx + normalized freq of its cell (the local distribution to predict)
        idx=np.full((N,TOPK), -1, np.int32); frq=np.zeros((N,TOPK), np.float32); ncell=np.zeros(N,np.int32)
        for i in range(N):
            dd=cell_sp[cell[i]]; items=sorted(dd.items(), key=lambda kv:-kv[1])[:TOPK]
            tot=sum(dd.values()); ncell[i]=len(dd)
            for j,(s,c) in enumerate(items): idx[i,j]=s; frq[i,j]=c/tot
        out[f"idx_{name}"]=idx; out[f"frq_{name}"]=frq; out[f"nsp_{name}"]=ncell
        rich=ncell[ncell>1]
        print(f"  {name}: cells={len(cell_sp)} | obs-in-multispecies-cells={int((ncell>1).sum())} | median richness(>1)={int(np.median(rich)) if len(rich) else 0} | max={ncell.max()}")
    np.savez_compressed("gbif_species_dist.npz", **out)
    print("wrote gbif_species_dist.npz")


# ============================================================================ build_lfmc
"""B34 Live Fuel Moisture Content label from field measurements (fire/lfmc_data_conus.csv). Per-species MEDIAN LFMC
(%) over the California measurements, mapped onto the plant vocab by binomial.

    python -m deepearth.data.deepcal.build_plant build_lfmc            # cache = data/deepcal (DEEPCAL_DATA_DIR override)
"""
_LFMC_LO, _LFMC_HI = 10.0, 400.0    # physical LFMC % window (drop bad / fresh-growth readings)

def cmd_build_lfmc(argv):
    argparse.ArgumentParser(prog="build_plant build_lfmc").parse_args(argv)
    import pandas as pd
    lfmc_here = Path(os.environ.get("DEEPCAL_DATA_DIR", Path(__file__).resolve().parent))
    df = pd.read_csv(lfmc_here / "fire/lfmc_data_conus.csv")
    df = df[(df["state_region"] == "California") & df["lfmc_value"].between(_LFMC_LO, _LFMC_HI)]
    med = df.assign(sp=df["species_collected"].map(_norm)).groupby("sp")["lfmc_value"].median()
    print(f"{len(df)} CA measurements -> {len(med)} species with a median LFMC", flush=True)

    vb = np.load(lfmc_here / "gbif_vocab.npz", allow_pickle=True)["binomial"]
    lfmc = np.zeros(len(vb), np.float32); has = np.zeros(len(vb), bool)
    for i, b in enumerate(vb):
        v = med.get(_norm(b))
        if v is not None and np.isfinite(v):
            lfmc[i] = float(v); has[i] = True
    np.savez(lfmc_here / "gbif_lfmc.npz", lfmc=lfmc, has_lfmc=has)
    lv = lfmc[has]
    print(f"gbif_lfmc.npz: {int(has.sum())}/{len(vb)} vocab species labeled | "
          f"LFMC {lv.min():.0f}..{lv.max():.0f}% (median {np.median(lv):.0f}%)", flush=True)


# ============================================================================ build_mycorrhiza
"""B42 plant-fungal SYMBIOSIS label from the FungalRoot database. Per occurrence, take the majority mycorrhizal type
per plant GENUS and map it onto the DeepCal plant vocab. Output: gbif_mycorrhiza.npz. Real published data."""
_MYCO_CLASSES = ["AM", "EcM", "ErM", "OM", "NM"]      # arbuscular / ecto / ericoid / orchid / non-mycorrhizal
# Explicit map over the 12 distinct FungalRoot "Mycorrhiza type" values — substring matching is unsafe
# ("undetermined" contains "erm"). Dual types resolve to the more specialized partner; genuinely undetermined /
# non-vascular / "Other" are left UNLABELED (None) rather than guessed.
_MYCO_MAP = {
    "am": "AM",
    "non-mycorrhizal": "NM",
    "ecm, am undetermined": "EcM", "ecm, no am colonization": "EcM", "ecm,am": "EcM",
    "erm": "ErM", "erm,ecm": "ErM", "erm,am": "ErM",
    "om": "OM",
    "other": None, "non-ectomycorrhizal (am undetermined)": None, "am-like (non-vascular plants)": None,
}

def _myco_canon(v):
    return _MYCO_MAP.get(v.strip().lower())

def cmd_build_mycorrhiza(argv):
    argparse.ArgumentParser(prog="build_plant build_mycorrhiza").parse_args(argv)
    myco_here = Path(__file__).resolve().parent
    ZIP = Path(os.environ.get("FUNGALROOT_ZIP", "/home/photon/4tb/deepcal_data/fungalroot.zip"))
    z = zipfile.ZipFile(ZIP)
    # occurrences.csv: ID -> genus (the real taxon; the measurements' "Name" field is a record id, not a name)
    id_genus = {}
    with z.open("occurrences.csv") as f:
        r = csv.reader(io.TextIOWrapper(f, "utf-8")); h = next(r)
        idi, gi = h.index("ID"), h.index("genus")
        for row in r:
            if len(row) > max(idi, gi) and row[gi]: id_genus[row[idi]] = row[gi].strip().lower()
    # measurements.csv: Core ID -> Mycorrhiza type ; link by id
    gen = defaultdict(Counter)
    with z.open("measurements.csv") as f:
        r = csv.reader(io.TextIOWrapper(f, "utf-8")); h = next(r)
        cid, ti, vi = h.index("Core ID"), h.index("measurementType"), h.index("measurementValue")
        for row in r:
            if len(row) <= max(cid, ti, vi) or row[ti] != "Mycorrhiza type": continue
            g = id_genus.get(row[cid]); cl = _myco_canon(row[vi])
            if g and cl: gen[g][cl] += 1
    genus_type = {g: cnt.most_common(1)[0][0] for g, cnt in gen.items() if cnt}
    print(f"FungalRoot: {len(id_genus)} occ with genus; {len(genus_type)} genera with a majority mycorrhizal type")

    # map onto the DeepCal plant vocab (species -> genus -> type)
    vocab = np.load(myco_here / "gbif_vocab.npz", allow_pickle=True)
    binomials = vocab["binomial"]
    myco_idx = np.full(len(binomials), -1, np.int64)
    for i, b in enumerate(binomials):
        g = str(b).strip().split()[0].lower()
        t = genus_type.get(g)
        if t is not None: myco_idx[i] = _MYCO_CLASSES.index(t)
    have = myco_idx >= 0
    dist = Counter(_MYCO_CLASSES[j] for j in myco_idx[have])
    print(f"plant vocab {len(binomials)}: {int(have.sum())} species with a mycorrhizal type | dist {dict(dist)}")
    np.savez(myco_here / "gbif_mycorrhiza.npz", myco=myco_idx, has_myco=have, classes=np.array(_MYCO_CLASSES, object))
    print(f"wrote {myco_here/'gbif_mycorrhiza.npz'}")


# ============================================================================ build_flower_gap
"""Build flowering probabilities for the gap observations (the ~171,656 California-plant GBIF/iNaturalist obs in
gbif_tokens/ that are NOT already labeled in gbif_flower.npz) via PhenoVision (phenobase/phenovision).

STAGES
  python -m deepearth.data.deepcal.build_plant build_flower_gap --stage urls
  python -m deepearth.data.deepcal.build_plant build_flower_gap --stage infer --nshards 5 --shard 0
  python -m deepearth.data.deepcal.build_plant build_flower_gap --stage merge
Resumable; GPU forced off (CPU only).
"""
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only; do not touch busy GPU
_FLOWER_WORK = os.path.join(HERE, "flower_gap_work")
_FLOWER_URLS_JSONL = os.path.join(_FLOWER_WORK, "media_urls.jsonl")
_FLOWER_RESULTS_DIR = os.path.join(_FLOWER_WORK, "results")
_FLOWER_NPZ = os.path.join(HERE, "gbif_flower.npz")
_FLOWER_TOKENS_GLOB = os.path.join(HERE, "gbif_tokens", "chunk*.npz")
_FLOWER_OUT_NPZ = os.path.join(HERE, "gbif_flower_all.npz")
_FLOWER_THRESHOLD = 0.48
_FLOWER_MODEL_ID = "phenobase/phenovision"

def _flower_all_token_ids():
    ids = [np.load(f)["gbifID"] for f in sorted(glob.glob(_FLOWER_TOKENS_GLOB))]
    return np.unique(np.concatenate(ids))

def _flower_gap_ids():
    tok = _flower_all_token_ids()
    have = np.load(_FLOWER_NPZ)["gbifID"]
    return tok[~np.isin(tok, have)]

# ---------------- Stage: urls ----------------
def _flower_stage_urls(workers=16):
    import requests
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(_FLOWER_WORK, exist_ok=True)
    gap = _flower_gap_ids()
    done = set()
    if os.path.exists(_FLOWER_URLS_JSONL):
        with open(_FLOWER_URLS_JSONL) as f:
            for line in f:
                try:
                    done.add(int(json.loads(line)["gbifID"]))
                except Exception:
                    pass
    todo = [int(g) for g in gap if int(g) not in done]
    print(f"gap={len(gap)} already_fetched={len(done)} todo={len(todo)}", flush=True)
    sess = requests.Session()

    def fetch(gid):
        try:
            r = sess.get(f"https://api.gbif.org/v1/occurrence/{gid}", timeout=30)
            d = r.json()
            urls = [m["identifier"] for m in d.get("media", []) if m.get("identifier")]
            return {"gbifID": gid, "inat": d.get("catalogNumber"), "urls": urls}
        except Exception as e:
            return {"gbifID": gid, "err": type(e).__name__}

    t0 = time.time(); n = 0
    with open(_FLOWER_URLS_JSONL, "a") as out, ThreadPoolExecutor(workers) as ex:
        for rec in ex.map(fetch, todo):
            out.write(json.dumps(rec) + "\n"); n += 1
            if n % 2000 == 0:
                out.flush()
                print(f"  {n}/{len(todo)}  {n/(time.time()-t0):.1f} req/s", flush=True)
    print(f"done urls: +{n} in {(time.time()-t0)/60:.1f} min", flush=True)

# ---------------- Stage: infer ----------------
def _flower_medium_variants(url):
    """Prefer small 'medium.jpg' variant; fall back to the exact GBIF original URL."""
    base = url.rsplit("/", 1)[0]
    return [base + "/medium.jpg", base + "/large.jpg", url]

def _flower_stage_infer(shard, nshards, dl_workers=8, batch=16, limit=None):
    import torch, requests
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    from PIL import Image
    torch.set_num_threads(max(1, 12 // max(1, nshards)))  # split cores across shards
    os.makedirs(_FLOWER_RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(_FLOWER_RESULTS_DIR, f"shard_{shard:02d}.csv")
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            for line in f:
                try:
                    done.add(int(line.split(",")[0]))
                except Exception:
                    pass
    # load work items for this shard
    items = []
    with open(_FLOWER_URLS_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            g = int(rec["gbifID"])
            if g % nshards != shard:
                continue
            if g in done:
                continue
            urls = rec.get("urls") or []
            if urls:
                items.append((g, urls))
            else:
                items.append((g, []))  # no media -> nan
    if limit:
        items = items[:limit]
    print(f"[shard {shard}/{nshards}] to_process={len(items)} already={len(done)}", flush=True)

    DEV = os.environ.get("FLOWER_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(_FLOWER_MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(_FLOWER_MODEL_ID).eval().to(DEV)
    print(f"[shard {shard}] device={DEV}", flush=True)
    sess = requests.Session()

    def dl(url):
        for c in _flower_medium_variants(url):
            try:
                r = sess.get(c, timeout=30)
                if r.status_code == 200 and r.content:
                    return Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception:
                continue
        return None

    fout = open(out_csv, "a")
    t0 = time.time(); n = 0; nimg = 0
    pool = ThreadPoolExecutor(dl_workers)
    for gid, urls in items:
        if not urls:
            fout.write(f"{gid},nan,0\n"); n += 1; continue
        imgs = [im for im in pool.map(dl, urls) if im is not None]
        if not imgs:
            fout.write(f"{gid},nan,0\n"); n += 1; continue
        # batch inference over this obs's photos, take max flower prob
        pmax = 0.0
        for i in range(0, len(imgs), batch):
            px = proc(imgs[i:i+batch], return_tensors="pt")["pixel_values"].to(DEV)
            with torch.no_grad():
                p = torch.sigmoid(model(px).logits)[:, 1]
            pmax = max(pmax, float(p.max()))
        nimg += len(imgs)
        fout.write(f"{gid},{pmax:.6f},{len(imgs)}\n"); n += 1
        if n % 100 == 0:
            fout.flush()
            el = time.time() - t0
            print(f"[shard {shard}] {n}/{len(items)} obs  {nimg/el:.2f} img/s  "
                  f"{n/el:.2f} obs/s  eta {(len(items)-n)/max(n/el,1e-9)/3600:.1f}h", flush=True)
    fout.flush(); fout.close()
    print(f"[shard {shard}] DONE {n} obs, {nimg} imgs in {(time.time()-t0)/3600:.2f}h", flush=True)

# ---------------- Stage: merge ----------------
def _flower_stage_merge():
    # existing labels (preserved verbatim)
    ex = np.load(_FLOWER_NPZ)
    ids = list(ex["gbifID"].astype(np.int64))
    pf = list(ex["p_flower"].astype(np.float32))
    src = ["phenovision_orig"] * len(ids)
    have = set(int(x) for x in ex["gbifID"])
    # recomputed gap
    add_g, add_p = [], []
    for csvf in sorted(glob.glob(os.path.join(_FLOWER_RESULTS_DIR, "shard_*.csv"))):
        with open(csvf) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                g = int(parts[0]); v = parts[1]
                if g in have:
                    continue
                have.add(g)
                add_g.append(g)
                add_p.append(float("nan") if v == "nan" else float(v))
    ids += add_g; pf += add_p; src += ["phenovision_recompute"] * len(add_g)
    ids = np.array(ids, np.int64)
    pf = np.array(pf, np.float32)
    src = np.array(src)
    flower = (pf >= _FLOWER_THRESHOLD).astype(np.float32)  # nan -> False
    np.savez(_FLOWER_OUT_NPZ, gbifID=ids, p_flower=pf, flower=flower, source=src)
    tok = _flower_all_token_ids()
    covered = np.isin(tok, ids[~np.isnan(pf)]).mean()
    print(f"wrote {_FLOWER_OUT_NPZ}: {len(ids)} rows "
          f"({np.isnan(pf).sum()} no-media nan). "
          f"orig={int((src=='phenovision_orig').sum())} "
          f"recompute={int((src=='phenovision_recompute').sum())}. "
          f"training-set coverage (non-nan) = {covered*100:.2f}% of {len(tok)}")

def cmd_build_flower_gap(argv):
    ap = argparse.ArgumentParser(prog="build_plant build_flower_gap")
    ap.add_argument("--stage", required=True, choices=["urls", "infer", "merge", "count"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if a.stage == "urls":
        _flower_stage_urls(a.workers)
    elif a.stage == "infer":
        _flower_stage_infer(a.shard, a.nshards, limit=a.limit)
    elif a.stage == "merge":
        _flower_stage_merge()
    elif a.stage == "count":
        print("gap ids:", len(_flower_gap_ids()))


# ============================================================================ build_patristic_ref
"""Regenerate derived/patristic_ref.npy — the reference cophenetic (branch-length path) distance matrix over the
dated tree's tips, indexed by global_idx. Self-check reference for phylogenomic._test_real_tree.

    python -m deepearth.data.deepcal.build_plant build_patristic_ref
"""

def _patristic_cophenetic(cparent, cblen, nsp, n):
    """All-pairs cophenetic over the nsp tips via one post-order sweep (each pair filled once, at its LCA)."""
    depth = np.zeros(n, int); changed = True
    while changed:                                                 # depth by relaxation (parents shallower than children)
        changed = False
        for c in range(n):
            if cparent[c] >= 0 and depth[c] != depth[cparent[c]] + 1:
                depth[c] = depth[cparent[c]] + 1; changed = True
    rootdist = np.zeros(n)
    for c in np.argsort(depth):
        if cparent[c] >= 0: rootdist[c] = rootdist[cparent[c]] + cblen[c]
    kids = defaultdict(list)
    for c in range(n):
        if cparent[c] >= 0: kids[cparent[c]].append(c)
    tips = [[c] if c < nsp else [] for c in range(n)]             # species indices < nsp are tips
    M = np.zeros((nsp, nsp), np.float32)
    for u in np.argsort(-depth):                                  # deepest first -> children ready before their parent
        ch = kids[u]
        for a in range(len(ch)):
            ia = np.array(tips[ch[a]], int)
            for b in range(a):                                    # pairs across two different children first meet at u
                jb = np.array(tips[ch[b]], int)
                blk = rootdist[ia][:, None] + rootdist[jb][None, :] - 2 * rootdist[u]
                M[np.ix_(ia, jb)] = blk; M[np.ix_(jb, ia)] = blk.T
        if u >= nsp:
            tips[u] = [t for c in ch for t in tips[c]]
    np.fill_diagonal(M, 0.0)
    return M

def cmd_build_patristic_ref(argv):
    argparse.ArgumentParser(prog="build_plant build_patristic_ref").parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))       # parent of the `deepearth` package -> importable
    from deepearth.encoders.biological.phylogenomic import build_tree_buffers
    cache = Path(os.environ.get("DEEPCAL_DATA_DIR", Path(__file__).resolve().parent))
    rows = list(csv.DictReader(open(cache / "derived/species_index.csv")))
    tree_toks = set(re.findall(r"[^(),:;\s]+", open(cache / "ca_subtree.dated.nwk").read()))
    tree_rows = [r for r in rows if r["tip_label"] in tree_toks]           # tree tips, in species_index (idx) order
    tips = [r["tip_label"] for r in tree_rows]
    gidx = np.array([int(r["idx"]) for r in tree_rows])                    # each tree tip's global_idx (species_index row)
    print(f"{len(rows)} species_index rows -> {len(tips)} tree tips (global_idx 0..{int(gidx.max())})", flush=True)
    b = build_tree_buffers(str(cache / "ca_subtree.dated.nwk"), tips)      # species position k <-> tips[k] <-> gidx[k]
    n, nsp = b["n_nodes"], b["n_species"]
    cparent = np.full(n, -1, int); cblen = np.zeros(n)
    for c, p, bl in zip(b["down_child"], b["down_parent"], b["down_blen"]):
        cparent[c] = p; cblen[c] = bl * b["branch_scale"]                 # undo the unit-mean scaling -> Myr
    Mpos = _patristic_cophenetic(cparent, cblen, nsp, n)                  # [nsp,nsp] indexed by tip position
    M = np.zeros((int(gidx.max()) + 1,) * 2, np.float32)                  # reindex to global_idx (like the shipped matrix)
    M[np.ix_(gidx, gidx)] = Mpos
    print(f"built cophenetic {M.shape}", flush=True)

    ref = cache / "derived/patristic_ref.npy"
    if ref.exists():                                                      # audit vs the shipped matrix before overwriting
        D = np.load(ref)
        err = float(np.abs(M[:D.shape[0], :D.shape[1]] - D).max()) if D.shape == M.shape else float("nan")
        print(f"audit vs shipped patristic_ref {D.shape}: max abs err {err:.3g}", flush=True)
        assert err < 1e-3, f"builder drifted from the shipped reference (max err {err:.3g})"
    np.save(ref, M)
    print(f"wrote {ref}", flush=True)


# ============================================================================ build_bioclip_text
"""Regenerate bioclip_taxon_text_emb.npy — the frozen BioCLIP-2.5 taxon-string prior (science.md rule 26), one
unit-normalized [1024] vector per vocab species, in gbif_vocab order.

    BIOCLIP_META=/home/photon/4tb/deepearth_gbif/observations_meta.parquet \
    python -m deepearth.data.deepcal.build_plant build_bioclip_text --device cuda:1     # audits cosine vs shipped emb
"""
_BIOCLIP_MODEL = "hf-hub:imageomics/bioclip-2.5-vith14"

def _bioclip_text_encode(strings, dev):
    """Encode taxonomy strings with the frozen BioCLIP-2.5 text tower (rule 26), unit-normalized [N,1024]."""
    import torch, open_clip, torch.nn.functional as F
    m, _, _ = open_clip.create_model_and_transforms(_BIOCLIP_MODEL); m = m.eval().to(dev)
    tok = open_clip.get_tokenizer(_BIOCLIP_MODEL)
    out = []
    with torch.no_grad():
        for i in range(0, len(strings), 128):
            t = m.encode_text(tok([f"a photo of {s}." for s in strings[i:i + 128]]).to(dev))
            out.append(F.normalize(t, dim=-1).float().cpu())
    return torch.cat(out).numpy()

def cmd_build_bioclip_text(argv):
    ap = argparse.ArgumentParser(prog="build_plant build_bioclip_text"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--write", action="store_true", help="write the emb (default: audit only)")
    a = ap.parse_args(argv)
    bt_here = Path(os.environ.get("DEEPCAL_DATA_DIR", Path(__file__).resolve().parent))
    META = Path(os.environ.get("BIOCLIP_META", "/home/photon/4tb/deepearth_gbif/observations_meta.parquet"))

    bino = np.load(bt_here / "gbif_vocab.npz", allow_pickle=True)["binomial"]          # [N], vocab order (authority)
    ofg = {r["binomial"]: (r["order"], r["family"], r["genus"])
           for r in csv.DictReader(open(bt_here / "derived/species_index.csv"))}       # O/F/G

    import pandas as pd
    df = pd.read_parquet(META, columns=["species", "kingdom", "phylum", "class"]).dropna(subset=["species"])
    kpc = {}                                                                        # species -> most-common (K,P,C)
    for sp, g in df.groupby("species"):
        kpc[sp] = (Counter(g["kingdom"].dropna()).most_common(1) or [("",)])[0][0], \
                  (Counter(g["phylum"].dropna()).most_common(1) or [("",)])[0][0], \
                  (Counter(g["class"].dropna()).most_common(1) or [("",)])[0][0]

    strings, miss = [], 0
    for b in bino:
        K, P, C = kpc.get(b, ("", "", ""))
        O, Fa, G = ofg.get(b, ("", "", ""))
        if b not in kpc or b not in ofg: miss += 1
        s = f"{K} {P} {C} {O} {Fa} {G} {b}".replace("  ", " ").strip()
        strings.append(s)
    print(f"{len(strings)} vocab species | {miss} missing taxonomy | e.g. {strings[0]!r}", flush=True)

    emb = _bioclip_text_encode(strings, a.device)
    print(f"encoded {emb.shape} norm mean {np.linalg.norm(emb, axis=1).mean():.4f}", flush=True)

    ref = bt_here / "bioclip_taxon_text_emb.npy"
    if ref.exists():
        D = np.load(ref).astype(np.float32)
        cos = (emb * D).sum(1) if D.shape == emb.shape else None
        if cos is not None:
            print(f"audit vs shipped {D.shape}: cosine min {cos.min():.4f} mean {cos.mean():.4f} "
                  f"median {np.median(cos):.4f} | <0.99: {int((cos < 0.99).sum())}", flush=True)
    if a.write:
        np.save(ref, emb); print(f"wrote {ref}", flush=True)


# ============================================================================ plant_dated_distance
"""Build the PLANT species-graph DATED distance from ca_subtree.dated.nwk (rules 7-12). Output gbif_plant_dist.npz.

    python -m deepearth.data.deepcal.build_plant plant_dated_distance
"""
_PLANT_RSCRIPT = "/home/photon/miniconda3/envs/rphylo/bin/Rscript"

def cmd_plant_dated_distance(argv):
    argparse.ArgumentParser(prog="build_plant plant_dated_distance").parse_args(argv)
    pd_here = Path(__file__).resolve().parent
    CACHE = pd_here
    DERIVED = CACHE / "derived"
    vocab = np.load(CACHE / "gbif_vocab.npz", allow_pickle=True)
    gi = vocab["global_idx"]
    rows = list(csv.DictReader(open(DERIVED / "species_index.csv")))
    tips = [rows[g]["tip_label"] for g in gi]                          # model species tip_labels in vocab order
    # clean reproducible intermediate the R builder consumes
    with open(DERIVED / "model_species.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model_idx", "tip_label"])
        for i, t in enumerate(tips): w.writerow([i, t])
    # cophenetic in R (ape) over the tree-covered model species
    r = subprocess.run([_PLANT_RSCRIPT, "deepearth/data/deepcal/plant_dated_patristic.R"],
                       cwd=str(pd_here.parents[2]), capture_output=True, text=True)
    sys.stdout.write(r.stdout);  sys.stderr.write(r.stderr)
    if "PLANT_DATED_PATRISTIC_DONE" not in r.stdout:
        raise SystemExit("R cophenetic failed")
    # read the dated cophenetic (tip_label-labelled) and map back to model indices
    rr = list(csv.reader(open(DERIVED / "plant_cophen.csv")))
    labels = [c.strip() for c in rr[0][1:]]
    M = np.array([[float(x) for x in row[1:]] for row in rr[1:]], np.float64)
    t2i = {t: i for i, t in enumerate(tips)}
    midx = np.array([t2i[l] for l in labels], np.int64)               # vocab-order index of each covered species
    assert M.shape[0] == M.shape[1] == len(midx)
    assert np.allclose(M, M.T, atol=1e-4) and np.allclose(np.diag(M), 0)
    np.savez(CACHE / "gbif_plant_dist.npz", dated=M.astype(np.float32), model_idx=midx)
    print(f"wrote {CACHE/'gbif_plant_dist.npz'}: {len(midx)}/{len(tips)} model species on real dated cophenetic "
          f"(mean {M[~np.eye(len(M),dtype=bool)].mean():.1f} Myr); {len(tips)-len(midx)} keep the embedding shadow")


# ============================================================================ fetch_photos_oot
"""Fetch iNat photo URLs for obs missing from photo_manifest (the recovered out-of-tree species), so rembed can
embed their ground vision. Resumable (skips obs already in the manifest).

    python -m deepearth.data.deepcal.build_plant fetch_photos_oot [LIMIT]
"""

def cmd_fetch_photos_oot(argv):
    import pandas as pd, requests
    GB = Path.home()/"deepearth/data/deepearth_gbif"; DER = Path.home()/"deepearth/data/cache/derived"
    LIMIT = int(argv[0]) if len(argv) > 0 else 0
    meta = pd.read_parquet(GB/"observations_meta.parquet").drop_duplicates("gbifID")
    mani = pd.read_parquet(GB/"photo_manifest.parquet"); have = set(mani.gbifID.tolist())
    b2g = set(r["binomial"] for r in csv.DictReader(open(DER/"species_index.csv")))
    todo = meta[meta.species.isin(b2g) & ~meta.gbifID.isin(have) & meta.occurrenceID.notna()].copy()
    todo["inat_id"] = todo.occurrenceID.str.extract(r"observations/(\d+)")[0]
    todo = todo[todo.inat_id.notna()]
    if LIMIT: todo = todo.head(LIMIT)
    print(f"{len(todo)} obs need photos (out-of-tree)", flush=True)
    sess = requests.Session()
    def api(ids):
        for k in range(4):
            try:
                r = sess.get("https://api.inaturalist.org/v1/observations?per_page=200&id="+",".join(ids), timeout=30)
                if r.status_code == 200:
                    return {str(o["id"]):[p["url"].replace("/square.","/medium.") for p in o.get("photos",[]) if p.get("url")] for o in r.json().get("results",[])}
            except Exception: pass
            time.sleep(2*(k+1))
        return {}
    id2r = {str(r.inat_id):r for r in todo.itertuples()}; ids = list(id2r); rows = []
    for i in range(0, len(ids), 200):
        for iid, purls in api(ids[i:i+200]).items():
            r = id2r[iid]
            for u in purls:
                rows.append(dict(gbifID=r.gbifID, photo_url=u, species=r.species, decimalLatitude=r.decimalLatitude, decimalLongitude=r.decimalLongitude, eventDate=str(r.eventDate), occurrenceID=r.occurrenceID))
        if i % 4000 == 0: print(f"  {i}/{len(ids)} | {len(rows)} photo rows", flush=True)
        time.sleep(0.5)
    if rows and not LIMIT:
        pd.concat([mani, pd.DataFrame(rows)], ignore_index=True).to_parquet(GB/"photo_manifest.parquet")
        print(f"extended photo_manifest: +{len(rows)} rows", flush=True)
    else:
        print(f"TEST: got {len(rows)} photo rows for {len(ids)} obs; sample: {rows[0] if rows else None}", flush=True)


if __name__ == "__main__":
    cmds = {
        "add_species": cmd_add_species,
        "add_observation": cmd_add_observation,
        "build_species_dist": cmd_build_species_dist,
        "build_lfmc": cmd_build_lfmc,
        "build_mycorrhiza": cmd_build_mycorrhiza,
        "build_flower_gap": cmd_build_flower_gap,
        "build_patristic_ref": cmd_build_patristic_ref,
        "build_bioclip_text": cmd_build_bioclip_text,
        "plant_dated_distance": cmd_plant_dated_distance,
        "fetch_photos_oot": cmd_fetch_photos_oot,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print("usage: python -m deepearth.data.deepcal.build_plant <cmd> [args]\n  cmds: " + ", ".join(cmds))
        sys.exit(1)
    cmds[sys.argv[1]](sys.argv[2:])
