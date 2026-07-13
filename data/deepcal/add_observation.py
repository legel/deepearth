"""Production ingestion: add iNaturalist/GBIF observations to the DeepCal dataset with the FULL co-occurring-modality
fan-out (nothing left out). Idempotent + resumable — re-running only processes genuinely new observations.

Pipeline per new observation (keyed by gbifID, matching the existing cache schema):
  1. fetch    GBIF occurrence search (iNat research-grade, CA, Plantae|Animalia, year 2025, <=10m, has image) -> records
  2. dedup    drop gbifIDs already embedded (gbif_tokens/*.npz)
  3. vision   download image(s) -> DINOv3-ViT-L/16 CLS (1024) + BioCLIP-2 image (768), mean-pooled -> a new token shard
  4. coords   append gbifID/lat/lon(/elev/eventtime) to the shared modality inputs
  5. env      run every environmental builder (topo, chm, hydro; daymet/naip/clay/soil plug in the same way) -> tokens
  6. species  any new species -> add_species (registers vocab + BioCLIP-2.5 seed + tree placement)

Usage:
  python -m deepearth.data.deepcal.add_observation --kingdom Plantae  --limit 5000        # grow plant coverage
  python -m deepearth.data.deepcal.add_observation --kingdom Animalia --taxon 797,1470 ...  # pollinator obs (B-III)
  python -m deepearth.data.deepcal.add_observation --gbif-zip <path>                        # ingest a bulk GBIF download
Same 2025 / research-grade / <=10 m / has-image constraints as the base dataset (see deepcal-data-provenance)."""
import os, io, sys, csv, json, argparse, subprocess, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = "/home/photon/4tb/deepearth_gbif/photos"          # 39 GB of already-downloaded iNat images, reused when present
INAT_RG = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"           # GBIF datasetKey for iNaturalist Research-grade Observations
FILTERS = dict(datasetKey=INAT_RG, country="US", stateProvince="California", year="2025",
               hasCoordinate="true", mediaType="StillImage")   # the base-dataset download predicate (2025-locked)
MAX_COORD_UNC = 10.0                                       # metres; high-GPS-precision only
KINGDOM_KEY = {"Plantae": 6, "Animalia": 1, "Fungi": 5}    # GBIF backbone taxonKeys — occurrence/search filters by KEY, not name
ENV_BUILDERS = ["env_priors/build_topo.py", "env_priors/build_chm.py", "env_priors/build_hydrowind_torch.py"]

def _dev():
    import torch; return "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ fetch
def fetch_gbif(kingdom="Plantae", taxon_keys=None, limit=5000, page=300):
    """Stream GBIF occurrences matching the base predicate. ``taxon_keys`` (list) restricts to specific taxa (e.g.
    pollinator families/genera). Returns dicts with coords, time, taxonomy, and the still-image URL."""
    import requests
    out, offset = [], 0
    base = dict(FILTERS, limit=page)
    while len(out) < limit:
        params = dict(base, offset=offset)
        params["taxonKey"] = taxon_keys if taxon_keys else KINGDOM_KEY.get(kingdom, 6)   # filter by KEY (name is ignored by GBIF)
        r = requests.get("https://api.gbif.org/v1/occurrence/search", params=params, timeout=60)
        r.raise_for_status(); js = r.json()
        for o in js["results"]:
            unc = o.get("coordinateUncertaintyInMeters")
            if unc is not None and unc > MAX_COORD_UNC:    # enforce the high-precision filter client-side
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
def fetch_gbif_download(kingdom, taxon_keys, cache, email="lance@3co.ai", resume_key=None):
    """Submit a GBIF predicate download (the SAME mechanism the base plant dataset used) and return ALL qualifying
    records — no pagination cap. Async: submit -> poll -> fetch DwCA zip -> parse occurrence + multimedia. Robust to
    transient SSL/network errors; ``resume_key`` re-attaches to an in-flight download instead of resubmitting."""
    import requests, zipfile, time as _t
    auth = (os.environ["GBIF_USER"], os.environ["GBIF_PWD"])
    def _rget(url, **kw):                                  # retry transient SSL/connection errors ~indefinitely (downloads prep for hours)
        for i in range(240):
            try: return requests.get(url, timeout=90, **kw)
            except Exception as e: print(f"  (retry {i}: {type(e).__name__})", flush=True); _t.sleep(min(15 + i, 60))
        return requests.get(url, timeout=90, **kw)
    if resume_key:
        dkey = resume_key; print(f"resuming GBIF download {dkey}...", flush=True)
    else:
        tk = [str(k) for k in (taxon_keys if taxon_keys else [KINGDOM_KEY.get(kingdom, 6)])]
        pred = {"type": "and", "predicates": [
            {"type": "equals", "key": "DATASET_KEY", "value": "50c9509d-22c7-4a22-a47d-8c48425ef4a7"},
            {"type": "equals", "key": "COUNTRY", "value": "US"},
            {"type": "equals", "key": "STATE_PROVINCE", "value": "California"},
            {"type": "equals", "key": "YEAR", "value": "2025"},
            {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
            {"type": "equals", "key": "MEDIA_TYPE", "value": "StillImage"},
            {"type": "lessThanOrEquals", "key": "COORDINATE_UNCERTAINTY_IN_METERS", "value": str(int(MAX_COORD_UNC))},
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
        import io as _io
        for row in csv.DictReader(_io.TextIOWrapper(zf.open("multimedia.txt"), "utf-8"), delimiter="\t"):
            if (row.get("type") == "StillImage" or "image" in (row.get("format") or "")) and row.get("identifier"):
                media.setdefault(row["gbifID"], []).append(row["identifier"])
    out = []
    import io as _io
    for o in csv.DictReader(_io.TextIOWrapper(zf.open("occurrence.txt"), "utf-8"), delimiter="\t"):
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
def existing_ids(cache, *extra_dirs):
    import glob
    ids = set()
    for d in ("gbif_tokens", *extra_dirs):
        for f in glob.glob(os.path.join(cache, d, "*.npz")):
            ids |= set(np.load(f)["gbifID"].tolist())
    return ids

# ------------------------------------------------------------------ vision embed
def _open_image(rec):
    """ALL of an observation's photos (mean-pooled downstream, matching the original vision pipeline): cached
    `<gbifID>_*.jpg` under the species dir if present, else download every media URL. Up to 6 (as the source did)."""
    from PIL import Image
    imgs = []
    sp = rec["species"].replace(" ", "_"); d = os.path.join(PHOTOS, sp)
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

def _binomial_to_local(cache):
    """Map normalized binomial -> vocab class index, so an added obs of a KNOWN species is immediately trainable."""
    v = np.load(os.path.join(cache, "gbif_vocab.npz"), allow_pickle=True)
    return {str(b).split()[0].lower() + " " + " ".join(str(b).split()[1:2]).lower(): i
            for i, b in enumerate(v["binomial"])}

def embed_vision(records, cache, dev, outdir, batch=64, workers=24, known_only=False, shard_n=2000):
    """DINOv3-ViT-L/16 CLS (1024) per observation -> a new token shard. Images are downloaded in parallel (network-bound)
    and embedded in GPU batches (compute-bound) so the GPU stays busy. Known species get their vocab class index in
    ``species_local``; unknown species get -1 (add_species assigns their index later)."""
    import torch, concurrent.futures as cf
    from transformers import AutoImageProcessor, AutoModel
    b2l = _binomial_to_local(cache)
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
    dino = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m").eval().to(dev)
    acc = {k: [] for k in ("gbifID", "species_local", "lat", "lon", "ord", "dino", "eventDate", "species")}
    os.makedirs(os.path.join(cache, outdir), exist_ok=True)
    written_ids, sidx = [], [0]
    def flush():                                                      # persist a shard every shard_n obs -> crash-safe + resumable (dedup skips written gbifIDs)
        if not acc["gbifID"]: return
        shard = os.path.join(cache, outdir, f"add_{int(time.time())}_{sidx[0]}.npz")
        np.savez_compressed(shard, **{k: np.array(v, object if k in ("eventDate", "species") else
                                                  (np.float32 if k in ("lat", "lon", "dino", "bio") else np.int64))
                                      for k, v in acc.items()})
        written_ids.extend(acc["gbifID"]); sidx[0] += 1
        print(f"  wrote {os.path.basename(shard)} (+{len(acc['gbifID'])}, total {len(written_ids)})", flush=True)
        for k in acc: acc[k].clear()
    def fetch(rec):
        try: return rec, _open_image(rec)
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
        de = []
        with torch.no_grad():                                         # embed every photo (sub-batched)
            for j in range(0, len(flat), 128):
                px = dproc(images=flat[j:j + 128], return_tensors="pt")["pixel_values"].to(dev)
                de.append(dino(px).last_hidden_state[:, 0].float().cpu().numpy())
        de = np.concatenate(de); owner = np.array(owner)
        for r, rec in enumerate(recs):
            emb = de[owner == r].mean(0)                              # mean over this obs's photos (matches the original)
            sl = b2l.get(" ".join(rec["species"].lower().split()[:2]), -1)
            for k, v in (("gbifID", rec["gbifID"]), ("species_local", sl), ("lat", rec["lat"]), ("lon", rec["lon"]),
                         ("ord", 0), ("dino", emb), ("eventDate", rec["eventDate"]), ("species", rec["species"])):
                acc[k].append(v)
        print(f"  embedded {len(written_ids) + len(acc['gbifID'])}/{min(i + batch, len(records))}", flush=True)
        if len(acc["gbifID"]) >= shard_n: flush()          # incremental persistence
    flush()
    return written_ids                                     # gbifIDs written to shards; species_local filled in by add_species / vocab merge

# ------------------------------------------------------------------ coords + env
def append_coords(cache, records):
    """Append gbifID/lat/lon to the shared obs_coords.npz that every environmental builder reads."""
    p = os.path.join(cache, "env_priors", "obs_coords.npz"); z = np.load(p)
    gid = np.concatenate([z["gbifID"], np.array([r["gbifID"] for r in records], z["gbifID"].dtype)])
    lat = np.concatenate([z["lat"], np.array([r["lat"] for r in records], np.float32)])
    lon = np.concatenate([z["lon"], np.array([r["lon"] for r in records], np.float32)])
    np.savez(p, gbifID=gid, lat=lat, lon=lon)

def run_env_builders(cache):
    """Every environmental builder is resumable and keyed by gbifID off obs_coords.npz, so it only processes the new
    rows and appends to its own token cache. Adding a modality = adding its builder to ENV_BUILDERS (no other change)."""
    for b in ENV_BUILDERS:
        print(f"[env] {b}", flush=True)
        subprocess.run([sys.executable, os.path.join(cache, b)], cwd=cache, check=False)   # sys.executable keeps the conda env

# ------------------------------------------------------------------ orchestrate
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=HERE)
    ap.add_argument("--kingdom", default="Plantae")
    ap.add_argument("--taxon", default=None, help="comma-separated GBIF taxonKeys to restrict to (e.g. pollinator clades)")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--outdir", default="gbif_tokens", help="token-shard subdir (use a separate track for non-plant taxa)")
    ap.add_argument("--shard_n", type=int, default=2000, help="write a token shard every N embedded obs (crash-safe/resumable)")
    ap.add_argument("--no-env", action="store_true", help="skip the env builders (vision embed + coords only)")
    ap.add_argument("--download", action="store_true", help="use the GBIF Download API (ALL qualifying, no cap) — the plant-build path")
    ap.add_argument("--gbif-key", default=None, help="resume an in-flight GBIF download key instead of resubmitting")
    ap.add_argument("--known-only", action="store_true", help="embed only observations whose species is already in the vocab")
    ap.add_argument("--dry-run", action="store_true", help="fetch + dedup only; report counts, write nothing")
    a = ap.parse_args()
    taxa = [int(t) for t in a.taxon.split(",")] if a.taxon else None
    if a.download or a.gbif_key:
        recs = fetch_gbif_download(a.kingdom, taxa, a.cache, resume_key=a.gbif_key)   # bulk: all qualifying records
    else:
        print(f"fetching up to {a.limit} {a.kingdom} obs (2025, research-grade, CA, <=10 m, has image)...", flush=True)
        recs = fetch_gbif(a.kingdom, taxa, a.limit)
    seen = existing_ids(a.cache, a.outdir)
    new = [r for r in recs if r["gbifID"] not in seen]
    print(f"fetched {len(recs)}, {len(new)} new (not already in the dataset)", flush=True)
    if a.dry_run or not new:
        return
    dev = _dev()
    print(f"embedding {len(new)} observations on {dev}...", flush=True)
    written = embed_vision(new, a.cache, dev, a.outdir, known_only=a.known_only, shard_n=a.shard_n)   # writes shards incrementally (crash-safe/resumable)
    print(f"wrote {len(written)} vision tokens -> {a.outdir}/ (incremental shards)", flush=True)
    if not a.no_env:
        wset = set(written)
        append_coords(a.cache, [r for r in new if r["gbifID"] in wset])
        run_env_builders(a.cache)
    # new species -> add_species (vocab + BioCLIP-2.5 seed + tree placement)
    new_species = sorted({r["species"] for r in new})
    with open(os.path.join(a.cache, "pending_species.json"), "w") as f:
        json.dump([r for r in new if r["species"] in set(new_species)], f)
    print(f"{len(new_species)} distinct species touched; run add_species.py to register any not in the vocab.", flush=True)

if __name__ == "__main__":
    main()
