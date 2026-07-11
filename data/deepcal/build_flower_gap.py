#!/usr/bin/env python3
"""
Build flowering probabilities for the gap observations (the ~171,656 California-plant
GBIF/iNaturalist obs in gbif_tokens/ that are NOT already labeled in gbif_flower.npz).

WHY THIS EXISTS
---------------
- All 207,431 training obs are from calendar year 2025 (verified: gbif_eventtime.npz
  min 2025-01-01, max 2025-12-31). The Zenodo PhenoVision iNat annotations
  (record 15306421) only cover observations through March 2024 -> 0% overlap.
  So there is NO pre-computed source; the only path is to run PhenoVision on the images.
- gbifID -> iNaturalist observation id is carried by GBIF occurrence `catalogNumber`
  (and occurrenceID/references). The image URLs come from the GBIF media array
  (media[].identifier -> inaturalist-open-data S3 .../photos/{id}/original.jpg).
- PhenoVision = huggingface phenobase/phenovision (ViT-L, 224px, multi-label sigmoid,
  id2label {0:fruit, 1:flower}, flower threshold 0.48).
- Aggregation: MAX flower prob over ALL photos of the observation. This was verified
  to reproduce the existing 35,775 labels (e.g. obs 5140987289 stored 0.9983; photo0
  gives 0.013 but a later photo gives 0.9985 -> max matches). photo0-only would produce
  false negatives.

STAGES
------
  python build_flower_gap.py --stage urls    # fetch media URLs for all gap ids (GBIF API)
  python build_flower_gap.py --stage infer --nshards 5 --shard 0   # run one worker
       ... launch shards 0..nshards-1 as separate processes to scale on CPU ...
  python build_flower_gap.py --stage merge   # write gbif_flower_all.npz

Resumable: every stage skips gbifIDs already recorded, so re-running continues.
GPU: forced off (CPU only) via CUDA_VISIBLE_DEVICES="" inside the script.
"""
import os, sys, json, glob, time, argparse, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only; do not touch busy GPU
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "flower_gap_work")
URLS_JSONL = os.path.join(WORK, "media_urls.jsonl")
RESULTS_DIR = os.path.join(WORK, "results")
FLOWER_NPZ = os.path.join(HERE, "gbif_flower.npz")
TOKENS_GLOB = os.path.join(HERE, "gbif_tokens", "chunk*.npz")
OUT_NPZ = os.path.join(HERE, "gbif_flower_all.npz")
THRESHOLD = 0.48
MODEL_ID = "phenobase/phenovision"


def all_token_ids():
    ids = [np.load(f)["gbifID"] for f in sorted(glob.glob(TOKENS_GLOB))]
    return np.unique(np.concatenate(ids))


def gap_ids():
    tok = all_token_ids()
    have = np.load(FLOWER_NPZ)["gbifID"]
    return tok[~np.isin(tok, have)]


# ---------------- Stage: urls ----------------
def stage_urls(workers=16):
    import requests
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(WORK, exist_ok=True)
    gap = gap_ids()
    done = set()
    if os.path.exists(URLS_JSONL):
        with open(URLS_JSONL) as f:
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
    with open(URLS_JSONL, "a") as out, ThreadPoolExecutor(workers) as ex:
        for rec in ex.map(fetch, todo):
            out.write(json.dumps(rec) + "\n"); n += 1
            if n % 2000 == 0:
                out.flush()
                print(f"  {n}/{len(todo)}  {n/(time.time()-t0):.1f} req/s", flush=True)
    print(f"done urls: +{n} in {(time.time()-t0)/60:.1f} min", flush=True)


# ---------------- Stage: infer ----------------
def _medium_variants(url):
    """Prefer small 'medium.jpg' variant; fall back to the exact GBIF original URL."""
    base = url.rsplit("/", 1)[0]
    return [base + "/medium.jpg", base + "/large.jpg", url]


def stage_infer(shard, nshards, dl_workers=8, batch=16, limit=None):
    import torch, requests
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    from PIL import Image
    torch.set_num_threads(max(1, 12 // max(1, nshards)))  # split cores across shards
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, f"shard_{shard:02d}.csv")
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
    with open(URLS_JSONL) as f:
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
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID).eval().to(DEV)
    print(f"[shard {shard}] device={DEV}", flush=True)
    sess = requests.Session()

    def dl(url):
        for c in _medium_variants(url):
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
def stage_merge():
    # existing labels (preserved verbatim)
    ex = np.load(FLOWER_NPZ)
    ids = list(ex["gbifID"].astype(np.int64))
    pf = list(ex["p_flower"].astype(np.float32))
    src = ["phenovision_orig"] * len(ids)
    have = set(int(x) for x in ex["gbifID"])
    # recomputed gap
    add_g, add_p = [], []
    for csv in sorted(glob.glob(os.path.join(RESULTS_DIR, "shard_*.csv"))):
        with open(csv) as f:
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
    flower = (pf >= THRESHOLD).astype(np.float32)  # nan -> False
    np.savez(OUT_NPZ, gbifID=ids, p_flower=pf, flower=flower, source=src)
    tok = all_token_ids()
    covered = np.isin(tok, ids[~np.isnan(pf)]).mean()
    print(f"wrote {OUT_NPZ}: {len(ids)} rows "
          f"({np.isnan(pf).sum()} no-media nan). "
          f"orig={int((src=='phenovision_orig').sum())} "
          f"recompute={int((src=='phenovision_recompute').sum())}. "
          f"training-set coverage (non-nan) = {covered*100:.2f}% of {len(tok)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["urls", "infer", "merge", "count"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.stage == "urls":
        stage_urls(a.workers)
    elif a.stage == "infer":
        stage_infer(a.shard, a.nshards, limit=a.limit)
    elif a.stage == "merge":
        stage_merge()
    elif a.stage == "count":
        print("gap ids:", len(gap_ids()))
