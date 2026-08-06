"""Fetch iNaturalist photos for the GloBI Tier-2 pollinator observations (B41). Reliable path: batched iNat API
(obs_id -> photo URLs) then parallel S3 downloads. Photos -> pollinator/photos/; manifest -> pollinator/photo_manifest.csv.
Embedding (DINOv2 + BioCLIP, to match gbif_tokens) is a separate GPU step. Resumable."""
import os, csv, json, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(HERE, "photos"); os.makedirs(PHOTOS, exist_ok=True)
MANIFEST = os.path.join(HERE, "photo_manifest.csv")
sess = requests.Session()

def load_obs():
    ids = []
    with open(os.path.join(HERE, "inat_photo_todo.csv")) as f:
        for r in csv.DictReader(f): ids.append(r["obs_id"])
    return ids

def api_photo_urls(batch_ids):
    """iNat API: up to 200 obs ids -> {obs_id: [photo_url(medium)]}."""
    url = "https://api.inaturalist.org/v1/observations?per_page=200&id=" + ",".join(batch_ids)
    for k in range(4):
        try:
            r = sess.get(url, timeout=30)
            if r.status_code == 200:
                out = {}
                for o in r.json().get("results", []):
                    ps = [p["url"].replace("/square.", "/medium.") for p in o.get("photos", []) if p.get("url")]
                    if ps: out[str(o["id"])] = ps
                return out
            time.sleep(2 * (k + 1))
        except Exception:
            time.sleep(2 * (k + 1))
    return {}

def dl(args):
    oid, pid, url = args
    d = os.path.join(PHOTOS, oid); os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, f"{pid}.jpg")
    if os.path.exists(fp) and os.path.getsize(fp) > 0: return (oid, fp)
    for c in (url, url.replace("/medium.", "/large."), url.replace("/medium.", "/original.")):
        try:
            r = sess.get(c, timeout=30)
            if r.status_code == 200 and r.content:
                open(fp, "wb").write(r.content); return (oid, fp)
        except Exception:
            continue
    return (oid, None)

def main():
    ids = load_obs()
    done = set()
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            for r in csv.reader(f): done.add(r[0])
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} obs, {len(todo)} to fetch", flush=True)
    mf = open(MANIFEST, "a"); t0 = time.time(); n = 0; np = 0
    for c0 in range(0, len(todo), 200):
        batch = todo[c0:c0 + 200]
        urls = api_photo_urls(batch)
        jobs = [(oid, str(pi), u) for oid, us in urls.items() for pi, u in enumerate(us)]
        with ThreadPoolExecutor(24) as ex:
            got = {}
            for oid, fp in ex.map(dl, jobs):
                if fp: got.setdefault(oid, []).append(fp)
        for oid in batch:
            mf.write(f"{oid},{'|'.join(got.get(oid, []))}\n"); n += 1; np += len(got.get(oid, []))
        mf.flush()
        el = time.time() - t0
        print(f"  {n}/{len(todo)} obs  {np} photos  {n/el:.1f} obs/s  eta {(len(todo)-n)/max(n/el,1e-9)/60:.1f}min", flush=True)
    mf.close(); print(f"DONE: {n} obs, {np} photos", flush=True)

if __name__ == "__main__":
    main()
