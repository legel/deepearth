"""Fetch SSURGO soil properties at every observation from USDA Soil Data Access (SDA). Aggregation matches the
existing gbif_soil_tokens.npz exactly: dominant component (max comppct_r), top horizon (min hzdept_r), representative
(_r) values. Two-phase, batched, threaded, resumable: (1) point -> mukey via the WktWgs84 intersection TVF,
(2) mukey -> 9 props (dedup: many obs share a map unit). Nulls -> nan; has_soil=False when no mukey/horizon row.

Output: gbif_soil_tokens.npz {gbifID, soil[N,9], has_soil[N], property_names[9]} with properties
[ph1to1h2o_r, om_r, claytotal_r, sandtotal_r, silttotal_r, awc_r, ksat_r, cec7_r, dbthirdbar_r]."""
import os, sys, json, time, glob, pickle, argparse, urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
SDA = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
PROPS = ["ph1to1h2o_r", "om_r", "claytotal_r", "sandtotal_r", "silttotal_r", "awc_r", "ksat_r", "cec7_r", "dbthirdbar_r"]
NPROP = len(PROPS)
CKPT = os.path.join(HERE, "soil_ckpt.pkl")
WORKERS = 12
PT_BATCH = 100          # points per phase-1 request
MU_BATCH = 100          # mukeys per phase-2 request

def sda(query, retries=5):
    body = json.dumps({"query": query, "format": "JSON"}).encode()
    for k in range(retries):
        try:
            req = urllib.request.Request(SDA, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read()).get("Table", [])
        except Exception:
            if k == retries - 1:
                return None
            time.sleep(2.0 * (k + 1))
    return None

def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan

def resolve_mukeys(batch):
    """batch: list[(key, lon, lat)] -> {key: mukey|None}. Points outside SSURGO (water/AK-off) get None."""
    u = " UNION ALL ".join(
        f"SELECT {i} AS pid,mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('point({lo} {la})')"
        for i, (k, lo, la) in enumerate(batch))
    t = sda(u)
    if t is None:
        return None
    hit = {int(r[0]): str(r[1]) for r in t}
    return {batch[i][0]: hit.get(i) for i in range(len(batch))}

def fetch_props(mukeys):
    """mukeys: list[str] -> {mukey: [9 floats]}. Each map unit is the comppct_r-weighted mean, across all its
    components, of the component's surface (min hzdept_r) horizon. Nulls are dropped and weights renormalized
    per property (matches SDA weighted-average aggregation and the existing cache exactly)."""
    q = ("SELECT mukey,comppct_r," + ",".join(PROPS) + " FROM (SELECT c.mukey,c.comppct_r," +
         ",".join("h." + p for p in PROPS) + ",ROW_NUMBER() OVER (PARTITION BY c.cokey ORDER BY h.hzdept_r ASC) rn "
         "FROM component c JOIN chorizon h ON h.cokey=c.cokey WHERE c.mukey IN (" + ",".join(mukeys) + ")) t "
         "WHERE rn=1")
    t = sda(q)
    if t is None:
        return None
    agg = {}
    for r in t:
        agg.setdefault(str(r[0]), []).append((_f(r[1]), [_f(x) for x in r[2:2 + NPROP]]))
    out = {}
    for mk, rows in agg.items():
        vec = np.full(NPROP, np.nan)
        for j in range(NPROP):
            num = den = 0.0
            for w, vals in rows:
                if not (np.isnan(w) or np.isnan(vals[j])):
                    num += w * vals[j]; den += w
            if den > 0:
                vec[j] = num / den
        out[mk] = vec.tolist()
    return out

def load_obs():
    """Union of obs_coords.npz and any gbif_pollinator_obs/*.npz, deduped by gbifID (first wins)."""
    gid, lat, lon = [], [], []
    seen = set()
    srcs = [os.path.join(HERE, "obs_coords.npz")] + sorted(glob.glob(os.path.join(HERE, "..", "gbif_pollinator_obs", "*.npz")))
    for p in srcs:
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        g, la, lo = z["gbifID"], z["lat"], z["lon"]
        for i in range(len(g)):
            k = int(g[i])
            if k in seen:
                continue
            seen.add(k); gid.append(k); lat.append(float(la[i])); lon.append(float(lo[i]))
    return np.array(gid, np.int64), np.array(lat, np.float64), np.array(lon, np.float64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only first N obs (test)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "gbif_soil_tokens.npz"))
    ap.add_argument("--ckpt", default=CKPT)
    a = ap.parse_args()

    gid, lat, lon = load_obs()
    if a.limit:
        gid, lat, lon = gid[:a.limit], lat[:a.limit], lon[:a.limit]
    N = len(gid)
    key = [(round(float(lat[i]), 6), round(float(lon[i]), 6)) for i in range(N)]   # coord dedup key

    ck = {"mukey": {}, "props": {}}
    if os.path.exists(a.ckpt):
        ck = pickle.load(open(a.ckpt, "rb"))
        print(f"resume: {len(ck['mukey'])} coords, {len(ck['props'])} mukeys cached", flush=True)

    # Phase 1: point -> mukey (unique unresolved coords, batched)
    ukeys = {k: (lon[i], lat[i]) for i, k in enumerate(key)}
    todo = [(k, float(v[0]), float(v[1])) for k, v in ukeys.items() if k not in ck["mukey"]]
    print(f"phase1: {len(ukeys)} unique coords, {len(todo)} to resolve", flush=True)
    t0 = time.time()
    batches = [todo[i:i + PT_BATCH] for i in range(0, len(todo), PT_BATCH)]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(resolve_mukeys, b): b for b in batches}
        for j, fut in enumerate(as_completed(futs)):
            res = fut.result()
            if res:
                ck["mukey"].update(res)
            if (j + 1) % 20 == 0:
                pickle.dump(ck, open(a.ckpt, "wb"))
                print(f"  {j+1}/{len(batches)} batches | {time.time()-t0:.0f}s", flush=True)
    pickle.dump(ck, open(a.ckpt, "wb"))

    # Phase 2: mukey -> props (unique unresolved mukeys, batched)
    mukeys = sorted({m for m in ck["mukey"].values() if m and m not in ck["props"]})
    print(f"phase2: {len({m for m in ck['mukey'].values() if m})} unique mukeys, {len(mukeys)} to fetch", flush=True)
    t0 = time.time()
    mb = [mukeys[i:i + MU_BATCH] for i in range(0, len(mukeys), MU_BATCH)]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_props, b): b for b in mb}
        for j, fut in enumerate(as_completed(futs)):
            res = fut.result()
            if res:
                ck["props"].update(res)
            if (j + 1) % 20 == 0:
                pickle.dump(ck, open(a.ckpt, "wb"))
                print(f"  {j+1}/{len(mb)} batches | {time.time()-t0:.0f}s", flush=True)
    pickle.dump(ck, open(a.ckpt, "wb"))

    # Assemble aligned arrays
    soil = np.full((N, NPROP), np.nan, np.float32)
    have = np.zeros(N, bool)
    for i in range(N):
        mk = ck["mukey"].get(key[i])
        if mk and mk in ck["props"]:
            soil[i] = ck["props"][mk]; have[i] = True
    np.savez(a.out, gbifID=gid, soil=soil, has_soil=have,
             property_names=np.array(PROPS, dtype=object))
    print(f"DONE: {have.sum()}/{N} obs have soil ({100*have.mean():.1f}%) -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
