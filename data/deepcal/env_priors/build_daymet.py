"""Fetch Daymet daily climate (ORNL single-pixel REST) for every observation: a 180-day window ending at the
obs eventDate (or a fixed 2025 window when no date). 7 vars in Daymet CSV order [dayl,prcp,srad,swe,tmax,tmin,vp].
One fetch per ~1km Daymet cell (deduped), reused for every obs+window in the cell. Missing days -> nan rows.

Output: gbif_daymet_tokens/chunk####.npz {gbifID[n], daymet[n,180,7] f16, ndays[n] i16}. ndays (non-nan days) is
the presence mask: ndays==0 means the fetch failed / no data. Resumable via already-written shards + a cell ckpt.
"""
import os, sys, time, pickle, argparse, warnings, glob, urllib.request
from datetime import date, timedelta, datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://daymet.ornl.gov/single-pixel/api/data"
VARS = "tmin,tmax,prcp,srad,vp,swe,dayl"           # request set (API returns canonical order below)
COLS = ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]   # cache column order (matches existing shards)
WIN = 180                                          # days per window
CELL = 0.01                                        # ~1.1km dedup cell (Daymet native grid is 1km)
FIXED_END = date(2025, 12, 31)                     # window end when an obs has no eventDate
WORKERS = 16
CHUNK = 8000                                       # obs per output shard (matches existing ~7990)

def fetch(lat, lon, years, retries=5):
    """Return {(year,yday): float32[7]} for the requested years, or None on failure."""
    url = f"{BASE}?lat={lat:.5f}&lon={lon:.5f}&vars={VARS}&years={','.join(map(str, sorted(years)))}"
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                lines = r.read().decode().splitlines()
        except Exception:
            if k == retries - 1:
                return None
            time.sleep(2.0 * (k + 1)); continue
        hi = next((i for i, l in enumerate(lines) if l.startswith("year,yday")), -1)
        if hi < 0:
            return None
        head = [h.split()[0] for h in lines[hi].split(",")]
        idx = [head.index(c) for c in COLS]                    # map header -> cache order
        lut = {}
        for l in lines[hi + 1:]:
            p = l.split(",")
            if len(p) < 9:
                continue
            lut[(int(p[0]), int(float(p[1])))] = np.array([float(p[j]) for j in idx], np.float32)
        return lut
    return None

def window(evd, lut):
    """180-day window ending (inclusive) at evd, oldest first. Daymet yday d in year Y == date(Y,1,1)+(d-1);
    leap Dec31 (d=366) and unavailable years fall through to nan rows."""
    a = np.full((WIN, 7), np.nan, np.float32); nd = 0
    for i in range(WIN):
        dt = evd - timedelta(days=WIN - 1 - i)
        row = lut.get((dt.year, (dt - date(dt.year, 1, 1)).days + 1))
        if row is not None:
            a[i] = row; nd += 1
    return a.astype(np.float16), nd

def process_cell(obs):
    """obs: [(gid, lat, lon, evd)]. One fetch (mean coord) covering the union of window years; slice per obs."""
    lat = float(np.mean([o[1] for o in obs])); lon = float(np.mean([o[2] for o in obs]))
    yrs = set()
    for _, _, _, evd in obs:
        yrs |= {(evd - timedelta(days=WIN - 1)).year, evd.year}
    lut = fetch(lat, lon, yrs)
    if not lut:
        return None
    return {gid: window(evd, lut) for gid, _, _, evd in obs}

def parse_date(v):
    if v is None:
        return FIXED_END
    s = str(v)
    for f in (s, s.split("T")[0]):
        try:
            return datetime.fromisoformat(f).date()
        except Exception:
            pass
    return FIXED_END

def load_coords(src):
    """Return [(gid, lat, lon, eventDate|None)]. src='obs' -> obs_coords.npz (no date); 'pollinator' -> token shards."""
    if src == "pollinator":
        rows = []
        for f in sorted(glob.glob(os.path.join(HERE, "..", "gbif_pollinator_obs", "*.npz"))):
            z = np.load(f, allow_pickle=True)
            g, la, lo = z["gbifID"], z["lat"], z["lon"]
            ed = z["eventDate"] if "eventDate" in z.files else [None] * len(g)
            rows += [(int(g[i]), float(la[i]), float(lo[i]), ed[i]) for i in range(len(g))]
        return rows
    z = np.load(os.path.join(HERE, "obs_coords.npz"))
    g, la, lo = z["gbifID"], z["lat"], z["lon"]
    return [(int(g[i]), float(la[i]), float(lo[i]), None) for i in range(len(g))]

def flush(buf, outdir, shard):
    g = np.array([b[0] for b in buf], np.int64)
    dm = np.stack([b[1] for b in buf]).astype(np.float16)
    nd = np.array([b[2] for b in buf], np.int16)
    np.savez(os.path.join(outdir, f"chunk{shard:04d}.npz"), gbifID=g, daymet=dm, ndays=nd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", default="obs", choices=["obs", "pollinator"])
    ap.add_argument("--out", default=os.path.join(HERE, "..", "gbif_daymet_tokens"))
    ap.add_argument("--sample", type=int, default=0, help="only first N obs (testing)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    obs = [(g, la, lo, parse_date(ed)) for g, la, lo, ed in load_coords(a.coords)]
    if a.sample:
        obs = obs[:a.sample]
    cells = {}
    for o in obs:
        cells.setdefault((round(o[1] / CELL), round(o[2] / CELL)), []).append(o)
    # resume: gbifIDs already in shards, and the next shard index
    shards = sorted(glob.glob(os.path.join(a.out, "chunk*.npz")))
    ckpt = os.path.join(a.out, "daymet_cells.pkl")
    done_cells = pickle.load(open(ckpt, "rb")) if os.path.exists(ckpt) else set()
    shard = len(shards); buf = []
    todo = [c for c in cells if c not in done_cells]
    print(f"{len(obs)} obs, {len(cells)} cells, {len(todo)} to fetch, resume from shard {shard}", flush=True)
    t0 = time.time(); nok = nfail = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_cell, cells[c]): c for c in todo}
        for i, fut in enumerate(as_completed(futs)):
            c = futs[fut]; res = fut.result()
            if res:
                nok += 1
                for gid, (arr, nd) in res.items():
                    buf.append((gid, arr, nd))
            else:                                              # fetch failed: emit empty (ndays=0) rows
                nfail += 1
                for gid, _, _, _ in cells[c]:
                    buf.append((gid, np.full((WIN, 7), np.nan, np.float16), 0))
            done_cells.add(c)
            while len(buf) >= CHUNK:
                flush(buf[:CHUNK], a.out, shard); shard += 1; buf = buf[CHUNK:]
            if (i + 1) % 200 == 0:
                pickle.dump(done_cells, open(ckpt, "wb")); el = time.time() - t0
                print(f"  {i+1}/{len(todo)} cells | ok {nok} fail {nfail} | {el:.0f}s | "
                      f"eta {el/(i+1)*(len(todo)-i-1)/60:.1f}min", flush=True)
    if buf:
        flush(buf, a.out, shard)
    pickle.dump(done_cells, open(ckpt, "wb"))
    print(f"DONE: {nok} cells ok, {nfail} fail, {len(obs)} obs -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
