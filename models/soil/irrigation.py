"""Irrigated ground, detected from its greenness through the record's dry season (2026-10-06: two city parks'
watered lawns read Dry under SOIL MOISTURE, because the balance was rain alone).

Turf that stays green through months with no rain is watered. The record names the site's dry season (the drought
window, exposure.water_modes.drought_window) and what fell in it. A herbaceous cell is judged only where that rain and
the water its root zone can hold could not keep turf growing through the window (`dry_cells`); there greenness is
evidence of watering, and elsewhere (a humid summer) it is not, so the cell stays rain-fed.

Greenness is Sentinel-2 L2A surface reflectance, 10 m, on every clear view of the window. A 10 m pixel over a park
mixes lawn with paths, paving and crowns, so each view is unmixed on the site's own 0.5 m classes (`unmix`): the
pixel's red and near infrared are the class shares times each class's reflectance, the classes' reflectances are
solved over the site's pixels (non-negative least squares), and the herbaceous part of each pixel is what is left.
A lawn, the 8-connected turf a watering zone covers, is judged whole: watered when the median over its readable cells
of their dry-season NDVI (each the 25th percentile of its views) is GREEN_NDVI or above, where rain-fed straw in the
same drought reads 0.28 (Tonzi). Its interior speaks for the edges a 10 m pixel half paving cannot read. The site's
crowns are no reference: Tonzi's sparse blue oaks read 0.40 to 0.47 in that drought.

Sentinel-2 L2A comes from Microsoft Planetary Computer and is ours after first use: each view's window over
the site (red, near infrared, scene classification) is kept under MIRROR and read from there by every later run. A
lawn with fewer than LAWN_MIN_READ readable cells (MIN_VIEWS clear views, pixels HERB_MIN turf) is left rain-fed
and counted as such; nothing is guessed.

    python3 -m exposure.irrigation probe --spec <site.json> --kit <kit.npz> --modes <modes.json> --work <dir> [--out <gs>]
"""
import calendar
import hashlib
import io
import json
import os
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
COLLECTION = "sentinel-2-l2a"
MIRROR = os.environ.get("S2_MIRROR", "s2_mirror")
SOURCE = "Sentinel-2 L2A (ESA Copernicus) through Microsoft Planetary Computer, mirrored at " + MIRROR
PAD_M = 30.0
"""Read beyond the grid's edge: three 10 m pixels, so every cell centre has its pixel and its neighbours."""
CLEAR_SCL = (4, 5, 6)
"""Scene classification classes read as a clear view of the ground: vegetation, not vegetated, water. Cloud (8, 9),
cirrus (10), shadow (3), dark area (2), unclassified (7), saturated (1) and no data (0) are not."""
OFFSET_BASELINE = "04.00"
"""From processing baseline 04.00 (25 January 2022) L2A digital numbers carry BOA_ADD_OFFSET = -1000 (ESA, Sentinel-2
L2A product format, PSD 14.9): reflectance = (DN - 1000) / 10000. Read without it, NDVI is pulled towards zero."""
KINDS = ("herbaceous", "crowns", "other", "water")
"""The unmixing's classes, from the six: pervious ground; tall vegetation; everything built, sealed or unclassified;
open water."""
FULL = 0.9
"""A pixel is read only when the site's grid covers this share of it (its class shares are then the pixel's own)."""
HERB_MIN = 0.25
"""Turf share a pixel needs for its turf part to be read. Each pixel's error is divided by this share, so one pixel at
a quarter turf can be far off; a lawn is judged on the median of all its readable cells (`lawn_values`), which a few
such pixels cannot move: at the lawn, Tonzi's straw read 0 m2 watered at every share from 0.25 to 0.5."""
MIN_VIEWS = 3
"""Clear views of the window a cell needs to be judged."""
LAWN_MIN_READ = 400
"""Readable cells a lawn needs to be judged as a whole: one 10 m pixel of turf. A smaller lawn has no evidence and
stays rain-fed."""
GREEN_NDVI = 0.4
"""Watered: the lawn's dry-season NDVI (unmixed turf, 25th percentile of its views, median over the lawn) at least
this. Measured in each record's drought window: rain-fed straw at Tonzi reads 0.28 as a lawn and 0.38 at its greenest
tenth of cells, US-SRM's 0.13; one city park's central lawn reads 0.50 and another's watered lawns 0.41 to 0.90. Dense
green leaf (0.6 to 0.8, Weier and Herring 2000) was too high a floor for summer turf at 10 m: it marked 10 % of Grand
Park's turf and missed its central lawn."""
LATTICE = 20
"""Cells between exact geodetic transforms; positions between are interpolated (sub-millimetre over 10 m)."""
READ_WORKERS = int(os.environ.get("S2_READ_WORKERS", "16"))
"""Views read at once: each is a gsutil read of our mirror or three ranged reads at the source, network-bound."""
MASK_FILE = "irrigated.npy"
"""The mask beside the job's work, for the water's per-GPU parts (exposure.product.water_part)."""


def _gs(*args: str, data: Optional[bytes] = None, timeout: float = 600) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(["gsutil", "-q", *args], input=data, capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def _key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()[:16]


def _cat(uri: str, timeout: float = 120) -> Optional[bytes]:
    """A mirror object's bytes, None where there is none (a local folder as the mirror: a test, a lab)."""
    if not uri.startswith("gs://"):
        p = Path(uri)
        return p.read_bytes() if p.is_file() else None
    r = _gs("cat", uri, timeout=timeout)
    return r.stdout if r is not None and r.returncode == 0 and r.stdout else None


def _put(uri: str, data: bytes, timeout: float = 300) -> None:
    if not uri.startswith("gs://"):
        p = Path(uri)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(p)
        return
    _gs("cp", "-", uri, data=data, timeout=timeout)


def _stac_search(bbox: Sequence[float], start: str, end: str) -> List[Dict]:
    """The STAC API's items (the Planetary Computer call)."""
    from sims import fetch as net
    feats, body = [], {"collections": [COLLECTION], "bbox": list(map(float, bbox)), "datetime": f"{start}/{end}",
                       "limit": 200}
    url = STAC
    while url:
        got = net.post(url, "Planetary Computer STAC (Sentinel-2 L2A)", timeout=120, json=body).json()
        feats += got.get("features", [])
        nxt = [ln for ln in got.get("links", []) if ln.get("rel") == "next"]
        url, body = (nxt[0]["href"], nxt[0].get("body") or body) if nxt else (None, None)
    return feats


def _bbox_key(bbox: Sequence[float]) -> str:
    return _key([round(float(v), 5) for v in bbox])


def _ls(prefix: str) -> List[str]:
    if not prefix.startswith("gs://"):
        p = Path(prefix)
        return sorted(str(q) for q in p.glob("*.json")) if p.is_dir() else []
    r = _gs("ls", f"{prefix}/*.json", timeout=120)
    return (r.stdout.decode() if isinstance(r.stdout, bytes) else r.stdout).split() if r and r.returncode == 0 else []


def _from_superset(bbox: Sequence[float], start: str, end: str) -> Optional[List[Dict]]:
    """[start, end]'s items from a wider search over the same box kept by `prefetch`, None where none answers exactly.
    The API reads a date-only start as its 00:00 UTC, inclusive (every kept search holds start-day items after
    midnight); how it reads a date-only end is not on record, so a wider search answers only where it reaches a whole
    day past `end` and holds no item on the end day itself: then the API's answer is the same under either reading."""
    from datetime import date, datetime, timedelta, timezone
    s0 = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    e0 = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    e1 = e0 + timedelta(days=1)
    for uri in _ls(f"{MIRROR}/search_super/{_bbox_key(bbox)}"):
        raw = _cat(uri)
        doc = json.loads(raw) if raw else None
        if not doc or date.fromisoformat(doc["start"]) > s0.date() or date.fromisoformat(doc["end"]) < e1.date():
            continue
        when = [datetime.fromisoformat(f["properties"]["datetime"].replace("Z", "+00:00")) for f in doc["features"]]
        if any(e0 < t < e1 for t in when):
            return None                                  # an end-day view: only the API says whether it is in
        return [f for f, t in zip(doc["features"], when) if s0 <= t <= e0]
    return None


def search(bbox: Sequence[float], start: str, end: str, wide: bool = False) -> List[Dict]:
    """Sentinel-2 L2A items over `bbox` (lon/lat) between two ISO dates: our copy, else a wider search over the same box
    kept by `prefetch` (`_from_superset`), else the STAC API (then kept). `wide`: a prefetch's own search, kept where
    later windows inside it are answered from."""
    uri = f"{MIRROR}/search/{_key([round(float(v), 5) for v in bbox], start, end)}.json"
    raw = _cat(uri)
    if raw:
        return json.loads(raw)["features"]
    if not wide:
        sub = _from_superset(bbox, start, end)
        if sub is not None:
            _put(uri, json.dumps({"at": time.time(), "bbox": list(bbox), "start": start, "end": end, "features": sub,
                                  "from": "a wider search kept by irrigation.prefetch"}).encode(), timeout=120)
            return sub
    feats = _stac_search(bbox, start, end)
    keep = [{"id": f["id"], "properties": {k: f["properties"].get(k) for k in
                                           ("datetime", "proj:epsg", "s2:processing_baseline", "eo:cloud_cover",
                                            "s2:mgrs_tile")},
             "assets": {b: {"href": f["assets"][b]["href"]} for b in ("B04", "B08", "SCL")}} for f in feats]
    doc = json.dumps({"at": time.time(), "bbox": list(bbox), "start": start, "end": end, "features": keep}).encode()
    _put(uri, doc, timeout=120)
    if wide:
        _put(f"{MIRROR}/search_super/{_bbox_key(bbox)}/{start}_{end}.json", doc, timeout=120)
    return keep


_TOKEN: Dict = {}


def _signed(href: str) -> str:
    from sims import fetch as net
    if not _TOKEN or _TOKEN["until"] < time.time() + 300:
        t = net.get(TOKEN, "Planetary Computer SAS (Sentinel-2 L2A)", timeout=60).json()
        exp = t.get("msft:expiry", "")
        until = calendar.timegm(time.strptime(exp[:19], "%Y-%m-%dT%H:%M:%S")) if exp else time.time() + 1800
        _TOKEN.update(token=t["token"], until=until)
    return f"{href}?{_TOKEN['token']}"


def scene_window(item: Dict, box: Sequence[float]) -> Optional[Dict]:
    """The item's red, near infrared and scene classification over `box` (x0, y0, x1, y1 in the item's CRS) plus PAD_M,
    as surface reflectance (offset removed) and SCL: our mirror's copy, else read from the source and kept."""
    epsg = int(item["properties"]["proj:epsg"])
    x0, y0 = np.floor((box[0] - PAD_M) / 20.0) * 20.0, np.floor((box[1] - PAD_M) / 20.0) * 20.0
    x1, y1 = np.ceil((box[2] + PAD_M) / 20.0) * 20.0, np.ceil((box[3] + PAD_M) / 20.0) * 20.0
    uri = window_uri(item, box)
    raw = _cat(uri)
    if raw:
        z = np.load(io.BytesIO(raw))
        return {k: z[k] for k in z.files} if "B04" in z.files else None
    out = _from_wide(item, x0, y0, x1, y1)              # an order's read-ahead covers it: cut, not read
    if out is None:
        out = _read_source(item, x0, y0, x1, y1)
    buf = io.BytesIO()
    if out["B04"].size == 0 or not out["SCL"].any():
        np.savez_compressed(buf, empty=np.array(1))           # kept too: the next run need not ask again
        _put(uri, buf.getvalue())
        return None
    off = 1000.0 if str(item["properties"].get("s2:processing_baseline") or "0") >= OFFSET_BASELINE else 0.0
    for band in ("B04", "B08"):
        out[band] = ((out[band].astype(np.float32) - off) / 10000.0).astype(np.float32)
    out["epsg"] = np.array(epsg)
    np.savez_compressed(buf, **out)
    _put(uri, buf.getvalue())
    return out


def _wide_prefix(item: Dict) -> str:
    return f"{MIRROR}/wide/{item['id']}"


def _read_source_wide(item: Dict, x0: float, y0: float, x1: float, y1: float) -> Dict:
    """The three bands over a wide box from the source, each with the dataset's own transform and the window's
    offsets, so any window inside it can be cut exactly as the source would have read it (`_from_wide`)."""
    import rasterio
    from rasterio.windows import from_bounds
    out = {}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT=60, GDAL_HTTP_MAX_RETRY=4,
                      GDAL_HTTP_RETRY_DELAY=2):
        for band in ("B04", "B08", "SCL"):
            with rasterio.open(_signed(item["assets"][band]["href"])) as ds:
                w = from_bounds(x0, y0, x1, y1, ds.transform).round_offsets().round_lengths()
                out[band] = ds.read(1, window=w, boundless=True, fill_value=0)
                out[band + "_T"] = np.array(tuple(ds.transform)[:6], np.float64)
                out[band + "_off"] = np.array([int(w.row_off), int(w.col_off)], np.int64)
    return out


def read_wide(item: Dict, box: Sequence[float]) -> bool:
    """One view's wide window over `box` (x0, y0, x1, y1 in its CRS, already padded), kept under the mirror's wide/
    with the box in its name; True when it is there (kept now or before)."""
    x0, y0, x1, y1 = (float(v) for v in box)
    uri = f"{_wide_prefix(item)}/{int(item['properties']['proj:epsg'])}_{x0:.0f}_{y0:.0f}_{x1:.0f}_{y1:.0f}.npz"
    if _cat(uri) is not None:
        return True
    out = _read_source_wide(item, x0, y0, x1, y1)
    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    _put(uri, buf.getvalue())
    return True


def _from_wide(item: Dict, x0: float, y0: float, x1: float, y1: float) -> Optional[Dict]:
    """The window `_read_source` would read, cut from a kept wide read that holds it: the same window arithmetic on the
    dataset's own transform (from_bounds, rounded offsets and lengths, rasterio.windows.transform), so the same pixels,
    the boundless zeros past the image included, and the same transforms. None where no wide read holds it."""
    from affine import Affine
    from rasterio import windows as W
    epsg = int(item["properties"]["proj:epsg"])
    for uri in _ls_npz(_wide_prefix(item)):
        try:
            e, a, b, c, d = Path(uri).stem.split("_")
            if int(e) != epsg or not (float(a) <= x0 and float(b) <= y0 and float(c) >= x1 and float(d) >= y1):
                continue
        except ValueError:
            continue
        raw = _cat(uri)
        if raw is None:
            continue
        z = np.load(io.BytesIO(raw))
        out = {}
        for band in ("B04", "B08", "SCL"):
            T = Affine(*z[band + "_T"])
            w = W.from_bounds(x0, y0, x1, y1, T).round_offsets().round_lengths()
            r0, c0 = int(w.row_off) - int(z[band + "_off"][0]), int(w.col_off) - int(z[band + "_off"][1])
            h, wd = int(w.height), int(w.width)
            arr = z[band]
            if r0 < 0 or c0 < 0 or r0 + h > arr.shape[0] or c0 + wd > arr.shape[1]:
                break
            t = W.transform(w, T)
            out[band] = arr[r0:r0 + h, c0:c0 + wd].copy()
            out[band + "_t"] = np.array([t.c, t.a, t.f, t.e], np.float64)
        else:
            return out
    return None


def _ls_npz(prefix: str) -> List[str]:
    if not prefix.startswith("gs://"):
        p = Path(prefix)
        return sorted(str(q) for q in p.glob("*.npz")) if p.is_dir() else []
    r = _gs("ls", f"{prefix}/*.npz", timeout=120)
    return (r.stdout.decode() if isinstance(r.stdout, bytes) else r.stdout).split() if r and r.returncode == 0 else []


ORDER_MARGIN_M = 300.0
"""How far past the order's cores its read-ahead reaches: past every block's wind ring (128 m) and water buffer (100 m),
so the order's decision frame lies inside it."""


def order_prefetch(spec, cores: Dict, start: str, end: str, pad_days: Optional[int] = None,
                   margin_m: float = ORDER_MARGIN_M, claim: Optional[Callable[[str], bool]] = None,
                   log: Callable = print) -> Dict:
    """A strided order's views read once, wide, over its whole ground, while its blocks solve their winds: the order's
    one decision (exposure.order_irrigation.decide) then cuts its windows from them (`_from_wide`) and reads nothing at
    the source. The first block to claim the order's read does it; the others skip. Never raises."""
    from datetime import date, timedelta
    t0 = time.time()
    pad_days = PREFETCH_PAD_D if pad_days is None else pad_days
    try:
        pts = np.array([p for parts in cores.values() for part in parts for ring in part for p in ring], float)
        bx0, by0 = pts[:, 0].min() - margin_m, pts[:, 1].min() - margin_m
        bx1, by1 = pts[:, 0].max() + margin_m, pts[:, 1].max() + margin_m
        s = (date.fromisoformat(start) - timedelta(days=pad_days)).isoformat()
        e = (date.fromisoformat(end) + timedelta(days=pad_days)).isoformat()
        key = _key(sorted(cores), [round(float(v), 2) for v in (bx0, by0, bx1, by1)], s, e)
        if claim is not None and not claim(key):
            rec = {"skipped": "another block of the order reads it", "seconds": round(time.time() - t0, 1)}
            log(f"irrigation: the order's views {json.dumps(rec)}")
            return rec
        dx = 10.0
        ny, nx = int(np.ceil((by1 - by0) / dx)) + 1, int(np.ceil((bx1 - bx0) / dx)) + 1
        lon, lat, bbox = read_plan(spec, (ny, nx), bx0, dx, by1, -dx)
        items = search(bbox, s, e, wide=True)
        xy = in_crs(items, lon, lat)

        def wide_box(it):
            x, y = xy[int(it["properties"]["proj:epsg"])]
            pad = PAD_M + 20.0
            return (np.floor((x.min() - pad) / 20.0) * 20.0, np.floor((y.min() - pad) / 20.0) * 20.0,
                    np.ceil((x.max() + pad) / 20.0) * 20.0, np.ceil((y.max() + pad) / 20.0) * 20.0)

        def read(it):
            try:
                return read_wide(it, wide_box(it))
            except Exception:  # noqa: BLE001  one unread view is read at the source by the decision, as before
                return False
        with ThreadPoolExecutor(max(1, min(READ_WORKERS, len(items) or 1))) as pool:
            got = list(pool.map(read, items))
        rec = {"window": [s, e], "views": len(items), "kept": int(sum(got)), "unread": int(len(got) - sum(got)),
               "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001  a head start, never a stop
        rec = {"error": f"{type(e).__name__}: {str(e)[:200]}", "seconds": round(time.time() - t0, 1)}
    log(f"irrigation: the order's views read ahead {json.dumps(rec)}")
    return rec


def claim_first(uri: str) -> bool:
    """True for the first writer of `uri` (GCS generation 0; a local file created exclusively)."""
    if not uri.startswith("gs://"):
        p = Path(uri)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(p, "x") as fh:
                fh.write(json.dumps({"at": time.time()}))
            return True
        except FileExistsError:
            return False
    r = _gs("-h", "x-goog-if-generation-match:0", "cp", "-", uri, data=json.dumps({"at": time.time()}).encode(),
            timeout=60)
    return r is not None and r.returncode == 0


def window_uri(item: Dict, box: Sequence[float]) -> str:
    """Where a view's window over `box` is kept: the item and the box on the 20 m grid, padded."""
    epsg = int(item["properties"]["proj:epsg"])
    x0, y0 = np.floor((box[0] - PAD_M) / 20.0) * 20.0, np.floor((box[1] - PAD_M) / 20.0) * 20.0
    x1, y1 = np.ceil((box[2] + PAD_M) / 20.0) * 20.0, np.ceil((box[3] + PAD_M) / 20.0) * 20.0
    return f"{MIRROR}/windows/{item['id']}/{_key(epsg, x0, x1, y0, y1)}.npz"


def _read_source(item: Dict, x0: float, y0: float, x1: float, y1: float) -> Dict:
    """The three bands over the box from the Planetary Computer (the read the mirror saves every later run)."""
    import rasterio
    from rasterio.windows import from_bounds
    out = {}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT=60, GDAL_HTTP_MAX_RETRY=4,
                      GDAL_HTTP_RETRY_DELAY=2):
        for band in ("B04", "B08", "SCL"):
            with rasterio.open(_signed(item["assets"][band]["href"])) as ds:
                w = from_bounds(x0, y0, x1, y1, ds.transform).round_offsets().round_lengths()
                a = ds.read(1, window=w, boundless=True, fill_value=0)
                t = ds.window_transform(w)
            out[band] = a
            out[band + "_t"] = np.array([t.c, t.a, t.f, t.e], np.float64)
    return out


def lonlat(spec, shape, x0: float, dx: float, y0: float, dy: float):
    """Every cell centre's longitude and latitude (scene ENU through sims.bundle.scene_to_naip, exact on a lattice of
    LATTICE cells, bilinear between)."""
    from scipy.interpolate import RegularGridInterpolator
    from sims.bundle import scene_to_naip
    ny, nx = shape
    rr = np.unique(np.r_[np.arange(0, ny, LATTICE), ny - 1])
    cc = np.unique(np.r_[np.arange(0, nx, LATTICE), nx - 1])
    R, C = np.meshgrid(rr, cc, indexing="ij")
    lo, la = scene_to_naip(spec, x0 + (C.ravel() + 0.5) * dx, y0 + (R.ravel() + 0.5) * dy, 4326)
    r, c = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    q = np.c_[r.ravel(), c.ravel()]
    out = []
    for v in (lo, la):
        out.append(RegularGridInterpolator((rr, cc), np.asarray(v, np.float64).reshape(R.shape))(q))
    return out[0], out[1]


def unmix(w: Dict, x: np.ndarray, y: np.ndarray, kind: np.ndarray, cell_m2: float) -> Optional[Dict]:
    """One view, unmixed on the site's classes. `x`, `y`: every cell centre in the view's CRS; `kind`: each cell's
    index in KINDS. Returns each cell's pixel herbaceous NDVI (NaN where its pixel is cloudy, not covered or too little
    herbaceous to read), the crowns' NDVI and each class's solved reflectance; None when the view saw no clear pixel."""
    from scipy.optimize import nnls
    c0, a, f0, e = w["B04_t"]
    H, W = w["B04"].shape
    col = np.floor((x - c0) / a).astype(np.int64)
    row = np.floor((y - f0) / e).astype(np.int64)
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    pid = np.where(ok, row * W + col, -1)
    counts = np.bincount(pid[ok] * len(KINDS) + kind[ok], minlength=H * W * len(KINDS)).reshape(H * W, len(KINDS))
    tot = counts.sum(1)
    full = tot >= FULL * abs(a * e) / cell_m2
    s = w["SCL_t"]
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    sc = np.floor((c0 + (jj.ravel() + 0.5) * a - s[0]) / s[1]).astype(np.int64)
    sr = np.floor((f0 + (ii.ravel() + 0.5) * e - s[2]) / s[3]).astype(np.int64)
    sin = (sr >= 0) & (sr < w["SCL"].shape[0]) & (sc >= 0) & (sc < w["SCL"].shape[1])
    clear = np.zeros(H * W, bool)
    clear[sin] = np.isin(w["SCL"][sr[sin], sc[sin]], CLEAR_SCL)
    red, nir = w["B04"].ravel().astype(np.float64), w["B08"].ravel().astype(np.float64)
    use = full & clear & (red + nir > 0)
    if use.sum() < len(KINDS):
        return None
    frac = counts[use] / tot[use, None]
    present = frac.sum(0) >= 2.0                    # a class on less than two pixels' worth of ground is not solved
    A = frac[:, present]
    ref = {}
    for band, v in (("red", red[use]), ("nir", nir[use])):
        sol, _ = nnls(A, v)
        full_sol = np.full(len(KINDS), np.nan)
        full_sol[present] = sol
        ref[band] = full_sol
    nd = lambda r_, n_: (n_ - r_) / np.where(n_ + r_ > 0, n_ + r_, np.nan)  # noqa: E731
    k_h, k_c = KINDS.index("herbaceous"), KINDS.index("crowns")
    out = {"ndvi_class": {k: (None if not present[i] else round(float(nd(ref["red"][i], ref["nir"][i])), 3))
                          for i, k in enumerate(KINDS)},
           "pixels": int(use.sum())}
    out["crowns_ndvi"] = out["ndvi_class"]["crowns"]
    fr = counts / np.maximum(tot, 1)[:, None]
    herb = np.full(H * W, np.nan)
    if present[k_h]:
        readable = use & (fr[:, k_h] > 0.0)                 # every pixel with turf; HERB_MIN applied below
        rest = [i for i in range(len(KINDS)) if i != k_h and present[i]]
        part = {}
        for band, v in (("red", red), ("nir", nir)):
            other = (fr[readable][:, rest] * ref[band][rest]).sum(1) if rest else 0.0
            part[band] = (v[readable] - other) / fr[readable, k_h]
        herb[readable] = nd(part["red"], part["nir"])
        out["herb_ndvi_any"] = np.where(ok, herb[np.maximum(pid, 0)], np.nan)     # at any share (diagnostics)
        herb[fr[:, k_h] < HERB_MIN] = np.nan
    cell = np.full(len(x), np.nan)
    cell[ok] = herb[pid[ok]]
    out["herb_ndvi"] = cell
    frac, raw = np.full(H * W, np.nan), np.full(H * W, np.nan)       # the pixel's herbaceous share and raw NDVI, read
    frac[use], raw[use] = fr[use, k_h], nd(red[use], nir[use])
    out["herb_frac"], out["raw_ndvi"] = (np.where(ok, a_[np.maximum(pid, 0)], np.nan) for a_ in (frac, raw))
    return out


def lawn_values(dry: np.ndarray, shape, idx: np.ndarray, g25: np.ndarray, judged: np.ndarray):
    """Each dry turf cell's lawn (8-connected turf at 0.5 m, one watering unit) and that lawn's greenness: the median of
    its readable cells' dry-season NDVI, with how many there were. A 10 m pixel half paving cannot be read, but the
    lawn it edges is watered or not as a whole, so its interior speaks for its edges (one city park: 53 % of turf cells
    lay in pixels under half turf). Returns (lawn value per idx cell, readable cells of its lawn per idx cell)."""
    from scipy import ndimage
    lab, _ = ndimage.label(dry.reshape(shape), structure=np.ones((3, 3), bool))
    li = lab.ravel()[idx]
    ok = judged & np.isfinite(g25)
    nl = int(li.max()) + 1
    cnt = np.bincount(li[ok], minlength=nl)
    med = np.full(nl, np.nan)
    if ok.any():
        o = np.lexsort((g25[ok], li[ok]))                # each lawn's readable values in order: its median by position
        ls, vs = li[ok][o], g25[ok][o]
        start = np.r_[0, np.cumsum(cnt)[:-1]]
        has = cnt > 0
        mid = start[has] + (cnt[has] - 1) / 2.0
        med[has] = (vs[np.floor(mid).astype(int)] + vs[np.ceil(mid).astype(int)]) / 2.0
    return med[li], cnt[li]


def dry_cells(taw_mm: np.ndarray, window: Dict) -> np.ndarray:
    """Cells where the window's rain and a full root zone meet less than (1 - p) of turf's demand, Kcb x ET0 over the
    window (FAO-56's turf Kcb, the pervious ground's in exposure.classes.PARAMS; p = water.P_STRESS, the depletion past
    which FAO-56 stresses it). There a rain-fed lawn spends the window stressed and browns, so greenness shows watering;
    where rain meets more (Fisher's 2025 summer: 122 mm of rain and 44 mm of root zone against 301 mm), it need not."""
    from exposure import classes
    from exposure.water import P_STRESS
    kcb = classes.PARAMS["pervious_ground"][3]
    supply = float(window["rain_mm"]) + np.nan_to_num(taw_mm, nan=np.inf)
    return supply < (1.0 - P_STRESS) * kcb * float(window["pet_mm"])


def taw(fields: Dict[str, np.ndarray]) -> np.ndarray:
    """Total available water of each cell's root zone, mm, as the balance holds it (exposure.water: layer 1 on its own
    soil to ZE, the root zone's below)."""
    from exposure.water import ZE
    fc1, wp1 = fields.get("fc1", fields["fc"]), fields.get("wp1", fields["wp"])
    return 1000.0 * ((fc1 - wp1) * ZE + (fields["fc"] - fields["wp"]) * np.maximum(fields["z_r"] - ZE, 0.0))


def read_plan(spec, shape, x0: float, dx: float, y0: float, dy: float):
    """(lon, lat, bbox): the grid's cell centres and the box searched; what `detect` and `prefetch` both read over."""
    lon, lat = lonlat(spec, shape, x0, dx, y0, dy)
    bbox = [float(lon.min()) - 0.002, float(lat.min()) - 0.002, float(lon.max()) + 0.002, float(lat.max()) + 0.002]
    return lon, lat, bbox


def in_crs(items: Sequence[Dict], lon: np.ndarray, lat: np.ndarray) -> Dict[int, tuple]:
    """The cell centres in each item's CRS, once a CRS."""
    from pyproj import Transformer
    xy: Dict[int, tuple] = {}
    for it in items:
        epsg = int(it["properties"]["proj:epsg"])
        if epsg not in xy:
            x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
            xy[epsg] = (np.asarray(x), np.asarray(y))
    return xy


PREFETCH_PAD_D = 21
"""Days either side of the guessed drought window that `prefetch` reads: the guess is made before the wind basis, whose
reference wind moves the window's potential evaporation a little."""


def prefetch(spec, shape, x0: float, dx: float, y0: float, dy: float, start: str, end: str,
             pad_days: int = PREFETCH_PAD_D, log: Callable = print) -> Dict:
    """Every view `detect` could read for a window near [start, end] (local dates), read now over the exact boxes
    `detect` reads and kept in the mirror, so `detect` itself reads the mirror: the same views, the same arrays, none
    of the Planetary Computer's seconds on the job's path Never raises."""
    from datetime import date, timedelta
    t0 = time.time()
    try:
        s = (date.fromisoformat(start) - timedelta(days=pad_days)).isoformat()
        e = (date.fromisoformat(end) + timedelta(days=pad_days)).isoformat()
        lon, lat, bbox = read_plan(spec, shape, x0, dx, y0, dy)
        items = search(bbox, s, e, wide=True)
        xy = in_crs(items, lon, lat)

        def read(it):
            try:
                scene_window(it, _box(xy, it))       # the mirror's copy where held, else the source's, then kept
                return True
            except Exception:  # noqa: BLE001  one unread view is read again by detect, as before
                return False
        with ThreadPoolExecutor(max(1, min(READ_WORKERS, len(items) or 1))) as pool:
            got = list(pool.map(read, items))
        rec = {"window": [s, e], "views": len(items), "kept": int(sum(got)), "unread": int(len(got) - sum(got)),
               "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001  the prefetch is a head start, never a stop
        rec = {"error": f"{type(e).__name__}: {str(e)[:200]}", "seconds": round(time.time() - t0, 1)}
    log(f"irrigation: views read ahead {json.dumps(rec)}")
    return rec


def _box(xy: Dict[int, tuple], it: Dict) -> tuple:
    x, y = xy[int(it["properties"]["proj:epsg"])]
    return (x.min(), y.min(), x.max(), y.max())


def detect(cls6: np.ndarray, taw_mm: np.ndarray, keep: np.ndarray, spec, shape, x0: float, dx: float, y0: float,
           dy: float, window: Dict, log: Callable = print, diag: bool = False) -> Dict:
    """The watered cells of a site's grid: {"mask": bool [cells] or None, "receipt": what was read and found}."""
    from exposure import classes
    t0 = time.time()
    herb = cls6 == classes.SIX.index("pervious_ground")
    kind = np.full(len(cls6), KINDS.index("other"), np.int64)
    kind[herb] = KINDS.index("herbaceous")
    kind[cls6 == classes.SIX.index("tall_vegetation")] = KINDS.index("crowns")
    kind[cls6 == classes.OPEN_WATER] = KINDS.index("water")
    dry = herb & dry_cells(taw_mm, window)
    rec = {"source": SOURCE, "window": {k: window.get(k) for k in ("start_local", "end_local", "rain_mm", "pet_mm")},
           "rule": f"open herbaceous ground (pervious ground, no crown) where the window's rain and a full root zone "
                   f"meet less than half of turf's demand (rain + TAW < (1 - p) Kcb_turf ET0), judged a lawn at a time "
                   f"(8-connected turf): watered when the median over its readable cells (pixels at least {HERB_MIN} "
                   f"turf, at least {MIN_VIEWS} clear views, at least {LAWN_MIN_READ} cells) of their turf NDVI, unmixed "
                   f"on the site's classes and the 25th percentile of the window's views, is {GREEN_NDVI} or above",
           "herbaceous_cells": int((herb & keep).sum()), "dry_cells": int((dry & keep).sum())}
    if not dry.any():
        rec.update(state="not dry", line="the dry season's rain and the root zone could keep turf green here, so "
                                         "greenness does not show which ground is watered; rain alone")
        log(f"irrigation: {rec['line']}")
        return {"mask": None, "receipt": rec}
    lon, lat, bbox = read_plan(spec, shape, x0, dx, y0, dy)
    items = sorted(search(bbox, window["start_local"], window["end_local"]), key=lambda f: f["properties"]["datetime"])
    xy = in_crs(items, lon, lat)

    def window_of(it):
        x, y = xy[int(it["properties"]["proj:epsg"])]
        try:
            return scene_window(it, (x.min(), y.min(), x.max(), y.max())), None
        except Exception as e:  # noqa: BLE001  one unreadable view is one fewer view, said in the receipt
            return None, e

    # THE READS ARE THE STAGE: 148 views one after another were 544 s of Denver's 627 s exposure, a GPU machine idle on
    # gsutil and range reads (2026-10-06). Read in parallel, kept in the items' order, so the record is the serial one's.
    t_read = time.time()
    with ThreadPoolExecutor(max(1, min(READ_WORKERS, len(items)))) as pool:
        windows = list(pool.map(window_of, items))
    rec.update(read_s=round(time.time() - t_read, 1), read_workers=READ_WORKERS)
    idx = np.flatnonzero(dry)
    G, views, skipped, D = [], [], [], {}
    for it, (w, err) in zip(items, windows):
        x, y = xy[int(it["properties"]["proj:epsg"])]
        try:
            if err is not None:
                raise err
            u = unmix(w, x, y, kind, abs(dx * dy)) if w is not None else None
        except Exception as e:  # noqa: BLE001  one unreadable view is one fewer view, said in the receipt
            skipped.append({"id": it["id"], "why": f"{type(e).__name__}: {str(e)[:160]}"})
            continue
        if u is None:
            skipped.append({"id": it["id"], "why": "no clear view of the site"})
            continue
        G.append(u["herb_ndvi"][idx].astype(np.float32))
        if diag:                                  # what each cell's pixel read, for the probe's breakdown
            for key in ("herb_ndvi_any", "herb_frac", "raw_ndvi"):
                D.setdefault(key, []).append(u.get(key, np.full(len(x), np.nan))[idx].astype(np.float32))
        views.append({"date": it["properties"]["datetime"][:10], "pixels": u["pixels"], **u["ndvi_class"]})
    n = np.zeros(len(idx), np.int64)
    g25 = np.full(len(idx), np.nan)
    if G:
        Gm = np.stack(G)
        n = np.isfinite(Gm).sum(0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            g25 = np.nanpercentile(Gm, 25, axis=0)
    judged = n >= MIN_VIEWS
    lawn, lawn_n = lawn_values(dry, shape, idx, g25, judged)
    wet = (lawn_n >= LAWN_MIN_READ) & (lawn >= GREEN_NDVI)
    mask = np.zeros(len(cls6), bool)
    mask[idx[wet]] = True
    ks = keep[idx]
    cell_m2 = abs(dx * dy)
    rec.update(views_found=len(items), views=views, views_skipped=skipped,
               judged_cells=int((judged & ks).sum()), unread_cells=int((~judged & ks).sum()),
               irrigated_cells=int((wet & ks).sum()), irrigated_m2=round(float((wet & ks).sum()) * cell_m2, 1),
               irrigated_share_of_herbaceous=round(float((wet & ks).sum()) / max(rec["herbaceous_cells"], 1), 4),
               green_p10_p50_p90=([round(float(v), 3) for v in np.nanpercentile(g25[judged & ks], [10, 50, 90])]
                                  if (judged & ks).any() else None),
               seconds=round(time.time() - t0, 1))
    if not views:
        rec.update(state="no imagery", line="no clear Sentinel-2 view covers the dry season here, so the model cannot "
                                            "tell which lawns are watered; rain alone")
    elif not (wet & ks).any():
        rec.update(state="none", line=f"no ground stayed green through the dry season ({len(views)} Sentinel-2 views), "
                                      f"so none is watered; rain alone")
    else:
        rec.update(state="detected", line=f"{rec['irrigated_m2']:,.0f} m2 of lawn stayed green through the dry season "
                                          f"with no rain to keep it so ({len(views)} Sentinel-2 views), so it is "
                                          f"watered: refilled to field capacity each day")
    log(f"irrigation: {rec['state']}: {rec['line']} ({rec['seconds']:.0f} s)")
    out = {"mask": mask if mask.any() else None, "receipt": rec}
    if diag:
        out.update(lawn=lawn, lawn_n=lawn_n, g25=g25, n=n, idx=idx, G=np.stack(G) if G else None, **{k: np.stack(v) for k, v in D.items()})
    return out


def detect_safe(*a, log: Callable = print, **kw) -> Dict:
    """`detect`, a failure named in the receipt and the log, never raised: the balance then runs on rain alone and the
    page says the detection did not run."""
    try:
        return detect(*a, log=log, **kw)
    except Exception as e:  # noqa: BLE001
        why = f"{type(e).__name__}: {str(e)[:300]}"
        log(f"irrigation: DETECTION FAILED, rain alone: {why}")
        return {"mask": None, "receipt": {"state": "failed", "error": why,
                                          "line": "the watered-ground detection did not run here; rain alone"}}


def breakdown(r: Dict, ks: np.ndarray) -> Dict:
    """For the probe: the site's dry herbaceous cells (in the disc) by why they are or are not watered under the rule,
    and the watered share across herbaceous shares and NDVI floors, unmixed and raw."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        A, Fr, Rw = r["herb_ndvi_any"][:, ks], r["herb_frac"][:, ks], r["raw_ndvi"][:, ks]
        frac = np.nanmedian(Fr, 0)
        na, nr = np.isfinite(A).sum(0), np.isfinite(Rw).sum(0)
        u25, u50 = np.nanpercentile(A, 25, 0), np.nanpercentile(A, 50, 0)
        r25 = np.nanpercentile(Rw, 25, 0)
    cells = int(ks.sum())
    mixed = ~(frac >= HERB_MIN)
    few = ~mixed & (na < MIN_VIEWS)
    read = ~mixed & ~few
    sh = lambda m: round(float(m.sum()) / max(cells, 1), 4)  # noqa: E731
    out = {"cells": cells, "watered": sh(read & (u25 >= GREEN_NDVI)), "mixed_pixel": sh(mixed),
           "too_few_views": sh(few), "floor": sh(read & (u50 < GREEN_NDVI)),
           "season_min": sh(read & (u50 >= GREEN_NDVI) & (u25 < GREEN_NDVI)),
           "herb_frac_p10_p50_p90": [round(float(v), 2) for v in np.nanpercentile(frac, [10, 50, 90])],
           "unmixed_p25_p10_p50_p90": [round(float(v), 3) for v in np.nanpercentile(u25, [10, 50, 90])],
           "raw_p25_p10_p50_p90": [round(float(v), 3) for v in np.nanpercentile(r25, [10, 50, 90])]}
    floors = (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
    out["sweep_unmixed"] = {f"min{hm}": {f"{t}": sh((frac >= hm) & (na >= MIN_VIEWS) & (u25 >= t)) for t in floors}
                            for hm in (0.1, 0.25, 0.35, 0.5)}
    out["sweep_raw"] = {f"{t}": sh((nr >= MIN_VIEWS) & (r25 >= t)) for t in floors}
    lw, ln = r["lawn"][ks], r["lawn_n"][ks]
    out["sweep_lawn"] = {f"{t}": sh((ln >= LAWN_MIN_READ) & (lw >= t)) for t in (0.3, 0.35) + floors}
    out["lawn_unjudged"] = sh(ln < LAWN_MIN_READ)
    v, c = np.unique(np.round(lw[ln >= LAWN_MIN_READ], 4), return_counts=True)
    out["lawns_by_area"] = [[float(a), int(b)] for a, b in sorted(zip(v, c), key=lambda t: -t[1])[:12]]
    out["unmixed_p25_hist_by_frac"] = {f"{lo}-{hi}": np.histogram(u25[(frac >= lo) & (frac < hi) & (na >= MIN_VIEWS)],
                                                                    bins=np.linspace(-0.2, 1.2, 15))[0].tolist()
                                       for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))}
    return out


def _probe(a) -> Dict:
    """`detect` on a published site, from its kit (classes by canopy and soil), its modes (grid and drought window)
    and its site spec: the numbers the rule is judged by, before it reaches an order."""
    from exposure import classes
    from sims.site import load
    work = os.path.abspath(a.work)
    os.makedirs(work, exist_ok=True)
    for name, uri in (("site.json", a.spec), ("kit.npz", a.kit), ("modes.json", a.modes)):
        subprocess.run(["gsutil", "-q", "cp", uri, os.path.join(work, name)], check=True)
    spec = load(os.path.join(work, "site.json"))
    modes = json.load(open(os.path.join(work, "modes.json")))
    kit = np.load(os.path.join(work, "kit.npz"))
    g = modes["grid"]
    t = g["transform"]
    soil = kit["no_soil"].astype(bool) == 0
    can = kit["canopy"].astype(bool)
    cls6 = np.full(len(soil), classes.SIX.index("sealed_hardscape"), np.int64)
    cls6[soil & ~kit["roof"].astype(bool) & (kit["perv"] > 0)] = classes.SIX.index("pervious_ground")
    cls6[soil & can] = classes.SIX.index("tall_vegetation")
    if "open_water" in kit.files:
        cls6[kit["open_water"] > 0] = classes.OPEN_WATER
    from exposure.water_modes import in_site
    keep = in_site(tuple(g["shape"]), t, spec.radius_m)
    T = taw({k: kit[k].astype(np.float64) for k in ("fc", "wp", "z_r", "fc1", "wp1") if k in kit.files})
    r = detect(cls6, T, keep, spec, tuple(g["shape"]), t["x0"], t["dx"], t["y0"], t["dy"], modes["drought"]["period"],
               diag=True)
    rec = r["receipt"]
    rec["site"] = modes["site"]
    if r.get("G") is not None:
        rec["breakdown"] = breakdown(r, keep[r["idx"]])
        import warnings
        dry = np.zeros(len(cls6), bool)
        dry[r["idx"]] = True
        ks = keep[r["idx"]]
        for hm in (0.25, 0.35, 0.5):               # the lawn judged on pixels down to this share of turf
            A = np.where(r["herb_frac"] >= hm, r["herb_ndvi_any"], np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                g = np.nanpercentile(A, 25, 0)
            n = np.isfinite(A).sum(0)
            lw, ln = lawn_values(dry, tuple(g_["shape"] for g_ in [modes["grid"]])[0], r["idx"], g, n >= MIN_VIEWS)
            rec["breakdown"][f"lawn_min{hm}"] = {
                f"{t}": round(float(((ln >= LAWN_MIN_READ) & (lw >= t))[ks].sum()) / max(int(ks.sum()), 1), 4)
                for t in (0.3, 0.35, 0.4, 0.45, 0.5)} | {"unjudged": round(float((ln < LAWN_MIN_READ)[ks].mean()), 4)}
    if "g25" in r:
        sel = keep[r["idx"]] & (r["n"] >= MIN_VIEWS)
        rec["green_hist"] = np.histogram(r["g25"][sel], bins=np.linspace(-0.5, 1.5, 21))[0].tolist()
        rec["taw_mm_p10_p50_p90"] = [round(float(v), 1) for v in np.percentile(T[cls6 == 2], [10, 50, 90])]
    if r["mask"] is not None:
        np.save(os.path.join(work, MASK_FILE), r["mask"])
    return rec


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    for k in ("spec", "kit", "modes", "work"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--out", default="")
    a = ap.parse_args(argv)
    res = _probe(a)
    txt = json.dumps(res, indent=1, default=float)
    print("IRRIGATION_PROBE " + json.dumps({k: res.get(k) for k in ("site", "state", "line", "irrigated_cells",
                                                                     "judged_cells", "dry_cells", "herbaceous_cells")}))
    if a.out:
        _gs("cp", "-", a.out, data=txt.encode(), timeout=120)
        mp = os.path.join(os.path.abspath(a.work), MASK_FILE)
        if os.path.exists(mp):
            _gs("cp", mp, a.out.rsplit(".", 1)[0] + "_mask.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
