"""Build a NAIP scene catalog from the public Planetary Computer STAC API."""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"


def load_points(root):
    files = sorted((root / "gbif_tokens").glob("*.npz"))
    if not files:
        raise SystemExit(f"no train/test token shards under {root / 'gbif_tokens'}")
    lat, lon = [], []
    for file in files:
        z = np.load(file)
        la = z["lat"].astype(np.float64)
        lo = z["lon"].astype(np.float64)
        ok = np.isfinite(la) & np.isfinite(lo)
        lat.append(la[ok])
        lon.append(lo[ok])
    if not lat:
        raise SystemExit("no finite coordinates in train/test token shards")
    return np.concatenate(lat), np.concatenate(lon)


def query_cells(lat, lon, cell_deg):
    ilat = np.floor(lat / cell_deg).astype(np.int64)
    ilon = np.floor(lon / cell_deg).astype(np.int64)
    cells = np.unique(np.stack([ilat, ilon], 1), axis=0)
    for ca, co in cells:
        lat0 = float(ca * cell_deg)
        lon0 = float(co * cell_deg)
        yield [lon0, lat0, lon0 + cell_deg, lat0 + cell_deg]


def save_catalog(out, rows, next_cell):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(rows, f)
    tmp.replace(out)
    with open(out.with_suffix(".progress.json"), "w") as f:
        json.dump({"next_cell": next_cell, "scenes": len(rows)}, f)


def stac_search(bbox, datetime, limit, timeout):
    body = {"collections": ["naip"], "bbox": bbox, "limit": limit}
    if datetime:
        body["datetime"] = datetime
    req = urllib.request.Request(
        STAC_SEARCH,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.load(response)
    return payload.get("features", [])


def feature_row(feature):
    assets = feature.get("assets") or {}
    image = assets.get("image")
    if not image or not image.get("href"):
        return None
    bbox = feature.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    return {
        "entityId": str(feature["id"]),
        "displayId": str(feature["id"]),
        "bbox": [float(x) for x in bbox],
        "url": image["href"],
        "source": "planetary-computer-stac",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=".")
    ap.add_argument("--out")
    ap.add_argument("--datetime", default="")
    ap.add_argument("--cell-deg", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    root = Path(args.cache).expanduser()
    out = Path(args.out).expanduser() if args.out else root / "env_priors" / "naip2024_tiles.json"
    lat, lon = load_points(root)
    rows, seen, start_cell = [], set(), 0
    progress = out.with_suffix(".progress.json")
    if args.resume and out.exists():
        with open(out) as f:
            rows = json.load(f)
        seen = {row["entityId"] for row in rows}
        if progress.exists():
            with open(progress) as f:
                start_cell = int(json.load(f).get("next_cell", 0))
        print(f"resume from cell {start_cell} with {len(rows)} scenes", flush=True)
    cells = list(query_cells(lat, lon, args.cell_deg))
    if args.max_cells:
        cells = cells[:args.max_cells]
    for i, bbox in enumerate(cells, 1):
        if i <= start_cell:
            continue
        try:
            features = stac_search(bbox, args.datetime, args.limit, args.timeout)
        except urllib.error.HTTPError as exc:
            print(f"cell {i}/{len(cells)} http {exc.code}: {bbox}", flush=True)
            features = []
        except Exception as exc:
            print(f"cell {i}/{len(cells)} failed: {exc}", flush=True)
            features = []
        for feature in features:
            row = feature_row(feature)
            if row and row["entityId"] not in seen:
                seen.add(row["entityId"])
                rows.append(row)
        if i == 1 or i % args.checkpoint_every == 0:
            save_catalog(out, rows, i)
            print(f"catalog {i}/{len(cells)} cells | {len(rows)} scenes", flush=True)
        time.sleep(args.sleep)
    save_catalog(out, rows, len(cells))
    print(f"wrote {out} ({len(rows)} scenes)")


if __name__ == "__main__":
    main()
