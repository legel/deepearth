"""Build the NAIP 2024 M2M scene catalog used by build_naip_m2m2024.py.

The output is a compact list of scenes:
  [{"entityId": ..., "displayId": ..., "bbox": [lon0, lat0, lon1, lat1]}, ...]
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import requests


HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("DEEPCAL_CACHE", HERE.parent)).expanduser()
TOKENS = CACHE / "gbif_tokens"
OUT = Path(os.environ.get("NAIP_TILES_JSON", str(CACHE / "env_priors" / "naip2024_tiles.json")))
BASE = "https://m2m.cr.usgs.gov/api/api/json/stable"
USERNAME = os.environ.get("M2M_USER", "ecological")
TOKEN_PATH = Path(os.environ.get("USGS_M2M_TOKEN", str(Path.home() / ".usgs_m2m_token")))
DATASET = os.environ.get("NAIP_M2M_DATASET", "naip")
START = os.environ.get("NAIP_START", "2024-01-01")
END = os.environ.get("NAIP_END", "2024-12-31")
PAGE = int(os.environ.get("NAIP_CATALOG_PAGE", "50000"))


def m2m(key, endpoint, body=None):
    headers = {"X-Auth-Token": key} if key else {}
    response = requests.post(f"{BASE}/{endpoint}", headers=headers, json=(body or {}), timeout=300)
    response.raise_for_status()
    payload = response.json()
    if payload.get("errorCode"):
        raise RuntimeError(f"M2M {endpoint}: {payload['errorCode']} {payload.get('errorMessage')}")
    return payload["data"]


def login():
    return m2m(None, "login-token", {"username": USERNAME, "token": TOKEN_PATH.read_text().strip()})


def bounds_from_tokens():
    files = sorted(TOKENS.glob("*.npz"))
    if not files:
        raise SystemExit(f"no train/test token shards under {TOKENS}")
    lo0 = la0 = float("inf")
    lo1 = la1 = float("-inf")
    for file in files:
        z = np.load(file)
        lat = z["lat"].astype(float)
        lon = z["lon"].astype(float)
        ok = np.isfinite(lat) & np.isfinite(lon)
        if not ok.any():
            continue
        lo0 = min(lo0, float(lon[ok].min()))
        lo1 = max(lo1, float(lon[ok].max()))
        la0 = min(la0, float(lat[ok].min()))
        la1 = max(la1, float(lat[ok].max()))
    if not np.isfinite([lo0, la0, lo1, la1]).all():
        raise SystemExit("could not derive finite bounds from train/test shards")
    pad = 0.05
    return lo0 - pad, la0 - pad, lo1 + pad, la1 + pad


def scene_bbox(scene):
    spatial = scene.get("spatialBounds") or {}
    coords = spatial.get("coordinates")
    if coords:
        flat = []
        stack = list(coords)
        while stack:
            value = stack.pop()
            if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
                flat.append(value)
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
        if flat:
            lon = [p[0] for p in flat]
            lat = [p[1] for p in flat]
            return [min(lon), min(lat), max(lon), max(lat)]
    if all(k in scene for k in ("lowerLeftCoordinate", "upperRightCoordinate")):
        ll, ur = scene["lowerLeftCoordinate"], scene["upperRightCoordinate"]
        return [ll["longitude"], ll["latitude"], ur["longitude"], ur["latitude"]]
    if all(k in scene for k in ("minX", "minY", "maxX", "maxY")):
        return [scene["minX"], scene["minY"], scene["maxX"], scene["maxY"]]
    return None


def search_page(key, bbox, start):
    lon0, lat0, lon1, lat1 = bbox
    body = {
        "datasetName": DATASET,
        "maxResults": PAGE,
        "startingNumber": start,
        "sceneFilter": {
            "spatialFilter": {
                "filterType": "mbr",
                "lowerLeft": {"latitude": lat0, "longitude": lon0},
                "upperRight": {"latitude": lat1, "longitude": lon1},
            },
            "acquisitionFilter": {"start": START, "end": END},
        },
    }
    return m2m(key, "scene-search", body)


def main():
    if not TOKEN_PATH.exists():
        raise SystemExit(f"{TOKEN_PATH} not found; set USGS_M2M_TOKEN to a token file")
    key = login()
    bbox = bounds_from_tokens()
    scenes, seen, start, total = [], set(), 1, None
    while total is None or len(scenes) < total:
        data = search_page(key, bbox, start)
        total = int(data.get("totalHits") or data.get("total") or 0)
        results = data.get("results") or []
        if not results:
            break
        for scene in results:
            entity = str(scene.get("entityId") or scene.get("entity_id") or "")
            display = str(scene.get("displayId") or scene.get("display_id") or entity)
            bbox_i = scene_bbox(scene)
            if not entity or not bbox_i or entity in seen:
                continue
            seen.add(entity)
            scenes.append({"entityId": entity, "displayId": display, "bbox": [float(x) for x in bbox_i]})
        print(f"catalog {len(scenes)}/{total or '?'} scenes", flush=True)
        start += len(results)
        time.sleep(0.2)
        if len(results) < PAGE:
            break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(scenes, f)
    tmp.replace(OUT)
    print(f"wrote {OUT} ({len(scenes)} scenes)")


if __name__ == "__main__":
    main()
