"""Global vascular-plant sampling effort from the GBIF map API (density vector tiles), aggregated to ~9 km cells.

EPSG:4326 tiles at zoom z form a 2^(z+1) x 2^z grid of square tiles, each 180/2^z degrees wide; features are
square bins with a ``total`` record count, in tile units (extent 4096, y up after decoding).
"""
import concurrent.futures as cf
import sys

import mapbox_vector_tile as mvt
import numpy as np
import pandas as pd
import requests

Z, SQUARE = 3, 2
URL = "https://api.gbif.org/v2/map/occurrence/density/{z}/{x}/{y}.mvt"


def tile(xy):
    x, y = xy
    for t in range(5):
        try:
            r = requests.get(URL.format(z=Z, x=x, y=y), params=dict(srs="EPSG:4326", taxonKey=7707728, bin="square",
                                                                    squareSize=SQUARE), timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException:
            if t == 4:
                raise
    if not r.content:
        return np.empty((0, 3))
    layer = mvt.decode(r.content).get("occurrence")
    if not layer:
        return np.empty((0, 3))
    span, ext = 180.0 / 2**Z, layer["extent"]
    lon0, lat_top = -180.0 + x * span, 90.0 - y * span
    out = []
    for f in layer["features"]:
        ring = np.asarray(f["geometry"]["coordinates"][0], float)
        cx, cy = ring[:, 0].mean(), ring[:, 1].mean()
        out.append((lon0 + cx / ext * span, lat_top - span + cy / ext * span, f["properties"]["total"]))
    return np.asarray(out)


if __name__ == "__main__":
    tiles = [(x, y) for x in range(2 ** (Z + 1)) for y in range(2 ** Z)]
    with cf.ThreadPoolExecutor(8) as ex:
        pts = np.vstack([a for a in ex.map(tile, tiles) if len(a)])
    df = pd.DataFrame({"xi": np.floor(pts[:, 0] * 12).astype(int), "yi": np.floor(pts[:, 1] * 12).astype(int),
                       "n": pts[:, 2].astype(np.int64)}).groupby(["yi", "xi"], as_index=False).n.sum()
    df.to_csv(sys.argv[1], index=False)
    print("bins", len(pts), "5' cells", len(df), "records", int(df.n.sum()))
