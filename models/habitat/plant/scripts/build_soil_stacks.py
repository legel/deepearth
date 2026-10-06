"""Unpack the warped SoilGrids layers (``fetch_soilgrids.sh``) into the memmaps ``soil.py`` and
``ConusStack(extra=...)`` read: North America ~230 m int16 (SoilGrids units, -32768 = no data) for training,
CONUS 240 m float32 (physical units, NaN = no data) for rendering."""
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402
from ranges import soil  # noqa: E402

D = config.data_root() / "work/soil"
for v in soil.VARS:
    src = v.replace("soil_", "")
    na, conus = D / f"{src}_na7p5s.tif", D / f"{src}_conus240.tif"
    if conus.exists() and not (D / f"{v}_conus240.f32").exists():
        with rasterio.open(conus) as r:
            out = np.memmap(D / f"{v}_conus240.f32", dtype=np.float32, mode="w+", shape=(r.height, r.width))
            for r0 in range(0, r.height, 2048):
                a = r.read(1, window=((r0, min(r0 + 2048, r.height)), (0, r.width)))
                out[r0:r0 + a.shape[0]] = np.where(a == soil.NODATA, np.nan, a * soil.SCALE[v])
            out.flush()
        print("conus", v, flush=True)
    if na.exists() and not (D / f"{v}_na7p5s.i16").exists():
        with rasterio.open(na) as r:
            out = np.memmap(D / f"{v}_na7p5s.i16", dtype=np.int16, mode="w+", shape=(r.height, r.width))
            for r0 in range(0, r.height, 2048):
                a = r.read(1, window=((r0, min(r0 + 2048, r.height)), (0, r.width)))
                out[r0:r0 + a.shape[0]] = a
            out.flush()
        print("na", v, flush=True)
