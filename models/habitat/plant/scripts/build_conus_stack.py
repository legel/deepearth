"""Unpack the 20 CONUS 240 m predictor GeoTIFFs into one uncompressed band-sequential float32 memmap
(20, 13053, 20149) so rendering reads raw rows (page-cache resident) instead of decompressing ~8 GB of ZSTD
GeoTIFF per species. Values are bit-identical to the GeoTIFFs."""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402
from ranges.predictors import VARIABLES  # noqa: E402

D = config.data_root() / "work/conus240"
with rasterio.open(D / f"{VARIABLES[0]}_conus240.tif") as r:
    H, W, transform, crs = r.height, r.width, r.transform, r.crs.to_string()
mm = np.memmap(D / "conus240_stack.f32", dtype=np.float32, mode="w+", shape=(len(VARIABLES), H, W))
for k, v in enumerate(VARIABLES):
    with rasterio.open(D / f"{v}_conus240.tif") as r:
        for r0 in range(0, H, 2048):
            h = min(2048, H - r0)
            mm[k, r0:r0 + h] = r.read(1, window=((r0, r0 + h), (0, W)))
    print("packed", v, flush=True)
mm.flush()
(D / "conus240_stack.json").write_text(json.dumps({"variables": VARIABLES, "shape": [len(VARIABLES), H, W],
                                                   "transform": list(transform)[:6], "crs": crs}))
