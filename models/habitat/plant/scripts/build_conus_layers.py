"""Shared CONUS 240 m layers used by rendering and range-card decoding (built once, then read-only):
ecoregion id per cell (RESOLVE Ecoregions 2017 ECO_ID, uint16, 0 = none) with Natural Earth 10m lakes set
to 0 (Daru masked lake pixels). Cell membership = cell centre inside the polygon (rasterio default)."""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ranges import config  # noqa: E402

R = config.data_root()
ref = rasterio.open(R / "work/conus240/wc2.1_30s_elev_conus240.tif")
eco = gpd.read_file(R / "raw/geo/ecoregions2017/Ecoregions2017.shp")[["ECO_ID", "geometry"]].to_crs(5070)
eco = eco.cx[ref.bounds.left:ref.bounds.right, ref.bounds.bottom:ref.bounds.top]
lakes = gpd.read_file(R / "raw/geo/ne_lakes/ne_10m_lakes.shp").to_crs(5070)
lakes = lakes.cx[ref.bounds.left:ref.bounds.right, ref.bounds.bottom:ref.bounds.top]
ids = rasterize(((g, int(i)) for g, i in zip(eco.geometry, eco.ECO_ID)), out_shape=ref.shape,
                transform=ref.transform, fill=0, dtype="uint16")
water = rasterize(((g, 1) for g in lakes.geometry), out_shape=ref.shape, transform=ref.transform, fill=0, dtype="uint8")
ids[water == 1] = 0
prof = ref.profile.copy()
prof.update(dtype="uint16", nodata=0, compress="zstd", predictor=2, tiled=True, blockxsize=512, blockysize=512)
with rasterio.open(R / "work/conus240/ecoregion_id_conus240.tif", "w", **prof) as d:
    d.write(ids, 1)
print("ecoregions in CONUS grid:", len(np.unique(ids)) - 1, "| lake cells masked:", int(water.sum()))
