"""Sentinel-2 L2A band groups, asset names per source, and digital-number scaling.

Bands are stored as the digital numbers (DN) of the source COGs, unchanged. Surface reflectance is

    reflectance = (DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE

with BOA_QUANTIFICATION_VALUE = 10000 for every baseline, and BOA_ADD_OFFSET = -1000 for processing baseline 04.00
and later (products from 2022-01-25 on, and the whole Collection 1 reprocessing, baselines 05.xx) and 0 before
(ESA, "Sentinel-2: new processing baseline 04.00", 2021). DN 0 is no data in every baseline.
"""

from dataclasses import dataclass

import numpy as np

QUANTIFICATION = 10000.0
NODATA = 0
CLD_UNAVAILABLE = 255
"""Fill for the cloud-probability band where the source does not provide it (Planetary Computer)."""


@dataclass(frozen=True)
class Group:
    name: str
    res: int          # metres
    bands: tuple      # ESA band names
    dtype: str

    @property
    def size(self) -> int:
        return CHIP_M // self.res


CHIP_M = 300
GROUPS = (
    Group("r10", 10, ("B02", "B03", "B04", "B08"), "uint16"),
    Group("r20", 20, ("B05", "B06", "B07", "B8A", "B11", "B12"), "uint16"),
    Group("r60", 60, ("B01", "B09"), "uint16"),
    Group("scl", 20, ("SCL",), "uint8"),
    Group("cld", 20, ("CLD",), "uint8"),
)
GROUP = {g.name: g for g in GROUPS}

# Asset keys in each STAC collection.
ASSETS = {
    "earth-search-c1": {"B01": "coastal", "B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1",
                        "B06": "rededge2", "B07": "rededge3", "B08": "nir", "B8A": "nir08", "B09": "nir09",
                        "B11": "swir16", "B12": "swir22", "SCL": "scl", "CLD": "cloud"},
    "planetary-computer": {b: b for b in ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09",
                                          "B11", "B12", "SCL")},
}

# Sen2Cor scene classification (SCL) classes.
SCL_CLASSES = {0: "no_data", 1: "saturated_or_defective", 2: "dark_area", 3: "cloud_shadow", 4: "vegetation",
               5: "not_vegetated", 6: "water", 7: "unclassified", 8: "cloud_medium_probability",
               9: "cloud_high_probability", 10: "thin_cirrus", 11: "snow_ice"}
SCL_CLOUD = (8, 9, 10)
SCL_SHADOW = (3,)
SCL_INVALID = (0, 1)


def baseline_tuple(baseline: str) -> tuple:
    """'04.00' -> (4, 0). Unknown or empty baselines sort first."""
    try:
        major, minor = str(baseline).split(".")
        return int(major), int(minor)
    except (ValueError, AttributeError):
        return (-1, -1)


def boa_add_offset(baseline: str) -> int:
    """BOA_ADD_OFFSET in DN for a processing baseline: -1000 from 04.00 on, else 0."""
    return -1000 if baseline_tuple(baseline) >= (4, 0) else 0


def offset_from_raster_bands(raster_bands: dict) -> int | None:
    """BOA_ADD_OFFSET in DN from a STAC raster:bands entry (offset in reflectance units, scale 1e-4), if present."""
    if not raster_bands or "offset" not in raster_bands or "scale" not in raster_bands:
        return None
    return int(round(raster_bands["offset"] / raster_bands["scale"]))


def reflectance(dn: np.ndarray, offset: int) -> np.ndarray:
    """Surface reflectance (float32) from stored DN; no-data pixels (DN 0) become NaN."""
    dn = np.asarray(dn)
    out = (dn.astype(np.float32) + np.float32(offset)) / np.float32(QUANTIFICATION)
    return np.where(dn == NODATA, np.float32(np.nan), out)


def scl_fractions(scl: np.ndarray) -> dict:
    """Cloud, shadow, snow, invalid and valid-clear fractions of one SCL chip (20 m pixels)."""
    scl = np.asarray(scl)
    n = scl.size
    cloud = np.isin(scl, SCL_CLOUD).sum() / n
    shadow = np.isin(scl, SCL_SHADOW).sum() / n
    snow = (scl == 11).sum() / n
    invalid = np.isin(scl, SCL_INVALID).sum() / n
    return {"cloud_frac": float(cloud), "shadow_frac": float(shadow), "snow_frac": float(snow),
            "invalid_frac": float(invalid), "clear_frac": float(1.0 - cloud - shadow - invalid)}
