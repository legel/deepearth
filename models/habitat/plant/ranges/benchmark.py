"""Compare a 240 m CONUS range map with Daru's (2024) published ~18 km rasters (Dryad SDM_set1).

Our map is aggregated onto Daru's grid (fraction of 240 m calibration cells present per ~18 km cell), and
agreement is scored only inside CONUS and inside both calibration areas:

* binary agreement (our fraction >= 0.5 vs Daru's 0/1): Jaccard, Cohen's kappa, sensitivity/specificity
* Spearman correlation of suitability (our mean per ~18 km cell vs Daru's median suitability)
* heterogeneity: share of Daru-present cells in which our 240 m map is mixed (0 < fraction < 1), i.e. where
  the finer map resolves structure that the ~18 km map cannot
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.stats import spearmanr


def _to_daru_grid(src_path: Path, daru: rasterio.DatasetReader, values: dict) -> np.ndarray:
    """Average a 240 m uint8 map (after mapping codes via ``values``) onto Daru's grid."""
    with rasterio.open(src_path) as s:
        a = s.read(1)
        lut = np.full(256, np.nan, np.float32)
        for code, v in values.items():
            lut[code] = v
        x = lut[a]
        out = np.full((daru.height, daru.width), np.nan, np.float32)
        reproject(x, out, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan,
                  dst_transform=daru.transform, dst_crs=daru.crs, dst_nodata=np.nan, resampling=Resampling.average)
    return out


def compare(ours_dir: Path, label: str, daru_binary: Path, daru_raw: Path) -> dict:
    with rasterio.open(daru_binary) as db, rasterio.open(daru_raw) as dr:
        d_bin, d_raw = db.read(1), dr.read(1)
        frac = _to_daru_grid(ours_dir / f"{label}.binary_vote.tif", db, {1: 0.0, 2: 1.0})
        suit = _to_daru_grid(ours_dir / f"{label}.suitability.tif", db,
                             {k: (k - 1) / 254.0 for k in range(1, 256)})
    both = np.isfinite(frac) & np.isfinite(d_bin)
    o, d = frac[both] >= 0.5, d_bin[both] == 1
    tp, fp, fn, tn = (o & d).sum(), (o & ~d).sum(), (~o & d).sum(), (~o & ~d).sum()
    n = tp + fp + fn + tn
    po, pe = (tp + tn) / n, ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
    ok = both & np.isfinite(suit) & np.isfinite(d_raw)
    rho = spearmanr(suit[ok], d_raw[ok]).statistic if ok.sum() > 2 else np.nan
    mixed = (frac[both] > 0) & (frac[both] < 1)
    return {"cells_compared": int(n), "jaccard": float(tp / max(tp + fp + fn, 1)),
            "kappa": float((po - pe) / (1 - pe)) if pe < 1 else np.nan,
            "sensitivity_vs_daru": float(tp / max(tp + fn, 1)), "specificity_vs_daru": float(tn / max(tn + fp, 1)),
            "spearman_suitability": float(rho), "share_daru_present_cells_mixed_at_240m": float(mixed[d].mean())
            if d.any() else np.nan}


def wcvp_consistency(binary_path: Path, present_values: tuple, native_geom, crs_geom: str = "EPSG:4326") -> dict:
    """Independent sanity check against the WCVP checklist: the share of the predicted range that lies inside
    the species' native TDWG level-3 areas (precision-like) and the share of those areas that is predicted
    (coverage-like). Cell areas are taken from the raster CRS (equal-area for EPSG:5070; cos-latitude
    weighted for geographic grids)."""
    import geopandas as gpd
    from rasterio.features import rasterize

    with rasterio.open(binary_path) as r:
        a, tr, crs = r.read(1), r.transform, r.crs
    geom = gpd.GeoSeries([native_geom], crs=crs_geom).to_crs(crs).iloc[0]
    nat = rasterize([(geom, 1)], out_shape=a.shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    pres = np.isin(a, present_values)
    if crs.is_geographic:
        lat = tr.f + tr.e * (np.arange(a.shape[0]) + 0.5)
        w = np.cos(np.radians(lat))[:, None] * np.ones(a.shape)
    else:
        w = np.ones(a.shape)
    inside = float((w * (pres & nat)).sum() / max((w * pres).sum(), 1e-12))
    coverage = float((w * (pres & nat)).sum() / max((w * nat).sum(), 1e-12))
    return {"share_predicted_inside_native_l3": inside, "share_native_l3_predicted": coverage}


def calibration_iou(ecoregion_ids, ecoregions, daru_raw: Path) -> dict:
    """Overlap of our calibration area (whole occupied ecoregions) with the footprint of Daru's published
    suitability raster (its non-NaN cells), on Daru's grid with cos-latitude cell weights (provenance D8)."""
    from rasterio.features import rasterize
    from shapely.ops import unary_union

    with rasterio.open(daru_raw) as r:
        foot = np.isfinite(r.read(1))
        tr, shape = r.transform, r.shape
    geom = unary_union(ecoregions[ecoregions.ECO_ID.isin(list(ecoregion_ids))].geometry.values)
    ours = rasterize([(geom, 1)], out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
    lat = tr.f + tr.e * (np.arange(shape[0]) + 0.5)
    w = np.cos(np.radians(lat))[:, None] * np.ones(shape)
    inter = (w * (ours & foot)).sum()
    return {"iou": float(inter / max((w * (ours | foot)).sum(), 1e-12)),
            "share_of_daru_covered": float(inter / max((w * foot).sum(), 1e-12)),
            "share_of_ours_inside_daru": float(inter / max((w * ours).sum(), 1e-12))}
