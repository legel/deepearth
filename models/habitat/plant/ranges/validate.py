"""Independent presence/absence validation at field plots (BLM AIM: complete species lists per plot visit).

A plot visit that lists the species is a presence; a visit that does not is an absence. Plots are collapsed
to unique locations (presence if any visit recorded the species). Our 240 m maps and Daru's ~18 km maps are
read at the same plots, so both are scored on identical ground truth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from sklearn.metrics import roc_auc_score


CONUS_LAT, CONUS_LON = (24.0, 50.0), (-125.0, -66.0)       # validation domain: where the CONUS product exists
DOMAINS = {"conus": (CONUS_LAT, CONUS_LON), "alaska": ((51.0, 72.0), (-180.0, -129.0)), "hawaii": ((18.5, 29.0), (-179.0, -154.5))}


def in_domain(lat, lon, region: str = "conus"):
    """True for points inside a region's validation box (the area its 240 m grid covers)."""
    (a, b), (c, d) = DOMAINS[region]
    return lat.between(a, b) & lon.between(c, d)


def plot_truth(aim_csv: Path, species: str) -> pd.DataFrame:
    a = pd.read_csv(aim_csv, usecols=["PrimaryKey", "ScientificName", "lon", "lat"])
    a["loc"] = a.lon.round(5).astype(str) + "," + a.lat.round(5).astype(str)
    pres = a.ScientificName.fillna("").str.startswith(species)
    locs = a.groupby("loc").agg(lon=("lon", "first"), lat=("lat", "first"))
    locs["present"] = a[pres].groupby("loc").size().reindex(locs.index).fillna(0).gt(0)
    locs = locs[locs.lat.between(*CONUS_LAT) & locs.lon.between(*CONUS_LON)]
    return locs.reset_index(drop=True)


_VEGBANK = {}


def vegbank_truth(vegbank_dir: Path, wcvp_dir: Path, species: str) -> pd.DataFrame | None:
    """VegBank presence/absence for one species (``PlotTruth`` built once per process)."""
    key = (str(vegbank_dir), str(wcvp_dir))
    if key not in _VEGBANK:
        _VEGBANK[key] = PlotTruth(vegbank_dir=vegbank_dir, wcvp_dir=wcvp_dir)
    return _VEGBANK[key].vegbank(species)


def _sample(path: Path, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Raster values at lon/lat (NaN outside the grid): one full read, then array indexing."""
    with rasterio.open(path) as r:
        a = r.read(1)
        x, y = (lon, lat) if r.crs.is_geographic else Transformer.from_crs(4326, r.crs, always_xy=True).transform(lon, lat)
        row, col = rasterio.transform.rowcol(r.transform, x, y)
    row, col = np.asarray(row, dtype=np.int64), np.asarray(col, dtype=np.int64)
    ok = (row >= 0) & (row < a.shape[0]) & (col >= 0) & (col < a.shape[1])
    out = np.full(len(row), np.nan)
    out[ok] = a[row[ok], col[ok]]
    return out


def _scores(present: np.ndarray, pred_bin: np.ndarray, score: np.ndarray | None) -> dict:
    tp, fn = (pred_bin & present).sum(), (~pred_bin & present).sum()
    tn, fp = (~pred_bin & ~present).sum(), (pred_bin & ~present).sum()
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    out = {"n_presence": int(present.sum()), "n_absence": int((~present).sum()), "sensitivity": float(sens),
           "specificity": float(spec), "TSS": float(sens + spec - 1)}
    if score is not None and present.any() and (~present).any():
        # plots outside the map grid / calibration area carry no score: treat them as unsuitable (0)
        out["AUC"] = float(roc_auc_score(present, np.nan_to_num(np.asarray(score, dtype=float), nan=0.0)))
    return out


def validate(truth: pd.DataFrame, maps_dir: Path, label: str, daru_binary: Path, daru_raw: Path) -> dict:
    lon, lat, present = truth.lon.values, truth.lat.values, truth.present.values
    vote = _sample(maps_dir / f"{label}.binary_vote.tif", lon, lat)
    p5 = _sample(maps_dir / f"{label}.binary_p5.tif", lon, lat)
    suit = _sample(maps_dir / f"{label}.suitability.tif", lon, lat)
    d_bin = _sample(daru_binary, lon, lat)
    d_raw = _sample(daru_raw, lon, lat)
    in_ours = vote > 0                                   # inside our calibration area (and CONUS grid)
    in_daru = np.isfinite(d_bin)
    both = in_ours & in_daru
    ours_s = np.where(suit > 0, (suit - 1) / 254, 0.0)
    return {
        "plots": int(len(truth)), "plots_in_both_calibration_areas": int(both.sum()),
        "ours_240m_all_plots": _scores(present, vote == 2, ours_s),
        "daru_10min_all_plots": _scores(present, np.nan_to_num(d_bin) == 1, np.nan_to_num(d_raw)),
        "ours_240m_p5_all_plots": _scores(present, p5 == 2, None),
        "ours_240m_shared_area": _scores(present[both], vote[both] == 2, ours_s[both]),
        "ours_240m_p5_shared_area": _scores(present[both], p5[both] == 2, None),
        "daru_10min_shared_area": _scores(present[both], d_bin[both] == 1, d_raw[both]),
    }


def fia_truth(fia_dir: Path, species: str) -> pd.DataFrame:
    """FIA plots as presence/absence for one tree species: presence = a live tree of the species on any
    inventory of the plot location; absence = every other sampled forested plot. Public FIA coordinates are
    perturbed (up to ~1.6 km), so this tests range placement rather than 240 m detail. Restricted to CONUS."""
    fia_dir = Path(fia_dir)
    plots = pd.read_csv(fia_dir / "fia_plots.csv", usecols=["CN", "LAT", "LON"])
    plots["loc"] = plots.LON.round(4).astype(str) + "," + plots.LAT.round(4).astype(str)
    pres = pd.read_csv(fia_dir / "fia_live_tree_presence.csv", usecols=["PLT_CN", "scientificName"])
    hit = set(pres.loc[pres.scientificName == species, "PLT_CN"])
    plots["present"] = plots.CN.isin(hit)
    locs = plots.groupby("loc").agg(lon=("LON", "first"), lat=("LAT", "first"), present=("present", "any"))
    locs = locs[locs.lat.between(*CONUS_LAT) & locs.lon.between(*CONUS_LON)]
    return locs.reset_index(drop=True)


class PlotTruth:
    """BLM AIM, FIA and VegBank presence/absence for many species, parsed once (per-species lookups are then cheap)."""

    def __init__(self, aim_csv: Path | None = None, fia_dir: Path | None = None, vegbank_dir: Path | None = None,
                 wcvp_dir: Path | None = None, vegbank_min_taxa: int = 10):
        """``vegbank_dir`` (with ``wcvp_dir``): VegBank plots (``scripts/fetch_vegbank.py``). Plant names are
        resolved through WCVP to their accepted species; only plots with exact public coordinates
        (confidentiality 0) inside CONUS are kept, and only plots listing >= ``vegbank_min_taxa`` taxa count as
        complete enough for an absence. Repeat visits of a plot are merged (present if seen on any visit)."""
        self.aim_locs = self.aim_sp = self.fia_locs = self.fia_sp = self.vb_locs = self.vb_sp = None
        if vegbank_dir is not None and (Path(vegbank_dir) / "taxa.parquet").exists():
            from . import names
            pl = pd.read_parquet(Path(vegbank_dir) / "plots.parquet")
            pl["lat"] = pd.to_numeric(pl.latitude, errors="coerce")
            pl["lon"] = pd.to_numeric(pl.longitude, errors="coerce")
            pl = pl[(pd.to_numeric(pl.confidentiality_status, errors="coerce") == 0) & pl.lat.between(*CONUS_LAT) & pl.lon.between(*CONUS_LON)]
            tx = pd.read_parquet(Path(vegbank_dir) / "taxa.parquet")
            tx["name"] = tx.int_curr_plant_sci_name_no_auth.where(~tx.int_curr_plant_sci_name_no_auth.isin(["None", "nan", ""]), tx.author_plant_name)
            tx = tx[tx.ob_code.isin(set(pl.ob_code))]
            n_taxa = tx.groupby("ob_code").name.nunique()
            pl = pl[pl.ob_code.isin(set(n_taxa[n_taxa >= vegbank_min_taxa].index))]
            tx = tx[tx.ob_code.isin(set(pl.ob_code))]
            r = names.resolve(pd.Series(sorted(tx.name.dropna().unique())), names.load_wcvp_names(wcvp_dir)).dropna(subset=["wcvp_accepted_name"])
            species_of = {q: " ".join(a.split()[:2]) for q, a in zip(r["query"], r["wcvp_accepted_name"])}
            tx["species"] = tx.name.map(species_of)
            tx["loc"] = tx.ob_code.map(dict(zip(pl.ob_code, pl.pl_code)))
            self.vb_locs = pl.groupby("pl_code").agg(lon=("lon", "first"), lat=("lat", "first"))
            self.vb_sp = tx.dropna(subset=["species", "loc"]).groupby("species")["loc"].apply(set)
        if aim_csv is not None and Path(aim_csv).exists():
            a = pd.read_csv(aim_csv, usecols=["ScientificName", "lon", "lat"])
            a["loc"] = a.lon.round(5).astype(str) + "," + a.lat.round(5).astype(str)
            self.aim_locs = a.groupby("loc").agg(lon=("lon", "first"), lat=("lat", "first"))
            a["binomial"] = a.ScientificName.fillna("").str.split().str[:2].str.join(" ")
            self.aim_sp = a.groupby("binomial")["loc"].apply(set)
        if fia_dir is not None and (Path(fia_dir) / "fia_live_tree_presence.csv").exists():
            p = pd.read_csv(Path(fia_dir) / "fia_plots.csv", usecols=["CN", "LAT", "LON"])
            p["loc"] = p.LON.round(4).astype(str) + "," + p.LAT.round(4).astype(str)
            self.fia_locs = p.groupby("loc").agg(lon=("LON", "first"), lat=("LAT", "first"))
            loc_of = dict(zip(p.CN, p["loc"]))
            t = pd.read_csv(Path(fia_dir) / "fia_live_tree_presence.csv", usecols=["PLT_CN", "scientificName"])
            t["loc"] = t.PLT_CN.map(loc_of)
            self.fia_sp = t.dropna(subset=["loc"]).groupby("scientificName")["loc"].apply(set)

    def _truth(self, locs, sp, species, region: str = "conus"):
        if locs is None:
            return None
        hit = sp.get(species, set())
        # score only where the product exists: AIM and FIA include Alaska plots (366 and 4,934), which a CONUS map
        # scores 0 while a global map (Daru's) covers them (Carex aquatilis: 213 of 226 AIM presences in Alaska)
        out = locs[in_domain(locs.lat, locs.lon, region)].copy()
        out["present"] = out.index.isin(hit)
        return out.reset_index(drop=True)

    def aim(self, species):
        return self._truth(self.aim_locs, self.aim_sp, species)

    def fia(self, species):
        return self._truth(self.fia_locs, self.fia_sp, species)

    def vegbank(self, species):
        return self._truth(self.vb_locs, self.vb_sp, species)
