"""Training and evaluation data of the joint range model.

A data directory (written by the data preparation of the national run) holds:

* ``species.csv``: one row per species in model order, with ``species`` (the tree's tip label) and
  ``calibration_ecoregions`` (space-separated RESOLVE Ecoregions 2017 ids: the species' calibration area);
* ``train_points_cache.npy``: float32 [n_points, n_variables], the predictors at every training point;
* ``joint_data.npz`` (members may instead sit beside it as ``<name>.npy`` memory maps):
  ``names`` (predictor names, the column order), ``sid`` (species index of each point), ``pres`` (1 = presence
  record, 0 = background point), and the VegBank evaluation plots: ``plot_X``, ``plot_lat``, ``plot_lon``,
  ``plot_eco`` (ecoregion id of each plot), ``eval_sid`` (species scored), ``eval_y`` (presence per species and
  plot) and optionally ``maxent_occ`` (the per-species MaxEnt maps at the plots, 0 outside their calibration area,
  for a paired comparison);
* ``plots_<source>.npz``: further independent plot sets (BLM AIM, FIA, ...) with the same plot members;
* ``tree/natives.dated.nwk``: the dated phylogeny whose tips are the species labels;

and, for the model with the landscape field, place pathway and learned calibration (docs/joint_model.md):

* ``shore_records.npz``: presences the per-species preparation dropped because their 30" WorldClim pixel has no
  climate (beaches, dunes, salt marshes), restored with the climate of the nearest pixel within 5 km
  (shoreline.py): ``lat``, ``lon``, ``sid``, ``X`` (the predictors), ``rc`` (their field positions);
* ``plot_fill_<source>.npz``: the plot predictors with the same shoreline fill (climate_fill.py), ``X`` and ``n``;
* ``field/``: the field pyramid (field.py) and the fractional (row, column) of every training row
  (``rc_train.npy``) and plot (``rc_plots_<source>.npy``) on its 240 m grid (geodesy.py);
* ``community/snap.npy``: for every training row, the row of the nearest record of another species (target-group
  background, prepare.snap_to_records).

Each species' training points are what a per-species MaxEnt would be fitted to: its presence records (thinned) and
background points drawn inside its calibration ecoregions with probability proportional to the sampling effort of
all records there, so that the model contrasts where the species was recorded with where anyone recorded anything.

Target-group background. The effort-weighted background points are cell centres of a 10 km effort density, while
records sit where people record: along roads and trails, at survey stops, at localities many species share. Any
pathway that can tell places apart at a finer scale (the landscape field, place) would learn "a record is here" and
separate every species' presences from its background, which is false at an independent plot. Moving each
background point to the nearest record of another species (Phillips et al. 2009, "target-group background") gives
presences and background the same fine-scale sampling: the comparison becomes where this species was recorded
against where other species were recorded nearby.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

CHUNK = 2_000_000                                    # rows per pass over the training points


class JointData:
    """Members of ``joint_data.npz``, or of ``<name>.npy`` beside it when the npz does not hold them (the national
    build keeps the per-point arrays as uncompressed memory maps)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.npz = np.load(self.root / "joint_data.npz")
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._cache:
            f = self.root / f"{key}.npy"
            self._cache[key] = (np.load(f, mmap_mode="r") if key not in self.npz.files and f.exists()
                                else self.npz[key])
        return self._cache[key]

    def __contains__(self, key: str) -> bool:
        return key in self.npz.files or (self.root / f"{key}.npy").exists()

    def points(self) -> np.ndarray:
        """The training predictors as a read-only memory map (never loaded whole)."""
        X = np.load(self.root / "train_points_cache.npy", mmap_mode="r")
        if X.shape[0] != len(self["pres"]):
            raise ValueError("train_points_cache.npy does not match joint_data.npz")
        return X

    def species(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "species.csv")

    def restored(self) -> dict | None:
        """The restored shoreline presences (``shore_records.npz``: lat, lon, sid, X, rc), None if not built."""
        f = self.root / "shore_records.npz"
        return dict(np.load(f)) if f.exists() else None

    def positions(self, field_dir: str | Path | None = None) -> np.ndarray:
        """Fractional (row, column) of every prepared training row on the field grid (memory map)."""
        return np.load(Path(field_dir or self.root / "field") / "rc_train.npy", mmap_mode="r")

    def snap(self) -> np.ndarray:
        """Row of the nearest record of another species, for every prepared row (memory map)."""
        return np.load(self.root / "community" / "snap.npy", mmap_mode="r")


def calibration_sets(species: pd.DataFrame) -> list[set[int]]:
    return [set(map(int, str(c).split())) for c in species.calibration_ecoregions.fillna("")]


@dataclass
class Standardizer:
    """Predictors -> network inputs. Heavy-tailed variables (precipitation, soil organic carbon) are log1p
    transformed; every variable is centred and scaled by its mean and standard deviation over the background
    points and clipped to +-6 standard deviations (no extrapolation beyond the training range); a missing value
    becomes 0 (the mean). One extra input flags a point where any of the ``flag_columns`` (soil) is missing:
    water, rock or built land, where SoilGrids has no value."""
    names: list[str]
    mu: np.ndarray
    sd: np.ndarray
    log_columns: np.ndarray
    flag_columns: np.ndarray
    clip: float = 6.0

    @property
    def n_inputs(self) -> int:
        return len(self.names) + 1

    @property
    def required_columns(self) -> np.ndarray:
        """Columns that must be finite for a cell to be scored (everything but the flagged soil columns)."""
        return np.setdiff1d(np.arange(len(self.names)), self.flag_columns)

    def _log(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float32)                       # copy
        X[:, self.log_columns] = np.log1p(np.clip(X[:, self.log_columns], 0, None))
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """float32 [n, n_variables] -> float32 [n, n_variables + 1]."""
        Z = np.clip((self._log(X) - self.mu) / self.sd, -self.clip, self.clip)
        flag = ~np.isfinite(Z[:, self.flag_columns]).all(1, keepdims=True)
        return np.concatenate([np.nan_to_num(Z, nan=0.0), flag.astype(np.float32)], 1).astype(np.float32)

    @classmethod
    def fit(cls, X: np.ndarray, background: np.ndarray, names: Sequence[str], log_variables: Sequence[str],
            flag_variables: Sequence[str], chunk: int = CHUNK) -> "Standardizer":
        """Mean and SD of every (log-transformed) variable over the background rows of ``X`` (memmap-friendly:
        accumulated in float64 over row chunks, ignoring missing values)."""
        names = list(names)
        unknown = [v for v in list(log_variables) + list(flag_variables) if v not in names]
        if unknown:
            raise KeyError(f"variables not among the predictors: {unknown}")
        st = cls(names, np.zeros(len(names)), np.ones(len(names)),
                 np.array([names.index(v) for v in log_variables], np.int64),
                 np.array([names.index(v) for v in flag_variables], np.int64))
        s1 = s2 = cnt = 0.0
        for i in range(0, X.shape[0], chunk):
            x = st._log(X[i:i + chunk])[np.asarray(background[i:i + chunk])].astype(np.float64)
            ok = np.isfinite(x)
            s1 = s1 + np.where(ok, x, 0).sum(0)
            s2 = s2 + np.where(ok, x * x, 0).sum(0)
            cnt = cnt + ok.sum(0)
        st.mu = s1 / cnt
        st.sd = np.sqrt(np.maximum(s2 / cnt - st.mu * st.mu, 0)) + 1e-6
        return st

    def save(self, path: str | Path) -> None:
        np.savez(path, mu=self.mu, sd=self.sd, logv=self.log_columns, flag=self.flag_columns,
                 names=np.array(self.names), clip=self.clip)

    @classmethod
    def load(cls, path: str | Path, flag_variables: Sequence[str] = ()) -> "Standardizer":
        """Read ``norm.npz``. A file without ``flag`` (written by earlier versions of the trainer, which flagged missing soil)
        takes the flagged columns from ``flag_variables``."""
        f = np.load(path)
        names = [str(n) for n in f["names"]]
        flag = (f["flag"] if "flag" in f.files else np.array([names.index(v) for v in flag_variables], np.int64))
        if not len(flag):
            raise ValueError(f"{path}: no flagged columns recorded; pass flag_variables")
        return cls(names, f["mu"], f["sd"], f["logv"].astype(np.int64), np.asarray(flag, np.int64),
                   float(f["clip"]) if "clip" in f.files else 6.0)


def ecoregion_at(positions: np.ndarray, raster: np.ndarray) -> np.ndarray:
    """Ecoregion id of the grid cell holding each fractional (row, column) position; -1 off the grid."""
    r = np.floor(np.asarray(positions[:, 0], np.float64)).astype(np.int64)
    c = np.floor(np.asarray(positions[:, 1], np.float64)).astype(np.int64)
    ok = (r >= 0) & (r < raster.shape[0]) & (c >= 0) & (c < raster.shape[1])
    out = np.full(len(r), -1, np.int32)
    out[ok] = raster[r[ok], c[ok]]
    return out


class TrainingPoints:
    """Every training row on the device, each species' presences and background addressable as contiguous ranges of
    a sort order: species s's presences are sorted positions [p_off[s], p_off[s] + n_p[s]), its background points
    [b_off[s], b_off[s] + n_b[s]); ``rows`` maps a sorted position to a row. Rows are the prepared points in file
    order, then the restored shoreline presences (``restored``). Per row, as requested: the standardized inputs ``X``
    (float16), the coordinates ``latlon`` (place), the field ``positions`` (field), the ``ecoregion`` of the row's
    240 m cell (learned calibration) and ``snap``, the row of the nearest record of another species (target-group
    background; a restored presence keeps itself). Host memory stays at a few GB however many points there are."""

    def __init__(self, data: JointData, standardizer: Standardizer | None, n_species: int, device: torch.device,
                 chunk: int = CHUNK, restored: dict | None = None, latlon: bool = False,
                 positions: np.ndarray | None = None, ecoregions: np.ndarray | None = None,
                 snap: np.ndarray | None = None):
        """``standardizer`` None: no inputs (the species stage reads cached features). ``positions``: (row, column) of
        the prepared rows (``JointData.positions``); ``ecoregions``: the ecoregion-id raster of that grid."""
        n0 = len(data["pres"])
        n_r = 0 if restored is None else len(restored["sid"])
        self.n_prepared, n = n0, n0 + n_r
        self.X = None
        if standardizer is not None:
            Xm = data.points()
            self.X = torch.empty((n, standardizer.n_inputs), dtype=torch.float16, device=device)
            for i in range(0, n0, chunk):
                j = min(i + chunk, n0)
                self.X[i:j] = torch.from_numpy(standardizer.transform(Xm[i:j])).to(device, torch.float16)
            if n_r:
                self.X[n0:] = torch.from_numpy(standardizer.transform(restored["X"])).to(device, torch.float16)
        sid = np.asarray(data["sid"])
        pres = np.asarray(data["pres"])
        if n_r:
            sid = np.concatenate([sid, restored["sid"].astype(sid.dtype)])
            pres = np.concatenate([pres, np.ones(n_r, pres.dtype)])
        self.latlon = None
        if latlon:
            lat, lon = np.asarray(data["lat"], np.float32), np.asarray(data["lon"], np.float32)
            if n_r:
                lat = np.concatenate([lat, restored["lat"].astype(np.float32)])
                lon = np.concatenate([lon, restored["lon"].astype(np.float32)])
            self.latlon = torch.from_numpy(np.stack([lat, lon], 1)).to(device)
            del lat, lon
        self.positions = self.ecoregion = None
        if positions is not None:
            rc = np.concatenate([np.asarray(positions, np.float32)] +
                                ([restored["rc"].astype(np.float32)] if n_r else []))
            if len(rc) != n:
                raise ValueError(f"{len(rc)} field positions for {n} training rows")
            if ecoregions is not None:
                self.ecoregion = torch.from_numpy(ecoregion_at(rc, ecoregions)).to(device)
            self.positions = torch.from_numpy(rc).to(device)
            del rc
        self.snap = None
        if snap is not None:
            self.snap = torch.empty(n, dtype=torch.int64, device=device)
            for i in range(0, n0, chunk):
                j = min(i + chunk, n0)
                self.snap[i:j] = torch.from_numpy(np.asarray(snap[i:j], np.int64)).to(device)
            self.snap[n0:] = torch.arange(n0, n, device=device)
        sid_t = torch.from_numpy(sid).long()
        pres_t = torch.from_numpy(pres).bool()
        del sid, pres
        order = torch.argsort(sid_t * 2 + (~pres_t).long(), stable=True)    # by species, presences first
        sid_o, pres_o = sid_t[order], pres_t[order]
        n_p = torch.bincount(sid_o[pres_o], minlength=n_species)
        n_b = torch.bincount(sid_o[~pres_o], minlength=n_species)
        start = torch.zeros(n_species, dtype=torch.long)
        start[1:] = torch.cumsum(n_p + n_b, 0)[:-1]
        self.rows = order.to(device)
        self.p_off, self.b_off = start.to(device), (start + n_p).to(device)
        self.n_p, self.n_b = n_p.to(device), n_b.to(device)
        self.active = torch.nonzero(self.n_p > 0).squeeze(1)              # species with at least one presence
        self.background = self.rows[torch.nonzero(~pres_o).squeeze(1).to(device)]   # every background row
        self.is_presence = pres_t.to(device)
        del sid_t, pres_t, sid_o, pres_o

    def __len__(self) -> int:
        return len(self.rows)

    def gather(self, positions: torch.Tensor) -> torch.Tensor:
        """Standardized inputs (float16) at sorted positions."""
        return self.X[self.rows[positions]]


@dataclass
class PlotSet:
    """Independent presence/absence plots for evaluation. ``present[i, j]``: species ``species[i]`` recorded at
    plot j; ``X``: the standardized inputs (float32: plots are few, and the map store reads cells in float32 too);
    ``valid[j]``: plot j has every required predictor (after the shoreline fill, if used);
    ``outside[i, j]``: plot j lies outside that species' calibration ecoregions; ``inside`` = valid and not outside
    (where a map under the hard calibration rule is not 0); ``baseline``: per-species MaxEnt map values at the
    plots, if available. ``latlon`` and ``rc`` (field positions) are filled for the place and field pathways."""
    name: str
    species: np.ndarray
    present: np.ndarray
    valid: np.ndarray
    outside: np.ndarray
    X: torch.Tensor
    lat: np.ndarray
    lon: np.ndarray
    eco: np.ndarray
    baseline: np.ndarray | None = None
    latlon: torch.Tensor | None = None
    rc: torch.Tensor | None = None

    @property
    def inside(self) -> np.ndarray:
        return ~self.outside & self.valid[None]


_PLOT_MEMBERS = ("plot_X", "plot_eco", "plot_lat", "plot_lon", "eval_sid", "eval_y")


def _plot_set(name: str, m: dict, calib: list[set[int]], st: Standardizer, device, rc=None) -> PlotSet:
    PX = np.asarray(m["plot_X"], np.float32)
    valid = np.isfinite(PX[:, st.required_columns]).all(1)
    sid = np.asarray(m["eval_sid"])
    peco = np.asarray(m["plot_eco"])
    outside = (np.stack([~np.isin(peco, list(calib[s])) for s in sid]) if len(sid)
               else np.zeros((0, len(PX)), bool))
    base = np.asarray(m["maxent_occ"], np.float32) if "maxent_occ" in m else None
    lat, lon = np.asarray(m["plot_lat"]), np.asarray(m["plot_lon"])
    return PlotSet(name, sid, np.asarray(m["eval_y"]).astype(bool), valid, outside,
                   torch.from_numpy(st.transform(PX)).to(device), lat, lon, peco, base,
                   torch.from_numpy(np.stack([lat, lon], 1).astype(np.float32)).to(device),
                   None if rc is None else torch.from_numpy(np.asarray(rc, np.float32)).to(device))


def plot_sets(data: JointData, calib: list[set[int]], st: Standardizer, device, fill: bool = False,
              field_dir: str | Path | None = None) -> list[PlotSet]:
    """VegBank (``joint_data.npz``) first, then every ``plots_<source>.npz`` of the data directory. ``fill``: the
    plots' predictors with the shoreline climate fill (``plot_fill_<source>.npz``); ``field_dir``: read the plots'
    field positions (``rc_plots_<source>.npy``)."""
    sources = [("vegbank", data, lambda k: k in data)]
    for f in sorted(data.root.glob("plots_*.npz")):
        E = np.load(f)
        sources.append((f.stem.split("_", 1)[1], E, lambda k, E=E: k in E.files))
    out = []
    for name, src, has in sources:
        m = {k: src[k] for k in _PLOT_MEMBERS}
        if has("maxent_occ"):
            m["maxent_occ"] = src["maxent_occ"]
        if fill:
            f = data.root / f"plot_fill_{name}.npz"
            if not f.exists():
                raise FileNotFoundError(f"{f}: the plots' shoreline fill is not built (scripts/build_shoreline.py)")
            z = np.load(f)
            if int(z["n"]) != len(m["plot_X"]):
                raise ValueError(f"{f} does not match the {name} plots")
            m["plot_X"] = z["X"]
        rc = np.load(Path(field_dir) / f"rc_plots_{name}.npy") if field_dir is not None else None
        out.append(_plot_set(name, m, calib, st, device, rc))
    return out


def block_halves(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """True for plots in the "dev" half: 1-degree blocks (~111 km north-south) assigned to halves by a hash of the
    block id, so dev and test plots are spatially separated (model selection never sees a test plot's block)."""
    blk = np.floor(lat).astype(int) * 1000 + np.floor(lon).astype(int)
    return pd.util.hash_array(blk.astype(np.int64)) % 2 == 0
