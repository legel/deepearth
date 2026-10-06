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
* ``tree/natives.dated.nwk``: the dated phylogeny whose tips are the species labels.

Each species' training points are what a per-species MaxEnt would be fitted to: its presence records (thinned) and
background points drawn inside its calibration ecoregions with probability proportional to the sampling effort of
all records there, so that the model contrasts where the species was recorded with where anyone recorded anything.
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


class TrainingPoints:
    """All training points standardized on the GPU (float16), with each species' presences and background
    addressable as contiguous ranges of a sort order: species s's presences are sorted positions
    [p_off[s], p_off[s] + n_p[s]), its background points [b_off[s], b_off[s] + n_b[s]); ``rows`` maps a sorted
    position to the row of ``X``. Host memory stays at a few GB however many points there are."""

    def __init__(self, data: JointData, standardizer: Standardizer, n_species: int, device: torch.device,
                 chunk: int = CHUNK):
        Xm = data.points()
        n = Xm.shape[0]
        self.X = torch.empty((n, standardizer.n_inputs), dtype=torch.float16, device=device)
        for i in range(0, n, chunk):
            self.X[i:i + chunk] = torch.from_numpy(standardizer.transform(Xm[i:i + chunk])).to(device, torch.float16)
        sid = torch.from_numpy(np.asarray(data["sid"])).long()
        pres = torch.from_numpy(np.asarray(data["pres"])).bool()
        order = torch.argsort(sid * 2 + (~pres).long(), stable=True)        # by species, presences first
        sid_o, pres_o = sid[order], pres[order]
        n_p = torch.bincount(sid_o[pres_o], minlength=n_species)
        n_b = torch.bincount(sid_o[~pres_o], minlength=n_species)
        del sid, pres, sid_o, pres_o
        start = torch.zeros(n_species, dtype=torch.long)
        start[1:] = torch.cumsum(n_p + n_b, 0)[:-1]
        self.rows = order.to(device)
        self.p_off, self.b_off = start.to(device), (start + n_p).to(device)
        self.n_p, self.n_b = n_p.to(device), n_b.to(device)
        self.active = torch.nonzero(self.n_p > 0).squeeze(1)              # species with at least one presence

    def gather(self, positions: torch.Tensor) -> torch.Tensor:
        """Standardized inputs (float16) at sorted positions."""
        return self.X[self.rows[positions]]


@dataclass
class PlotSet:
    """Independent presence/absence plots for evaluation. ``present[i, j]``: species ``species[i]`` recorded at
    plot j; ``inside[i, j]``: plot j lies in that species' calibration ecoregions and has every required
    predictor (a map is 0 elsewhere); ``baseline``: per-species MaxEnt map values at the plots, if available."""
    name: str
    species: np.ndarray
    present: np.ndarray
    inside: np.ndarray
    X: torch.Tensor
    lat: np.ndarray
    lon: np.ndarray
    baseline: np.ndarray | None = None


_PLOT_MEMBERS = ("plot_X", "plot_eco", "plot_lat", "plot_lon", "eval_sid", "eval_y")


def _plot_set(name: str, m: dict, calib: list[set[int]], st: Standardizer, device) -> PlotSet:
    PX = np.asarray(m["plot_X"], np.float32)
    ok = np.isfinite(PX[:, st.required_columns]).all(1)
    sid = np.asarray(m["eval_sid"])
    peco = np.asarray(m["plot_eco"])
    inside = np.stack([np.isin(peco, list(calib[s])) & ok for s in sid]) if len(sid) else np.zeros((0, len(PX)), bool)
    base = np.asarray(m["maxent_occ"], np.float32) if "maxent_occ" in m else None
    return PlotSet(name, sid, np.asarray(m["eval_y"]).astype(bool), inside,
                   torch.from_numpy(st.transform(PX)).to(device, torch.float16),
                   np.asarray(m["plot_lat"]), np.asarray(m["plot_lon"]), base)


def plot_sets(data: JointData, calib: list[set[int]], st: Standardizer, device) -> list[PlotSet]:
    """VegBank (``joint_data.npz``) first, then every ``plots_<source>.npz`` of the data directory."""
    sources = [("vegbank", data, lambda k: k in data)]
    for f in sorted(data.root.glob("plots_*.npz")):
        E = np.load(f)
        sources.append((f.stem.split("_", 1)[1], E, lambda k, E=E: k in E.files))
    out = []
    for name, src, has in sources:
        m = {k: src[k] for k in _PLOT_MEMBERS}
        if has("maxent_occ"):
            m["maxent_occ"] = src["maxent_occ"]
        out.append(_plot_set(name, m, calib, st, device))
    return out


def block_halves(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """True for plots in the "dev" half: 1-degree blocks (~111 km north-south) assigned to halves by a hash of the
    block id, so dev and test plots are spatially separated (model selection never sees a test plot's block)."""
    blk = np.floor(lat).astype(int) * 1000 + np.floor(lon).astype(int)
    return pd.util.hash_array(blk.astype(np.int64)) % 2 == 0
