"""The shared features of a trained model, computed once for every row the species stage can read and every plot.

In the species stage (train.py, ``freeze_shared``) the networks of a trained run are fixed, so each location's
shared features F(x) = h(x) + G(C(x)) and P(x) (model.py) never change. With target-group background every row a
training step reads is a record: a presence, or the record of another species that a background point (of the
species' own pool or of the continental pool) moves to. So the features of exactly those rows are computed here once,
with the run's own inputs and arithmetic (``JointRangeModel.shared``, bfloat16 as in training), and training
then fits only the species parameters on them: a step costs the dot products of 256 species with their rows (18 ms
for the 16,448-species CONUS model instead of seconds with the landscape field). The plots of every evaluation set are
cached the same way.

Layout (``<out>/``): ``rows.npy`` int64, the training rows cached, in the data's row order (prepared rows, then the
restored shoreline presences at n_prepared + i); ``F.npy`` and ``P.npy`` float16 [n_rows, width] and
[n_rows, place_dim]; ``plots_<source>_F.npy`` and ``_P.npy`` for each plot set; ``cache.json`` (run, rows,
shoreline_records, fill_plots).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import JointData, PlotSet, Standardizer, calibration_sets, plot_sets
from .model import JointRangeModel


def record_rows(data: JointData, restored: dict | None) -> np.ndarray:
    """Every row a training step with target-group background can read: the presences, the rows background points
    move to, and the restored presences (after the prepared rows)."""
    pres = np.asarray(data["pres"]).astype(bool)
    rows = np.unique(np.concatenate([np.flatnonzero(pres), np.asarray(data.snap())[~pres]]))
    if restored is not None:
        rows = np.concatenate([rows, len(pres) + np.arange(len(restored["sid"]))])
    return rows


@torch.no_grad()
def build_cache(run: str | Path, data_dir: str | Path, out: str | Path, flag_variables=(), field_dir=None,
                shoreline_records: bool = True, fill_plots: bool = True, device: str = "cuda", chunk: int = 65536,
                log=print) -> dict:
    """Cache the shared features of the run ``run`` (its ``model_best.pt`` and ``norm.npz``) for the training rows
    and plots of ``data_dir`` in ``out``. ``flag_variables``: the flagged predictors, for a ``norm.npz`` that does
    not record them."""
    run, data_dir, out = Path(run), Path(data_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    field_dir = Path(field_dir) if field_dir else data_dir / "field"
    model = JointRangeModel.load(run / "model_best.pt", dev, field_dir)
    st = Standardizer.load(run / "norm.npz", flag_variables)
    data = JointData(data_dir)
    if st.names != [str(n) for n in data["names"]]:
        raise ValueError(f"{run}: its predictors differ from {data_dir}'s")
    restored = data.restored() if shoreline_records else None
    rows = record_rows(data, restored)
    n0 = len(data["pres"])
    Xm, lat, lon = data.points(), data["lat"], data["lon"]
    rc = data.positions(field_dir) if model.field is not None else None
    F = np.lib.format.open_memmap(out / "F.npy", mode="w+", dtype=np.float16, shape=(len(rows), model.width))
    P = (np.lib.format.open_memmap(out / "P.npy", mode="w+", dtype=np.float16, shape=(len(rows), model.place_dim))
         if model.place is not None else None)
    t0 = time.time()
    autocast = torch.autocast(dev.type, dtype=torch.bfloat16)
    for i in range(0, len(rows), chunk):
        j = rows[i:i + chunk]
        base, ext = j[j < n0], j[j >= n0] - n0
        X = np.concatenate([np.asarray(Xm[base], np.float32)] + ([restored["X"][ext]] if len(ext) else []))
        ll = np.stack([np.concatenate([np.asarray(lat[base])] + ([restored["lat"][ext]] if len(ext) else [])),
                       np.concatenate([np.asarray(lon[base])] + ([restored["lon"][ext]] if len(ext) else []))], 1)
        r = None
        if rc is not None:
            r = torch.from_numpy(np.concatenate([np.asarray(rc[base])] + ([restored["rc"][ext]] if len(ext) else []))
                                 .astype(np.float32)).to(dev)
        with autocast:
            f, p = model.shared(torch.from_numpy(st.transform(X)).to(dev),
                                torch.from_numpy(ll.astype(np.float32)).to(dev), r)
        F[i:i + len(j)] = f.half().cpu().numpy()
        if P is not None:
            P[i:i + len(j)] = p.half().cpu().numpy()
        if (i // chunk) % 50 == 0:
            log(f"rows {i + len(j):,}/{len(rows):,} ({time.time() - t0:.0f} s)")
    np.save(out / "rows.npy", rows)
    F.flush()
    if P is not None:
        P.flush()
    from .train import plot_features
    sets = plot_sets(data, calibration_sets(data.species()), st, dev, fill=fill_plots,
                     field_dir=field_dir if model.field is not None else None)
    for ps in sets:
        f, p = plot_features(model, ps)
        np.save(out / f"plots_{ps.name}_F.npy", f.half().cpu().numpy())
        if p is not None:
            np.save(out / f"plots_{ps.name}_P.npy", p.half().cpu().numpy())
        log(f"{ps.name}: {len(ps.X):,} plots cached")
    n_total = n0 + (0 if restored is None else len(restored["sid"]))
    info = {"run": str(run.resolve()), "rows": int(len(rows)), "rows_total": int(n_total),
            "shoreline_records": bool(shoreline_records), "fill_plots": bool(fill_plots),
            "width": model.width, "place_dim": model.place_dim, "seconds": round(time.time() - t0)}
    (out / "cache.json").write_text(json.dumps(info, indent=1))
    log(f"cached {len(rows):,} rows and {len(sets)} plot sets in {time.time() - t0:.0f} s -> {out}")
    return info


class FeatureCache:
    """A cache in device memory: ``rows(r)`` returns the float32 features (F, P) of training rows ``r`` (data row
    order), ``plot_features(ps)`` those of a plot set."""

    def __init__(self, directory: str | Path, device):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / "cache.json").read_text())
        self.device = torch.device(device)
        rows = torch.from_numpy(np.load(self.dir / "rows.npy"))
        self.pos = torch.full((self.meta["rows_total"],), -1, dtype=torch.int32)
        self.pos[rows] = torch.arange(len(rows), dtype=torch.int32)
        self.pos = self.pos.to(self.device)
        self.F = torch.from_numpy(np.load(self.dir / "F.npy")).to(self.device)
        self.place = (torch.from_numpy(np.load(self.dir / "P.npy")).to(self.device)
                      if (self.dir / "P.npy").exists() else None)
        self._checked = False

    @property
    def n_rows(self) -> int:
        return len(self.F)

    def check(self, run: str | Path, n_rows_total: int) -> None:
        """The cache must come from ``run`` and cover the same training rows (with or without restored records)."""
        if Path(self.meta["run"]).resolve() != Path(run).resolve():
            raise ValueError(f"{self.dir} caches {self.meta['run']}, not {run}")
        if self.meta["rows_total"] != n_rows_total:
            raise ValueError(f"{self.dir} was built for {self.meta['rows_total']:,} training rows, not "
                             f"{n_rows_total:,} (shoreline records on in one and off in the other?)")

    def rows(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        i = self.pos[rows].long()
        if not self._checked:                               # a step may only read cached rows
            if bool((i < 0).any()):
                raise RuntimeError("a training step read a row without cached features (cache built without "
                                   "target-group background?)")
            self._checked = True
        return self.F[i].float(), (self.place[i].float() if self.place is not None else None)

    def plot_features(self, ps: PlotSet) -> tuple[torch.Tensor, torch.Tensor | None]:
        F = torch.from_numpy(np.load(self.dir / f"plots_{ps.name}_F.npy")).to(self.device).float()
        p = self.dir / f"plots_{ps.name}_P.npy"
        return F, (torch.from_numpy(np.load(p)).to(self.device).float() if p.exists() else None)
