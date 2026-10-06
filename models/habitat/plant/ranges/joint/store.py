"""The national map store: every species' 240 m map as one shared, transform-coded field plus a short code per
species.

Why it is small. The joint model scores species s at a cell x as f_s(x) = <h(x), w_s> + b_s: all species read the
same 256 numbers h(x) per cell. Storing h once per cell and (w_s, b_s) once per species stores every map exactly;
what remains is to store h compactly and to know how much each rounding error in h costs the maps.

Transform coding (Karhunen-Loeve transform, KLT, in the metric of the species). Let W be the matrix of all
species' vectors and M = W^T W. A change e in h(x) moves the scores of all species by a total squared amount
|W e|^2 = e^T M e, so the right measure of error in feature space is the M-metric, not the plain one. Write

    y(x) = (h(x) - mu) M^(1/2) V,       r_s = V^T M^(-1/2) w_s,       o_s = <mu, w_s> + b_s,

with mu the mean feature vector over a sample of land cells and V the eigenvectors (largest variance first) of the
feature covariance in that metric, M^(1/2) C M^(1/2). Then f_s(x) = <y(x), r_s> + o_s exactly, and the r_s form an
orthonormal frame (sum_s r_s r_s^T = I): an error vector e in y changes the scores of all species by a total
squared error of exactly |e|^2. So one uniform quantizer step ``delta`` (3.2 in y units for the national store)
serves every channel and every species, and no channel is dropped: low-variance channels simply round to mostly
zeros and cost almost nothing after compression.

Layout and reading: reader.py. Building (``build_store``) writes q(x) = round(y(x) / delta) per cell and, per
species, its code delta * r_s, offset, background quantiles, P5 threshold and calibration ecoregions;
``update_zero_shot`` replaces the species without records in place; ``recode_store`` rewrites the tiles in the
current layout.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from pyproj import Transformer  # noqa: I001  (before rasterio: its bundled PROJ can break pyproj's array transforms)
import torch
import zstandard

from .data import JointData, Standardizer
from .model import JointRangeModel
from .reader import TILE, Store
from .zero_shot import ZeroShot


# ------------------------------------------------------------------------------------------------ inputs
class GridInputs:
    """Predictors of the cells of a 240 m region grid, in the model's input order: WorldClim (and any other
    variable the region stack holds) from the stack, SoilGrids sampled at the cell centres from the ~230 m North
    America layers, as at the training points."""

    def __init__(self, grid_dir: str | Path, soil_dir: str | Path, names: Sequence[str]):
        from ..predictors import ConusStack
        from ..soil import VARS as SOIL_VARS, SoilPoints
        self.stack = ConusStack(grid_dir)
        self.names = list(names)
        self.source = [("stack", self.stack.variables.index(n)) if n in self.stack.variables else
                       ("soil", SOIL_VARS.index(n)) for n in self.names
                       if n in self.stack.variables or n in SOIL_VARS]
        if len(self.source) != len(self.names):
            missing = [n for n in self.names if n not in self.stack.variables and n not in SOIL_VARS]
            raise KeyError(f"{grid_dir}: no source for predictors {missing}")
        self.soil = SoilPoints(soil_dir) if any(src == "soil" for src, _ in self.source) else None
        self.t = self.stack.profile["transform"]
        self.to_lonlat = Transformer.from_crs(self.stack.crs, 4326, always_xy=True)
        self.shape = self.stack.shape

    def block(self, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        """float32 [(r1 - r0) * (c1 - c0), n_variables], row-major cells."""
        n = (r1 - r0) * (c1 - c0)
        X = np.empty((n, len(self.names)), np.float32)
        soil = None
        if self.soil is not None:
            cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
            x = self.t.c + (cc.ravel() + 0.5) * self.t.a
            y = self.t.f + (rr.ravel() + 0.5) * self.t.e
            lon, lat = self.to_lonlat.transform(x, y)
            if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
                raise RuntimeError("non-finite coordinate transform (import pyproj before rasterio)")
            soil = self.soil.at(lon, lat).astype(np.float32)
        for j, (src, k) in enumerate(self.source):
            X[:, j] = (np.asarray(self.stack.data[k][r0:r1, c0:c1], np.float32).ravel() if src == "stack"
                       else soil[:, k])
        return X


@torch.no_grad()
def features(model: JointRangeModel, Xs: np.ndarray, device, chunk: int = 262144) -> torch.Tensor:
    """h(x) (float32, on ``device``) of standardized inputs, evaluated in bfloat16 as in training."""
    dev = torch.device(device)
    out = []
    for i in range(0, len(Xs), chunk):
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            out.append(model.features(torch.from_numpy(Xs[i:i + chunk]).to(dev)).float())
    return torch.cat(out) if out else torch.zeros(0, model.width, device=dev)


# ------------------------------------------------------------------------------------------------- transform
def klt(H: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
    """Metric-weighted KLT (module docstring) from sampled features H [n, d], species vectors W [S, d] and offsets
    b [S]. Returns mu [d], T [d, d] (y = (h - mu) T), R [S, d] (rows r_s), offsets o [S] and the channel
    variances (descending)."""
    Hd = H.double()
    mu = Hd.mean(0)
    e, U = torch.linalg.eigh((W.T @ W).double())
    e = e.clamp_min(float(e.max()) * 1e-12)
    Ms, Mi = (U * e.sqrt()) @ U.T, (U / e.sqrt()) @ U.T                 # M^(1/2), M^(-1/2)
    ev, V = torch.linalg.eigh(Ms @ torch.cov((Hd - mu).T) @ Ms)
    ev, V = ev.flip(0), V.flip(1)
    off = (mu @ W.T.double() + b.double()).float()
    return mu.float(), (Ms @ V).float(), (W.double() @ Mi @ V).float(), off, ev


def sample_features(model: JointRangeModel, st: Standardizer, grids: dict[str, GridInputs], device,
                    blocks: int = 32, block: int = 512, cells: int = 12000, seed: int = 0) -> torch.Tensor:
    """Features of up to ``cells`` random land cells in each of ``blocks`` random ``block`` x ``block`` windows of
    every region grid (in the order of ``grids``): the sample the transform is fitted to."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    out = []
    for gi in grids.values():
        H_, W_ = gi.shape
        for _ in range(blocks):
            r0 = int(rng.integers(0, max(1, H_ - block)))
            c0 = int(rng.integers(0, max(1, W_ - block)))
            X = gi.block(r0, min(H_, r0 + block), c0, min(W_, c0 + block))
            ok = np.isfinite(X[:, st.required_columns]).all(1)
            if ok.sum():
                out.append(features(model, st.transform(X[ok]), device)[torch.randperm(int(ok.sum()))[:cells]])
    return torch.cat(out)


# ---------------------------------------------------------------------------------------------- tile codec
class KltWriter:
    """Writes one region's quantized field, strip by strip (``put`` rows r0 .. r0 + tile of the full width)."""
    limit = 32767

    def __init__(self, out: Path, region: str, H: int, W: int, d: int, tile: int = TILE, level: int = 9,
                 threads: int = 12):
        self.tile = tile
        self.blob = open(Path(out) / f"field_{region}.zst", "wb")
        self.index = np.zeros(((H + tile - 1) // tile, (W + tile - 1) // tile, 4), np.int64)   # offset, bytes, k, k16
        self.path, self.channels, self.nbytes = Path(out) / f"index_{region}.npy", d, 0
        self._local, self._level = threading.local(), level                # one compressor per thread
        self.pool = ThreadPoolExecutor(threads)

    def encode_tile(self, q: np.ndarray) -> tuple[bytes, int, int]:
        """int16 [h, w, d] -> (compressed payload, k, k16)."""
        c = np.ascontiguousarray(np.moveaxis(q, -1, 0))                    # channel-major (d, h, w)
        nz = np.flatnonzero(c.reshape(c.shape[0], -1).any(1))
        if not len(nz):
            return b"", 0, 0
        k = int(nz[-1]) + 1
        big = np.flatnonzero((np.abs(c[:k].reshape(k, -1)) > 127).any(1))
        k16 = int(big[-1]) + 1 if len(big) else 0
        planes = np.ascontiguousarray(c[:k16]).view(np.uint8).reshape(-1, 2).T      # low bytes, then high bytes
        zc = getattr(self._local, "zc", None)
        if zc is None:
            zc = self._local.zc = zstandard.ZstdCompressor(level=self._level)
        payload = np.ascontiguousarray(planes).tobytes() + c[k16:k].astype(np.int8).tobytes()
        return zc.compress(payload), k, k16

    def put(self, r0: int, q: np.ndarray) -> None:
        ty = r0 // self.tile
        jobs = [q[:, c0:c0 + self.tile] for c0 in range(0, q.shape[1], self.tile)]
        for tx, (b, k, k16) in enumerate(self.pool.map(self.encode_tile, jobs)):
            self.index[ty, tx] = (self.blob.tell(), len(b), k, k16)
            self.blob.write(b)
            self.nbytes += len(b)

    def close(self) -> None:
        self.blob.close()
        self.pool.shutdown()
        np.save(self.path, self.index)


# --------------------------------------------------------------------------------------------------- build
def write_field(model: JointRangeModel, st: Standardizer, gi: GridInputs, out: Path, region: str,
                encode: Callable[[torch.Tensor], torch.Tensor], d: int, device, tile: int = TILE,
                log=print) -> dict:
    """Quantize and write one region's field and valid mask (``encode``: features -> rounded y / delta)."""
    H_, W_ = gi.shape
    writer = KltWriter(out, region, H_, W_, d, tile)
    valid = np.zeros((H_, W_), bool)
    t = time.time()
    for r0 in range(0, H_, tile):
        r1 = min(H_, r0 + tile)
        X = gi.block(r0, r1, 0, W_)
        ok = np.isfinite(X[:, st.required_columns]).all(1)
        q = torch.zeros((len(X), d), dtype=torch.int16)
        if ok.any():
            for i in range(0, int(ok.sum()), 1 << 20):
                sel = torch.from_numpy(np.flatnonzero(ok)[i:i + (1 << 20)])
                z = encode(features(model, st.transform(X[sel.numpy()]), device))
                q[sel] = torch.clamp(z, -writer.limit, writer.limit).to(torch.int16).cpu()
        writer.put(r0, q.numpy().reshape(r1 - r0, W_, d))
        valid[r0:r1] = ok.reshape(r1 - r0, W_)
    writer.close()
    np.save(out / f"valid_{region}.npy", np.packbits(valid, axis=1))
    log(f"field {region}: {H_} x {W_}, {int(valid.sum()):,} land cells, {writer.nbytes / 1e9:.2f} GB, "
        f"{time.time() - t:.0f} s")
    return {"cells": int(valid.sum()), "bytes": writer.nbytes}


class _SpeciesPoints:
    """Training rows of every species (grouped once), for the background quantiles and P5 of the species table."""

    def __init__(self, data: JointData, n_species: int):
        self.X = data.points()
        sid = np.asarray(data["sid"])
        self.pres = np.asarray(data["pres"]).astype(bool)
        self.order = np.argsort(sid, kind="stable")
        self.bounds = np.searchsorted(sid[self.order], np.arange(n_species + 1))

    def own(self, s: int) -> np.ndarray:
        return np.sort(self.order[self.bounds[s]:self.bounds[s + 1]])

    def relatives(self, rel: Sequence[int], n: int, seed: int) -> np.ndarray:
        """Up to ``n`` of the relatives' points, drawn with the generator ``seed``."""
        idx = np.concatenate([self.order[self.bounds[r]:self.bounds[r + 1]] for r in rel])
        return np.sort(np.random.default_rng(seed).choice(idx, min(len(idx), n), replace=False))

    @torch.no_grad()
    def scale(self, model: JointRangeModel, st: Standardizer, idx: np.ndarray, w: torch.Tensor, b: torch.Tensor,
              device) -> tuple[np.ndarray, float]:
        """(254 quantiles of f over the background rows of ``idx``, 5th percentile over its presence rows)."""
        fs = (features(model, st.transform(np.asarray(self.X[idx])), device) @ w + b).cpu().numpy()
        bg, pr = fs[~self.pres[idx]], fs[self.pres[idx]]
        return (np.quantile(bg, np.linspace(0, 1, 256)[1:-1]) if len(bg) else 0,
                np.quantile(pr, 0.05) if len(pr) else np.inf)


@torch.no_grad()
def species_table(model: JointRangeModel, st: Standardizer, data: JointData, W: torch.Tensor, b: torch.Tensor,
                  relatives: Sequence[list[int] | None], device, inferred_points: int = 30000,
                  seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Per species: 254 quantiles of f_s over its background points and the 5th percentile of f_s over its
    presences (float32 [S, 254], [S]). Rows of W beyond the model's species (zero-shot species) use a random
    sample of up to ``inferred_points`` of their relatives' points (``relatives[s]``), drawn with the generator
    seeded ``seed + s`` (s = the species' row in the table), so any row can be recomputed on its own."""
    S0, S = model.n_species, W.shape[0]
    pts = _SpeciesPoints(data, S0)
    quant = np.zeros((S, 254), np.float32)
    p5 = np.zeros(S, np.float32)
    for s in range(S):
        idx = pts.own(s) if s < S0 else pts.relatives(relatives[s], inferred_points, seed + s)
        if len(idx):
            quant[s], p5[s] = pts.scale(model, st, idx, W[s], b[s], device)
    return quant, p5


@torch.no_grad()
def update_zero_shot(path: str | Path, model: JointRangeModel, st: Standardizer, data: JointData, zs: ZeroShot,
                     device="cuda", inferred_points: int = 30000, seed: int = 1, recompute_all: bool = False,
                     log=print) -> int:
    """Replace the species without records of an existing store by ``zs`` (e.g. after a change of the zero-shot
    rule) without rebuilding its fields. Returns the number of species changed.

    A species' code and offset are linear in its vector: code_s = w_s M^-1/2 V delta, offset_s = <mu, w_s> + b_s.
    The linear map w -> (code, offset - b) is recovered from the store's trained species by least squares (the
    store must have been built from ``model``: relative code error < 1e-4, offset error < 1e-3 score units, else
    nothing is written) and applied to the new vectors. Quantiles and P5 are recomputed for the species whose code
    or offset changed (every species of ``zs`` with ``recompute_all``, e.g. to bring a table built with other random
    draws to the per-row seeds), from the generator seeded ``seed + row`` as in ``species_table``. species.npz is
    replaced atomically."""
    import os
    path = Path(path)
    meta = json.loads((path / "store.json").read_text())
    if meta.get("codec") != "klt":
        raise ValueError(f"{path}: codec {meta.get('codec')!r} is not supported")
    T = dict(np.load(path / "species.npz"))
    S0 = model.n_species
    labels = list(data.species().species)
    if list(T["species"][:S0]) != labels:
        raise ValueError(f"{path}: the store's species order differs from the model's")
    W = model.species_vectors().detach().double().cpu().numpy()
    b = model.b.detach().double().cpu().numpy()
    codes0 = T["codes"][:S0].astype(np.float64)
    L = np.linalg.lstsq(W, codes0, rcond=None)[0]
    m = np.linalg.lstsq(W, T["offsets"][:S0].astype(np.float64) - b, rcond=None)[0]
    err_c = np.abs(W @ L - codes0).max() / np.abs(codes0).max()
    err_o = np.abs(W @ m + b - T["offsets"][:S0]).max()
    log(f"{path.name}: linear code map recovered; max relative code error {err_c:.1e}, max offset error "
        f"{err_o:.1e} score units")
    if err_c > 1e-4 or err_o > 1e-3:
        raise RuntimeError(f"{path}: codes are not linear in this model's species vectors (built from another model?)")
    row = {n: i for i, n in enumerate(T["species"])}
    missing = [n for n in zs.names if n not in row]
    if missing:
        raise KeyError(f"{path}: {len(missing)} species are not in the store, e.g. {missing[:3]}")
    Wz = zs.W.detach().double().cpu().numpy()
    pts = _SpeciesPoints(data, S0)
    changed = 0
    for j, name in enumerate(zs.names):
        i = row[name]
        c = (Wz[j] @ L).astype(np.float32)
        o = np.float32(Wz[j] @ m + zs.b[j])
        if (not recompute_all and np.abs(c - T["codes"][i]).max() <= 1e-4 * np.abs(T["codes"][i]).max()
                and abs(o - T["offsets"][i]) < 1e-4):
            continue
        T["codes"][i], T["offsets"][i] = c, o
        idx = pts.relatives(zs.relatives[j], inferred_points, seed + i)
        T["quantiles"][i], T["p5"][i] = pts.scale(model, st, idx, torch.tensor(Wz[j], dtype=torch.float32,
                                                                               device=device),
                                                  float(zs.b[j]), device)
        changed += 1
    tmp = path / "species.tmp.npz"
    np.savez(tmp, **T)
    os.replace(tmp, path / "species.npz")
    log(f"{path.name}: {changed} of {len(zs.names)} species without records updated")
    return changed


def build_store(model: JointRangeModel, st: Standardizer, data: JointData, out: str | Path,
                grids: dict[str, GridInputs], delta: float = 3.2, zero_shot: ZeroShot | None = None,
                sample: dict | None = None, tile: int = TILE, device="cuda",
                calibration: Callable[[list[str], list[str]], list[str]] | None = None, log=print) -> dict:
    """Write a map store for every species of the model (and the zero-shot species, if given) over the region
    grids ``grids`` (region -> GridInputs). ``sample``: keyword arguments of ``sample_features``. ``calibration``:
    (labels, calibration areas) -> calibration areas as stored, e.g. ``scope.extend_calibration`` so that every
    species native to the mapped region has a calibration area on its grid."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sp = data.species()
    with torch.no_grad():
        W = model.species_vectors().detach().float()
        b = model.b.detach().float()
    labels = list(sp.species)
    calib = sp.calibration_ecoregions.fillna("").astype(str).tolist()
    inferred = [False] * len(labels)
    relatives: list[list[int] | None] = [None] * len(labels)
    if zero_shot is not None and zero_shot.names:
        W = torch.cat([W, zero_shot.W.float().to(W.device)])
        b = torch.cat([b, torch.tensor(zero_shot.b, device=b.device)])
        labels += zero_shot.names
        calib += zero_shot.calibration
        inferred += [True] * len(zero_shot.names)
        relatives += zero_shot.relatives
        log(f"zero-shot: {len(zero_shot.names)} species without records mapped from relatives")
    if calibration is not None:
        calib = calibration(labels, calib)

    Hs = sample_features(model, st, grids, device, **(sample or {}))
    mu, T, R, off, ev = klt(Hs, W, b)
    shown = [i for i in (0, 9, 63, 127) if i < len(ev)]
    log(f"transform: d {T.shape[0]}, step {delta}, sample {len(Hs):,} cells; SD of channels {shown}: "
        f"{ev[shown].clamp_min(0).sqrt().cpu().numpy().round(2)}")
    del Hs
    meta = {"codec": "klt", "delta": delta, "tile": tile, "d": int(T.shape[0]), "shape": {}, "valid": "packbits"}
    fields = {}
    for region, gi in grids.items():
        meta["shape"][region] = list(gi.shape)
        fields[region] = write_field(model, st, gi, out, region, lambda h: torch.round(((h - mu) @ T) / delta),
                                     int(T.shape[0]), device, tile, log)
    (out / "store.json").write_text(json.dumps(meta))

    quant, p5 = species_table(model, st, data, W, b, relatives, device)
    np.savez(out / "species.npz", species=np.array(labels, dtype=str),
             codes=(R * delta).cpu().numpy().astype(np.float32), offsets=off.cpu().numpy().astype(np.float32),
             quantiles=quant, p5=p5, calibration=np.array(calib, dtype=str), inferred=np.array(inferred, bool))
    log(f"species table: {len(labels)} species, {(out / 'species.npz').stat().st_size / 1e6:.1f} MB")
    return {"species": len(labels), "fields": fields}


def recode_store(src: str | Path, dst: str | Path, log=print) -> None:
    """Rewrite a store's tiles with the current writer (lossless: the decoded field is unchanged); the species
    table, masks and metadata are copied."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    st = Store(src)
    d, t = int(st.meta["d"]), st.tile
    for region in st.regions():
        H, W = st.shape(region)
        w = KltWriter(dst, region, H, W, d, t)
        t0 = time.time()
        for ty in range((H + t - 1) // t):
            r0, r1 = ty * t, min(H, ty * t + t)
            strip = np.zeros((r1 - r0, W, d), np.int16)
            for tx in range((W + t - 1) // t):
                q16, q8 = st.planes(region, ty, tx)
                c = np.concatenate([q16, q8.astype(np.int16)])
                strip[:, tx * t:tx * t + c.shape[2], :len(c)] = np.moveaxis(c, 0, -1)
            w.put(r0, strip)
        w.close()
        log(f"{region}: {(src / f'field_{region}.zst').stat().st_size / 1e9:.2f} -> {w.nbytes / 1e9:.2f} GB, "
            f"{time.time() - t0:.0f} s")
    for f in src.iterdir():
        if f.is_file() and not f.name.startswith(("field_", "index_")):
            shutil.copy2(f, dst / f.name)
