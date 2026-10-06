"""Range cards: a species' 240 m range maps stored as the model that generates them.

A rendered range map is a deterministic function of (a) the replicate MaxEnt models, (b) their thresholds,
(c) the calibration mask and (d) the predictor stack, which is shared by every species. A range card stores
(a)–(c) only:

* each replicate's ``.lambdas`` text (a few kB, the exact fitted coefficients);
* per-replicate thresholds (equal training sensitivity/specificity; 5th-percentile training presence);
* the calibration area as a list of ecoregion ids, resolved against the shared 240 m ecoregion-id layer
  (lakes = 0) that rendering itself uses, so the mask is identical by construction.

``decode`` re-renders suitability / binary_vote / binary_p5 with ``project``'s arithmetic; ``verify`` checks
byte equality against the stored rasters.
"""
from __future__ import annotations

import functools
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd

from .maxent import MaxentModel, default_device


def encode(card_path: Path, label: str, final_dir: Path, variables: list[str], thresholds_ess: list[float],
           thresholds_p5: list[float], ecoregion_ids: list[int], maps: dict) -> int:
    """Write a range card (zstd-compressed tar) and return its size in bytes. ``maps``: {region: {layer: uint8
    array}} for every region grid the species is rendered on (a flat {layer: array} means the CONUS grid). The
    SHA-256 of each map is stored so decoding can be verified without any raster file."""
    if maps and isinstance(next(iter(maps.values())), np.ndarray):
        maps = {"conus": maps}
    crs = {"conus": "EPSG:5070", "alaska": "EPSG:3338", "hawaii": "ESRI:102007"}
    grids = {r: {"crs": crs.get(r), "cell_m": 240, "shape": list(next(iter(m.values())).shape),
                 "sha256_uint8": {n: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for n, a in m.items()}}
             for r, m in maps.items()}
    meta = {"format": "deepearth-range-card/3", "label": label, "variables": variables, "grids": grids,
            "ecoregion_ids": sorted(int(i) for i in ecoregion_ids),
            "thresholds_ess": thresholds_ess, "thresholds_p5": thresholds_p5}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        def add(name, data: bytes):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
        add("meta.json", json.dumps(meta).encode())
        for lam in sorted(final_dir.glob(f"{label}_[0-9]*.lambdas")):
            add(lam.name, lam.read_bytes())
    data = zstd.ZstdCompressor(level=19).compress(buf.getvalue())
    card_path.write_bytes(data)
    return len(data)


def grids(meta: dict) -> dict:
    """Per-region grid records of a card (format 2 cards hold only the CONUS grid)."""
    if "grids" in meta:
        return meta["grids"]
    return {"conus": {"crs": "EPSG:5070", "cell_m": 240, "shape": meta["shape"], "sha256_uint8": meta["sha256_uint8"]}}


def decode(card_path: Path, stack, device: str | None = None) -> dict[str, np.ndarray]:
    """Re-render the three uint8 maps on one region grid (``stack``) from a range card."""
    device = default_device(device)
    from .project import compute_maps

    meta, reps = load(Path(card_path), device)
    outs, _ = compute_maps(reps, meta["variables"], meta["ecoregion_ids"], stack, device=device)
    return outs


def verify(card_path: Path, stacks) -> dict[str, dict[str, bool]]:
    """Decode a card on each region grid it records and check every map against its stored SHA-256.
    ``stacks``: one stack or {region: stack}."""
    raw = zstd.ZstdDecompressor().decompress(Path(card_path).read_bytes())
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        g = grids(json.loads(tar.extractfile("meta.json").read()))
    if not isinstance(stacks, dict):
        stacks = {getattr(stacks, "region", "conus"): stacks}
    out = {}
    for r, rec in g.items():
        dec = decode(card_path, stacks[r])
        out[r] = {n: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() == rec["sha256_uint8"][n] for n, a in dec.items()}
    return out


@functools.lru_cache(maxsize=64)
def load(card_path: Path, device: str | None = None) -> tuple[dict, list]:
    """A card's metadata and replicate models (cached: a card is usually decoded window after window)."""
    device = default_device(device)
    from .project import Replicate

    raw = zstd.ZstdDecompressor().decompress(Path(card_path).read_bytes())
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        meta = json.loads(tar.extractfile("meta.json").read())
        texts = {m: tar.extractfile(m).read() for m in sorted(n for n in tar.getnames() if n.endswith(".lambdas"))}
    reps = [Replicate(MaxentModel.from_text(t.decode(), variables=meta["variables"], dtype=torch.float32, device=device),
                      te, tp) for t, te, tp in zip(texts.values(), meta["thresholds_ess"], meta["thresholds_p5"])]
    return meta, reps


def _classify(meta: dict, reps: list, x: torch.Tensor) -> dict[str, np.ndarray]:
    """Suitability (1 + round(254 * median cloglog)), replicate-majority ESS binary and P5 binary (1 = absent,
    2 = present) for predictor rows ``x``: the arithmetic of ``project.compute_maps``, value for value."""
    p = torch.stack([r.model.cloglog(x) for r in reps], 0)
    med = p.median(0).values
    t_ess = torch.tensor(meta["thresholds_ess"], device=x.device)
    return {"suitability": (1 + torch.round(med * 254)).clamp(1, 255).to(torch.uint8).cpu().numpy(),
            "binary_vote": 1 + ((p >= t_ess[:, None]).float().mean(0) > 0.5).cpu().numpy().astype(np.uint8),
            "binary_p5": 1 + (med >= float(np.median(meta["thresholds_p5"]))).cpu().numpy().astype(np.uint8)}


def decode_window(card_path: Path, stack, row0: int, row1: int, col0: int, col1: int,
                  device: str | None = None) -> dict[str, np.ndarray]:
    """Decode only rows [row0, row1) x cols [col0, col1) of a card's three maps. MaxEnt evaluation is
    batch-invariant (see ``MaxentModel.linear_predictor``), so the window is byte-identical to the same window
    of the full render."""
    device = default_device(device)
    meta, reps = load(Path(card_path), device)
    ids = torch.tensor(meta["ecoregion_ids"], dtype=torch.int32, device=device)
    bands = stack.band_index(meta["variables"])
    h, w = row1 - row0, col1 - col0
    outs = {n: np.zeros((h, w), np.uint8) for n in ("suitability", "binary_vote", "binary_p5")}
    mask = torch.isin(torch.from_numpy(stack.eco[row0:row1, col0:col1].astype(np.int32)).to(device), ids)
    if not bool(mask.any()):
        return outs
    x = torch.stack([torch.from_numpy(np.ascontiguousarray(stack.data[k, row0:row1, col0:col1])).to(device)
                     for k in bands], dim=-1)
    ok = mask & torch.isfinite(x).all(-1)
    if not bool(ok.any()):
        return outs
    okc = ok.cpu().numpy()
    for n, v in _classify(meta, reps, x[ok]).items():
        outs[n][okc] = v
    return outs


def decode_cells(card_path: Path, stack, rows: np.ndarray, cols: np.ndarray,
                 device: str | None = None) -> dict[str, np.ndarray]:
    """Decode a card's three maps at arbitrary grid cells (0 = outside the calibration area or the grid).
    Only the distinct cells are evaluated, so the cost is set by the number of cells asked for, not by the
    area they span; values are identical to the full render (batch invariance)."""
    device = default_device(device)
    meta, reps = load(Path(card_path), device)
    rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
    H, W = stack.shape
    outs = {n: np.zeros(rows.shape, np.uint8) for n in ("suitability", "binary_vote", "binary_p5")}
    inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    flat = np.where(inside, rows * W + cols, -1)
    cells, inverse = np.unique(flat[inside], return_inverse=True)
    if len(cells) == 0:
        return outs
    r, c = cells // W, cells % W
    keep = np.isin(stack.eco[r, c], meta["ecoregion_ids"])
    order = np.argsort(cells[keep])                          # read the memmap in file order
    rk, ck = r[keep][order], c[keep][order]
    x = np.stack([stack.data[k][rk, ck] for k in stack.band_index(meta["variables"])], -1)
    ok = np.isfinite(x).all(-1)
    vals = {n: np.zeros(len(cells), np.uint8) for n in outs}
    if ok.any():
        idx = np.nonzero(keep)[0][order][ok]
        for n, v in _classify(meta, reps, torch.from_numpy(x[ok]).to(device)).items():
            vals[n][idx] = v
    for n in outs:
        outs[n][inside] = vals[n][inverse]
    return outs
