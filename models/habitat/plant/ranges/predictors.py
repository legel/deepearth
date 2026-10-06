"""Predictor stack (Daru 2024 step 6b): WorldClim 2.1 bio1–19 + elevation at ~1 km (global).

The 20 global layers are packed once into a single pixel-interleaved float32 memory map (rows, cols, 20) so
that extracting all predictors at a point is one contiguous read and a calibration window is an array slice.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

VARIABLES = [f"wc2.1_30s_bio_{i}" for i in range(1, 20)] + ["wc2.1_30s_elev"]
CELL = 1.0 / 120.0                       # 30 arc-seconds
NROW, NCOL = 21600, 43200                # -180..180, 90..-90


class GlobalStack:
    def __init__(self, path: str | Path):
        meta = json.loads(Path(path).with_suffix(".json").read_text())
        self.variables = meta["variables"]
        self.data = np.memmap(path, dtype=np.float32, mode="r", shape=(NROW, NCOL, len(self.variables)))

    @staticmethod
    def rowcol(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = np.clip(np.floor((90.0 - np.asarray(lat)) / CELL).astype(np.int64), 0, NROW - 1)
        c = np.clip(np.floor((np.asarray(lon) + 180.0) / CELL).astype(np.int64), 0, NCOL - 1)
        return r, c

    def at(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        r, c = self.rowcol(lon, lat)
        return np.asarray(self.data[r, c, :])

    def focal(self, lon: np.ndarray, lat: np.ndarray, k: int = 9) -> np.ndarray:
        """Mean of every band over the k x k block of 30" cells centred on each point (NaN cells ignored): the
        neighbourhood context of a location (k = 9: ~8 km N-S; ledger L16)."""
        r, c = self.rowcol(lon, lat)
        h = k // 2
        out = np.empty((len(r), len(self.variables)))
        for i, (ri, ci) in enumerate(zip(r, c)):
            blk = np.asarray(self.data[max(ri - h, 0):ri + h + 1, max(ci - h, 0):ci + h + 1, :], dtype=np.float64)
            with np.errstate(invalid="ignore"):
                out[i] = np.nanmean(blk.reshape(-1, blk.shape[-1]), axis=0) if np.isfinite(blk).any() else np.nan
        return out

    @staticmethod
    def cell_centres(r: np.ndarray, c: np.ndarray) -> np.ndarray:
        return np.column_stack([-180.0 + (c + 0.5) * CELL, 90.0 - (r + 0.5) * CELL])


def build(worldclim_dir: str | Path, out_path: str | Path, rows_per_block: int = 1200) -> None:
    out_path = Path(out_path)
    mm = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(NROW, NCOL, len(VARIABLES)))
    srcs = [rasterio.open(Path(worldclim_dir) / f"{v}.tif") for v in VARIABLES]
    for s in srcs:
        assert (s.height, s.width) == (NROW, NCOL), s.name
    for r0 in range(0, NROW, rows_per_block):
        h = min(rows_per_block, NROW - r0)
        block = np.empty((h, NCOL, len(VARIABLES)), np.float32)
        for k, s in enumerate(srcs):
            a = s.read(1, window=((r0, r0 + h), (0, NCOL))).astype(np.float32)
            if s.nodata is not None:
                a[a == s.nodata] = np.nan
            a[a < -1e30] = np.nan
            block[:, :, k] = a
        mm[r0:r0 + h] = block
    mm.flush()
    out_path.with_suffix(".json").write_text(json.dumps({"variables": VARIABLES, "shape": [NROW, NCOL, len(VARIABLES)],
                                                         "cell_deg": CELL, "origin": [-180, 90], "dtype": "float32"}))


class _Bands:
    """Band-indexed view over several 2-D arrays (``data[k]``, ``data[k, rows, cols]``), so a stack can combine
    the WorldClim memmap with separately stored layers without copying them into one file."""

    def __init__(self, bands: list):
        self.bands = bands

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.bands[key[0]][key[1:]]
        return self.bands[key]


class ConusStack:
    """The CONUS 240 m predictor stack (band-sequential float32 memmap) plus the shared ecoregion-id layer held
    in memory with each ecoregion's bounding window, so a species renders only the rows/columns its
    calibration ecoregions span."""

    def __init__(self, directory: str | Path, stack: str | Path | None = None, extra: dict | None = None):
        """``stack``: path of a band-sequential float32 memmap with a sibling .json (default: the WorldClim
        CONUS stack in ``directory``; the fine stack is work/fine/conus240_fine.f32). ``extra``: further
        variables, name -> single-band float32 memmap on the same grid (e.g. soil), appended as bands."""
        d = Path(directory)
        if stack is None:                                # <region>240_stack.f32: conus240, alaska240, hawaii240
            found = sorted(d.glob("*240_stack.f32"))
            stack = found[0] if found else d / "conus240_stack.f32"
        stack = Path(stack)
        meta = json.loads(stack.with_suffix(".json").read_text())
        self.region = d.name.replace("240", "")
        self.crs = meta.get("crs", "EPSG:5070")
        self.shape = tuple(meta["shape"][1:])
        base = np.memmap(stack, dtype=np.float32, mode="r", shape=tuple(meta["shape"]))
        extra = extra or {}
        self.variables = list(meta["variables"]) + list(extra)
        self.data = _Bands([base[k] for k in range(base.shape[0])] +
                           [np.memmap(p, dtype=np.float32, mode="r", shape=self.shape) for p in extra.values()])
        with rasterio.open(d / "ecoregion_id_conus240.tif") as r:
            self.eco = r.read(1)
            self.profile = r.profile.copy()
        self.bbox = {}
        for i in np.unique(self.eco):
            if i == 0:
                continue
            rows = np.nonzero((self.eco == i).any(1))[0]
            cols = np.nonzero((self.eco == i).any(0))[0]
            self.bbox[int(i)] = (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)

    def window(self, ecoregion_ids) -> tuple[int, int, int, int] | None:
        boxes = [self.bbox[i] for i in ecoregion_ids if i in self.bbox]
        if not boxes:
            return None
        b = np.array(boxes)
        return int(b[:, 0].min()), int(b[:, 1].max()), int(b[:, 2].min()), int(b[:, 3].max())

    def band_index(self, variables) -> list[int]:
        return [self.variables.index(v) for v in variables]


GridStack = ConusStack                                   # the same class serves every region grid

REGIONS = ("conus", "alaska", "hawaii")


def region_stacks(work: str | Path, regions=REGIONS) -> dict:
    """The 240 m grid stacks that exist under ``work`` (work/<region>240), keyed by region."""
    return {r: GridStack(Path(work) / f"{r}240") for r in regions if (Path(work) / f"{r}240").exists()}


if __name__ == "__main__":
    import sys
    build(sys.argv[1], sys.argv[2])
