"""The binary frame formats the solver writes and the viewer reads.

Two layouts, both little-endian. `SIMF` is the contract every solver writes to, laid out below.

Depth only, read by `viewer/static/js/flood.js`, whose layout comment must match::

    [0:4]                       b'SIML'
    [4:16]                      uint32 n_frames, rows, cols
    [16:16+4n]                  float32 times_min[n_frames]
    [16+4n:]                    float32 depth[n_frames][rows][cols], metres

Named fields on one grid, shared by every solver and streamed one record at a time::

    [0:4]                       b'SIMF'
    [4:8]                       uint32 version (2)
    [8:12]                      uint32 n_frames, patched on close
    [12:16]                     uint32 nx (columns, eastward)
    [16:20]                     uint32 ny (rows, southward from the origin)
    [20:24]                     uint32 nz (levels, upward; 1 for a surface field)
    [24:28]                     uint32 n_fields
    [28:32]                     uint32 sample: 0 float32, 1 uint16 (value = sample x scale), 2 float16
    [32:40]                     float64 cell_m
    [40:64]                     float64 origin x, y, z: scene metres of the north-west corner of
                                cell [row 0, col 0] and of level 0
    [64:72]                     float64 scale (1 for float32 and float16)
    [72:72+16f]                 ascii name[8] unit[8] per field, NUL padded
    [..:..+4(nz+1)]             float32 zf[nz+1] level face heights above origin z
    then n_frames records of:   float64 t_s, sample[n_fields][nz][ny][nx]

Solver arrays are north-up rasters, so row 0 is already the north edge and rows are written as
they are. A sidecar `<file>.json` carries everything that is not a number per cell.
"""

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

MAGIC = b"SIML"
HEADER = struct.Struct("<III")

FIELDS_MAGIC = b"SIMF"
FIELDS_VERSION = 2
FIELDS_HEADER = struct.Struct("<IIIIIIIddddd")
NAME_BYTES = 8
SAMPLE_DTYPES = {0: np.dtype("<f4"), 1: np.dtype("<u2"), 2: np.dtype("<f2")}


def write(path: Path, frames: Sequence[np.ndarray], times_min: Sequence[float]) -> Path:
    """Write depth frames and their timestamps.

    Args:
        path: Destination.
        frames: Depth arrays [m], all the same shape.
        times_min: Simulated minutes for each frame.

    Returns:
        `path`, for chaining.
    """
    arr = np.asarray(frames, dtype=np.float32)
    assert arr.ndim == 3, f"expected (n, rows, cols), got {arr.shape}"
    assert len(times_min) == arr.shape[0], (
        f"{len(times_min)} times for {arr.shape[0]} frames")
    with open(path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(HEADER.pack(*arr.shape))
        fh.write(np.asarray(times_min, dtype="<f4").tobytes())
        fh.write(arr.astype("<f4").tobytes())
    return path


def read(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a depth-frame file.

    Returns:
        (times_min [n], depth [n, rows, cols]).
    """
    raw = Path(path).read_bytes()
    assert raw[:4] == MAGIC, f"{path} is not a {MAGIC.decode()} file"
    n, rows, cols = HEADER.unpack(raw[4:16])
    times = np.frombuffer(raw, dtype="<f4", count=n, offset=16)
    depth = np.frombuffer(raw, dtype="<f4", count=n * rows * cols,
                          offset=16 + 4 * n).reshape(n, rows, cols)
    return times, depth


@dataclass(frozen=True)
class Header:
    """Everything a reader needs to place and scale a SIMF file.

    Args:
        n_frames: Records in the file.
        shape: (nz, ny, nx).
        fields: (name, unit) per field.
        cell_m: Cell size [m].
        origin: Scene metres (x, y, z) of the north-west corner of cell [0, 0], level 0.
        zf: Level face heights above origin z, nz + 1 of them.
        sample: Sample code, see `SAMPLE_DTYPES`.
        scale: Multiplier from stored sample to physical value.
        version: Layout version the file was read with.
    """

    n_frames: int
    shape: Tuple[int, int, int]
    fields: Tuple[Tuple[str, str], ...]
    cell_m: float
    origin: Tuple[float, float, float]
    zf: Tuple[float, ...]
    sample: int = 0
    scale: float = 1.0
    version: int = FIELDS_VERSION

    @property
    def size(self) -> int:
        """Header length in bytes."""
        faces = 4 * (self.shape[0] + 1) if self.version >= 2 else 0
        return 4 + FIELDS_HEADER.size + 2 * NAME_BYTES * len(self.fields) + faces

    @property
    def record(self) -> np.dtype:
        """One frame record."""
        return np.dtype([("t", "<f8"),
                         ("f", SAMPLE_DTYPES[self.sample], (len(self.fields),) + self.shape)])

    def pack(self) -> bytes:
        names = b"".join(n.encode("ascii").ljust(NAME_BYTES, b"\0")
                         + u.encode("ascii").ljust(NAME_BYTES, b"\0") for n, u in self.fields)
        nz, ny, nx = self.shape
        return (FIELDS_MAGIC + FIELDS_HEADER.pack(
            FIELDS_VERSION, self.n_frames, nx, ny, nz, len(self.fields), self.sample,
            self.cell_m, *self.origin, self.scale) + names
            + np.asarray(self.zf, dtype="<f4").tobytes())

    @classmethod
    def unpack(cls, raw: bytes) -> "Header":
        assert raw[:4] == FIELDS_MAGIC, f"not a {FIELDS_MAGIC.decode()} file"
        version, n, nx, ny, nz, nf, sample, cell, ox, oy, oz, scale = FIELDS_HEADER.unpack(
            raw[4:4 + FIELDS_HEADER.size])
        assert version in (1, 2), f"SIMF version {version}"
        start = 4 + FIELDS_HEADER.size
        fields = []
        for i in range(nf):
            at = start + 2 * NAME_BYTES * i
            fields.append((raw[at:at + NAME_BYTES].rstrip(b"\0").decode("ascii"),
                           raw[at + NAME_BYTES:at + 2 * NAME_BYTES].rstrip(b"\0").decode("ascii")))
        at = start + 2 * NAME_BYTES * nf
        zf = (np.frombuffer(raw, "<f4", count=nz + 1, offset=at) if version >= 2
              else np.arange(nz + 1) * cell)
        return cls(n_frames=n, shape=(nz, ny, nx), fields=tuple(fields), cell_m=cell,
                   origin=(ox, oy, oz), zf=tuple(float(v) for v in zf), sample=sample,
                   scale=scale, version=version)


class Writer:
    """Streams named field frames to a SIMF file and its sidecar.

    Args:
        path: Destination; the sidecar is `path` with `.json` appended.
        shape: (rows, cols) or (nz, rows, cols) of every field, row 0 north.
        fields: (name, unit) per field, at most 8 ASCII bytes each.
        cell_m: Cell size [m].
        origin: Scene metres (x, y, z) of the north-west corner of cell [0, 0], level 0.
        sidecar: Everything the viewer needs that is not a number per cell: t0, timezone,
            site, epsg, anchor_utm, surface, fields, marks, provenance.
        zf: Level face heights above origin z; None means one level `cell_m` thick.
    """

    def __init__(self, path: Path, shape: Sequence[int], fields: Sequence[Tuple[str, str]],
                 cell_m: float, origin: Tuple[float, float, float], sidecar: Dict[str, object],
                 zf: Optional[Sequence[float]] = None):
        shape = tuple(shape) if len(shape) == 3 else (1,) + tuple(shape)
        for name, unit in fields:
            assert len(name.encode("ascii")) <= NAME_BYTES, name
            assert len(unit.encode("ascii")) <= NAME_BYTES, unit
        zf = tuple(float(v) for v in (zf if zf is not None else np.arange(shape[0] + 1) * cell_m))
        assert len(zf) == shape[0] + 1, f"{len(zf)} faces for {shape[0]} levels"
        missing = SIDECAR_KEYS - set(sidecar)
        assert not missing, f"sidecar lacks {sorted(missing)}"
        self.path = Path(path)
        self.header = Header(n_frames=0, shape=shape, fields=tuple(fields), cell_m=float(cell_m),
                             origin=tuple(float(v) for v in origin), zf=zf)
        self.n = 0
        Path(f"{self.path}.json").write_text(json.dumps(sidecar, indent=1))
        self.fh = open(self.path, "wb")
        self.fh.write(self.header.pack())
        self.fh.flush()

    def append(self, t_s: float, fields: np.ndarray) -> None:
        """Write one record.

        Args:
            t_s: Seconds since the start of the run.
            fields: [n_fields, rows, cols] or [n_fields, nz, rows, cols] array, row 0 north.
        """
        arr = np.asarray(fields, dtype=np.float32).reshape(
            (len(self.header.fields),) + self.header.shape)
        self.fh.write(struct.pack("<d", float(t_s)))
        self.fh.write(arr.astype("<f4").tobytes())
        self.fh.flush()
        self.n += 1

    def close(self) -> Path:
        """Patch the frame count into the header and close the file."""
        self.fh.seek(0)
        self.fh.write(Header(**{**vars(self.header), "n_frames": self.n}).pack())
        self.fh.close()
        return self.path

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


SIDECAR_KEYS = {"t0", "timezone", "site", "epsg", "anchor_utm", "surface", "fields", "marks",
                "provenance"}


def sidecar(t0: str, timezone: str, site: str, epsg: int, anchor_utm: Sequence[float],
            fields: Dict[str, Dict[str, object]], provenance: Dict[str, object],
            surface: str = "ground", marks: Sequence[Dict[str, object]] = ()) -> Dict[str, object]:
    """The sidecar document a `Writer` requires."""
    return {"t0": t0, "timezone": timezone, "site": site, "epsg": epsg,
            "anchor_utm": list(anchor_utm), "surface": surface, "fields": fields,
            "marks": list(marks), "provenance": provenance}


def write_fields(path: Path, fields: Sequence[Tuple[str, str]], times_s: Sequence[float],
                 frames: Sequence[np.ndarray], cell_m: float, origin: Tuple[float, float, float],
                 sidecar: Dict[str, object]) -> Path:
    """Write every frame at once. `frames` are [n_fields, rows, cols] arrays."""
    assert len(times_s) == len(frames), f"{len(times_s)} times for {len(frames)} frames"
    with Writer(path, frames[0].shape[1:], fields, cell_m, origin, sidecar) as w:
        for t, f in zip(times_s, frames):
            w.append(t, f)
    return Path(path)


def read_fields(path: Path) -> Tuple[Header, np.ndarray, np.ndarray]:
    """Read a SIMF file, version 1 or 2.

    Returns:
        (header, times_s [n] float64, data [n, n_fields, nz, ny, nx] float32).
    """
    raw = Path(path).read_bytes()
    assert raw[:4] == FIELDS_MAGIC, f"{path} is not a {FIELDS_MAGIC.decode()} file"
    header = Header.unpack(raw)
    n = header.n_frames or (len(raw) - header.size) // header.record.itemsize
    recs = np.frombuffer(raw, dtype=header.record, count=n, offset=header.size)
    data = recs["f"].astype(np.float32) * np.float32(header.scale) if header.sample else recs["f"]
    return header, recs["t"].astype(np.float64), data
