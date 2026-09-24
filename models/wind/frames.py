"""The time-series binaries the solvers write and the viewer reads, little-endian throughout.

`SIML`, the 2D scalar frames `models/hydro/frames.py` defines, byte for byte::

    [0:4]                       b'SIML'
    [4:16]                      uint32 n_frames, rows, cols
    [16:16+4n]                  float32 times_min[n_frames]
    [16+4n:]                    float32 depth[n_frames][rows][cols], metres

`SIMF`, named fields on one grid, shared by every solver and streamed one record at a time.
The header is `models/hydro`'s; version 2 appends the level faces so a stretched vertical can
be declared::

    [0:4]                       b'SIMF'
    [4:8]                       uint32 version (2)
    [8:12]                      uint32 n_frames
    [12:16]                     uint32 nx (columns, eastward)
    [16:20]                     uint32 ny (rows, southward)
    [20:24]                     uint32 nz (levels, upward; 1 for a surface field)
    [24:28]                     uint32 n_fields
    [28:32]                     uint32 sample code: 0 float32, 1 uint16 scaled by `scale`
    [32:40]                     float64 cell_m
    [40:64]                     float64 origin x, y, z: scene metres of the top-left corner of
                                cell [row 0, col 0] and of level 0; rows run south from y
    [64:72]                     float64 scale (value = sample * scale; 1 for float32, float16)
    [72:72+16f]                 ascii name[8] unit[8] per field, NUL padded
    [..:..+4(nz+1)]             version 2: float32 zf[nz+1], level face heights above origin z;
                                NaN in both faces of a level marks one that follows the surface
    then n_frames records of:   float64 t_s, sample[n_fields][nz][ny][nx]

Sample code 2 is float16. A version 1 file declares no faces; its levels are `cell_m` thick.
`n_frames` is written on close, so a writer can stream frames it has not counted yet. The
sidecar `<file>.json` is required and carries what is not a number per cell: `t0`, `timezone`,
`site`, `epsg`, `anchor_utm`, `surface`, `fields`, `marks`, `provenance`.
Wind writes six float32 fields: u, v, w in m/s and vort_x,
vort_y, vort_z in 1/s.
"""

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MAGIC = b"SIML"
HEADER = struct.Struct("<III")

FIELDS_MAGIC = b"SIMF"
FIELDS_VERSION = 2
FIELDS_HEADER = struct.Struct("<IIIIIIIddddd")
NAME_BYTES = 8
SAMPLE_DTYPES = {0: np.dtype("<f4"), 1: np.dtype("<u2"), 2: np.dtype("<f2")}
EPSG = 32610
"""WGS84 / UTM zone 10N, the scene projection."""

WIND_FIELDS = (("u", "m/s"), ("v", "m/s"), ("w", "m/s"),
               ("vort_x", "1/s"), ("vort_y", "1/s"), ("vort_z", "1/s"))


def write(path: Path, frames: Sequence[np.ndarray], times_min: Sequence[float]) -> Path:
    """Write 2D scalar frames and their timestamps.

    Args:
        path: Destination.
        frames: Arrays [m], all the same (rows, cols) shape.
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
    """Read a 2D scalar frame file.

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

    Attributes:
        n_frames: Records in the file.
        shape: (nz, ny, nx).
        fields: (name, unit) per field, at most 8 ASCII bytes each.
        cell_m: Cell size [m].
        origin: Scene metres (x, y, z) of the top-left corner of cell [0, 0], level 0.
        zf: Level face heights above `origin[2]` [m], nz + 1 of them.
        sample: Sample code, see `SAMPLE_DTYPES`.
        scale: Multiplier from stored sample to physical value.
    """

    n_frames: int
    shape: Tuple[int, int, int]
    fields: Tuple[Tuple[str, str], ...]
    cell_m: float
    origin: Tuple[float, float, float]
    zf: Tuple[float, ...]
    sample: int = 0
    scale: float = 1.0

    @property
    def size(self) -> int:
        """Header length in bytes."""
        return 4 + FIELDS_HEADER.size + 2 * NAME_BYTES * len(self.fields) + 4 * (self.shape[0] + 1)

    @property
    def record(self) -> np.dtype:
        """One frame record."""
        return np.dtype([("t", "<f8"),
                         ("f", SAMPLE_DTYPES[self.sample], (len(self.fields),) + self.shape)])

    def pack(self) -> bytes:
        assert len(self.zf) == self.shape[0] + 1, f"{len(self.zf)} faces for {self.shape[0]} levels"
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
        assert version in (1, FIELDS_VERSION), f"SIMF version {version}"
        start = 4 + FIELDS_HEADER.size
        fields = []
        for i in range(nf):
            at = start + 2 * NAME_BYTES * i
            fields.append((raw[at:at + NAME_BYTES].rstrip(b"\0").decode("ascii"),
                           raw[at + NAME_BYTES:at + 2 * NAME_BYTES].rstrip(b"\0").decode("ascii")))
        at = start + 2 * NAME_BYTES * nf
        zf = (np.frombuffer(raw, "<f4", count=nz + 1, offset=at) if version == 2
              else cell * np.arange(nz + 1))
        return cls(n_frames=n, shape=(nz, ny, nx), fields=tuple(fields), cell_m=cell,
                   origin=(ox, oy, oz), zf=tuple(float(z) for z in zf), sample=sample,
                   scale=scale)


class Writer:
    """Streams field frames to a SIMF file, patching `n_frames` on close.

    Solver arrays run north with the row index; records are written with rows running south
    from the origin, so `add` flips them.

    Args:
        path: Destination.
        header: Layout; its `n_frames` is ignored and counted.
    """

    def __init__(self, path: Path, header: Header):
        self.path, self.header, self.n = Path(path), header, 0
        self.fh = open(self.path, "wb")
        self.fh.write(header.pack())

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def add(self, t_s: float, fields: np.ndarray) -> None:
        """Append one record of shape (n_fields, nz, ny, nx), rows running north."""
        shape = (len(self.header.fields),) + self.header.shape
        assert fields.shape == shape, f"expected {shape}, got {fields.shape}"
        self.fh.write(struct.pack("<d", t_s))
        self.fh.write(np.ascontiguousarray(fields[:, :, ::-1, :] / self.header.scale)
                      .astype(SAMPLE_DTYPES[self.header.sample]).tobytes())
        self.n += 1

    def close(self) -> Path:
        if not self.fh.closed:
            self.fh.seek(8)
            self.fh.write(struct.pack("<I", self.n))
            self.fh.close()
        return self.path


def write_fields(path: Path, header: Header, times_s: Sequence[float],
                 frames: Sequence[np.ndarray]) -> Path:
    """Write every frame at once."""
    with Writer(path, header) as w:
        for t, f in zip(times_s, frames):
            w.add(t, f)
    return w.path


def sidecar(path: Path, t0: str, timezone: str, site: str, anchor_utm: Sequence[float],
            fields: Dict[str, Dict[str, object]], marks: List[Dict[str, object]],
            provenance: Dict[str, object], surface: str = "ground") -> Path:
    """Write the required `<path>.json` beside a SIMF file.

    Args:
        path: The SIMF file.
        t0: ISO 8601 instant with offset that every record's `t_s` is added to.
        timezone: IANA zone the viewer displays in.
        site: Site registry key.
        anchor_utm: (E, N, h) of the scene origin in `EPSG`.
        fields: Per field name, its display `domain` [lo, hi] and `lut`.
        marks: [{"t_s", "label"}] instants worth labelling.
        provenance: How the frames were made.
        surface: Terrain a surface field or a NaN level sits on: "ground" or "dsmtop".

    Returns:
        The sidecar path.
    """
    out = Path(str(path) + ".json")
    out.write_text(json.dumps({
        "t0": t0, "timezone": timezone, "site": site, "epsg": EPSG,
        "anchor_utm": list(anchor_utm), "surface": surface, "fields": fields,
        "marks": marks, "provenance": provenance}, indent=1))
    return out


def compress(path: Path) -> Dict[str, int]:
    """Write `<path>.gz` beside a product, for a viewer fetching it over the wire.

    Returns:
        Bytes before and after.
    """
    import gzip
    import shutil

    out = Path(str(path) + ".gz")
    with open(path, "rb") as src, gzip.open(out, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)
    return {"bytes": path.stat().st_size, "gz_bytes": out.stat().st_size}


def read_fields(path: Path, frame: Optional[int] = None) -> Tuple[Header, np.ndarray, np.ndarray]:
    """Read a SIMF file, or one frame of it, with rows flipped back to run north.

    Returns:
        (header, t_s [n], values [n, n_fields, nz, ny, nx]) in physical units.
    """
    raw = Path(path).read_bytes()
    header = Header.unpack(raw)
    count = header.n_frames if frame is None else 1
    offset = header.size + (0 if frame is None else frame * header.record.itemsize)
    records = np.frombuffer(raw, dtype=header.record, count=count, offset=offset)
    values = records["f"].astype(np.float32) * header.scale
    return header, records["t"].copy(), np.ascontiguousarray(values[:, :, :, ::-1, :])
