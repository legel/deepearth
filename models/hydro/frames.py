"""The depth-frame binary the solver writes and the viewer reads.

One format with two writers and three readers is how a format drifts, so it lives here once.
The JavaScript reader in `viewer/static/js/flood.js` is the fourth participant and cannot share
this code; its layout comment must be kept in step with `HEADER` below.

Layout, little-endian throughout::

    [0:4]                       b'SIML'
    [4:16]                      uint32 n_frames, rows, cols
    [16:16+4n]                  float32 times_min[n_frames]
    [16+4n:]                    float32 depth[n_frames][rows][cols], metres
"""

import struct
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

MAGIC = b"SIML"
HEADER = struct.Struct("<III")


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
    """Read a frame file.

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
