"""A measured rain series (a flux tower's P, a rain gauge) as the solver's forcing.

The file is CSV with a header, one row per interval:

    time,rain_mm
    2022-09-28T00:00:00Z,0.0
    2022-09-28T01:00:00Z,4.2

`time` is the interval's start in ISO 8601 (UTC unless an offset is given); `rain_mm` is the depth that fell between
it and the next row's time, the last row over the series' median spacing. Hourly or finer. The depth is integrated
exactly onto the solver's steps, so the rain the solver applies totals the series' own.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np


def read(path: Path) -> Tuple[str, np.ndarray, np.ndarray]:
    """(start as ISO 8601, interval starts [s] from it, depth [mm] per interval)."""
    with open(path, newline="") as fh:
        rows = [(r["time"].strip(), float(r["rain_mm"] or 0.0)) for r in csv.DictReader(fh)]
    assert rows, f"{path} holds no rows"
    stamps = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t, _ in rows]
    stamps = [s if s.tzinfo else s.replace(tzinfo=timezone.utc) for s in stamps]
    start = np.array([(s - stamps[0]).total_seconds() for s in stamps])
    assert np.all(np.diff(start) > 0), f"{path}: times must increase"
    depth = np.array([d for _, d in rows])
    assert np.all(depth >= 0), f"{path}: negative rain"
    return stamps[0].isoformat(), start, depth


def rates(start_s: np.ndarray, depth_mm: np.ndarray, dt_s: float) -> np.ndarray:
    """Rain rate [m/s] on each solver step of `dt_s`: the cumulative depth interpolated to the step edges, so each
    interval's rain falls uniformly across it and the steps' total is the series' total."""
    start = np.asarray(start_s, dtype=float) - float(start_s[0])
    last = float(np.median(np.diff(start))) if len(start) > 1 else 3600.0
    edges_in = np.append(start, start[-1] + last)
    cum_in = np.concatenate([[0.0], np.cumsum(depth_mm)]) / 1000.0
    n = int(np.ceil(edges_in[-1] / dt_s - 1e-9))
    cum = np.interp(np.arange(n + 1) * dt_s, edges_in, cum_in)
    return np.diff(cum) / dt_s
