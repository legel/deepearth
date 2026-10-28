"""FluxFill-Bench mask families on one 180-day window.

A window is a (V, T) field with a boolean `measured` array (True where the source value is a real sensor value or a
real reanalysis or imagery value). A mask is a (V, T) boolean array: True hides the value from the method. Scores
use `mask & measured` only. Every family is drawn from a numpy Generator seeded per (site, window, family), so the
benchmark masks are reproducible and frozen.

M1 distribution-matched gaps (Irvin et al. 2021) live in `irvin_gaps.py`; this file holds M2 to M4.
"""

from typing import Iterable, Sequence

import numpy as np

STEPS_PER_DAY = 48
WINDOW_DAYS = 180
BLOCK_DAYS = (1, 7, 30, 90, 180)
"""M2 block lengths in days; 180 masks the whole window."""


def seed_for(site: str, window_start: str, family: str, k: int = 0) -> int:
    """A stable 63-bit seed from the identifiers (independent of Python's hash randomisation)."""
    import hashlib
    h = hashlib.sha256(f"{site}|{window_start}|{family}|{k}".encode()).digest()
    return int.from_bytes(h[:8], "little") >> 1


def contiguous_block(rows: Sequence[int], V: int, T: int, days: int, rng: np.random.Generator) -> np.ndarray:
    """M2: one block of `days` x 48 steps over the given variable rows, start uniform over the window."""
    n = min(days * STEPS_PER_DAY, T)
    start = int(rng.integers(0, T - n + 1))
    m = np.zeros((V, T), bool)
    m[np.asarray(rows), start:start + n] = True
    return m


def transplanted_outage(rows: Sequence[int], V: int, outage: np.ndarray) -> np.ndarray:
    """M3: a donor's all-tower-channels-missing series (a (T,) boolean from `cut_outage`) applied to every row in
    `rows`."""
    outage = np.asarray(outage, bool)
    m = np.zeros((V, outage.shape[-1]), bool)
    m[np.asarray(rows)] = outage[None]
    return m


def cut_outage(record_missing_all: np.ndarray, T: int, rng: np.random.Generator) -> np.ndarray:
    """A random T-step slice of a donor record's all-channels-missing series, drawn among slices that contain at
    least one outage (so the family always masks something). Returns a (T,) boolean."""
    rec = np.asarray(record_missing_all, bool)
    if rec.shape[0] < T:
        raise ValueError("donor record shorter than the window")
    csum = np.concatenate(([0], np.cumsum(rec)))
    starts = np.flatnonzero(csum[T:] - csum[:-T] > 0)
    if starts.size == 0:
        raise ValueError("donor record has no outage")
    s = int(rng.choice(starts))
    return rec[s:s + T].copy()


def modality_dropout(rows_of_modality: Iterable[int], V: int, T: int) -> np.ndarray:
    """M4: every value of one modality hidden for the whole window."""
    m = np.zeros((V, T), bool)
    m[np.asarray(list(rows_of_modality))] = True
    return m
