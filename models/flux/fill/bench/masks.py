"""FluxFill-Bench mask families on one 180-day window.

A window is a (V, T) field with a boolean `measured` array (True where the source value is a real sensor value or a
real reanalysis or imagery value). A mask is a (V, T) boolean array: True hides the value from the method. Scores
use `mask & measured` only. Every family is drawn from a numpy Generator seeded per (site, window, family), so the
benchmark masks are reproducible and frozen.

Families (docs/BENCHMARK.md section 5):
- M1, distribution-matched gaps (Irvin et al. 2021): learned and drawn over a site's whole record in `irvin_gaps.py`;
  `window_from_record` cuts the record mask to a window.
- M2, one contiguous block on all tower channels: 1 to 180 days (`contiguous_block`), and the sub-daily blocks of 2 h
  and 6 h (`contiguous_steps`).
- M3, a transplanted real outage (`cut_outage`, `transplanted_outage`).
- M4, one modality hidden for the whole window (`modality_dropout`).
- M5, mixed: an independent block per variable with a log-uniform length and its own placement (`mixed_blocks`),
  plus one modality (tower meteorology, reanalysis or fluxes) hidden for the whole window, chosen by the seed
  (`M5_MODALITIES`).
"""

from typing import Iterable, Sequence

import numpy as np

STEPS_PER_DAY = 48
WINDOW_DAYS = 180
BLOCK_DAYS = (1, 7, 30, 90, 180)
"""M2 block lengths in days; 180 masks the whole window."""
SUBDAILY_HOURS = (2, 6)
"""Sub-daily M2 block lengths in hours (4 and 12 half-hours)."""
M5_MIN_STEPS, M5_MAX_STEPS = 1, 90 * STEPS_PER_DAY
"""M5 block lengths: log-uniform from one half-hour to 90 days."""
M5_MODALITIES = ("tower_met", "reanalysis", "fluxes")
"""The modality M5 hides for the whole window: tower meteorology and soil, reanalysis drivers, or all tower fluxes
(NEE, LE, H, FCH4, G). Imagery is not a window row in the method protocol, so its dropout is not part of M5 in
release 0.2."""


def seed_for(site: str, window_start: str, family: str, k: int = 0) -> int:
    """A stable 63-bit seed from the identifiers (independent of Python's hash randomisation)."""
    import hashlib
    h = hashlib.sha256(f"{site}|{window_start}|{family}|{k}".encode()).digest()
    return int.from_bytes(h[:8], "little") >> 1


def contiguous_block(rows: Sequence[int], V: int, T: int, days: int, rng: np.random.Generator) -> np.ndarray:
    """M2: one block of `days` x 48 steps over the given variable rows, start uniform over the window."""
    return contiguous_steps(rows, V, T, days * STEPS_PER_DAY, rng)


def contiguous_steps(rows: Sequence[int], V: int, T: int, steps: int, rng: np.random.Generator) -> np.ndarray:
    """One block of `steps` half-hours over the given variable rows, start uniform over the window."""
    n = min(int(steps), T)
    start = int(rng.integers(0, T - n + 1))
    m = np.zeros((V, T), bool)
    m[np.asarray(rows), start:start + n] = True
    return m


def mixed_blocks(rows: Sequence[int], V: int, T: int, rng: np.random.Generator, lo: int = M5_MIN_STEPS,
                 hi: int = M5_MAX_STEPS):
    """M5 blocks: for each row, in order, one block of length round(exp(U(ln lo, ln hi))) half-hours (log-uniform)
    with its own uniform start. Returns the (V, T) mask and the {row: (start, length)} placements."""
    m = np.zeros((V, T), bool)
    place = {}
    for r in rows:
        n = int(np.clip(np.rint(np.exp(rng.uniform(np.log(lo), np.log(hi)))), lo, min(hi, T)))
        a = int(rng.integers(0, T - n + 1))
        m[r, a:a + n] = True
        place[int(r)] = (a, n)
    return m, place


def window_from_record(record_mask: np.ndarray, record_t: np.ndarray, window_t: np.ndarray) -> np.ndarray:
    """(T,) boolean of a window from a record mask on the record's own time grid (`record_t`, sorted datetime64): each
    window half-hour takes the value of the record step that contains it (an hourly record step covers two
    half-hours); outside the record it is False."""
    rt = np.asarray(record_t, "datetime64[m]")
    wt = np.asarray(window_t, "datetime64[m]")
    step = np.median(np.diff(rt)) if rt.size > 1 else np.timedelta64(30, "m")
    j = np.searchsorted(rt, wt, side="right") - 1
    ok = (j >= 0) & (wt < rt[np.clip(j, 0, None)] + step)
    out = np.zeros(wt.shape, bool)
    out[ok] = np.asarray(record_mask, bool)[j[ok]]
    return out


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
