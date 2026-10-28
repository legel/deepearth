"""Training masks for FluxFill, drawn from the benchmark mask families (docs/BENCHMARK.md section 5) plus the states
that masked generation passes through.

A mask is a (V, T) boolean array over the window's tabular rows (tower, reanalysis, reanalysis-land) (True hides the value); the imagery row has its own
(D,) day mask. Families:
- block_all: one contiguous block of every tower channel, 1 to 180 days, log-uniform length, any start (M2, M3);
- block_var: an independent block per tower variable, 1 half-hour to 90 days (the per-variable M2 variant);
- short: many gaps of 1 to 64 half-hours (the short classes of M1), per variable or shared by all tower channels;
- r3: every tower channel hidden for the whole window (regime R3);
- mixed: block_all plus short gaps.
Modality dropout (M4) is drawn on top: all tower meteorology, all ONEFlux reanalysis, all reanalysis-land drivers, or
all imagery hidden for the window (independently; never a reanalysis modality under r3).
Reveal (for masked generation, Li et al. 2024 arXiv:2406.11838): with probability `p_reveal`, a random fraction
r ~ U(0, 1) of the hidden tower day-tokens is shown again, so training covers the partly generated states that random-
order generation visits.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from data import IS_LAND, IS_REAN, IS_TOWER, IS_TOWER_MET, STEPS_PER_DAY, T_WIN, WINDOW_DAYS


@dataclass
class MaskConfig:
    p_block_all: float = 0.35
    p_block_var: float = 0.20
    p_short: float = 0.15
    p_r3: float = 0.10
    p_mixed: float = 0.20
    p_drop_met: float = 0.15
    p_drop_rean: float = 0.15
    p_drop_land: float = 0.15
    p_drop_img: float = 0.30
    p_img_acq: float = 0.15      # each imagery day hidden (a reconstruction target)
    p_reveal: float = 0.5


FAMILIES = ("block_all", "block_var", "short", "r3", "mixed")


def _loguniform_int(rng, lo: int, hi: int) -> int:
    return int(np.clip(np.round(np.exp(rng.uniform(np.log(lo), np.log(hi)))), lo, hi))


def _block(rng, T: int, n: int) -> slice:
    a = int(rng.integers(0, T - n + 1))
    return slice(a, a + n)


def draw(rng: np.random.Generator, present: np.ndarray, cfg: MaskConfig = MaskConfig()) -> Tuple[np.ndarray, np.ndarray, str]:
    """One training mask. present: (V,) rows the site measures in the window. Returns hide (V, T), imagery day
    hide (D,) and the family name."""
    V, T = present.shape[0], T_WIN
    hide = np.zeros((V, T), bool)
    tower = np.flatnonzero(present & IS_TOWER)
    p = np.array([cfg.p_block_all, cfg.p_block_var, cfg.p_short, cfg.p_r3, cfg.p_mixed])
    fam = FAMILIES[int(rng.choice(len(FAMILIES), p=p / p.sum()))]
    if fam in ("block_all", "mixed"):
        hide[tower, _block(rng, T, _loguniform_int(rng, STEPS_PER_DAY, T))] = True
    if fam == "block_var":
        for v in tower:
            if rng.random() < 0.5:
                hide[v, _block(rng, T, _loguniform_int(rng, 1, 90 * STEPS_PER_DAY))] = True
        if not hide.any() and tower.size:
            v = rng.choice(tower)
            hide[v, _block(rng, T, _loguniform_int(rng, 1, 90 * STEPS_PER_DAY))] = True
    if fam in ("short", "mixed"):
        shared = rng.random() < 0.5
        for v in ([tower] if shared else [[v] for v in tower]):
            for _ in range(rng.poisson(12)):
                hide[np.asarray(v), _block(rng, T, _loguniform_int(rng, 1, 64))] = True
    if fam == "r3":
        hide[tower] = True
    reveal = rng.random() < cfg.p_reveal and fam != "r3"
    if reveal:
        r = rng.uniform()
        tok = hide.reshape(V, WINDOW_DAYS, STEPS_PER_DAY).any(-1)
        show = tok & (rng.random(tok.shape) < r)
        if show.sum() < tok.sum():      # keep at least one hidden token
            hide &= ~np.repeat(show, STEPS_PER_DAY, axis=1)
    if rng.random() < cfg.p_drop_met:
        hide[present & IS_TOWER_MET] = True
    if fam != "r3" and rng.random() < cfg.p_drop_rean:
        hide[present & IS_REAN] = True
    if fam != "r3" and rng.random() < cfg.p_drop_land:
        hide[present & IS_LAND] = True
    img_hide = np.full(WINDOW_DAYS, rng.random() < cfg.p_drop_img)
    img_hide |= rng.random(WINDOW_DAYS) < cfg.p_img_acq
    return hide, img_hide, fam + ("+reveal" if reveal else "")
