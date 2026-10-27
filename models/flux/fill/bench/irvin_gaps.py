"""M1: distribution-matched artificial gaps (Irvin et al. 2021, Agric. For. Meteorol. 308-309:108528).

The gap-length sampling distribution is learned per site so that artificial gaps merged with the site's real gaps
reproduce the observed gap-length distribution. This module calls the released implementation (FluxGapfill,
github.com/stanfordmlgroup/methane-gapfill-ml, Apache-2.0; Irvin et al. 2021, Zenodo 10.5281/zenodo.5515761) with
an exact, faster reformulation of its `sample_artificial_gaps` (identical output and random state; verified site by
site). M1 is drawn once over a site's whole record on its primary flux, then cut into windows, so it keeps the
paper's semantics: about `eval_frac` of the measured half-hours are hidden.
"""

import numpy as np


def learn_and_sample(flux_record: np.ndarray, eval_frac: float = 0.1, seed: int = 1000, dist: str = "CramerVonMises",
                     n_grid: int = 10, n_mc: int = 50) -> np.ndarray:
    """Boolean mask over the record (True = artificial gap on a measured half-hour), with the paper's defaults."""
    from fluxgapfill.preprocess.artificial import learn_gap_dist, sample_artificial_gaps
    flux = np.asarray(flux_record, np.float64)
    pmf = learn_gap_dist(flux_data=flux, dist=dist, n_grid=n_grid, n_mc=n_mc, seed=seed)
    masked = sample_artificial_gaps(flux_data=flux, sampling_pmf=pmf, eval_frac=eval_frac, seed=seed)
    return np.isnan(masked) & np.isfinite(flux)
