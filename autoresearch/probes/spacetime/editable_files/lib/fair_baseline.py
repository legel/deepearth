"""The fair baseline — what every `vs RFF` gain is measured against.

This is not a helper. It defines what "fair-gain" MEANS on this board, and therefore what every EARNING
or ENCODER-LIMITED read is asserting. It lives apart from probe.py so it can be tested without CUDA,
because a baseline nobody can test is how the previous one stayed broken.
"""
import numpy as np
import torch

# The control's width is now FIXED instead of tracking the encoder's.
#
# Both call sites used to pass `e4d.shape[1]`, so the baseline's width moved whenever any arm changed
# the encoder's output width -- and RFF accuracy is non-monotone in width. Measured: padding the encoder
# with columns of LITERAL ZEROS, adding no information at all, moved family_from_spacetime's share from
# 20.7% (dim 2592) to 27.2% (dim 3024) to 15.1% (dim 3744). Since `share = fair_gain / score` is the
# number that chooses DATA vs ARCHITECTURE, an unknown fraction of every EARNING / ENCODER-LIMITED read
# on this board was an artifact of output width.
#
# 2592 is the width family_from_spacetime's record was set at. It is a PROTOCOL CONSTANT, not a
# per-capability fit: rows whose encoder is narrower (flowering_peak_month is 144) now face a control
# far wider than themselves. That is deliberate -- the control is the same everywhere, so gains are
# comparable ACROSS rows for the first time -- but it means every stored gain is non-comparable and the
# whole board re-baselines under v4.
FAIR_CONTROL_DIM = 2592


def fair_rff(rn: np.ndarray, out_dim: int, train_mask=None, seed: int = 0,
             bandwidths=(1.0, 4.0, 16.0, 64.0, 256.0, 1024.0)):
    """A random-Fourier control that is actually FAIR.

    The old control was `rn @ N(0, 8)` where rn is (lat/90, lon/180): coordinates normalized to the
    GLOBE. On a regional corpus that is degenerate. California spans ~9.5 deg of latitude, i.e. ~0.05 of
    the normalized range, so at sigma=8 the projection varies ~0.04 CYCLES across the entire dataset --
    every sample gets nearly the same feature. Measured: it scored 0.008, BELOW the raw-coordinate
    baseline at 0.0166. A nonlinear control that loses to raw coordinates is not a control; it is a
    handicap, and every `vs RFF` gain computed against it was inflated by that handicap.

    Two fixes, both of which just extend to the baseline the courtesy the encoder already gets:

      1. Normalize to the TRAIN extent, exactly as the encoder's GeoAdaptiveRange does, so the control
         sees the data at its actual scale rather than as a speck of the globe. Fit on train rows only —
         using the full extent would leak the evaluation range into the control's features.
      2. Select the bandwidth. The encoder gets its hyperparameters chosen; a baseline pinned to one
         arbitrary sigma is not the strongest fair baseline, it is a straw man. Pick the sigma that
         maximizes the control's own held-out fit, and let the encoder beat THAT.

    Returns (features, chosen_sigma). Selection is on the same split the arms are scored on, which is
    generous to the baseline by design: an encoder gain that survives a baseline tuned on the evaluation
    split is not an artifact of the baseline being weak.
    """
    rn = np.asarray(rn, dtype=np.float32)
    fit = np.ones(len(rn), dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool)
    lo = rn[fit].min(0)
    span = np.maximum(rn[fit].max(0) - lo, 1e-6)
    scaled = ((rn - lo) / span * 2.0 - 1.0).astype(np.float32)      # train extent -> [-1, 1]
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, (scaled.shape[1], max(out_dim // 2, 1))).astype(np.float32)
    return scaled, base, tuple(bandwidths)


def _rff_features(scaled: np.ndarray, base: np.ndarray, sigma: float) -> torch.Tensor:
    proj = scaled @ (base * float(sigma))
    return torch.tensor(np.concatenate([np.sin(proj), np.cos(proj)], 1).astype(np.float32))
