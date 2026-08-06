"""Gradient of the hash-grid output w.r.t. its per-level log2 resolutions.

Kept free of the CUDA backend so both the standard and the sparse update paths can share one
definition, and so the numerical domain guard can be tested without a compiled kernel.
"""

import torch

_LN2 = 0.6931471805599453


def resolution_grad(per_level_scale: torch.Tensor, base_resolution: torch.Tensor,
                    contrib: torch.Tensor) -> torch.Tensor:
    """Resolution gradient from the contracted per-level ``dy_dx`` term ``contrib`` [L, D].

    For scale_l[d] = exp2(pls[l,d])*base[d] - 1 the factor is ln2*(scale+1)/scale, which diverges as
    scale -> 0.  Clamping pls to [1-log2(base), 20] keeps scale >= 1, so the factor stays in (ln2, 2*ln2].
    """
    L, D = contrib.shape
    base = base_resolution.view(1, D).float()
    pls = per_level_scale.view(L, D).float().clamp_min(1.0 - torch.log2(base)).clamp_max(20.0)
    scale = torch.exp2(pls) * base - 1.0
    grad = _LN2 * (scale + 1.0) / scale * contrib
    return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).to(per_level_scale.dtype)
