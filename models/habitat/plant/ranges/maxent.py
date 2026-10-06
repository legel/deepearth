"""Evaluate fitted MaxEnt models (maxent.jar ``.lambdas`` files) on GPU with PyTorch.

A MaxEnt model is a log-linear density over environmental space. maxent.jar writes the fitted model as a
``.lambdas`` file: one line per feature (``name, lambda, min, max``) followed by four normalizing constants.
This module parses that file into tensors and reproduces maxent.jar's raw, logistic and cloglog outputs for
any number of locations, so a range map can be rendered at any resolution directly from the few kilobytes of
coefficients instead of being stored as a raster.

Feature types (as written by maxent.jar 3.4.x):
    ``var``          linear:          (x - min) / (max - min)
    ``var^2``        quadratic:       ((x^2) - min) / (max - min)
    ``a*b``          product:         ((a * b) - min) / (max - min)
    ``(t<var)``      threshold:       1 if x > t else 0
    ``'var``         forward hinge:   (x - min) / (max - min) if x > min else 0      (min = knot)
    ```var``         reverse hinge:   (max - x) / (max - min) if x < max else 0      (max = knot)

Outputs, with S(x) = sum_j lambda_j f_j(x):
    raw      = exp(S(x) - linearPredictorNormalizer) / densityNormalizer
    cloglog  = 1 - exp(-exp(entropy) * raw)
    logistic = exp(entropy) * raw / (1 + exp(entropy) * raw)

Clamping (maxent.jar default ``doclamp=true``): each variable is clipped to the [min, max] range of its linear
feature (the training range) before features are computed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch


def default_device(device: str | torch.device | None = None) -> str | torch.device:
    """``device`` if given, else ``cuda`` when a CUDA GPU is usable and ``cpu`` otherwise."""
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


_THRESHOLD = re.compile(r"^\((?P<t>[-+0-9.eE]+)<(?P<var>.+)\)$")


@dataclass
class MaxentModel:
    """A parsed maxent.jar model, held as tensors grouped by feature type."""

    variables: List[str]
    clamp_lo: torch.Tensor          # (V,) training minimum per variable
    clamp_hi: torch.Tensor          # (V,) training maximum per variable
    linear: Dict[str, torch.Tensor]       # var_index, lam, lo, hi
    quadratic: Dict[str, torch.Tensor]
    product: Dict[str, torch.Tensor]      # var_a, var_b, lam, lo, hi
    threshold: Dict[str, torch.Tensor]    # var_index, lam, t
    forward_hinge: Dict[str, torch.Tensor]
    reverse_hinge: Dict[str, torch.Tensor]
    linear_predictor_normalizer: float
    density_normalizer: float
    entropy: float

    @classmethod
    def from_lambdas(cls, path: str | Path, variables: Sequence[str] | None = None,
                     dtype: torch.dtype = torch.float64, device: str | torch.device = "cpu") -> "MaxentModel":
        """Parse a ``.lambdas`` file. ``variables`` fixes the column order of the inputs to ``evaluate``."""
        return cls.from_text(Path(path).read_text(), variables, dtype, device)

    @classmethod
    def from_text(cls, text: str, variables: Sequence[str] | None = None,
                  dtype: torch.dtype = torch.float64, device: str | torch.device = "cpu") -> "MaxentModel":
        """Parse the contents of a ``.lambdas`` file."""
        rows, consts = [], {}
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                consts[parts[0]] = float(parts[1])
            elif len(parts) == 4:
                rows.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
        lin = [r for r in rows if _kind(r[0]) == "linear"]
        names = list(variables) if variables is not None else [r[0] for r in lin]
        index = {n: i for i, n in enumerate(names)}
        lo = torch.full((len(names),), -torch.inf, dtype=dtype)
        hi = torch.full((len(names),), torch.inf, dtype=dtype)
        for name, _, mn, mx in lin:
            lo[index[name]], hi[index[name]] = mn, mx
        groups: Dict[str, list] = {k: [] for k in ("linear", "quadratic", "product", "threshold", "fhinge", "rhinge")}
        for name, lam, mn, mx in rows:
            if lam == 0.0:
                continue
            kind = _kind(name)
            if kind == "linear":
                groups["linear"].append((index[name], lam, mn, mx))
            elif kind == "quadratic":
                groups["quadratic"].append((index[name[:-2]], lam, mn, mx))
            elif kind == "product":
                a, b = name.split("*")
                groups["product"].append((index[a], index[b], lam, mn, mx))
            elif kind == "threshold":
                m = _THRESHOLD.match(name)
                groups["threshold"].append((index[m["var"]], lam, float(m["t"])))
            elif kind == "fhinge":
                groups["fhinge"].append((index[name[1:]], lam, mn, mx))
            elif kind == "rhinge":
                groups["rhinge"].append((index[name[1:]], lam, mn, mx))

        def pack(items, fields):
            cols = list(zip(*items)) if items else [[] for _ in fields]
            return {f: torch.tensor(c, dtype=torch.long if f.startswith("var") else dtype, device=device)
                    for f, c in zip(fields, cols)}

        return cls(
            variables=names, clamp_lo=lo.to(device), clamp_hi=hi.to(device),
            linear=pack(groups["linear"], ("var", "lam", "lo", "hi")),
            quadratic=pack(groups["quadratic"], ("var", "lam", "lo", "hi")),
            product=pack(groups["product"], ("var_a", "var_b", "lam", "lo", "hi")),
            threshold=pack(groups["threshold"], ("var", "lam", "t")),
            forward_hinge=pack(groups["fhinge"], ("var", "lam", "lo", "hi")),
            reverse_hinge=pack(groups["rhinge"], ("var", "lam", "lo", "hi")),
            linear_predictor_normalizer=consts["linearPredictorNormalizer"],
            density_normalizer=consts["densityNormalizer"],
            entropy=consts["entropy"],
        )

    def linear_predictor(self, x: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        """S(x) for x of shape (N, V) in ``self.variables`` order.

        Per-feature contributions are formed element-wise and summed along the feature axis in a fixed order,
        so each cell's value is independent of how many cells share the batch (a matrix-vector product is
        not: cuBLAS picks algorithms by batch size). Any tile of a map therefore decodes bit-identically to
        the full render."""
        x = x.to(self.clamp_lo.dtype)
        if clamp:
            x = torch.maximum(torch.minimum(x, self.clamp_hi), self.clamp_lo)
        parts = []
        g = self.linear
        if g["lam"].numel():
            parts.append((x[:, g["var"]] - g["lo"]) / (g["hi"] - g["lo"]) * g["lam"])
        g = self.quadratic
        if g["lam"].numel():
            parts.append((x[:, g["var"]] ** 2 - g["lo"]) / (g["hi"] - g["lo"]) * g["lam"])
        g = self.product
        if g["lam"].numel():
            parts.append((x[:, g["var_a"]] * x[:, g["var_b"]] - g["lo"]) / (g["hi"] - g["lo"]) * g["lam"])
        g = self.threshold
        if g["lam"].numel():
            parts.append((x[:, g["var"]] > g["t"]).to(x.dtype) * g["lam"])
        g = self.forward_hinge
        if g["lam"].numel():
            parts.append(torch.clamp(x[:, g["var"]] - g["lo"], min=0.0) / (g["hi"] - g["lo"]) * g["lam"])
        g = self.reverse_hinge
        if g["lam"].numel():
            parts.append(torch.clamp(g["hi"] - x[:, g["var"]], min=0.0) / (g["hi"] - g["lo"]) * g["lam"])
        if not parts:
            return x.new_zeros(x.shape[0])
        return torch.cat(parts, dim=1).sum(dim=1)

    def raw(self, x: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        return torch.exp(self.linear_predictor(x, clamp) - self.linear_predictor_normalizer) / self.density_normalizer

    def cloglog(self, x: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        return 1.0 - torch.exp(-torch.exp(torch.tensor(self.entropy, dtype=x.dtype if x.is_floating_point() else torch.float64)) * self.raw(x, clamp))

    def logistic(self, x: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        r = torch.exp(torch.tensor(self.entropy, dtype=torch.float64)) * self.raw(x, clamp)
        return r / (1.0 + r)


def _kind(name: str) -> str:
    if name.startswith("("):
        return "threshold"
    if name.startswith("'"):
        return "fhinge"
    if name.startswith("`"):
        return "rhinge"
    if name.endswith("^2"):
        return "quadratic"
    if "*" in name:
        return "product"
    return "linear"
