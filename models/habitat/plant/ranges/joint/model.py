"""The joint range model: one shared environment network and one vector per species.

For species s at a location x with standardized environment X(x) the model's log relative intensity of occurrence
(the MaxEnt "raw" score: log of how many more records per unit area the species makes at x than on average) is

    f_s(x) = <h(x), w_s> + b_s.

* h(x) = trunk(env(X(x))) in R^width is shared by every species. ``env`` is a stack of ``depth`` linear layers
  (LayerNorm and SiLU between them); ``trunk`` is LayerNorm, SiLU, two linear layers and a final LayerNorm without a
  learned scale or shift. That last LayerNorm bounds the features (|h(x)| = sqrt(width) for every x), so a logit is at
  most |w_s| sqrt(width): no single location can dominate a species' normalizer by extrapolating, and the prior on
  w_s (below) bounds |w_s|.
* w_s, the species' niche vector, has a Brownian-motion phylogenetic prior. With A the tree's path matrix
  (A[s, e] = sqrt(l_e / l_mean) on every branch e between the root and species s; tree.py), one vector z_e per
  branch and one species-specific vector u_s,

      w_s = sum_e A[s, e] z_e + u_s.

  Weight decay on z (a Gaussian prior) makes A z a Brownian random walk along the tree: two species share every
  branch above their common ancestor, so their vectors are correlated in proportion to the time they evolved
  together. A species with few records therefore starts from its clade's niche and departs from it only as far as
  its own data demand. u_s (weaker decay) lets a well-recorded species move away from its clade.
* b_s is the species' offset (its overall record density; it cancels in the MaxEnt likelihood within a species but
  sets the common scale of the scores).

Parameter names follow the formulas (``A``, ``z``, ``u``, ``b``) and are the names the trained national checkpoint
uses, so ``JointRangeModel.from_state_dict`` loads it directly.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def _mlp(n_in: int, width: int, n_out: int, layers: int) -> nn.Sequential:
    """``layers`` linear layers with LayerNorm and SiLU between them (none after the last)."""
    mods, d = [], n_in
    for _ in range(layers - 1):
        mods += [nn.Linear(d, width), nn.LayerNorm(width), nn.SiLU()]
        d = width
    return nn.Sequential(*mods, nn.Linear(d, n_out))


class JointRangeModel(nn.Module):
    def __init__(self, n_inputs: int, paths: torch.Tensor, width: int = 256, depth: int = 3):
        """``paths``: sparse [n_species, n_branches] path matrix (``tree.path_matrix``)."""
        super().__init__()
        n_species, n_branches = paths.shape
        self.env = _mlp(n_inputs, width, width, depth)
        self.trunk = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), _mlp(width, width, width, 2),
                                   nn.LayerNorm(width, elementwise_affine=False))
        self.register_buffer("A", paths.coalesce())
        self.z = nn.Parameter(torch.randn(n_branches, width) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_species))
        self.u = nn.Parameter(torch.zeros(n_species, width))

    @property
    def n_species(self) -> int:
        return self.A.shape[0]

    @property
    def width(self) -> int:
        return self.z.shape[1]

    def species_vectors(self) -> torch.Tensor:
        """w = A z + u, [n_species, width]."""
        return torch.sparse.mm(self.A, self.z) + self.u

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """h(x) for standardized inputs x [n, n_inputs] -> [n, width]."""
        return self.trunk(self.env(x))

    def forward(self, x: torch.Tensor, species: torch.Tensor | None = None) -> torch.Tensor:
        """Scores f_s(x), [n, n_species] (or [n, len(species)])."""
        w, b = self.species_vectors(), self.b
        if species is not None:
            w, b = w[species], b[species]
        return self.features(x) @ w.T + b

    # ------------------------------------------------------------------------------------------- persistence
    @classmethod
    def from_state_dict(cls, state: dict) -> "JointRangeModel":
        """Rebuild a model from its state dict alone: the input count and width come from the first layer, the
        depth from the number of linear layers in ``env``, the tree from the stored path matrix ``A``. The global
        random number generator is left untouched."""
        w0 = state["env.0.weight"]
        depth = sum(1 for k, v in state.items() if k.startswith("env.") and k.endswith(".weight") and v.dim() == 2)
        with torch.random.fork_rng(devices=[]):
            model = cls(w0.shape[1], state["A"].cpu(), width=w0.shape[0], depth=depth)
        model.load_state_dict(state)
        return model

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "JointRangeModel":
        """A saved checkpoint (``torch.save(model.state_dict())``), in evaluation mode on ``device``."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls.from_state_dict(state).to(device).eval()
