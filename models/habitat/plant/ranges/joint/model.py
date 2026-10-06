"""The joint range model: shared representations of a location and one set of vectors per species.

For species s at a location x the model's log relative intensity of occurrence (the MaxEnt "raw" score: log of how
many more records per unit area the species makes at x than on average) is

    f_s(x) = <h(x) + G(C(x)), w_s> + <P(x), v_s> + b_s - pi_s [x outside the calibration area of s].

Each term is optional (an environment-only model has f_s = <h(x), w_s> + b_s); every product is linear in a species
vector, so all maps are read off a few shared numbers per location (store.py).

* h(x) = trunk(env(X(x))) in R^width, the environment features, is shared by every species. ``env`` is a stack of
  ``depth`` linear layers (LayerNorm and SiLU between them); ``trunk`` is LayerNorm, SiLU, two linear layers and a
  final LayerNorm without a learned scale or shift. That last LayerNorm bounds the features (|h(x)| = sqrt(width) for
  every x), so a logit is at most |w_s| sqrt(width): no single location can dominate a species' normalizer by
  extrapolating, and the prior on w_s (below) bounds |w_s|.
* G(C(x)), the landscape around x (field.py: Entropy3D ring harmonics of the raw 240 m field, pooled by attention),
  is added to h(x): the species read their surroundings through the same niche vector w_s.
* P(x), place features (place.py: SINR's coordinate network), read by each species through its place vector v_s:
  range geometry that no predictor explains (barriers, coastlines, history).
* w_s, the species' niche vector, has a Brownian-motion phylogenetic prior. With A the tree's path matrix
  (A[s, e] = sqrt(l_e / l_mean) on every branch e between the root and species s; tree.py), one vector z_e per
  branch and one species-specific vector u_s,

      w_s = sum_e A[s, e] z_e + u_s.

  Weight decay on z (a Gaussian prior) makes A z a Brownian random walk along the tree: two species share every
  branch above their common ancestor, so their vectors are correlated in proportion to the time they evolved
  together. A species with few records therefore starts from its clade's niche and departs from it only as far as
  its own data demand. u_s (weaker decay) lets a well-recorded species move away from its clade.
* v_s = sum_e A[s, e] z_p,e: the place vectors under the same prior (relatives share range geometry).
* pi_s = softplus(sum_e A[s, e] z_c,e + c_0) >= 0: a learned penalty applied outside the species' calibration area
  (the RESOLVE ecoregions holding its records). A per-species MaxEnt is fitted inside that area only, and its map is 0
  outside; the soft penalty keeps what the area says (a species is mostly found where its records are) without
  making it absolute (records miss part of many ranges). Relatives share it through the tree; c_0 starts it at
  ``penalty_init``.
* b_s is the species' offset (its overall record density; it cancels in the MaxEnt likelihood within a species but
  sets the common scale of the scores).

Parameter names follow the formulas (``A``, ``z``, ``u``, ``b``, ``zp``, ``zc``, ``c0``, modules ``env``, ``trunk``,
``field``, ``place``) and are the names of the trained checkpoints, so ``JointRangeModel.from_state_dict`` rebuilds
any of them from the state dict alone.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .field import FieldPyramid, HarmonicField
from .place import SinrPlace

SPECIES_TENSORS = ("A", "z", "u", "b", "zp", "zc")   # sized by the species and tree of a data set


def _mlp(n_in: int, width: int, n_out: int, layers: int) -> nn.Sequential:
    """``layers`` linear layers with LayerNorm and SiLU between them (none after the last)."""
    mods, d = [], n_in
    for _ in range(layers - 1):
        mods += [nn.Linear(d, width), nn.LayerNorm(width), nn.SiLU()]
        d = width
    return nn.Sequential(*mods, nn.Linear(d, n_out))


class JointRangeModel(nn.Module):
    def __init__(self, n_inputs: int, paths: torch.Tensor, width: int = 256, depth: int = 3,
                 place: dict | None = None, field: dict | None = None, penalty_init: float | None = None):
        """``paths``: sparse [n_species, n_branches] path matrix (``tree.path_matrix``). ``place``: keyword arguments
        of ``SinrPlace`` (``d_place``, ...), None for no place pathway. ``field``: keyword arguments of
        ``HarmonicField`` but ``d`` (``n_channels``, ...), None for no field. ``penalty_init``: initial
        outside-area penalty (score units), None for the hard calibration rule (maps 0 outside the area)."""
        super().__init__()
        n_species, n_branches = paths.shape
        self.env = _mlp(n_inputs, width, width, depth)
        self.trunk = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), _mlp(width, width, width, 2),
                                   nn.LayerNorm(width, elementwise_affine=False))
        self.register_buffer("A", paths.coalesce())
        self.z = nn.Parameter(torch.randn(n_branches, width) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_species))
        self.u = nn.Parameter(torch.zeros(n_species, width))
        self.place = SinrPlace(**place) if place else None
        if self.place is not None:
            self.zp = nn.Parameter(torch.randn(n_branches, self.place.out.out_features) * 0.02)
        self.field = HarmonicField(width, **field) if field else None
        if penalty_init is not None:
            self.zc = nn.Parameter(torch.zeros(n_branches))
            self.c0 = nn.Parameter(torch.tensor(math.log(math.expm1(penalty_init))))

    # ----------------------------------------------------------------------------------------------- sizes
    @property
    def n_species(self) -> int:
        return self.A.shape[0]

    @property
    def width(self) -> int:
        return self.z.shape[1]

    @property
    def place_dim(self) -> int:
        return 0 if self.place is None else self.zp.shape[1]

    @property
    def has_penalty(self) -> bool:
        return hasattr(self, "zc")

    def shared_parameters(self):
        """(name, parameter) of everything not sized by the species: the networks and c0."""
        return [(n, p) for n, p in self.named_parameters() if n not in SPECIES_TENSORS]

    # ------------------------------------------------------------------------------------- species vectors
    def _path_sum(self, z: torch.Tensor) -> torch.Tensor:
        with torch.autocast(z.device.type, enabled=False):                   # the sparse product runs in float32
            return torch.sparse.mm(self.A, z.float())

    def species_vectors(self) -> torch.Tensor:
        """w = A z + u, [n_species, width]."""
        return self._path_sum(self.z) + self.u

    def place_vectors(self) -> torch.Tensor | None:
        """v = A z_p, [n_species, place_dim] (None without a place pathway)."""
        return self._path_sum(self.zp) if self.place is not None else None

    def outside_penalty(self) -> torch.Tensor | None:
        """pi = softplus(A z_c + c_0) [n_species] (None with the hard calibration rule)."""
        if not self.has_penalty:
            return None
        return F.softplus(self._path_sum(self.zc[:, None])[:, 0] + self.c0)

    # ---------------------------------------------------------------------------------- shared features
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """h(x) for standardized inputs x [n, n_inputs] -> [n, width]."""
        return self.trunk(self.env(x))

    def attach_field(self, pyramid: FieldPyramid) -> "JointRangeModel":
        if self.field is not None:
            self.field.attach(pyramid)
        return self

    def shared(self, x: torch.Tensor, latlon: torch.Tensor | None = None, rc: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The species-independent features of locations: F = h(x) + G(C(x)) [n, width] and P(x) [n, place_dim]
        (None without a place pathway). ``latlon`` [n, 2] degrees (place), ``rc`` [n, 2] fractional (row, column)
        of the field pyramid's level 0 (field). Each pathway may run in reduced precision; their sum and the outputs
        are float32."""
        Fx = self.features(x).float()
        if self.field is not None:
            Fx = Fx + self.field(rc).float()
        return Fx, (self.place(latlon).float() if self.place is not None else None)

    def scores(self, Fx: torch.Tensor, P: torch.Tensor | None = None, species: torch.Tensor | None = None,
               outside: torch.Tensor | None = None) -> torch.Tensor:
        """f_s(x) [n, n_species] (or [n, len(species)]) from shared features. ``outside`` [n, n_species] (or
        [n, len(species)]) bool: x lies outside the species' calibration area (applies the learned penalty)."""
        W, b, V, pen = self.species_vectors(), self.b, self.place_vectors(), self.outside_penalty()
        if species is not None:
            W, b = W[species], b[species]
            V = V[species] if V is not None else None
            pen = pen[species] if pen is not None else None
        f = Fx @ W.T + b
        if V is not None:
            f = f + P @ V.T
        if pen is not None and outside is not None:
            f = f - pen * outside.float()
        return f

    def forward(self, x: torch.Tensor, species: torch.Tensor | None = None, latlon: torch.Tensor | None = None,
                rc: torch.Tensor | None = None, outside: torch.Tensor | None = None) -> torch.Tensor:
        """Scores f_s(x), [n, n_species] (or [n, len(species)])."""
        return self.scores(*self.shared(x, latlon, rc), species, outside)

    # ------------------------------------------------------------------------------------------- persistence
    @classmethod
    def from_state_dict(cls, state: dict) -> "JointRangeModel":
        """Rebuild a model from its state dict alone: the input count and width come from the first layer, the
        depth from the number of linear layers in ``env``, the tree from the stored path matrix ``A``, the place and
        field pathways from their tensors, the calibration rule from the presence of ``zc``. The global random number
        generator is left untouched. A field still needs its pyramid (``attach_field``)."""
        w0 = state["env.0.weight"]
        depth = sum(1 for k, v in state.items() if k.startswith("env.") and k.endswith(".weight") and v.dim() == 2)
        place = SinrPlace.spec_from_state(state) if "place.out.weight" in state else None
        field = HarmonicField.spec_from_state(state) if "field.log_r" in state else None
        with torch.random.fork_rng(devices=[]):
            model = cls(w0.shape[1], state["A"].cpu(), width=w0.shape[0], depth=depth, place=place, field=field,
                        penalty_init=1.0 if "zc" in state else None)
        model.load_state_dict(state)
        return model

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu", field_dir: str | Path | None = None
             ) -> "JointRangeModel":
        """A saved checkpoint (``torch.save(model.state_dict())``), in evaluation mode on ``device``; a model with a
        field pathway gets the pyramid of ``field_dir`` attached."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls.from_state_dict(state).to(device).eval()
        if model.field is not None:
            if field_dir is None:
                raise ValueError(f"{path}: the model reads a landscape field; give its pyramid (field_dir)")
            model.attach_field(FieldPyramid(field_dir, device))
        return model
