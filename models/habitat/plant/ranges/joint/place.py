"""Place: a learned function of where a location is, shared by all species, read by each species through its own
place vector.

Why. Climate, soil and terrain do not explain everything about a range. A species may be absent from suitable
climate because it never got there (a mountain range or a desert in between), present only along a coastline or a
river system, or limited by a history no predictor records. Such range geometry is a function of position itself.
SINR (Cole et al. 2023, "Spatial implicit neural representations for global-scale species mapping", ICML) learns it
with one coordinate network shared by tens of thousands of species; its environment-plus-coordinates variant was the
strongest published competitor of this model on independent plots (docs/scientific_provenance.md, 2026-10-05).

Network (SINR's location encoder, "sin_cos" input). The position enters as
x = [sin(pi lon / 180), cos(pi lon / 180), sin(pi lat / 90), cos(pi lat / 90)] (continuous across the antimeridian),
then Linear(4, 256) + ReLU, four residual blocks (Linear-ReLU-Linear-ReLU with a skip connection), and a linear
projection to ``d_place`` place features P(x). The projection starts at zero, so adding the pathway to a trained model
leaves its scores unchanged at the start.

Use in the score (model.py). Place is its own pathway beside the environment: f_s(x) = ... + <P(x), v_s>, with a
place vector v_s per species under the same Brownian-motion prior as the niche vectors (v = A z_p), so related
species share range geometry as they share niches.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinrPlace(nn.Module):
    def __init__(self, d_place: int, hidden: int = 256, blocks: int = 4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(4, hidden), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                                   nn.ReLU()) for _ in range(blocks)])
        self.out = nn.Linear(hidden, d_place)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def spec_from_state(state: dict, prefix: str = "place.") -> dict:
        """Constructor arguments of a saved place network, read off its tensors."""
        blocks = len({k[len(prefix + "blocks."):].split(".")[0] for k in state if k.startswith(prefix + "blocks.")})
        return dict(d_place=int(state[prefix + "out.weight"].shape[0]),
                    hidden=int(state[prefix + "inp.0.weight"].shape[0]), blocks=blocks)

    @staticmethod
    def encode(latlon: torch.Tensor) -> torch.Tensor:
        """[n, 2] (latitude, longitude) in degrees -> [n, 4] sin/cos input."""
        lat, lon = latlon[:, 0].float() / 90.0, latlon[:, 1].float() / 180.0
        return torch.stack([torch.sin(math.pi * lon), torch.cos(math.pi * lon), torch.sin(math.pi * lat),
                            torch.cos(math.pi * lat)], 1)

    def forward(self, latlon: torch.Tensor) -> torch.Tensor:
        """P(x) [n, d_place] at [n, 2] (latitude, longitude) in degrees."""
        h = self.inp(self.encode(latlon))
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)
