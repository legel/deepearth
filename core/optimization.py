"""Small optimizer groups for staged DeepEarth training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass
class OptimizerGroup:
    parameters: list[nn.Parameter]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler


class Optimizers:
    def __init__(self, **groups: OptimizerGroup | None):
        self.groups = {name: group for name, group in groups.items() if group}

    def __getitem__(self, name: str) -> OptimizerGroup:
        return self.groups[name]

    def add(self, name: str, group: OptimizerGroup | None) -> None:
        if group is not None:
            self.groups[name] = group

    def zero_grad(self) -> None:
        for group in self.groups.values():
            group.optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        for group in self.groups.values():
            group.optimizer.step()
            group.scheduler.step()

    def clip_grad_norm(self, maximum: float) -> None:
        for group in self.groups.values():
            torch.nn.utils.clip_grad_norm_(group.parameters, maximum)


def adamw_group(
    parameters: Iterable[nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
    steps: int,
    device: str,
    parameter_groups=None,
) -> OptimizerGroup | None:
    parameters = list(parameters)
    if not parameters:
        return None
    optimizer = torch.optim.AdamW(
        parameter_groups or parameters,
        lr=lr,
        weight_decay=weight_decay,
        fused=device.startswith("cuda"),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    return OptimizerGroup(parameters, optimizer, scheduler)
