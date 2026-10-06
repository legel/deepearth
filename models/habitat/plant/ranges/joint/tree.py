"""The dated phylogeny of the joint range model: Newick parsing and the Brownian-motion path matrix.

A rooted tree is held as three arrays indexed by node id: ``parent`` (-1 at the root), ``length`` (the branch from
the node up to its parent, in the tree's time unit, here millions of years) and ``label`` ("" for an unlabelled
node). Node ids are assigned in pre-order (a node before its children, children left to right), so a parent's id is
always smaller than its children's. Every node except the root owns exactly one branch, the one above it, so a node
id also names that branch ("edge").

The path matrix A ties species to branches: A[s, e] = sqrt(l_e / l_mean) when branch e lies on the path from the
root to species s's tip, else 0. Under Brownian motion a trait drifts along every branch with variance proportional
to its length, so with one independent standard-normal vector z_e per branch, w = A z has exactly the Brownian
covariance between species: the shared root-to-ancestor time of any two species (model.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def parse_newick(text: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Newick text -> (parent int64, length float64, label list), node ids in pre-order.

    Iterative, so the depth of the tree is unbounded. Accepts a leading rooting comment (``[&U]``), internal-node
    labels and missing branch lengths (0)."""
    s = text.strip()
    if s.startswith("[&"):
        s = s[s.index("]") + 1:].strip()
    if s.endswith(";"):
        s = s[:-1]
    parent: list[int] = []
    length: list[float] = []
    label: list[str] = []
    n = len(s)

    def new(p: int) -> int:
        parent.append(p)
        length.append(0.0)
        label.append("")
        return len(parent) - 1

    pos, stack, cur = 0, [], new(-1)
    while True:
        if pos < n and s[pos] == "(":                            # descend: the current node gets its first child
            stack.append(cur)
            cur = new(cur)
            pos += 1
            continue
        start = pos                                              # label and/or ":length" of the current node
        while pos < n and s[pos] not in "(),:;":
            pos += 1
        if pos > start:
            label[cur] = s[start:pos]
        if pos < n and s[pos] == ":":
            pos += 1
            start = pos
            while pos < n and s[pos] not in "(),:;":
                pos += 1
            length[cur] = float(s[start:pos])
        if pos >= n:
            break
        if s[pos] == ",":                                        # next sibling
            if not stack:
                raise ValueError(f"malformed Newick: ',' outside parentheses at position {pos}")
            cur = new(stack[-1])
        elif s[pos] == ")":                                      # back to the parent, whose label may follow
            if not stack:
                raise ValueError(f"malformed Newick: unbalanced ')' at position {pos}")
            cur = stack.pop()
        else:
            raise ValueError(f"malformed Newick near position {pos}")
        pos += 1
    if stack:
        raise ValueError("malformed Newick: unclosed '('")
    return np.asarray(parent, np.int64), np.asarray(length, np.float64), label


@dataclass
class Tree:
    parent: np.ndarray
    length: np.ndarray
    label: list[str]
    _children: list[list[int]] | None = field(default=None, repr=False)

    @classmethod
    def read(cls, path: str | Path) -> "Tree":
        return cls(*parse_newick(Path(path).read_text()))

    def __len__(self) -> int:
        return len(self.parent)

    def node_of(self) -> dict[str, int]:
        """Label -> node id (labelled nodes only)."""
        return {lab: i for i, lab in enumerate(self.label) if lab}

    def children(self) -> list[list[int]]:
        if self._children is None:
            kids: list[list[int]] = [[] for _ in range(len(self.parent))]
            for c, p in enumerate(self.parent):
                if p >= 0:
                    kids[p].append(c)
            self._children = kids
        return self._children

    def tips_under(self, node: int) -> list[str]:
        """Labels of the labelled leaves in the clade below ``node`` (``node`` itself if it is a leaf)."""
        kids = self.children()
        out, todo = [], [node]
        while todo:
            x = todo.pop()
            if self.label[x] and not kids[x]:
                out.append(self.label[x])
            todo.extend(kids[x])
        return out


def path_matrix(tree: Tree, tips: Sequence[str]) -> torch.Tensor:
    """Sparse float32 [len(tips), len(tree)]: sqrt(l_e / l_mean) on every branch e of each tip's root-to-tip path.

    l_mean is the mean positive branch length of the whole tree, so the loadings are dimensionless and a branch of
    average length contributes one unit of prior variance."""
    node = tree.node_of()
    missing = [t for t in tips if t not in node]
    if missing:
        raise KeyError(f"{len(missing)} species are not tips of the tree, e.g. {missing[:3]}")
    scale = tree.length[tree.length > 0].mean()
    rows, cols, vals = [], [], []
    for s, t in enumerate(tips):
        x = node[t]
        while tree.parent[x] >= 0:
            rows.append(s)
            cols.append(x)
            vals.append(math.sqrt(tree.length[x] / scale))
            x = tree.parent[x]
    return torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.tensor(vals, dtype=torch.float32),
                                   (len(tips), len(tree))).coalesce()
