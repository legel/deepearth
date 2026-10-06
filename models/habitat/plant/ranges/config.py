"""Where the data live and what the production run was.

Every input and output of this package sits under one directory, the data root, in the layout the README lists
(``raw/`` for downloaded sources, ``work/`` for everything built from them, ``tools/`` for maxent.jar). The root is
the environment variable ``DEEPEARTH_HABITAT_DATA`` when it is set, else the configuration's ``data_root`` (relative
to this model's directory). Paths inside a configuration file are relative to the data root, so one file describes
a run wherever its data are kept.

``configs/conus.json`` is the configuration of the published product: all native vascular plants of the contiguous
United States, trained as one joint model and stored as one transform-coded map store.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MODEL_DIR / "configs" / "conus.json"
ENV = "DEEPEARTH_HABITAT_DATA"


class Config(dict):
    """A run configuration (the parsed JSON) with its data root; ``path`` resolves a configured path."""

    def __init__(self, values: dict, root: Path, source: Path | None = None):
        super().__init__(values)
        self.root = Path(root)
        self.source = source

    def path(self, rel: str | os.PathLike | None) -> Path | None:
        """A configured path under the data root (an absolute path is kept as it is; None stays None)."""
        if rel is None:
            return None
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def paths(self, rels) -> list[Path]:
        return [self.path(r) for r in rels]


def data_root(values: dict | None = None) -> Path:
    """The data root: ``$DEEPEARTH_HABITAT_DATA``, else ``values["data_root"]`` relative to the model directory,
    else ``<model>/data``."""
    env = os.environ.get(ENV)
    if env:
        return Path(env).expanduser()
    rel = Path((values or {}).get("data_root", "data")).expanduser()
    return rel if rel.is_absolute() else MODEL_DIR / rel


def load(path: str | os.PathLike | None = None) -> Config:
    """Read a configuration file (default ``configs/conus.json``)."""
    path = Path(path) if path else DEFAULT_CONFIG
    values = json.loads(path.read_text())
    return Config(values, data_root(values), path)
