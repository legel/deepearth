"""Fixed data contract for the mesh experiment. Do not edit during research."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))

from deepearth.autoresearch.main.editable_files.lib import data as base_data


CONTINUOUS_SIGNALS = (
    "vision_dino",
    "vision_bio",
    "phylo",
    "climate",
    "soil",
    "naip_rgb",
    "naip_ir",
    "clay",
    "topo",
    "chm",
    "hydro",
    "phenology",
)


def load(cache_dir: str, device: str, *, subset: dict | None = None):
    """Restore the canonical California observations and spatial holdout."""
    settings = {
        "adapter": "california",
        "cache_dir": str(Path(cache_dir).expanduser().resolve()),
        "n_neighbors": 16,
        "holdout": "spatial",
        "subset": subset,
        "time_axis": True,
        "time_km": 50.0,
        "clay_v2": False,
    }
    cache = Path(settings["cache_dir"])
    # Reuse the canonical assembled dataset when the cache ships one. Rebuilding
    # this artifact duplicates ~15 GB without changing a single observation.
    existing = sorted(cache.glob("prepared_*.pt"))
    prepared = existing[0] if existing else cache / f"prepared_{base_data.prepared_tag(settings)}.pt"
    source = base_data.build(
        settings["adapter"],
        cache_dir=settings["cache_dir"],
        n_neighbors=settings["n_neighbors"],
        device=device,
        holdout=settings["holdout"],
        subset=settings["subset"],
        time_axis=settings["time_axis"],
        time_km=settings["time_km"],
        clay_v2=settings["clay_v2"],
        prepared=str(prepared),
    )

    dims = source.variable_dims()
    variables = [
        {"name": name, "kind": "continuous", "dim": int(dims[name])}
        for name in CONTINUOUS_SIGNALS
        if int(dims.get(name, 0)) > 0
    ]
    variables.insert(2, {
        "name": "identity",
        "kind": "categorical",
        "num_classes": int(dims["identity_classes"]),
    })
    variables.extend(
        {"name": name, "kind": "categorical", "num_classes": int(classes)}
        for name, classes in dims["trait_classes"].items()
    )

    always = {}
    if "alphaearth" in source.extra:
        always["alphaearth"] = int(source.extra["alphaearth"][2])
    return source, variables, always
