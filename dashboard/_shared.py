"""Shared helpers for tools that call into the trainer. One definition, no drift."""
import hashlib, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def normalize_config(config):
    """Mirror train.main: absolutize cache_dir against the repo root (the cache tag hashes it)."""
    cd = config.get("data", {}).get("cache_dir")
    if cd and not Path(cd).is_absolute():
        config["data"]["cache_dir"] = str(REPO / cd)
    return config


def prepared_path(config):
    """The trainer's prepared-cache path for this config — byte-identical tag computation."""
    d = config["data"]
    keyparts = {k: d.get(k) for k in ("adapter", "cache_dir", "n_neighbors", "holdout",
                                      "subset", "time_axis", "time_km")}
    tag = hashlib.md5(json.dumps(keyparts, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return str(REPO / "data" / "deepcal" / f"prepared_{tag}.pt")
