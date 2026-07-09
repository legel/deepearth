"""One-time setup for DeepCal autoresearch. Idempotent: safe to run repeatedly.

    python -m deepearth.core.prepare              # uses core/deepcal.yaml
    python -m deepearth.core.prepare --config core/deepcal.yaml

It ensures, in order:
  1. DATA   -- the fully pre-processed DeepCal cache (embeddings, tokens, tree, splits) is present under
     ``deepearth/data/deepcal/`` (or an existing local cache), downloading + extracting the NERSC-hosted zips if not.
  2. KERNEL -- the Earth4D hash-grid CUDA kernel is importable, compiling it (``encoders/spacetime/install.sh``) if not.
  3. PREPARED -- the assembled dataset (glob + KD-tree neighbor index, ~2.7 min to build) is cached to a single
     ``.pt`` so every training run / experiment spins up in ~1 second.
  4. TEST I/O -- a small held-out inference bundle is materialized so the benchmark harness is ready to score.

Run this ONCE before launching the autoresearch loop. It is not modified by experiments (see ``autoresearch.md``).
"""
from __future__ import annotations
import argparse, os, subprocess, sys, zipfile
from pathlib import Path
import urllib.request
import yaml

REPO = Path(__file__).resolve().parents[1]                       # .../deepearth
DATA_DIR = REPO / "data" / "deepcal"                             # where prepare.py places the data
SPACETIME = REPO / "encoders" / "spacetime"
# NERSC data portal (project m5239) -- plain HTTPS, no auth needed to download. Override with DEEPCAL_DATA_URL.
DATA_URL_BASE = os.environ.get("DEEPCAL_DATA_URL", "https://portal.nersc.gov/cfs/m5239/deepcal")
DATA_ZIPS = ["deepcal_data.zip"]                                 # the pre-processed cache, split if large

# Files that must exist for the cache to count as "ready" (a representative subset of the full manifest).
REQUIRED = ["gbif_vocab.npz", "ca_subtree.dated.nwk", "derived/species_index.csv", "derived/patristic_ref.npy"]


def _has_required(d: Path) -> bool:
    return d.exists() and all((d / r).exists() for r in REQUIRED)


def resolve_data_dir(cache_dir_cfg: str) -> Path:
    """Pick the data directory: prefer the download target, then an existing local cache, else download."""
    if _has_required(DATA_DIR):
        print(f"[data] ready at {DATA_DIR}")
        return DATA_DIR
    cfg = Path(cache_dir_cfg) if cache_dir_cfg else None
    if cfg and _has_required(cfg):
        print(f"[data] using existing local cache {cfg}")
        return cfg
    print(f"[data] not found locally; downloading from {DATA_URL_BASE} -> {DATA_DIR}")
    download_data()
    if not _has_required(DATA_DIR):
        raise SystemExit(f"[data] download did not yield required files under {DATA_DIR}; "
                         f"check DEEPCAL_DATA_URL / NERSC hosting")
    return DATA_DIR


def download_data() -> None:
    """Download + extract the pre-processed data zips from the NERSC data portal."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in DATA_ZIPS:
        url = f"{DATA_URL_BASE}/{name}"; dst = DATA_DIR / name
        if not dst.exists():
            print(f"[data] GET {url}")
            _download(url, dst)
        print(f"[data] extracting {name}")
        with zipfile.ZipFile(dst) as z:
            z.extractall(DATA_DIR)
        dst.unlink()                                            # drop the zip after extraction


def _download(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0)); done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); done += len(chunk)
            if total:
                print(f"\r  {done/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
    print()
    tmp.rename(dst)


def ensure_kernel() -> None:
    """Import the Earth4D CUDA hash-grid kernel; build it in place if the compiled extension is absent."""
    sys.path.insert(0, str(SPACETIME))
    try:
        from hashencoder.hashgrid import HashEncoder  # noqa: F401
        print("[kernel] Earth4D hash-grid CUDA kernel importable")
        return
    except Exception as e:
        print(f"[kernel] not importable ({e}); building via install.sh")
    install = SPACETIME / "install.sh"
    if install.exists():
        subprocess.run(["bash", str(install)], cwd=str(SPACETIME), check=True)
    else:
        subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"],
                       cwd=str(SPACETIME / "hashencoder"), check=True)
    from hashencoder.hashgrid import HashEncoder  # noqa: F401  (raise if still broken)
    print("[kernel] built and importable")


def ensure_prepared(config: dict, data_dir: Path, device: str) -> str:
    """Build the assembled-dataset cache (the slow glob + KD-tree) once, so runs spin up instantly."""
    import hashlib, json
    from deepearth.autoresearch import data as data_module
    d = dict(config["data"]); d["cache_dir"] = str(data_dir)
    keyparts = {k: d.get(k) for k in ("adapter", "cache_dir", "n_neighbors", "holdout", "subset", "time_axis", "time_km")}
    tag = hashlib.md5(json.dumps(keyparts, sort_keys=True, default=str).encode()).hexdigest()[:10]
    prepared = str(DATA_DIR / f"prepared_{tag}.pt")
    if Path(prepared).exists():
        print(f"[prepared] cache present: {prepared}")
        return prepared
    print(f"[prepared] building assembled dataset (one-time; ~2-3 min) -> {prepared}")
    src = data_module.build(d["adapter"], cache_dir=str(data_dir), n_neighbors=d.get("n_neighbors", 24),
                            device=device, holdout=d.get("holdout", "spatial"), subset=d.get("subset"),
                            time_axis=d.get("time_axis", False), time_km=d.get("time_km", 50.0), prepared=prepared)
    print(f"[prepared] {src.n} observations, {len(src.train)} train / {len(src.test)} held-out")
    return prepared


def ensure_test_io(prepared: str, device: str) -> None:
    """Materialize a small held-out inference bundle (indices + truth) so the benchmark harness is ready to score
    and results are inspectable. Lightweight; the full suite runs at eval time in ``benchmarks.py``."""
    import torch
    from deepearth.autoresearch import data as data_module
    bundle = DATA_DIR / "test_io_sample.pt"
    if bundle.exists():
        print(f"[test-io] present: {bundle}")
        return
    src = data_module.build("california", cache_dir=str(DATA_DIR), device=device, prepared=prepared)
    n = min(256, len(src.test))
    idx = src.test[:n]
    torch.save({"test_index_sample": idx, "n_test": len(src.test), "n_train": len(src.train),
                "variables": list(src.variable_dims().keys())}, bundle)
    print(f"[test-io] wrote held-out inference sample ({n} rows) -> {bundle}")


def main():
    ap = argparse.ArgumentParser(description="One-time DeepCal setup (data, kernel, prepared cache, test I/O).")
    ap.add_argument("--config", default=str(Path(__file__).with_name("deepcal.yaml")))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    config = yaml.safe_load(open(a.config))
    print(f"=== DeepCal prepare ({config.get('name', 'DeepCal')}) ===")
    ensure_kernel()
    data_dir = resolve_data_dir(config["data"].get("cache_dir"))
    prepared = ensure_prepared(config, data_dir, a.device)
    ensure_test_io(prepared, a.device)
    print("=== prepare complete: ready for `python -m deepearth.core.train` and the autoresearch loop ===")


if __name__ == "__main__":
    main()
