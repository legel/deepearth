"""One-time setup for DeepCal autoresearch. Idempotent: safe to run repeatedly.

    python -m deepearth.autoresearch.prepare                              # uses autoresearch/deepcal.yaml
    python -m deepearth.autoresearch.prepare --config autoresearch/deepcal.yaml

Run from the repo's PARENT directory (the one containing ``deepearth/``) so the ``deepearth`` package imports.

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
import argparse, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import urllib.request, urllib.error
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


def _download(url: str, dst: Path, retries: int = 20) -> None:
    """Resumable download: HTTP Range-resume the .tmp partial after any drop, retry transient failures with backoff,
    and verify the final byte count before promoting to dst — never rename a short/partial file as complete
    (robust on flaky networks; a plain urlopen loop silently ships a truncated zip when the connection drops)."""
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    # Fast path: the NERSC portal throttles each HTTP connection (~70 KB/s), so a single stream crawls; aria2c's
    # parallel range requests aggregate ~13x. Use it when present (resumable via -c), else fall back to the loop below.
    if shutil.which("aria2c"):
        rc = subprocess.run(["aria2c", "-x16", "-s16", "-k5M", "-c", "--file-allocation=none",
                             "--retry-wait=5", "--max-tries=0", "--console-log-level=warn",
                             "-d", str(tmp.parent), "-o", tmp.name, url]).returncode
        if rc == 0 and tmp.exists():
            tmp.rename(dst); return
        print("[data] aria2c failed; falling back to single-stream resumable download", flush=True)
    for attempt in range(retries):
        pos = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-"} if pos else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if pos and r.status != 206:                       # server ignored the Range -> restart the file
                    pos = 0
                total = pos + int(r.headers.get("Content-Length", 0))
                with open(tmp, "ab" if pos else "wb") as f:
                    done = pos
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk); done += len(chunk)
                        if total:
                            print(f"\r  {done/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
            print()
            if total and tmp.stat().st_size < total:              # short read = connection dropped -> resume
                raise IOError(f"incomplete {tmp.stat().st_size}/{total}")
            tmp.rename(dst); return
        except urllib.error.HTTPError as e:
            if e.code == 416 and tmp.exists():                    # Range-not-satisfiable = already fully downloaded
                tmp.rename(dst); return
            _resume_wait(e, pos, attempt, retries)
        except (urllib.error.URLError, TimeoutError, IOError, ConnectionError) as e:
            _resume_wait(e, pos, attempt, retries)
    raise SystemExit(f"[data] download failed after {retries} attempts — check the network or DEEPCAL_DATA_URL")


def _resume_wait(err, pos, attempt, retries):
    wait = min(2 * (attempt + 1), 30)
    print(f"\n  [resume] {type(err).__name__} at {pos/1e6:.0f} MB — retry in {wait}s ({attempt+1}/{retries})", flush=True)
    time.sleep(wait)


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
    # Line-buffer stdout so progress is visible in real time even under `> prepare.log 2>&1` (block buffering otherwise hides every step until exit — the run looks hung during the ~2-3 min cache build).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="One-time DeepCal setup (data, kernel, prepared cache, test I/O).")
    ap.add_argument("--config", default=str(Path(__file__).with_name("deepcal.yaml")))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    config = yaml.safe_load(open(a.config))
    print(f"=== DeepCal prepare ({config.get('name', 'DeepCal')}) ===")
    ensure_kernel()
    # Portability: resolve a relative cache_dir against the repo root, so a from-scratch download lands in-repo on any device.
    cache_dir_cfg = config["data"].get("cache_dir")
    if cache_dir_cfg and not os.path.isabs(cache_dir_cfg):
        cache_dir_cfg = str(REPO / cache_dir_cfg)
    data_dir = resolve_data_dir(cache_dir_cfg)
    prepared = ensure_prepared(config, data_dir, a.device)
    ensure_test_io(prepared, a.device)
    print("=== prepare complete: ready for `python -m deepearth.autoresearch.train` and the autoresearch loop ===")


if __name__ == "__main__":
    main()
