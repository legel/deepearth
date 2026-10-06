"""MaxEnt fitting following Daru (2024) step 6.

* Predictor screening: stepwise variance-inflation factor, threshold 5 (port of ``usdm::vifstep``).
* maxent.jar with linear, threshold and hinge features; regularization multiplier beta chosen from
  {2, 5, 10, 15, 20} by mean test AUC under 5-fold cross-validation; then five replicate models on random
  75/25 splits at the chosen beta. The replicate median is the species' model (see ``project``).

Two engines run that protocol and write the same files: ``torch`` (``maxent_torch``: a GPU reimplementation of
maxent.jar's fitting, validated in docs/maxent_torch.md; 7-8x faster, reproducible replicate splits) and ``java``
(maxent.jar). The engine is chosen by the ``engine`` argument or the MAXENT_ENGINE environment variable; unset, it
is ``torch`` when a CUDA GPU is present and ``java`` otherwise (``default_engine``; provenance 2026-10-04).
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import data_root

MAXENT_JAR = Path(os.environ["MAXENT_JAR"]) if os.environ.get("MAXENT_JAR") else data_root() / "tools/maxent.jar"
BETAS = (2, 5, 10, 15, 20)


def vif(x: np.ndarray) -> np.ndarray:
    """Variance inflation factor of each column: diagonal of the inverse correlation matrix."""
    c = np.corrcoef(x, rowvar=False)
    return np.diag(np.linalg.pinv(c))


def vifstep(df: pd.DataFrame, threshold: float = 5.0, seed: int = 0) -> list[str]:
    """Drop the variable with the largest VIF while it is >= ``threshold`` (usdm::vifstep, which also draws a
    random 5,000 rows when given 6,000 or more)."""
    df = df.dropna()
    if len(df) >= 6000:
        df = df.sample(5000, random_state=seed)
    keep = [c for c in df.columns if df[c].std() > 0]
    while len(keep) > 2:
        v = vif(df[keep].values.astype(np.float64))
        i = int(np.argmax(v))
        if v[i] < threshold:
            break
        keep.pop(i)
    return keep


@dataclass
class FitResult:
    beta: float
    cv_auc: dict
    replicate_dirs: list[Path]
    lambdas: list[Path]


def _write_swd(path: Path, label: str, xy: np.ndarray, env: pd.DataFrame) -> None:
    df = env.copy()
    df.insert(0, "y", xy[:, 1])
    df.insert(0, "x", xy[:, 0])
    df.insert(0, "species", label)
    df.to_csv(path, index=False)


def _maxent(outdir: Path, samples: Path, background: Path, beta: float, extra: Sequence[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    args = ["java", f"-mx{os.environ.get('MAXENT_HEAP', '4g')}", "-jar", str(MAXENT_JAR), "nowarnings", "noprefixes", "-a", "-z",
            f"samplesfile={samples}", f"environmentallayers={background}", f"outputdirectory={outdir}",
            f"betamultiplier={beta}", "linear=true", "quadratic=false", "product=false", "threshold=true",
            "hinge=true", "autofeature=false", "responsecurves=false", "jackknife=false", "pictures=false",
            "plots=false", "outputformat=cloglog", *extra]
    subprocess.run(args, check=True, capture_output=True)
    if not (outdir / "maxentResults.csv").exists():
        raise RuntimeError(f"maxent.jar produced no results in {outdir}")


KEEP_SUFFIXES = (".lambdas", "maxentResults.csv", "_samplePredictions.csv", "_omission.csv", "maxent.log",
                 "samples.csv", "background.csv")


def slim(directory: Path) -> None:
    """Drop maxent.jar's bulky per-run outputs (background prediction CSVs, HTML, batch files); keep the
    model inputs (SWD files), coefficients, results tables, sample predictions and omission tables."""
    for f in directory.rglob("*"):
        if f.is_file() and not f.name.endswith(KEEP_SUFFIXES):
            f.unlink()


@contextlib.contextmanager
def _gpu_slot():
    """Limit concurrent GPU fits across worker processes to MAXENT_TORCH_GPU_SLOTS (0 or unset: no limit), using
    lock files, so many CPU workers can share one GPU without exhausting its memory."""
    n = int(os.environ.get("MAXENT_TORCH_GPU_SLOTS", "0"))
    if n <= 0:
        yield
        return
    lockdir = Path(os.environ.get("MAXENT_TORCH_LOCKDIR", "/tmp"))
    while True:
        for i in range(n):
            fh = open(lockdir / f"maxent_torch_gpu_slot_{i}.lock", "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
            return
        time.sleep(0.5)


def default_engine() -> str:
    """``torch`` when a CUDA GPU is usable, else ``java`` (CPU-only machines)."""
    try:
        import torch
        return "torch" if torch.cuda.is_available() else "java"
    except ImportError:
        return "java"


def write_swd_inputs(workdir: Path, label: str, pres_xy: np.ndarray, pres_env: pd.DataFrame, bg_xy: np.ndarray,
                     bg_env: pd.DataFrame) -> tuple[Path, Path]:
    """MaxEnt's samples-with-data inputs: samples.csv (presences) and background.csv, columns species, x, y and the
    selected predictors. Both engines fit from these files; the joint range model reads them too."""
    workdir.mkdir(parents=True, exist_ok=True)
    s, b = workdir / "samples.csv", workdir / "background.csv"
    _write_swd(s, label, pres_xy, pres_env)
    _write_swd(b, "background", bg_xy, bg_env)
    return s, b


def fit_species(workdir: Path, label: str, pres_xy: np.ndarray, pres_env: pd.DataFrame, bg_xy: np.ndarray,
                bg_env: pd.DataFrame, betas: Sequence[float] = BETAS, replicates: int = 5, seed: int = 0,
                engine: str | None = None) -> FitResult:
    engine = engine or os.environ.get("MAXENT_ENGINE") or default_engine()
    s, b = write_swd_inputs(workdir, label, pres_xy, pres_env, bg_xy, bg_env)
    if engine == "torch":
        from . import maxent_torch
        with _gpu_slot():
            best, cv = maxent_torch.fit_species_dir(workdir, label, betas, replicates, seed,
                                                    device=os.environ.get("MAXENT_TORCH_DEVICE", "cuda"))
        d = workdir / "final"
        lambdas = sorted(d.glob(f"{label}_[0-9]*.lambdas"))
        if len(lambdas) != replicates:
            raise RuntimeError(f"expected {replicates} replicate models, found {len(lambdas)} in {d}")
        return FitResult(beta=best, cv_auc=cv, replicate_dirs=[d], lambdas=lambdas)
    if engine != "java":
        raise ValueError(f"unknown MaxEnt engine {engine!r} (java or torch)")

    def cv_auc(beta):                       # one JVM per beta, run concurrently (identical results)
        d = workdir / f"cv_beta{beta}"
        _maxent(d, s, b, beta, ["replicates=5", "replicatetype=crossvalidate", "randomseed=false", "threads=1"])
        res = pd.read_csv(d / "maxentResults.csv")
        return beta, float(res.loc[res.Species.str.contains("average"), "Test AUC"].iloc[0])

    with ThreadPoolExecutor(len(betas)) as ex:
        cv = dict(ex.map(cv_auc, betas))
    best = max(cv, key=cv.get)
    d = workdir / "final"
    _maxent(d, s, b, best, [f"replicates={replicates}", "replicatetype=subsample", "randomtestpoints=25",
                            "randomseed=false", "writebackgroundpredictions=false", f"threads={replicates}"])
    if os.environ.get("KEEP_FULL_MAXENT", "0") != "1":
        slim(workdir)
    lambdas = sorted(d.glob(f"{label}_[0-9]*.lambdas"))
    if len(lambdas) != replicates:
        raise RuntimeError(f"expected {replicates} replicate models, found {len(lambdas)} in {d}")
    return FitResult(beta=best, cv_auc=cv, replicate_dirs=[d], lambdas=lambdas)
