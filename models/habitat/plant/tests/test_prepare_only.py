"""The prepare-only path and grouped cleaning change nothing about the data a model is fitted to.

* ``clean_coordinates_many`` (several record sets in one R session) flags exactly what one session per set flags,
  including when one set is large enough (>= 10,000 records) for CoordinateCleaner's outlier test to switch to its
  raster approximation, which in a single call would apply to every species in that call. Needs the R environment.
* ``run_species(prepare_only=True)`` writes samples.csv and background.csv byte-identical to the stored full run
  of a fitted species (Abies procera: the native-first cleaning order, provenance D11). Needs the project data.
* A prepare-only summary does not count as a fitted species for a fitting run (``run_fit.finished``).
"""
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ranges.config import data_root

ROOT = data_root()
REPO = Path(__file__).resolve().parents[1]


def _r_available() -> bool:
    try:
        conda_sh = os.path.expanduser(os.environ.get("CONDA_SH", "~/miniconda3/etc/profile.d/conda.sh"))
        cmd = f"source {conda_sh} && conda activate {os.environ.get('R_ENV', 'daru')} && Rscript -e 'library(CoordinateCleaner)'"
        return subprocess.run(["bash", "-lc", cmd], capture_output=True).returncode == 0
    except Exception:
        return False


def _run_fit():
    spec = importlib.util.spec_from_file_location("run_fit", REPO / "scripts/run_fit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_prepared_summary_is_not_a_fit(tmp_path):
    rf = _run_fit()
    p = tmp_path / "summary.json"
    assert not rf.finished(p)
    p.write_text(json.dumps({"species": "X y", "stage": "prepared", "n_presence": 10}))
    assert rf.finished(p, prepare_only=True)
    assert not rf.finished(p)                       # a fitting run refits it
    p.write_text(json.dumps({"species": "X y", "beta": 2}))
    assert rf.finished(p) and rf.finished(p, prepare_only=True)
    p.write_text("")                                # interrupted write
    assert not rf.finished(p, prepare_only=True)


@pytest.mark.skipif(not _r_available(), reason="needs the R environment with CoordinateCleaner")
def test_grouped_cleaning_equals_separate_sessions():
    seas = ROOT / "raw/geo/ne_50m_land/ne_50m_land.shp"
    if seas.exists():                                # the local copy of CoordinateCleaner's sea reference
        os.environ.setdefault("CC_SEAS_REF", str(seas))
    if not os.environ.get("CC_SEAS_REF"):
        pytest.skip("needs CoordinateCleaner's sea reference: CC_SEAS_REF or raw/geo/ne_50m_land (fetch_sources.sh geo)")
    from ranges import occurrences
    rng = np.random.default_rng(0)

    def species(name, n, lon, lat, spread):
        d = pd.DataFrame({"species": name, "decimalLongitude": lon + rng.normal(0, spread, n),
                          "decimalLatitude": lat + rng.normal(0, spread, n)})
        d.loc[: n // 50, ["decimalLongitude", "decimalLatitude"]] = [lon + 40, lat - 30]   # far outliers
        d.loc[n // 50 + 1: n // 25, ["decimalLongitude", "decimalLatitude"]] = 0.0          # zero coordinates
        return d.round(4)

    sets = [species("Alpha one", 300, -120.0, 38.0, 1.0), species("Beta two", 12_000, -95.0, 40.0, 3.0),
            species("Alpha one", 40, -120.5, 38.5, 0.5).iloc[0:0], species("Gamma three", 60, -80.0, 35.0, 0.3)]
    grouped = occurrences.clean_coordinates_many(sets)
    for f, g in zip(sets, grouped):
        if len(f):
            alone = occurrences.clean_coordinates(f)
            assert np.array_equal(alone.cc_valid.values, g.cc_valid.values)
            assert not g.cc_valid.all() and g.cc_valid.any()
        else:
            assert len(g) == 0 and "cc_valid" in g


@pytest.mark.skipif(not (ROOT / "work/natives/occurrences/Abies_procera/maxent/samples.csv").exists()
                    or not (ROOT / "raw/gbif_full/parquet").exists() or not _r_available(),
                    reason="needs the project data, the stored Abies procera fit and the R environment")
def test_prepare_only_reproduces_fitted_inputs(tmp_path):
    import pyarrow.dataset as ds
    from ranges import pipeline
    rf = _run_fit()
    os.environ.setdefault("CC_SEAS_REF", str(ROOT / "raw/geo/ne_50m_land/ne_50m_land.shp"))
    res = pipeline.Resources.load(ROOT, ROOT / "work/sbm/sbm_combined.csv", ROOT / "work/bias_grid_behrmann10km.npz")
    parts = [str(f) for f in sorted((ROOT / "raw/gbif_full/parquet").rglob("*")) if f.is_file() and f.stat().st_size > 0]
    g = ds.dataset(parts, format="parquet").to_table(columns=list(rf.COLS), filter=ds.field("species") == "Abies procera")
    g = g.to_pandas().rename(columns=rf.COLS)
    s = pipeline.run_species("Abies procera", g, res, tmp_path, presence_mode="occurrences", background_mode="bias",
                             prepare_only=True)
    ref = ROOT / "work/natives/occurrences/Abies_procera"
    stored = json.loads((ref / "summary.json").read_text())
    assert s["stage"] == "prepared" and "beta" not in s and not (tmp_path / "hull.gpkg").exists()
    assert s["cleaning_order"] == stored["cleaning_order"] == "native_first"
    for k in ("n_gbif", "n_native", "n_thinned", "n_presence", "n_background", "calibration_ecoregions", "predictors"):
        assert s[k] == stored[k], k
    for f in ("samples.csv", "background.csv"):
        assert (tmp_path / "maxent" / f).read_bytes() == (ref / "maxent" / f).read_bytes(), f
    with pytest.raises(ValueError):
        pipeline.run_species("Abies procera", g, res, tmp_path / "x", presence_mode="hull", prepare_only=True)
