"""Parity of the PyTorch MaxEnt evaluator with maxent.jar's own predictions.

Requires a maxent.jar run directory produced with ``writebackgroundpredictions=true`` and the SWD files it
was fitted on, in tools/bench under the data root (DEEPEARTH_HABITAT_DATA).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from ranges.config import data_root
from ranges.maxent import MaxentModel

BENCH = data_root() / "tools/bench"
pytestmark = pytest.mark.skipif(not (BENCH / "runs/ref").exists(), reason="maxent.jar reference run not available")


def _load():
    pres = pd.read_csv(BENCH / "pres.csv")
    variables = [c for c in pres.columns if c not in ("species", "x", "y")]
    model = MaxentModel.from_lambdas(BENCH / "runs/ref/Cercis_canadensis.lambdas", variables=variables)
    return pres, variables, model


def test_sample_predictions_match_to_machine_precision():
    pres, variables, model = _load()
    sp = pd.read_csv(BENCH / "runs/ref/Cercis_canadensis_samplePredictions.csv")
    keyed = pres.assign(k=list(zip(pres.x.round(6), pres.y.round(6)))).drop_duplicates("k").set_index("k")
    sp["k"] = list(zip(sp.X.round(6), sp.Y.round(6)))
    sp = sp[sp.k.isin(keyed.index)]
    x = torch.tensor(keyed.loc[sp.k, variables].values, dtype=torch.float64)
    raw = model.raw(x).numpy()
    assert np.max(np.abs(raw - sp["Raw prediction"].values) / sp["Raw prediction"].values) < 1e-12
    assert np.max(np.abs(model.cloglog(x).numpy() - sp["Cloglog prediction"].values)) < 1e-12


def test_background_predictions_match_printed_precision():
    _, variables, model = _load()
    bg = pd.read_csv(BENCH / "bg.csv")
    ref = pd.read_csv(BENCH / "runs/ref/Cercis_canadensis_backgroundPredictions.csv")
    x = torch.tensor(bg[variables].values[: len(ref)], dtype=torch.float64)
    # maxent.jar prints cloglog with 9 decimals
    assert np.max(np.abs(model.cloglog(x).numpy() - ref["Cloglog"].values)) <= 5e-10 + 1e-15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU invariance check")
def test_cell_values_independent_of_batch_size():
    _, variables, _ = _load()
    model = MaxentModel.from_lambdas(BENCH / "runs/ref/Cercis_canadensis.lambdas", variables=variables,
                                     dtype=torch.float32, device="cuda")
    bg = pd.read_csv(BENCH / "bg.csv")
    x = torch.tensor(bg[variables].values, dtype=torch.float32, device="cuda").repeat(40, 1)
    full = model.cloglog(x)
    for n in (1, 7, 1000, 12345, 200_000):
        assert torch.equal(model.cloglog(x[:n]), full[:n])
