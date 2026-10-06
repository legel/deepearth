"""The PyTorch MaxEnt fitter (maxent_torch) against maxent.jar.

Unit checks use values printed by Java itself (java.util.Random, Double.toString, maxent's
SortedFeatureGenerator.precision). Parity checks refit stored maxent.jar runs on the same SWD inputs and compare
the models: a single run (betamultiplier 2) and 5-fold cross-validation (betamultiplier 5), both deterministic in
maxent.jar. Reference runs (maxent.jar 3.4.4, Arctostaphylos auriculata, 101 presences, 10,000 background points,
8 WorldClim variables) live in tools/bench/torch_parity under the data root; the parity tests are skipped without
them.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ranges import maxent_torch as mt
from ranges.config import data_root

REF = data_root() / "tools/bench/torch_parity"
LABEL = "Arctostaphylos_auriculata"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
needs_ref = pytest.mark.skipif(not (REF / "single_b2").exists(), reason="maxent.jar reference runs not available")


def test_java_random_matches_java():
    r = mt.JavaRandom(0)
    assert [r.next_double() for _ in range(4)] == [0.730967787376657, 0.24053641567148587, 0.6374174253501083,
                                                    0.5504370051176339]
    assert mt.JavaRandom(11111).next_double() == 0.14877869593833193


@pytest.mark.parametrize("x,s", [(0.001, "0.001"), (1e-4, "1.0E-4"), (1234567.0, "1234567.0"), (1.0e7, "1.0E7"),
                                 (3.1356288845074687e-4, "3.1356288845074687E-4"), (777.8402100000001, "777.8402100000001"),
                                 (-0.0, "-0.0"), (100.0, "100.0"), (0.1 + 0.2, "0.30000000000000004"),
                                 (-2.2911633964871294, "-2.2911633964871294")])
def test_double_to_string_matches_java(x, s):
    assert mt.jstr(x) == s


def test_precision_matches_java():
    vals = [12.5, 0.0, 617.73267, 0.6666667, 1000.0, 0.3, 13.8458325, -3.2333333, 4276.0, 1.97021e-9, 159.0]
    java = [0.05, 0.0, 5.0e-4, 5.000000000000001e-7, 500.0, 0.05, 5.0e-5, 5.000000000000001e-6, 0.5,
            5.000000000000001e-15, 0.5]
    assert list(mt.precision(vals)) == java


def test_group_starts_matches_greedy_scan():
    def greedy(u, prec):
        st = [0]
        for i in range(1, len(u)):
            if u[i] - u[st[-1]] > prec:
                st.append(i)
        return u[st]
    rng = np.random.default_rng(0)
    for _ in range(200):
        u = np.unique(np.round(rng.random(rng.integers(2, 2000)) * rng.choice([1, 10, 1000]), rng.integers(1, 6)))
        prec = float(rng.choice([1e-3, 5e-3, 0.02, 0.3, 0.0]))
        assert np.array_equal(mt.group_starts(u, prec), greedy(u, prec))


def test_cv_folds_and_betas():
    assert mt.betas_for(5, 1.0) == (1.0, 1.95, 0.5)
    assert mt.betas_for(101, 2.0) == (0.1, 2.0, 1.0)
    assert np.isclose(mt.betas_for(20, 1.0)[0], 0.6)
    f = mt.cv_folds(101)
    assert sorted(np.bincount(f)) == [20, 20, 20, 20, 21]


def _lambdas(path):
    rows, consts = {}, {}
    for line in Path(path).read_text().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 2:
            consts[p[0]] = float(p[1])
        else:
            rows[(p[0], p[2], p[3]) if p[0][0] in "'`" else p[0]] = float(p[1])
    return rows, consts


def _assert_same_model(java, torch_, lam_tol=1e-8):
    a, ca = _lambdas(java)
    b, cb = _lambdas(torch_)
    ka, kb = [k for k, v in a.items() if v != 0], [k for k, v in b.items() if v != 0]
    assert ka == kb                                   # same features, in the same (export) order
    assert max(abs(a[k] - b[k]) for k in ka) < lam_tol
    for k in ("linearPredictorNormalizer", "densityNormalizer", "entropy"):
        assert abs(ca[k] - cb[k]) <= 1e-9 * max(1.0, abs(ca[k])), k
    assert ca["numBackgroundPoints"] == cb["numBackgroundPoints"]


@needs_ref
def test_single_run_matches_maxent_jar(tmp_path):
    variables, B, S, xy = mt.read_swd(REF / "samples.csv", REF / "background.csv")
    fits = mt.fit_runs([mt.Run(LABEL, 2.0, B, S, xy)], DEVICE)
    mt.write_run_dir(tmp_path, LABEL, fits, variables)
    _assert_same_model(REF / f"single_b2/{LABEL}.lambdas", tmp_path / f"{LABEL}.lambdas")
    rj = pd.read_csv(REF / "single_b2/maxentResults.csv").iloc[0]
    rt = pd.read_csv(tmp_path / "maxentResults.csv").iloc[0]
    for col in ("Regularized training gain", "Unregularized training gain", "Iterations", "Training AUC",
                "#Background points", "Entropy", "Equal training sensitivity and specificity Cloglog threshold",
                "10 percentile training presence Cloglog threshold"):
        assert rj[col] == rt[col], col
    spj = pd.read_csv(REF / f"single_b2/{LABEL}_samplePredictions.csv")
    spt = pd.read_csv(tmp_path / f"{LABEL}_samplePredictions.csv")
    assert np.allclose(spj["Cloglog prediction"], spt["Cloglog prediction"], rtol=0, atol=1e-9)


@needs_ref
def test_cross_validation_matches_maxent_jar(tmp_path):
    from ranges import evaluate, project
    variables, B, S, xy = mt.read_swd(REF / "samples.csv", REF / "background.csv")
    fits = mt.fit_runs(mt.cv_runs(LABEL, 5.0, B, S, xy), DEVICE)
    mt.write_run_dir(tmp_path, LABEL, fits, variables)
    for j in range(5):
        _assert_same_model(REF / f"cv_b5/{LABEL}_{j}.lambdas", tmp_path / f"{LABEL}_{j}.lambdas")
    rj = pd.read_csv(REF / "cv_b5/maxentResults.csv").set_index("Species")
    rt = pd.read_csv(tmp_path / "maxentResults.csv").set_index("Species")
    assert list(rj.columns[:10]) == list(rt.columns[:10])
    for col in ("Test AUC", "Training AUC", "Regularized training gain", "Iterations", "#Test samples",
                "Test gain", "Entropy"):
        assert (rj.loc[rt.index, col] == rt[col]).all(), col
    # Threshold rules compare each training presence with its own copy among the background points, whose density
    # maxent.jar takes from its running linear predictor: such exact ties break by rounding, so the rules may
    # differ by one background point (area 1e-4) and the ESS threshold slightly.
    for rule in mt.RULES:
        assert (rj.loc[rt.index, f"{rule} area"] - rt[f"{rule} area"]).abs().max() <= 2e-4, rule
        assert (rj.loc[rt.index, f"{rule} Cloglog threshold"] - rt[f"{rule} Cloglog threshold"]).abs().max() <= 2e-3, rule
    assert mt.cv_test_auc(tmp_path / "maxentResults.csv") == mt.cv_test_auc(REF / "cv_b5/maxentResults.csv")
    # the files feed the rest of the pipeline unchanged
    reps = project.load_replicates(tmp_path, LABEL, variables, device="cpu")
    assert len(reps) == 5
    assert evaluate.daru_metrics(tmp_path, LABEL, variables, REF / "background.csv")["replicates"] == 5
