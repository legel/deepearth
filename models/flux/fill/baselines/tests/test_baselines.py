"""Protocol tests of the FluxFill-Bench baselines: output shape, no use of hidden values, determinism under a seed.

Every method is run on a synthetic site (three years of half-hourly data with diurnal and seasonal cycles) with
small configurations so the tests take seconds. The no-leak test passes a window whose hidden entries hold extreme
sentinel values instead of NaN: a method that reads only the visible entries returns the same samples.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "bench"))

from masks import contiguous_block, seed_for  # noqa: E402
from windows import Window, load_window, lst_index  # noqa: E402


def synthetic_table(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2018-01-01", "2020-12-31 23:30", freq="30min")
    n = len(t)
    hour = (t.hour + t.minute / 60).to_numpy()
    doy = t.dayofyear.to_numpy()
    season = np.sin(2 * np.pi * (doy - 100) / 365)
    sw = np.clip(800 * np.sin(np.pi * (hour - 6) / 12), 0, None) * (0.6 + 0.4 * season)
    ta = 10 + 10 * season + 5 * np.sin(np.pi * (hour - 9) / 12) + rng.normal(0, 1, n)
    vpd = np.clip(0.5 + 0.1 * ta + rng.normal(0, 0.3, n), 0, None)
    d = {"t_lst": ((t - pd.Timestamp("1970-01-01")) // pd.Timedelta(minutes=1)).astype(np.int64),
         "TA": ta, "SW_IN": sw, "VPD": vpd, "WS": np.abs(rng.normal(3, 1, n)), "PA": 100 + rng.normal(0, 0.3, n),
         "SWC_1": 30 + 5 * season + rng.normal(0, 1, n), "TS_1": 8 + 8 * season + rng.normal(0, 0.5, n)}
    d["NEE"] = 2 + 0.1 * ta - 0.012 * sw * (0.5 + 0.5 * season) + rng.laplace(0, 1, n)
    d["LE"] = 0.25 * sw + 10 * vpd + rng.normal(0, 15, n)
    d["H"] = 0.3 * sw - 2 * ta + rng.normal(0, 15, n)
    for k in ("TA", "SW_IN", "VPD", "WS", "PA"):
        d["ERA5_" + k] = d[k] + rng.normal(0, 0.05 * (np.std(d[k]) + 1e-9), n)
    tab = pd.DataFrame(d)
    for k in ("NEE", "LE", "H", "TA", "SW_IN", "VPD", "WS", "PA", "SWC_1", "TS_1"):
        gaps = rng.uniform(size=n) < (0.3 if k in ("NEE", "LE", "H") else 0.05)
        tab.loc[gaps, k] = np.nan
        tab[f"{k}_flag"] = np.where(gaps, 1, 0)
    return tab


@pytest.fixture(scope="module")
def case():
    tab = synthetic_table()
    w = load_window(tab, "XX-Syn", "2019-04-01")
    idx = lst_index(tab)
    years = set(range(w.t[0].year, w.t[-1].year + 1))
    ctx = {"table": tab[~idx.year.isin(list(years))], "index": idx[~idx.year.isin(list(years))]}
    V, T = w.values.shape
    mask = contiguous_block(w.rows("tower"), V, T, 7, np.random.default_rng(seed_for("XX-Syn", "2019-04-01", "M2", 7)))
    vis = Window(w.site, w.start, w.t, w.keys, np.where(mask, np.nan, w.values), w.measured & ~mask, w.provenance)
    sentinel = Window(w.site, w.start, w.t, w.keys, np.where(mask, 1e6, w.values), w.measured, w.provenance)
    return w, mask, vis, sentinel, ctx


def make(name):
    if name == "irvin_rf":
        from irvin_rf import IrvinRF
        return IrvinRF(n_members=3, rf_params=dict(n_estimators=5, max_features=1.0, min_samples_leaf=4,
                                                   max_samples=0.2, bootstrap=True), n_jobs=1)
    if name == "vekuri_de":
        from vekuri_de import HPARAMS, VekuriDE
        return VekuriDE(hparams=dict(HPARAMS, epochs=2, members=2, batch=1024), threads=1)
    if name == "liu_xgb":
        from liu_xgb import XGB_PARAMS, LiuXGB
        return LiuXGB(params=dict(XGB_PARAMS, n_estimators=20), n_jobs=1, use_clay=False)
    raise KeyError(name)


PER_SITE = ["irvin_rf", "vekuri_de", "liu_xgb"]


@pytest.mark.parametrize("name", PER_SITE)
def test_shape_and_fill(case, name):
    w, mask, vis, _, ctx = case
    s = make(name).predict(vis, mask, ctx)
    assert s.ndim == 3 and s.shape[1:] == w.values.shape and s.shape[0] >= 64
    for i, k in enumerate(w.keys):
        if k in ("NEE", "LE", "H"):
            assert np.isfinite(s[:, i, mask[i]]).all(), k
            assert np.isnan(s[:, i, ~mask[i]]).all(), k          # nothing written outside the mask


@pytest.mark.parametrize("name", PER_SITE)
def test_no_hidden_access(case, name):
    w, mask, vis, sentinel, ctx = case
    a = make(name).predict(vis, mask, ctx)
    b = make(name).predict(sentinel, mask, ctx)
    np.testing.assert_array_equal(np.nan_to_num(a, nan=-7.0), np.nan_to_num(b, nan=-7.0))


@pytest.mark.parametrize("name", PER_SITE)
def test_deterministic(case, name):
    w, mask, vis, _, ctx = case
    a, b = make(name).predict(vis, mask, ctx), make(name).predict(vis, mask, ctx)
    np.testing.assert_array_equal(np.nan_to_num(a, nan=-7.0), np.nan_to_num(b, nan=-7.0))


def test_irvin_delta_fix():
    from irvin_rf import temporal, temporal_released
    t = pd.date_range("2019-03-01", periods=500, freq="30min")
    full, part = temporal(t), temporal(t[200:])
    np.testing.assert_allclose(full[200:], part)                   # same time, same delta in any frame
    rel = temporal_released(t[200:])
    assert rel[0, 2] == 0.0 and not np.allclose(rel[:, 2], full[200:, 2])   # released: origin moves with the frame


def test_irvin_platt_from_context_only(case):
    w, mask, vis, _, ctx = case
    m = make("irvin_rf")
    m.predict(vis, mask, ctx)
    assert m.log and all(r["n_cal"] > 0 and np.isfinite(r["platt"]) for r in m.log)


def test_vekuri_member_trajectories(case):
    """Samples of one member share its mean trajectory: the sample mean over a member's samples is smooth in the
    member index structure (members differ, samples of a member differ only by Laplace noise)."""
    w, mask, vis, _, ctx = case
    m = make("vekuri_de")
    s = m.predict(vis, mask, ctx)
    i = w.keys.index("NEE")
    hid = mask[i]
    g0, g1 = s[0::2, i, hid].mean(0), s[1::2, i, hid].mean(0)
    assert not np.allclose(g0, g1)


def test_quantile_samples():
    from common import quantile_samples
    lv = np.array([0.1, 0.5, 0.9])
    q = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
    s = quantile_samples(lv, q, 20000, np.random.default_rng(0))
    assert s.shape == (20000, 2)
    np.testing.assert_allclose(np.median(s, 0), [0.0, 1.0], atol=0.03)
    np.testing.assert_allclose(np.quantile(s, 0.9, 0), [1.0, 2.0], atol=0.05)


def test_holdout_blocks_fraction():
    from common import holdout_blocks
    t = pd.date_range("2018-01-01", periods=48 * 400, freq="30min")
    y = np.where(np.random.default_rng(1).uniform(size=len(t)) < 0.5, 1.0, np.nan)
    h = holdout_blocks(t, y, np.random.default_rng(0))
    frac = np.isfinite(y[h]).sum() / np.isfinite(y).sum()
    assert 0.1 <= frac < 0.14


def _tiny_ckpt(tmp_path, kind, monkeypatch):
    import torch
    import pypots_imputers as P
    if kind == "csdi":
        monkeypatch.setitem(P.CSDI_CFG, "n_diffusion_steps", 3)
        monkeypatch.setitem(P.CSDI_CFG, "n_layers", 1)
    else:
        monkeypatch.setitem(P.SAITS_CFG, "n_layers", 1)
        monkeypatch.setitem(P.SAITS_CFG, "d_model", 32)
    V = len(P.vars_()[0])
    torch.manual_seed(0)
    net = P.build(kind, V)
    p = tmp_path / f"{kind}.pt"
    torch.save({"kind": kind, "state": net.state_dict(), "V": V, "stats": {"mean": [0.0] * V, "std": [1.0] * V}}, p)
    return str(p)


@pytest.mark.parametrize("kind", ["saits", "csdi"])
def test_pypots_protocol(case, tmp_path, monkeypatch, kind):
    import json
    import pypots_imputers as P
    w, mask, vis, sentinel, ctx = case
    ck = _tiny_ckpt(tmp_path, kind, monkeypatch)
    if kind == "saits":
        sp = {k: {"all": 1.0, "by_class": [1.0] * 5} for k in w.keys}
        Path(ck).with_name("saits_spread.json").write_text(json.dumps(sp))
        mk = lambda: P.SAITSMethod(ck, device="cpu")   # noqa: E731
    else:
        mk = lambda: P.CSDIMethod(ck, device="cpu", n_samples=4, batch=4)   # noqa: E731
    a = mk().predict(vis, mask, ctx)
    b = mk().predict(sentinel, mask, ctx)
    c = mk().predict(vis, mask, ctx)
    assert a.shape[1:] == w.values.shape
    for i in w.rows("tower"):
        if mask[i].any():
            assert np.isfinite(a[:, i, mask[i]]).all()
    np.testing.assert_array_equal(np.nan_to_num(a, nan=-7.0), np.nan_to_num(b, nan=-7.0))
    np.testing.assert_array_equal(np.nan_to_num(a, nan=-7.0), np.nan_to_num(c, nan=-7.0))


def test_chronos2_protocol(case):
    from pathlib import Path as _P
    hub = _P("/home/photon/4tb/hf/hub/models--amazon--chronos-2")
    if not any(hub.glob("snapshots/*/model.safetensors")):
        pytest.skip("Chronos-2 weights not downloaded")
    import os
    os.environ.setdefault("HF_HOME", "/home/photon/4tb/hf")
    from chronos2 import Chronos2
    w, mask, vis, sentinel, ctx = case
    m = Chronos2(device="cpu", context_steps=512)
    a = m.predict(vis, mask, ctx)
    b = m.predict(sentinel, mask, ctx)
    assert a.shape[1:] == w.values.shape and a.shape[0] >= 64
    i = w.keys.index("NEE")
    assert np.isfinite(a[:, i, mask[i]]).all()
    np.testing.assert_allclose(np.nan_to_num(a, nan=-7.0), np.nan_to_num(b, nan=-7.0))
