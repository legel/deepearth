"""Every option of recipe.OPTIONS: forward and backward run, values stay finite, and an option built to start from the
base network does so. The ablation study depends on a flag changing one thing."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from data import STEPS_PER_DAY, V_TAB, WINDOW_DAYS, Stats, time_features
from net import FluxFillNet, QUANTILE_KNOTS, marginal_mean, marginal_nll, mixture_cdf, monotone_quantiles, quantile_inverse, row_modalities, sample_marginal_params, sincos_1d
from recipe import COMPONENTS, FINAL, OPTIONS, net_kwargs
from solar import sun_features
from train import PowerFunctionEMA, losses

KW = dict(d=64, depth=2, heads=4, flow_width=32, memory=True, mem_residual=True, gated_base=True, gate_scale=True,
          edge_continue=True, year_anomaly=True, img_nll=True, sinr_place=0)
ARMS = [dict(mot="ffn"), dict(mot="all"), dict(schedule="time,time,time,time_var", depth=1), dict(attn_gate=True),
        dict(rope_partial=0.5), dict(rmsnorm=True), dict(swiglu=True), dict(plr=True), dict(time_enc="prithvi"),
        dict(time_enc="sinr"), dict(solar=True), dict(off=["rope"]), dict(off=["time_attn"]), dict(off=["var_attn"]),
        dict(off=["day_emb", "doy", "site", "prov_emb", "var_emb", "mask_token"]),
        dict(quantile_head=True, gated_base=False, gate_scale=False, edge_continue=False, year_anomaly=False),
        dict(fgn_noise=8, gated_base=False, gate_scale=False, edge_continue=False, year_anomaly=False)]


def batch(n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    D, K = WINDOW_DAYS, STEPS_PER_DAY
    z = torch.randn(n, V_TAB, D, K, generator=g)
    obs = torch.rand(n, V_TAB, D, K, generator=g) > 0.3
    tm = ~obs & (torch.rand(n, V_TAB, D, K, generator=g) > 0.5)
    return {"z": z * obs, "obs": obs, "tgt": z, "tmask": tm, "present": torch.ones(n, V_TAB, dtype=torch.bool),
            "img_cls": torch.randn(n, D, 1024, generator=g), "img_meta": torch.zeros(n, D, 8),
            "img_obs": torch.zeros(n, D, dtype=torch.bool), "img_tgt_cls": torch.zeros(n, D, 1024),
            "img_tgt": torch.zeros(n, D, dtype=torch.bool), "img_present": torch.zeros(n, dtype=torch.bool),
            "doy": torch.randn(n, D, 4, generator=g), "site": torch.tensor([[0.5, 0.5, 0.5, 0.1]] * n),
            "time": torch.tensor(time_features(pd.Timestamp("2019-07-15")))[None].expand(n, D, 2),
            "sun": torch.tensor(sun_features(pd.Timestamp("2019-07-15"), D, 45.0, -100.0, -6.0))[None].expand(n, D, 96),
            "mem_m": torch.zeros(n, V_TAB, D, K), "mem_s": torch.ones(n, V_TAB, D, K), "mem_k": torch.zeros(n, V_TAB, D, K),
            "mem_h": torch.ones(n, V_TAB, D, K, dtype=torch.bool)}


@pytest.mark.parametrize("arm", ARMS, ids=lambda a: "-".join(f"{k}" for k in a))
def test_option_trains(arm):
    torch.manual_seed(0)
    net = FluxFillNet(**{**KW, **arm})
    L = losses(net, batch(), False)
    L["total"].backward()
    assert torch.isfinite(L["total"])
    used = [p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None]
    assert used and all(used)


@pytest.mark.parametrize("arm", [dict(mot="ffn"), dict(mot="all"), dict(solar=True), dict(plr=True), dict(time_enc="sinr")],
                         ids=lambda a: "-".join(f"{k}={v}" for k, v in a.items()))
def test_option_starts_from_the_base_network(arm):
    torch.manual_seed(0)
    base = FluxFillNet(**KW).eval()
    torch.manual_seed(0)
    net = FluxFillNet(**{**KW, **arm}).eval()
    b = batch()
    with torch.no_grad():
        a, c = base(b)[1], net(b)[1]
    for x, y in zip(a, c):
        assert torch.allclose(x, y, atol=1e-5), arm


def test_base_network_ignores_the_new_inputs():
    torch.manual_seed(0)
    net = FluxFillNet(**KW).eval()
    b, c = batch(), batch()
    c["sun"], c["time"] = torch.zeros_like(c["sun"]), torch.zeros_like(c["time"])
    with torch.no_grad():
        assert torch.equal(net(b)[1][0], net(c)[1][0])


def test_modalities_cover_every_row():
    m = row_modalities()
    assert m.shape == (V_TAB + 1,) and m[-1] == 4 and set(m) == {0, 1, 2, 3, 4}


def test_quantile_head_interface_is_consistent():
    torch.manual_seed(0)
    raw = torch.randn(3, 2, 21)
    q = monotone_quantiles(raw, QUANTILE_KNOTS)
    assert (q[..., 1:] >= q[..., :-1]).all()
    t = torch.tensor(QUANTILE_KNOTS)
    for i in (0, 10, 20):
        assert torch.allclose(quantile_inverse(q, t[i].expand(3, 2)), q[..., i], atol=1e-5)
    y = q[..., 10]
    assert np.allclose(mixture_cdf((q.numpy(),), y.numpy()), 0.5, atol=1e-6)
    draws = sample_marginal_params((q,), 2000)
    assert torch.allclose(draws.median(0).values, q[..., 10], atol=0.25)
    assert marginal_mean((q,)).shape == (3, 2) and marginal_nll(y, (q,)).shape == (3, 2)


def test_fgn_members_differ_and_share_noise_per_window():
    torch.manual_seed(0)
    arm = dict(fgn_noise=8, gated_base=False, gate_scale=False, edge_continue=False, year_anomaly=False)
    net = FluxFillNet(**{**KW, **arm}).eval()
    for p in net.parameters():                                  # a visible effect of the noise
        if p.dim() > 1:
            p.data.add_(torch.randn_like(p) * 0.05)
    b = batch()
    b["fgn_z"] = torch.ones(2, 8)
    with torch.no_grad():
        a, c = net(b)[1][0], net(b)[1][0]
        b["fgn_z"] = -torch.ones(2, 8)
        d = net(b)[1][0]
    assert torch.equal(a, c) and not torch.allclose(a, d)


def test_asinh_stats_round_trip():
    st = Stats(np.array([1.0, 2.0]), np.array([2.0, 4.0]), True)
    x = np.array([[1.0, 100.0], [-50.0, 2.0]])
    assert np.allclose(st.denorm(st.norm(x)), x)


def test_solar_geometry():
    s = sun_features(pd.Timestamp("2020-06-21"), 1, 40.0, -105.0, -7.0)
    alt, toa = s[0, :48], s[0, 48:]
    noon = alt.argmax() / 2                                     # local standard time of the highest sun
    assert 11.0 < noon < 13.0
    assert alt.max() == pytest.approx(np.cos(np.radians(40 - 23.44)), abs=0.01) and toa.min() == 0.0
    assert (sun_features(pd.Timestamp("2020-06-21"), 2, np.nan, 0.0, 0.0) == 0).all()


def test_ema_follows_edm2():
    net = torch.nn.Linear(2, 2)
    e = PowerFunctionEMA(net, 0.05)
    assert e.exp == pytest.approx(float(np.roots([1, 7, 16 - 400, 12 - 400]).real.max()))
    with torch.no_grad():
        net.weight.add_(1.0)
    before = e.ema.weight.clone()
    e.update(1000, 8)
    assert not torch.equal(before, e.ema.weight)


def test_recipe_builds_old_configs():
    kw = net_kwargs({"d": 192, "depth": 6, "heads": 6, "memory": True, "no_imagery": True, "memory_skew_fluxes": False})
    assert kw["use_imagery"] is False and kw["d"] == 192 and "schedule" not in kw
    FluxFillNet(**kw)
    assert all(o.source for o in OPTIONS) and len({c.key for c in COMPONENTS}) == len(COMPONENTS)
    final = FINAL.split()
    for c in COMPONENTS:
        for f in c.remove.split():
            assert f in final, (c.key, f)
