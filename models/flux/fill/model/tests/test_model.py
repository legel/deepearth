"""Unit tests of FluxFill tokens, masks, heads and the no-leak property, on a synthetic site."""

import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats as sst

import masking
from data import (IS_REAN, IS_TOWER, IS_TOWER_MET, STEPS_PER_DAY, T_WIN, V_TAB, VARS, SiteRecord, Stats,
                  heldout_year, valid_starts, window_tokens, window_values)
from net import FluxFillNet, sample_student_t, student_t_nll
from sampler import collate, make_example


def synthetic_record(days=800, seed=0, absent=("FCH4", "G")):
    rng = np.random.default_rng(seed)
    T = days * STEPS_PER_DAY
    tt = np.arange(T) / STEPS_PER_DAY
    vals = np.stack([np.sin(2 * np.pi * tt) * (1 + i) + 0.1 * rng.standard_normal(T) + i for i in range(V_TAB)])
    vals = vals.astype(np.float32)
    for v in absent:
        vals[VARS.index(v)] = np.nan
    vals[0, rng.random(T) < 0.3] = np.nan
    return SiteRecord("XX-Syn", pd.Timestamp("2016-01-01"), vals, lat=45.0, lon=-100.0, elev=300.0)


def stats_of(rec):
    m = np.nanmean(rec.values, 1).astype(np.float64)
    s = np.nanstd(rec.values, 1).astype(np.float64)
    return Stats(np.nan_to_num(m), np.where(np.isfinite(s) & (s > 0), s, 1.0))


def test_tokens_roundtrip_and_day_alignment():
    rec = synthetic_record()
    st = stats_of(rec)
    vals = window_values(rec, 10)
    z, meas = window_tokens(vals, st)
    assert z.shape == (V_TAB, 180, STEPS_PER_DAY)
    back = st.denorm(z.reshape(V_TAB, -1).astype(np.float64))
    ok = meas.reshape(V_TAB, -1)
    assert np.allclose(back[ok], vals[ok], atol=1e-4)
    # token (v, d, k) is half-hour k of day d
    assert z[5, 3, 7] == pytest.approx((vals[5, 3 * 48 + 7] - st.mean[5]) / st.std[5], abs=1e-5)


def test_heldout_year_and_valid_starts():
    rec = synthetic_record(days=1200)
    rec.heldout_year = heldout_year(rec)
    assert rec.heldout_year == 2018            # 2016-01-01 + 1200 d ends in April 2019
    st = valid_starts(rec, exclude_heldout=True)
    days = rec.t0 + pd.to_timedelta(st, unit="D")
    ends = days + pd.Timedelta(days=179)
    assert not ((days.year == 2018) | (ends.year == 2018)).any()
    assert valid_starts(rec, exclude_heldout=False).size > st.size


def test_masks_families():
    rng = np.random.default_rng(1)
    present = np.ones(V_TAB, bool)
    present[VARS.index("FCH4")] = False
    for _ in range(300):
        hide, img_hide, fam = masking.draw(rng, present)
        assert hide.shape == (V_TAB, T_WIN) and img_hide.shape == (180,)
        assert not hide[~present].any()
        r = hide[IS_REAN]
        assert r.all() or not r.any()           # reanalysis only by whole-window dropout
        if fam.startswith("r3"):
            assert hide[present & IS_TOWER].all()
            assert not r.any()


def make_batch(rec, st, hide, seed=0):
    ex = make_example(rec, 20, st, hide, np.zeros(180, bool), None)
    return collate([ex])


def test_no_leak_of_hidden_values():
    torch.manual_seed(0)
    rec = synthetic_record()
    st = stats_of(rec)
    hide = np.zeros((V_TAB, T_WIN), bool)
    hide[IS_TOWER, 1000:3000] = True
    net = FluxFillNet(d=32, depth=2, heads=4).eval()
    b1 = make_batch(rec, st, hide)
    rec2 = synthetic_record()
    a = 20 * STEPS_PER_DAY
    rec2.values[IS_TOWER, a + 1000:a + 3000] += 50.0      # change only hidden values
    b2 = make_batch(rec2, st, hide)
    assert not torch.equal(b1["tgt"], b2["tgt"])
    with torch.no_grad():
        o1, o2 = net(b1)[1], net(b2)[1]
    for x, y in zip(o1, o2):
        assert torch.equal(x, y)


def test_absent_rows_do_not_affect_present_rows():
    torch.manual_seed(0)
    rec = synthetic_record()
    st = stats_of(rec)
    net = FluxFillNet(d=32, depth=2, heads=4).eval()
    b = make_batch(rec, st, np.zeros((V_TAB, T_WIN), bool))
    i = VARS.index("FCH4")
    assert not b["present"][0, i]
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    b2["z"][0, i] = 3.0
    b2["obs"][0, i] = True                      # content of an absent row changes, presence does not
    with torch.no_grad():
        h1, h2 = net.encode(b), net.encode(b2)
    keep = b["present"][0].tolist() + [False]
    keep = torch.tensor(keep)
    # rows other than the absent one see it only through variable attention, where it is not a key
    assert torch.allclose(h1[0, keep], h2[0, keep], atol=1e-5)


def test_student_t_nll_matches_scipy_and_sampler():
    y = torch.tensor([0.3, -2.0, 5.0])
    loc, sc, df = torch.tensor([0.0, 1.0, -1.0]), torch.tensor([1.0, 0.5, 2.0]), torch.tensor([2.5, 4.0, 30.0])
    ref = -sst.t.logpdf(y.numpy(), df.numpy(), loc.numpy(), sc.numpy())
    assert np.allclose(student_t_nll(y, loc, sc, df).numpy(), ref, atol=1e-5)
    s = sample_student_t(loc, sc, df, 200000)
    assert torch.allclose(s.median(0).values, loc, atol=0.05)


def test_flow_head_shapes_and_loss():
    net = FluxFillNet(d=32, depth=1, heads=4, flow_width=64, flow_depth=2)
    cond = torch.randn(10, 32)
    x1 = torch.randn(10, 48)
    tm = (torch.rand(10, 48) > 0.5).float()
    L = net.head_flow.loss(cond, x1, tm, torch.zeros(10, 48), torch.zeros(10, 48))
    assert torch.isfinite(L)
    L.backward()
    x = net.head_flow.sample(cond, torch.zeros(10, 48), torch.zeros(10, 48), steps=4)
    assert x.shape == (10, 48) and torch.isfinite(x).all()


@pytest.mark.parametrize("memory,mem_residual,components", [(False, False, 1), (True, False, 1), (True, True, 1),
                                                           (True, True, 3)])
@pytest.mark.parametrize("local_base,gated_base,gate_scale,edge_continue", [(False, False, False, False),
                                                                            (True, False, False, False),
                                                                            (False, True, False, False),
                                                                            (False, True, True, False),
                                                                            (False, True, True, True)])
@pytest.mark.parametrize("memory_skew", [False, True])
def test_method_protocol(tmp_path, memory, mem_residual, components, local_base, gated_base, gate_scale, edge_continue,
                         memory_skew):
    """FluxFill.predict returns (S, V, T) samples, the visible truth outside the mask, finite values inside it, in
    both sampling modes."""
    from data import Stats
    from method import FluxFill
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
    from windows import Window
    rec = synthetic_record()
    st = stats_of(rec)
    net = FluxFillNet(d=32, depth=1, heads=4, flow_width=32, flow_depth=1, memory=memory, mem_residual=mem_residual,
                      components=components, local_base=local_base, gated_base=gated_base,
                      gate_scale=gate_scale, edge_continue=edge_continue, memory_skew=memory_skew)
    ck = tmp_path / "ck.pt"
    torch.save({"model": net.state_dict(), "config": {"stats": st.to_dict(), "d": 32, "depth": 1, "heads": 4,
                                                                 "flow_width": 32, "flow_depth": 1, "memory": memory,
                                                                 "mem_residual": mem_residual,
                                                                 "components": components,
                                                                 "local_base": local_base,
                                                                 "gated_base": gated_base,
                                                                 "gate_scale": gate_scale,
                                                                 "edge_continue": edge_continue,
                                                                 "memory_skew": memory_skew}}, ck)
    keys = [v for v in VARS if v not in ("FCH4", "G")]
    rows = [VARS.index(k) for k in keys]
    vals = window_values(rec, 30)[rows].astype(np.float64)
    t = pd.date_range(rec.t0 + pd.Timedelta(days=30), periods=T_WIN, freq="30min")
    prov = ["tower" if IS_TOWER[r] else "reanalysis" for r in rows]
    mask = np.zeros(vals.shape, bool)
    tw = [i for i, p in enumerate(prov) if p == "tower"]
    mask[tw, 500:900] = True
    w = Window("XX-Syn", t[0], t, keys, np.where(mask, np.nan, vals), np.isfinite(vals) & ~mask, prov)
    for mode in ("marginal", "joint"):
        m = FluxFill(str(ck), mode=mode, n_samples=4, steps=3, flow_steps=2, device="cpu")
        m.flow_steps = 2
        tt = rec.t0 + pd.to_timedelta(np.arange(rec.values.shape[1]) * 30, unit="m")
        other = ~tt.year.isin([2016])
        ctx_tab = pd.DataFrame({"t_lst": ((tt[other] - pd.Timestamp("1970-01-01")) // pd.Timedelta(minutes=1)).astype(np.int64)})
        for i, v in enumerate(VARS):
            ctx_tab[v] = rec.values[i, other]
        s = m.predict(w, mask, {"table": ctx_tab})
        assert s.shape == (4,) + vals.shape
        assert np.isfinite(s[:, mask]).all()
        ok = ~mask & np.isfinite(vals)
        assert np.allclose(s[:, ok], np.broadcast_to(vals[ok], (4, ok.sum())))


def test_memory_excludes_window_years_and_matches_bench_path():
    from memory import memory_from_table
    from sampler import WindowSampler
    rec = synthetic_record(days=1500)
    st = stats_of(rec)
    smp = WindowSampler({"XX-Syn": rec}, ["XX-Syn"], st, {}, train=False, memory=True)
    d = 400                                                      # 2017-02-04 .. 2017-08-02
    m1 = smp.window_memory("XX-Syn", d)
    # change every value of the window's calendar year: memory must not move
    rec2 = synthetic_record(days=1500)
    t = rec2.t0 + pd.to_timedelta(np.arange(rec2.values.shape[1]) * 30, unit="m")
    rec2.values[:, t.year == 2017] += 7.0
    smp2 = WindowSampler({"XX-Syn": rec2}, ["XX-Syn"], st, {}, train=False, memory=True)
    m2 = smp2.window_memory("XX-Syn", d)
    for a, b in zip(m1, m2):      # float32 totals: equal up to rounding, far below any real signal (+7 here)
        assert np.allclose(a.astype(np.float64), b.astype(np.float64), atol=5e-3)
    # the benchmark path (context table without the window's years) gives the same channels
    ctx = rec.values.copy()
    ctx[:, t.year == 2017] = np.nan
    m3 = memory_from_table(ctx, rec.t0, st, rec.t0 + pd.Timedelta(days=d))
    for a, b in zip(m1, m3):
        assert np.allclose(a.astype(np.float32), b, atol=2e-3)


def test_window_memory_direct_equals_cached_tables():
    """The compiled direct memory (sampler fast_memory) equals the cached day-of-year tables: windows across year
    boundaries, a leap year, a never-used (held-out) year and windows inside it."""
    from sampler import WindowSampler
    rec = synthetic_record(days=1500)            # 2016 (leap) .. 2020-02
    rec.heldout_year = 2018
    st = stats_of(rec)
    a = WindowSampler({"XX-Syn": rec}, ["XX-Syn"], st, {}, train=False, memory=True)
    a._mem["XX-Syn"] = __import__("memory").SiteMemory(rec, [2018])
    b = WindowSampler({"XX-Syn": rec}, ["XX-Syn"], st, {}, train=False, memory=True, fast_memory=True)
    b._mem["XX-Syn"] = a._mem["XX-Syn"]
    n_diff = n_tot = 0
    for d in (0, 1, 200, 280, 330, 365, 366, 600, 731, 800, 1000, 1300):
        x, y = a.window_memory("XX-Syn", d), b.window_memory("XX-Syn", d)
        assert (x[2] == y[2]).all()
        for u, v in zip(x[:2], y[:2]):
            assert np.allclose(u, v, rtol=1e-3, atol=1e-6)        # at most one float16 unit
            n_diff += int((u != v).sum()); n_tot += u.size
    assert n_diff <= 1e-4 * n_tot


def test_memory_mean_is_climatology_pool_mean():
    from memory import memory_from_table
    rec = synthetic_record(days=1500)
    st = stats_of(rec)
    start = pd.Timestamp("2017-03-01")
    ctx = rec.values.copy()
    t = rec.t0 + pd.to_timedelta(np.arange(ctx.shape[1]) * 30, unit="m")
    ctx[:, t.year == 2017] = np.nan
    mz, sz, has = memory_from_table(ctx, rec.t0, st, start)
    v, day, k = 5, 10, 20                                         # 2017-03-11, slot 20
    target_doy = (start + pd.Timedelta(days=day)).day_of_year
    slot = t.hour * 2 + t.minute // 30
    dd = np.abs(t.day_of_year - target_doy)
    dd = np.minimum(dd, 366 - dd)
    pool = ctx[v, (slot == k) & (dd <= 15)]
    pool = pool[np.isfinite(pool)]
    assert has[v, day, k]
    assert mz[v, day, k] == pytest.approx((pool.mean() - st.mean[v]) / st.std[v], abs=0.05)


def test_land_rows_and_dropout():
    from data import IS_LAND, LAND_VARS, land_matrix
    assert IS_LAND.sum() == len(LAND_VARS) == 8
    rng = np.random.default_rng(3)
    present = np.ones(V_TAB, bool)
    for _ in range(200):
        hide, _, fam = masking.draw(rng, present)
        r = hide[IS_LAND]
        assert r.all() or not r.any()
        if fam.startswith("r3"):
            assert not r.any()
    t = pd.date_range("2018-01-01", periods=96, freq="30min")
    assert np.isnan(land_matrix("XX-None", t, 0.0)).all()


def test_mem_residual_location_is_memory_mean_at_zero_head():
    rec = synthetic_record(days=1500)
    st = stats_of(rec)
    from sampler import WindowSampler
    smp = WindowSampler({"XX-Syn": rec}, ["XX-Syn"], st, {}, train=False, memory=True)
    hide = np.zeros((V_TAB, T_WIN), bool)
    hide[IS_TOWER, 2000:4000] = True
    ex = make_example(rec, 400, st, hide, np.zeros(180, bool), None, smp.window_memory("XX-Syn", 400))
    b = collate([ex])
    net = FluxFillNet(d=32, depth=1, heads=4, memory=True, mem_residual=True).eval()
    torch.nn.init.zeros_(net.head_t.mlp[-1].weight); torch.nn.init.zeros_(net.head_t.mlp[-1].bias)
    with torch.no_grad():
        loc = net(b)[1][0]
    want = b["mem_m"] * b["mem_h"].float()
    assert torch.allclose(loc, want, atol=1e-6)
    assert b["mem_h"].any() and (loc[~b["mem_h"]] == 0).all()


def test_imagery_gaussian_nll_matches_scipy():
    from net import imagery_scores
    torch.manual_seed(0)
    net = FluxFillNet(d=16, depth=1, heads=4, img_nll=True)
    h = torch.randn(5, 16)
    y = torch.randn(5, 1024)
    nll, mse, cos = imagery_scores(net, h, y)
    o = net.head_img(h).detach().numpy()
    mu, s = o[:, :1024], np.exp(np.clip(o[:, 1024:], -7, 5))
    assert float(nll) == pytest.approx(-sst.norm.logpdf(y.numpy(), mu, s).mean(), rel=1e-5)
    assert float(mse) == pytest.approx(((mu - y.numpy()) ** 2).mean(), rel=1e-5)
    assert -1 <= float(cos) <= 1
    net0 = FluxFillNet(d=16, depth=1, heads=4)          # point head (old checkpoints): no NLL
    assert imagery_scores(net0, h, y)[0] is None


def test_student_t_mixture_nll_sampling_and_memres():
    from net import marginal_mean, marginal_nll, sample_marginal_params
    torch.manual_seed(0)
    loc = torch.tensor([[-1.0, 2.0, 0.5]]); scale = torch.tensor([[0.5, 1.0, 2.0]]); df = torch.tensor([[3.0, 5.0, 30.0]])
    logw = torch.log(torch.tensor([[0.2, 0.5, 0.3]]))
    y = torch.tensor([0.7])
    ref = -np.log(sum(w * sst.t.pdf(0.7, n, l, s) for w, l, s, n in zip([0.2, 0.5, 0.3], [-1, 2, 0.5], [0.5, 1, 2], [3, 5, 30])))
    assert float(marginal_nll(y, (loc, scale, df, logw))) == pytest.approx(ref, rel=1e-5)
    d = sample_marginal_params((loc, scale, df, logw), 200000)
    assert d.shape == (200000, 1)
    assert float(d.mean()) == pytest.approx(float(marginal_mean((loc, scale, df, logw))), abs=0.03)
    # head and memory residual: every component location is shifted by the memory mean
    rec = synthetic_record(days=1500)
    st = stats_of(rec)
    from sampler import WindowSampler
    smp = WindowSampler({"XX-Syn": rec}, ["XX-Syn"], st, {}, train=False, memory=True)
    hide = np.zeros((V_TAB, T_WIN), bool); hide[IS_TOWER, 2000:4000] = True
    b = collate([make_example(rec, 400, st, hide, np.zeros(180, bool), None, smp.window_memory("XX-Syn", 400))])
    net = FluxFillNet(d=32, depth=1, heads=4, memory=True, mem_residual=True, components=3).eval()
    torch.nn.init.zeros_(net.head_t.mlp[-1].weight); torch.nn.init.zeros_(net.head_t.mlp[-1].bias)
    with torch.no_grad():
        p = net(b)[1]
    assert len(p) == 4 and p[0].shape[-1] == 3
    off = (b["mem_m"] * b["mem_h"].float())[..., None]
    assert torch.allclose(p[0] - off, torch.linspace(-0.5, 0.5, 3).expand_as(p[0]), atol=1e-6)


def test_local_diurnal_course_reference_and_no_leak():
    from net import local_diurnal_course
    rng = np.random.default_rng(0)
    z = torch.tensor(rng.standard_normal((1, 2, 10, 48)), dtype=torch.float32)
    obs = torch.tensor(rng.random((1, 2, 10, 48)) > 0.4)
    mean, ok = local_diurnal_course(z, obs)
    zn, on = z.numpy(), obs.numpy()
    for (v, d, k) in [(0, 0, 0), (1, 5, 17), (0, 9, 47), (1, 3, 30)]:
        nb = [d + j for j in (-3, -2, -1, 1, 2, 3) if 0 <= d + j < 10]
        vals = [zn[0, v, n, k] for n in nb if on[0, v, n, k]]
        assert bool(ok[0, v, d, k]) == (len(vals) >= 2)
        if len(vals) >= 2:
            assert float(mean[0, v, d, k]) == pytest.approx(np.mean(vals), abs=1e-5)
    z2 = torch.where(obs, z, z + 100.0)              # change only hidden values
    m2, ok2 = local_diurnal_course(z2, obs)
    assert torch.equal(ok, ok2) and torch.allclose(mean, m2)


def test_nearest_visible_distance_reference():
    from net import nearest_visible_distance
    obs = torch.zeros(1, 1, 2, 48, dtype=torch.bool)
    obs[0, 0, 0, 10] = True; obs[0, 0, 1, 5] = True               # visible at t = 10 and t = 53
    d = nearest_visible_distance(obs).reshape(-1)
    t = np.arange(96)
    ref = np.minimum(np.abs(t - 10), np.abs(t - 53))
    assert np.array_equal(d.numpy().astype(int), ref)
    assert (nearest_visible_distance(torch.zeros(1, 1, 2, 48, dtype=torch.bool)) == 96).all()


def test_gated_base_anchors_gate_and_no_leak():
    from net import local_diurnal_course
    rec = synthetic_record(days=1500)
    st = stats_of(rec)
    from sampler import WindowSampler
    smp = WindowSampler({"XX-Syn": rec}, ["XX-Syn"], st, {}, train=False, memory=True)
    hide = np.zeros((V_TAB, T_WIN), bool); hide[IS_TOWER, 2000:2100] = True; hide[IS_TOWER, 5000:8000] = True
    ex = make_example(rec, 400, st, hide, np.zeros(180, bool), None, smp.window_memory("XX-Syn", 400))
    b = collate([ex])
    net = FluxFillNet(d=32, depth=1, heads=4, memory=True, mem_residual=True, gated_base=True).eval()
    torch.nn.init.zeros_(net.head_t.mlp[-1].weight); torch.nn.init.zeros_(net.head_t.mlp[-1].bias)
    torch.nn.init.zeros_(net.gate_in.weight); torch.nn.init.zeros_(net.gate_in.bias)
    with torch.no_grad():
        loc, scale, df, logw = net(b)[1]
    lm, ok = local_diurnal_course(b["z"], b["obs"])
    mem = b["mem_m"] * b["mem_h"].float()
    assert torch.allclose(loc[..., 1], mem, atol=1e-6)
    assert torch.allclose(loc[..., 0], torch.where(ok, lm, mem), atol=1e-6)
    w0 = logw[..., 0].exp()
    assert torch.allclose(w0[ok], torch.full_like(w0[ok], 0.5), atol=1e-5)    # zero gate: equal weights
    assert (w0[~ok] < 1e-10).all()                                              # no local course: memory only
    ex2 = dict(ex); ex2["tgt"] = ex["tgt"] + 9.0                               # hidden truth changes, inputs do not
    with torch.no_grad():
        p2 = net(collate([ex2]))[1]
    for x, y in zip((loc, scale, df, logw), p2):
        assert torch.equal(x, y)


def test_scaling_power_law_fit_recovers_parameters():
    from scaling import fit, power_law
    x = np.array([30, 60, 120, 240, 488], float)
    y = power_law(x, 0.5, 0.35, 0.15)
    a, alpha, c = fit(x, y)
    assert alpha == pytest.approx(0.35, abs=0.02) and c == pytest.approx(0.15, abs=0.01)


def test_gate_scale_zero_init_equals_gated_and_scales_with_evidence():
    rec = synthetic_record(days=1500)
    st = stats_of(rec)
    from sampler import WindowSampler
    smp = WindowSampler({"XX-Syn": rec}, ["XX-Syn"], st, {}, train=False, memory=True)
    hide = np.zeros((V_TAB, T_WIN), bool); hide[IS_TOWER, 2000:2100] = True; hide[IS_TOWER, 5000:8000] = True
    b = collate([make_example(rec, 400, st, hide, np.zeros(180, bool), None, smp.window_memory("XX-Syn", 400))])
    torch.manual_seed(0)
    a = FluxFillNet(d=32, depth=1, heads=4, memory=True, mem_residual=True, gated_base=True).eval()
    torch.manual_seed(0)
    g = FluxFillNet(d=32, depth=1, heads=4, memory=True, mem_residual=True, gated_base=True, gate_scale=True).eval()
    g.load_state_dict({**a.state_dict(), **{k: v for k, v in g.state_dict().items() if k.startswith("scale_in")}})
    with torch.no_grad():
        pa, pg = a(b)[1], g(b)[1]
    for x, y in zip(pa, pg):
        assert torch.allclose(x, y)                                   # zero-initialised: identical to --gated-base
    with torch.no_grad():
        g.scale_in.bias.copy_(torch.tensor([0.0, 0.5]))
        pg2 = g(b)[1]
    assert torch.allclose(pg2[1][..., 1], pa[1][..., 1] * np.exp(0.5), rtol=1e-5)


@pytest.mark.parametrize("gated", [False, True])
def test_method_exact_pit_matches_samples_and_runner_uses_it(tmp_path, gated):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
    from method import FluxFill
    from windows import Window
    import run as bench
    rec = synthetic_record()
    st = stats_of(rec)
    net = FluxFillNet(d=32, depth=1, heads=4, memory=True, mem_residual=True, gated_base=gated)
    ck = tmp_path / "ck.pt"
    torch.save({"model": net.state_dict(), "config": {"stats": st.to_dict(), "d": 32, "depth": 1, "heads": 4,
                                                       "memory": True, "mem_residual": True, "gated_base": gated}}, ck)
    keys = [v for v in VARS if v not in ("FCH4", "G")]
    rows = [VARS.index(k) for k in keys]
    vals = window_values(rec, 30)[rows].astype(np.float64)
    t = pd.date_range(rec.t0 + pd.Timedelta(days=30), periods=T_WIN, freq="30min")
    prov = ["tower" if IS_TOWER[r] else "reanalysis" for r in rows]
    mask = np.zeros(vals.shape, bool)
    mask[[i for i, p in enumerate(prov) if p == "tower"], 500:1500] = True
    w = Window("XX-Syn", t[0], t, keys, vals, np.isfinite(vals), prov)
    vis = Window("XX-Syn", t[0], t, keys, np.where(mask, np.nan, vals), np.isfinite(vals) & ~mask, prov)
    m = FluxFill(str(ck), mode="marginal", n_samples=200, device="cpu")       # small: the mixture sampler expands per component
    s = m.predict(vis, mask, {})
    pit = m.pit(vals)
    sel = mask & np.isfinite(vals)
    emp = (s[:, sel] <= vals[sel][None]).mean(0)                     # empirical CDF of the samples at the truth
    assert np.abs(emp - pit[sel]).mean() < 0.05                      # sampling error of a 200-draw ECDF is about 0.03
    out = bench.score("ff", "R1", "M2-x", w, mask, s, pit)
    assert {r.get("cov_method") for r in out if r["kind"] == "values" and r.get("n", 0) > 0} == {"exact"}


def test_edge_anomaly_reference_no_leak_and_anchor():
    from net import edge_anomaly, local_course_stats
    z = torch.zeros(1, 1, 3, 48); obs = torch.zeros(1, 1, 3, 48, dtype=torch.bool)
    obs[0, 0, 1, :10] = True; obs[0, 0, 1, 30:] = True                 # day 1 visible except half-hours 10-29
    z[0, 0, 1, 9] = 2.0; z[0, 0, 1, 30] = -1.0
    course = torch.zeros_like(z); ok = torch.ones_like(obs)
    A, d = edge_anomaly(z, obs, course, ok)
    t = 48 + np.arange(10, 30)
    assert np.array_equal(d[0, 0].reshape(-1)[t].numpy(), np.minimum(t - (48 + 9), (48 + 30) - t).astype(np.float32))
    assert float(A[0, 0, 1, 12]) == 2.0 and float(A[0, 0, 1, 28]) == -1.0   # nearest edge's anomaly
    z2 = torch.where(obs, z, z + 50.0)                                     # hidden values changed
    A2, d2 = edge_anomaly(z2, obs, course, ok)
    assert torch.equal(A, A2) and torch.equal(d, d2)
    # anchor with a zero head: local component = course + A * 0.5 ** d where a local course exists
    rec = synthetic_record(days=1500); st = stats_of(rec)
    from sampler import WindowSampler
    smp = WindowSampler({"XX-Syn": rec}, ["XX-Syn"], st, {}, train=False, memory=True)
    hide = np.zeros((V_TAB, T_WIN), bool); hide[IS_TOWER, 2000:2048] = True
    b = collate([make_example(rec, 400, st, hide, np.zeros(180, bool), None, smp.window_memory("XX-Syn", 400))])
    net = FluxFillNet(d=32, depth=1, heads=4, memory=True, mem_residual=True, gated_base=True, edge_continue=True).eval()
    torch.nn.init.zeros_(net.head_t.mlp[-1].weight); torch.nn.init.zeros_(net.head_t.mlp[-1].bias)
    with torch.no_grad():
        loc = net(b)[1][0]
    lm, okk, _ = local_course_stats(b["z"], b["obs"])
    Ae, de = edge_anomaly(b["z"], b["obs"], lm, okk)
    want = torch.where(okk, lm + Ae * 0.5 ** de, b["mem_m"] * b["mem_h"].float())
    assert torch.allclose(loc[..., 0], want, atol=1e-5)


def test_memory_skew_accuracy_stability_no_leak_and_consistency():
    from memory import memory_from_table, memory_from_table_skew
    from sampler import WindowSampler
    rng = np.random.default_rng(5)
    def make():
        r = synthetic_record(days=1500)
        T = r.values.shape[1]
        g = np.random.default_rng(7)
        r.values[VARS.index("NEE")] = g.gamma(2.0, 1.0, T).astype(np.float32)              # skewness 1.41
        r.values[VARS.index("PA")] = (100.0 + 0.5 * g.standard_normal(T)).astype(np.float32)  # large mean, small spread
        tt = r.t0 + pd.to_timedelta(np.arange(T) * 30, unit="m")
        r.values[VARS.index("TS_1")] = (10 + 12 * np.sin(2 * np.pi * tt.day_of_year / 365.25) + g.gamma(2.0, 0.3, T)).astype(np.float32)
        return r, tt
    rec, t = make()
    st = stats_of(rec)
    start = pd.Timestamp("2017-03-01")
    ctx = rec.values.copy(); ctx[:, t.year == 2017] = np.nan
    mz, sz, kz, has = memory_from_table_skew(ctx, rec.t0, st, start)
    mz0, sz0, has0 = memory_from_table(ctx, rec.t0, st, start)
    assert np.array_equal(has, has0)
    assert np.allclose(mz, mz0, atol=2e-3) and np.allclose(sz, sz0, atol=2e-3)       # same mean and sd as before
    day, k = 10, 20
    doy = (start + pd.Timedelta(days=day)).day_of_year
    dd = np.abs(t.day_of_year - doy); dd = np.minimum(dd, 366 - dd)
    slot = t.hour * 2 + t.minute // 30
    for v, tol in (("NEE", 0.05), ("PA", 0.1)):
        i = VARS.index(v)
        pool = ctx[i, (slot == k) & (dd <= 15)]; pool = pool[np.isfinite(pool)].astype(np.float64)
        assert abs(kz[i, day, k] - sst.skew(pool)) < tol, v
    # no leak: change every value of the window's year; all four channels stay within float16 resolution
    smp = WindowSampler({"X": rec}, ["X"], st, {}, train=False, memory=True, memory_skew=True)
    rec2, t2 = make(); rec2.values[:, t2.year == 2017] += 7.0
    smp2 = WindowSampler({"X": rec2}, ["X"], st, {}, train=False, memory=True, memory_skew=True)
    for a, b in zip(smp.window_memory("X", 400), smp2.window_memory("X", 400)):
        assert np.allclose(a.astype(np.float64), b.astype(np.float64), atol=2e-2)


def test_driver_anomaly_zero_init_formula_and_no_leak():
    from net import local_course_stats
    rec = synthetic_record(days=1500); st = stats_of(rec)
    from sampler import WindowSampler
    smp = WindowSampler({"XX-Syn": rec}, ["XX-Syn"], st, {}, train=False, memory=True)
    hide = np.zeros((V_TAB, T_WIN), bool); hide[IS_TOWER, 2000:2048] = True
    ex = make_example(rec, 400, st, hide, np.zeros(180, bool), None, smp.window_memory("XX-Syn", 400))
    b = collate([ex])
    torch.manual_seed(0)
    a = FluxFillNet(d=32, depth=1, heads=4, memory=True, mem_residual=True, gated_base=True).eval()
    torch.manual_seed(0)
    g = FluxFillNet(d=32, depth=1, heads=4, memory=True, mem_residual=True, gated_base=True, driver_anomaly=True).eval()
    g.load_state_dict({**a.state_dict(), **{k: v for k, v in g.state_dict().items() if k.startswith("beta_out")}})
    with torch.no_grad():
        pa, pg = a(b)[1], g(b)[1]
    for x, y in zip(pa, pg):
        assert torch.allclose(x, y)                                     # zero-initialised: identical
    with torch.no_grad():
        g.beta_out.bias.copy_(torch.tensor([0.5, -0.2, 0.1]))
        pg2 = g(b)[1]
    lm, ok, _ = local_course_stats(b["z"], b["obs"])
    dr = g.driver_rows
    danom = torch.where(b["obs"][:, dr] & ok[:, dr], b["z"][:, dr] - lm[:, dr], torch.zeros_like(lm[:, dr]))
    shift = 0.5 * danom[:, 0] - 0.2 * danom[:, 1] + 0.1 * danom[:, 2]                # (B, D, 48), same for every row
    assert torch.allclose(pg2[0][..., 1] - pa[0][..., 1], shift[:, None].expand_as(pa[0][..., 1]), atol=1e-5)
    ex2 = dict(ex); ex2["tgt"] = ex["tgt"] + 9.0
    with torch.no_grad():
        p3 = g(collate([ex2]))[1]
    for x, y in zip(pg2, p3):
        assert torch.equal(x, y)


def test_memory_skew_fluxes_only_ignores_other_rows():
    from data import IS_FLUX
    rec = synthetic_record(days=1500); st = stats_of(rec)
    from sampler import WindowSampler
    smp = WindowSampler({"XX-Syn": rec}, ["XX-Syn"], st, {}, train=False, memory=True, memory_skew=True)
    hide = np.zeros((V_TAB, T_WIN), bool); hide[IS_TOWER, 2000:2048] = True
    ex = make_example(rec, 400, st, hide, np.zeros(180, bool), None, smp.window_memory("XX-Syn", 400))
    b = collate([ex])
    net = FluxFillNet(d=32, depth=1, heads=4, memory=True, mem_residual=True, gated_base=True,
                      memory_skew_fluxes_only=True).eval()
    assert net.memory_skew and net.tab_in.in_features == 5 * 48 and net.skew_in.in_features == 48
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    b2["mem_k"][:, torch.from_numpy(~IS_FLUX)] += 3.0                     # change the skewness of non-flux rows only
    with torch.no_grad():
        p1, p2 = net(b)[1], net(b2)[1]
    for x, y in zip(p1, p2):
        assert torch.equal(x, y)
    b3 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    b3["mem_k"][:, torch.from_numpy(IS_FLUX)] += 3.0                       # flux rows: the output changes
    with torch.no_grad():
        net.skew_in.weight.normal_()                                       # (zero at initialisation by design)
        p1 = net(b)[1]
        p3 = net(b3)[1]
    assert not torch.equal(p1[0], p3[0])



@pytest.mark.parametrize("option", [dict(driver_anomaly=True), dict(memory_skew=True), dict(memory_skew_fluxes_only=True),
                                    dict(driver_anomaly=True, memory_skew_fluxes_only=True), dict(year_anomaly=True),
                                    dict(sinr_place=0), dict(sinr_place=4)])
def test_option_arms_start_from_the_base_weights(option):
    """An A/B arm with an option has exactly the base model's initial weights (and output) at the same seed."""
    base = dict(d=32, depth=2, heads=4, memory=True, mem_residual=True, gated_base=True, gate_scale=True, img_nll=True,
                edge_continue=True)
    torch.manual_seed(0); A = FluxFillNet(**base).eval()
    torch.manual_seed(0); B = FluxFillNet(**base, **option).eval()
    sa, sb = A.state_dict(), B.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)                      # every base tensor equal
    extra = [k for k in sb if k not in sa]
    zeroed = [k for k in extra if not k.startswith("place.") or k.startswith("place.out.")]   # SINR: output layer
    assert extra and zeroed and all(torch.count_nonzero(sb[k]) == 0 for k in zeroed)   # option output starts at zero
    rec = synthetic_record(days=1500); st = stats_of(rec)
    from sampler import WindowSampler
    smp = WindowSampler({"XX-Syn": rec}, ["XX-Syn"], st, {}, train=False, memory=True, memory_skew=True)
    hide = np.zeros((V_TAB, T_WIN), bool); hide[IS_TOWER, 2000:2048] = True
    b = collate([make_example(rec, 400, st, hide, np.zeros(180, bool), None, smp.window_memory("XX-Syn", 400))])
    with torch.no_grad():
        for x, y in zip(A(b)[1], B(b)[1]):
            assert torch.allclose(x, y, atol=1e-6)


def test_scale_anchor_same_weights_scales_by_memory_sd():
    """E27: the arm has the base weights (no parameters); scales are the base scales times (mem_s + floor) where
    memory exists and (1 + floor) where it does not; locations, df and weights unchanged."""
    from net import SCALE_FLOOR
    from sampler import WindowSampler
    base = dict(d=32, depth=2, heads=4, memory=True, mem_residual=True, gated_base=True, gate_scale=True, img_nll=True,
                edge_continue=True)
    torch.manual_seed(0); A = FluxFillNet(**base).eval()
    torch.manual_seed(0); B = FluxFillNet(**base, scale_anchor=True).eval()
    sa, sb = A.state_dict(), B.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    rec = synthetic_record(days=1500); st = stats_of(rec)
    smp = WindowSampler({"XX-Syn": rec}, ["XX-Syn"], st, {}, train=False, memory=True)
    hide = np.zeros((V_TAB, T_WIN), bool); hide[IS_TOWER, 2000:2100] = True
    b = collate([make_example(rec, 400, st, hide, np.zeros(180, bool), None, smp.window_memory("XX-Syn", 400))])
    b["mem_h"][:, :, :20] = False                                          # some days without memory
    with torch.no_grad():
        pa, pb = A(b)[1], B(b)[1]
    s = torch.where(b["mem_h"], b["mem_s"], torch.ones_like(b["mem_s"])) + SCALE_FLOOR
    assert torch.allclose(pb[1], pa[1] * s[..., None], rtol=1e-5)
    for i in (0, 2, 3):
        assert torch.equal(pa[i], pb[i])


def test_year_anomaly_formula_and_no_leak():
    """E28: A_win is the mean of (visible z - memory mean) over the window's visible days at each half-hour; the
    memory anchor moves by c * A; hidden values never enter."""
    from net import year_anomalies
    from sampler import WindowSampler
    base = dict(d=32, depth=2, heads=4, memory=True, mem_residual=True, gated_base=True, gate_scale=True, img_nll=True,
                edge_continue=True)
    rec = synthetic_record(days=1500); st = stats_of(rec)
    smp = WindowSampler({"XX-Syn": rec}, ["XX-Syn"], st, {}, train=False, memory=True)
    hide = np.zeros((V_TAB, T_WIN), bool); hide[IS_TOWER, 2000:4000] = True
    b = collate([make_example(rec, 400, st, hide, np.zeros(180, bool), None, smp.window_memory("XX-Syn", 400))])
    an, ok = year_anomalies(b["z"], b["obs"], b["mem_m"], b["mem_h"])
    o = (b["obs"] & b["mem_h"]).float()
    ref = ((b["z"] - b["mem_m"]) * o).sum(2, keepdim=True) / o.sum(2, keepdim=True).clamp_min(1)
    assert torch.allclose(an[1], ref.expand_as(an[1]), atol=1e-5)
    torch.manual_seed(0); A = FluxFillNet(**base).eval()
    torch.manual_seed(0); B = FluxFillNet(**base, year_anomaly=True).eval()
    B.load_state_dict(A.state_dict(), strict=False)
    with torch.no_grad():
        B.yanom_coef.fill_(0.5)
        pa, pb = A(b)[1], B(b)[1]
        shift = torch.where(b["mem_h"], 0.5 * (an[0] + an[1]), torch.zeros_like(an[0]))
        assert torch.allclose(pb[0][..., 1], pa[0][..., 1] + shift, atol=1e-4)      # the memory component
        b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
        b2["tgt"] = b2["tgt"] + 5.0 * b2["tmask"].float()                          # hidden truth changed
        b2["z"] = torch.where(b2["obs"], b2["z"], b2["z"] + 3.0)                     # values behind the mask changed
        for x, y in zip(B(b)[1], B(b2)[1]):
            assert torch.allclose(x, y, atol=1e-5)


def test_sinr_place_encoding_learns_and_unknown_location():
    """E29: SINR input at n_freq = 0 is sin/cos of longitude and latitude (as the habitat model's SinrPlace) plus
    elevation; hidden layers keep a random initialisation (so the zero output layer receives a gradient); an
    all-zero site (unknown location) gives zero; replace mode drops site_in."""
    from net import SinrPlace
    from data import site_features
    p = SinrPlace(8, n_freq=2)
    s = torch.from_numpy(np.stack([site_features(45.0, -100.0, 300.0), site_features(-33.0, 151.0, np.nan),
                                   np.zeros(4, np.float32)]))
    e = p.encode(s)
    assert e.shape == (3, 1 + 4 * 3)
    lat, lon = math.radians(45.0), math.radians(-100.0)
    assert torch.allclose(e[0, 1:5], torch.tensor([math.sin(lon), math.cos(lon), math.sin(2 * lat), math.cos(2 * lat)]),
                          atol=1e-5)                                    # sin(pi lon / 180), sin(pi lat / 90)
    assert torch.allclose(e[0, 5:7], torch.tensor([math.sin(2 * lon), math.cos(2 * lon)]), atol=1e-5)
    assert abs(float(e[0, 0]) - 0.3) < 1e-6
    out = p(s)
    assert torch.count_nonzero(out) == 0
    (p(s)[:2] * torch.randn(2, 8)).sum().backward()
    assert torch.count_nonzero(p.out.weight.grad) > 0                      # learns from the zero start
    with torch.no_grad():
        p.out.bias.fill_(1.0)
    assert torch.count_nonzero(p(s)[2]) == 0                               # unknown location: zero
    base = dict(d=32, depth=1, heads=4, memory=True, mem_residual=True, gated_base=True)
    torch.manual_seed(0); A = FluxFillNet(**base)
    torch.manual_seed(0); R = FluxFillNet(**base, sinr_place=0, sinr_replace=True)
    assert all(torch.equal(A.state_dict()[k], v) for k, v in R.state_dict().items() if k in A.state_dict())
    assert R.sinr_replace and not A.sinr_replace


def test_fluxes_off_mask_option():
    from data import IS_FLUX
    present = np.ones(V_TAB, bool)
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    for _ in range(50):                                                       # default: the same draws as before
        a = masking.draw(r1, present); b = masking.draw(r2, present, masking.MaskConfig(p_fluxes_off=0.0))
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    rng = np.random.default_rng(4)
    cfg = masking.MaskConfig(p_fluxes_off=1.0, p_drop_met=0.0)
    seen_met_visible = False
    for _ in range(100):
        hide, _, fam = masking.draw(rng, present, cfg)
        assert hide[IS_FLUX].all()                                            # every flux row, whole window
        seen_met_visible |= not hide[IS_TOWER_MET].all()
    assert seen_met_visible


def test_extended_channels_pair_with_base_weights(tmp_path):
    """E30: with FLUXFILL_CHANNELS=extended (a separate process: the row layout is fixed at import), the network has
    the base model's weights in every base tensor at the same seed, a zero table for the extra rows, and runs on an
    example with the extra rows."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parents[1]
    base = dict(d=32, depth=2, heads=4, memory=True, mem_residual=True, gated_base=True, gate_scale=True, img_nll=True,
                edge_continue=True)
    code = f"""
import sys, numpy as np, torch
sys.path.insert(0, {str(here)!r}); sys.path.insert(0, {str(here / 'tests')!r})
import data
assert data.CHANNELS == "extended" and data.V_TAB == data.V_BASE + len(data.EXTRA_VARS) and data.IS_TOWER[data.V_BASE:].all()
from net import FluxFillNet
from sampler import collate, make_example
from test_model import synthetic_record, stats_of
torch.manual_seed(0); net = FluxFillNet(**{base!r}).eval()
torch.save(net.state_dict(), {str(tmp_path / 'ext.pt')!r})
assert torch.count_nonzero(net.extra_emb.weight) == 0
rec = synthetic_record(days=400); st = stats_of(rec)
hide = np.zeros((data.V_TAB, data.T_WIN), bool); hide[data.IS_TOWER, 2000:2100] = True
b = collate([make_example(rec, 10, st, hide, np.zeros(180, bool), None, None)])
with torch.no_grad():
    h, p = net(b)
assert h.shape[1] == data.V_TAB + 1 and torch.isfinite(p[0]).all()
print("OK")
"""
    r = subprocess.run([sys.executable, "-c", code], env={**os.environ, "FLUXFILL_CHANNELS": "extended",
                       "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, cwd=str(here))
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-2000:]
    ext = torch.load(tmp_path / "ext.pt", weights_only=True)
    torch.manual_seed(0)
    A = FluxFillNet(**base)
    sa = A.state_dict()
    for k, v in sa.items():
        if v.shape == ext[k].shape:
            assert torch.equal(v, ext[k]), k
        else:                                   # per-variable zero-initialised vectors grow with the row count
            assert k in ("edge_logit",) and torch.count_nonzero(ext[k]) == 0, k
    assert set(ext) - set(sa) == {"extra_emb.weight"}
