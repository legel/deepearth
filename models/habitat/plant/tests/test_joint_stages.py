"""The joint model with the landscape field, place pathway and learned calibration, trained in stages (ranges/joint).

* AUC: the batched rank computation (``auc_table``, on the CPU and, when there is one, the GPU) equals scikit-learn's
  roc_auc_score per species, with ties, plots outside the calibration area, a plot selection and unscorable rows;
* target-group snapping: every row moves to the nearest record of another species (brute force on the sphere);
* the model: the learned penalty starts at its initial value and applies only outside the area; place starts at zero
  and is continuous across the antimeridian; a full model is rebuilt from its state dict alone;
* end to end on synthetic data (CPU): an environment model; the representation started from it (target-group and
  continental background, restored shoreline presences, filled plots, field, place, penalty); the feature cache of
  the representation equals the features recomputed by the model for its rows and plots; the species stage on the
  cache keeps the shared networks fixed and its in-training plot scores equal the full model's; a map store of it
  decodes within the quantization bound of the full model, and serves maps outside the calibration area lowered by
  the species' penalty, for trained and inferred species alike.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from ranges.joint import store as ST
from ranges.joint.cache import FeatureCache, build_cache, record_rows
from ranges.joint.data import JointData, Standardizer, calibration_sets, plot_sets
from ranges.joint.field import build_pyramid
from ranges.joint.model import SPECIES_TENSORS, JointRangeModel
from ranges.joint.place import SinrPlace
from ranges.joint.prepare import snap_to_records, unit_vectors
from ranges.joint.reader import ECOREGION_RASTER
from ranges.joint.train import (TrainConfig, auc_table, auc_table_reference, plot_features, plot_scores,
                                scores_from_features, train)
from ranges.joint.tree import Tree
from ranges.joint.zero_shot import infer_species

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------------------------------------------- AUC
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_auc_table_equals_roc_auc_score(device):
    rng = np.random.default_rng(0)
    S, n = 60, 3000
    y = rng.random((S, n)) < rng.uniform(0.002, 0.3, (S, 1))
    score = rng.standard_normal((S, n)).astype(np.float32) + y * rng.uniform(0, 2, (S, 1))
    score[:, ::7] = np.round(score[:, ::7], 1)                           # ties
    score[3] = np.nan                                                    # no map: NaN
    inside = rng.random((S, n)) < 0.9
    sel = rng.random(n) < 0.6
    for ins in (inside, np.ones((S, n), bool)):
        a = auc_table_reference(y, score, ins, sel)
        b = auc_table(y, score, ins, sel, device=device)
        ok = np.isfinite(a)
        assert (np.isfinite(b) == ok).all() and ok.sum() > 30
        assert np.abs(a[ok] - b[ok]).max() < 1e-12


# ------------------------------------------------------------------------------------------------- snapping
def test_snap_to_the_nearest_record_of_another_species(tmp_path):
    rng = np.random.default_rng(0)
    N = 3000
    lat, lon = rng.uniform(30, 40, N).astype(np.float32), rng.uniform(-110, -100, N).astype(np.float32)
    sid = rng.integers(0, 12, N).astype(np.int32)
    pres = (rng.random(N) < 0.3).astype(np.int8)
    np.savez(tmp_path / "joint_data.npz", lat=lat, lon=lon, sid=sid, pres=pres)
    snap = np.asarray(snap_to_records(tmp_path, chunk=700, log=lambda m: None))
    rec = np.flatnonzero(pres)
    P, Q = unit_vectors(lat, lon), unit_vectors(lat[rec], lon[rec])
    for i in rng.choice(N, 300, replace=False):
        d = ((Q - P[i]) ** 2).sum(1)
        d[sid[rec] == sid[i]] = np.inf
        assert snap[i] == rec[np.argmin(d)]


# ---------------------------------------------------------------------------------------------------- model
def _paths(S=6):
    from ranges.joint.tree import parse_newick, path_matrix
    nwk = "((" + ",".join(f"t{i}:1" for i in range(S // 2)) + "):1,(" + \
        ",".join(f"t{i}:2" for i in range(S // 2, S)) + "):0.5);"
    return path_matrix(Tree(*parse_newick(nwk)), [f"t{i}" for i in range(S)])


def test_penalty_place_and_state_dict():
    torch.manual_seed(0)
    m = JointRangeModel(7, _paths(), width=16, depth=2, place=dict(d_place=8, hidden=16, blocks=2),
                        field=dict(n_channels=3, d_c=16, heads=2, layers=1, radii_cells=(1, 2, 4)),
                        penalty_init=3.0)
    assert torch.allclose(m.outside_penalty(), torch.full((6,), 3.0))
    ll = torch.tensor([[35.0, 179.999], [35.0, -179.999], [10.0, 20.0]])
    assert m.place(ll).abs().max() == 0                                       # place starts at zero
    torch.nn.init.normal_(m.place.out.weight)
    p = m.place(ll)
    assert (p[0] - p[1]).abs().max() < 1e-3 * p.abs().max()                  # continuous across the antimeridian
    assert torch.allclose(SinrPlace.encode(ll)[2], torch.tensor([math.sin(math.pi / 9), math.cos(math.pi / 9),
                                                                 math.sin(math.pi / 9), math.cos(math.pi / 9)]))
    Fx, P = torch.randn(5, 16), torch.randn(5, 8)
    out = torch.zeros(5, 6, dtype=torch.bool)
    out[2, 3] = True
    d = m.scores(Fx, P) - m.scores(Fx, P, outside=out)
    assert torch.allclose(d, 3.0 * out.float(), atol=1e-5)
    r = JointRangeModel.from_state_dict(m.state_dict())
    assert r.place_dim == 8 and r.has_penalty and r.field.n_ch == 3 and len(r.field.blocks) == 1
    assert torch.equal(r.scores(Fx, P, outside=out), m.scores(Fx, P, outside=out))


# ---------------------------------------------------------------------------------------------- end to end
NAMES = ["v0", "v1", "v2", "v3"]          # v2 heavy-tailed (log-transformed), v3 sometimes missing (flagged)
H, W = 150, 300


def _env(rng, n):
    X = rng.standard_normal((n, 4)).astype(np.float32)
    X[:, 2] = np.exp(X[:, 2])
    X[rng.random(n) < 0.2, 3] = np.nan
    return X


def _suitability(X, opt):
    return np.exp(-((X[:, None, :2] - opt[None]) ** 2).sum(-1) / 0.5)


def _balanced(tips):
    if len(tips) == 1:
        return tips[0]
    h = len(tips) // 2
    return f"({_balanced(tips[:h])}:1.0,{_balanced(tips[h:])}:1.0)"


def _where(rng, n):
    """Positions on the synthetic grid and coordinates to match (any consistent lat/lon serves the place pathway)."""
    rc = np.c_[rng.uniform(0, H, n), rng.uniform(0, W, n)].astype(np.float32)
    return rc, (40 - rc[:, 0] * 0.02).astype(np.float32), (-110 + rc[:, 1] * 0.03).astype(np.float32)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    import rasterio
    from rasterio.transform import from_origin
    root = tmp_path_factory.mktemp("stages")
    rng = np.random.default_rng(0)
    S = 24
    tips = [f"Sp_{i:02d}" for i in range(S)] + ["Sp_new"]
    order = tips[:12] + ["Sp_new"] + tips[12:S]
    (root / "tree").mkdir()
    (root / "tree/natives.dated.nwk").write_text(_balanced(order) + ";")
    opt = np.c_[np.where(np.arange(S) < 12, -1.0, 1.0), np.zeros(S)] + rng.normal(0, 0.4, (S, 2))
    calib = ["1 2 3 4"] * S
    calib[5] = calib[17] = "1 2"
    pd.DataFrame({"species": tips[:S], "calibration_ecoregions": calib}).to_csv(root / "species.csv", index=False)
    Xs, sid, pres = [], [], []
    for s in range(S):
        pool = _env(rng, 4000)
        p = _suitability(pool, opt[s:s + 1])[:, 0]
        pick = rng.choice(len(pool), 80, replace=False, p=p / p.sum())
        Xs += [pool[pick], _env(rng, 400)]
        sid += [np.full(80, s), np.full(400, s)]
        pres += [np.ones(80, np.int8), np.zeros(400, np.int8)]
    X = np.concatenate(Xs)
    N = len(X)
    np.save(root / "train_points_cache.npy", X)
    rc, lat, lon = _where(rng, N)

    def plots(n, seed):
        r = np.random.default_rng(seed)
        PX = _env(r, n)
        PX[:12, 0] = np.nan                                              # shoreline plots without climate
        y = r.random((n, S)) < 0.8 * _suitability(np.nan_to_num(PX), opt)
        prc, plat, plon = _where(r, n)
        return dict(plot_X=PX, plot_lat=plat, plot_lon=plon, plot_eco=r.integers(1, 5, n).astype(np.int32),
                    eval_sid=np.arange(S, dtype=np.int32), eval_y=y.T.astype(np.int8)), prc
    vb, rc_vb = plots(1500, 1)
    aim, rc_aim = plots(800, 2)
    np.savez(root / "joint_data.npz", names=np.array(NAMES), sid=np.concatenate(sid).astype(np.int32),
             pres=np.concatenate(pres), lat=lat, lon=lon, **vb, maxent_occ=np.full((S, 1500), np.nan, np.float16))
    np.savez(root / "plots_aim.npz", **aim)
    for name, z in (("vegbank", vb), ("aim", aim)):                     # the shoreline fill of the plots
        Xf = z["plot_X"].copy()
        Xf[:10, 0] = 0.1                                                 # 10 of the 12 within reach
        dist = np.where(np.isfinite(z["plot_X"][:, 0]), 0.0, 1.0)
        np.savez(root / f"plot_fill_{name}.npz", X=Xf, dist=dist, n=len(Xf))
    # restored shoreline presences of a few species
    n_r = 30
    r_rc, r_lat, r_lon = _where(rng, n_r)
    r_sid = np.sort(rng.integers(0, S, n_r)).astype(np.int32)
    np.savez(root / "shore_records.npz", lat=r_lat, lon=r_lon, sid=r_sid, X=_env(rng, n_r), rc=r_rc,
             fill_km=np.full(n_r, 0.5, np.float32), names=np.array(NAMES))
    # the 240 m grid (stack + ecoregions) and the field pyramid on it
    grid = root / "test240"
    grid.mkdir()
    G = _env(rng, H * W)
    G[:2000, :] = np.nan
    np.ascontiguousarray(G.T.reshape(4, H, W)).tofile(grid / "test240_stack.f32")
    transform = [240.0, 0.0, -1e6, 0.0, -240.0, 1.5e6]
    (grid / "test240_stack.json").write_text(json.dumps({"variables": NAMES, "shape": [4, H, W], "crs": "EPSG:5070",
                                                         "transform": transform}))
    eco = np.repeat(np.repeat(rng.integers(1, 5, (6, 6)), 25, 0), 50, 1).astype(np.uint16)
    with rasterio.open(grid / ECOREGION_RASTER, "w", driver="GTiff", height=H, width=W, count=1, dtype="uint16",
                       crs="EPSG:5070", transform=from_origin(-1e6, 1.5e6, 240, 240)) as r:
        r.write(eco, 1)
    build_pyramid(grid / "test240_stack.f32", ["v0", "v1", "v2"], root / "field", levels=4, log_channels=("v2",),
                  log=lambda m: None)
    np.save(root / "field" / "rc_train.npy", rc)
    np.save(root / "field" / "rc_plots_vegbank.npy", rc_vb)
    np.save(root / "field" / "rc_plots_aim.npy", rc_aim)
    snap_to_records(root, log=lambda m: None)
    return {"root": root, "grid": grid, "S": S, "tips": tips}


SMALL = dict(width=16, depth=2, batch_species=8, n_presence=16, n_background=64, log_variables=["v2"],
             flag_variables=["v3"])
FULL = dict(target_group=True, continental_background=16, shoreline_records=True, fill_plots=True, place_dim=8,
            place_hidden=16, place_blocks=2, field=True, field_dim=16, field_heads=2, field_layers=1,
            field_radii=[1, 2, 4, 8], field_angles=8, field_orders=2, field_chunk=256, calibration_penalty=3.0)


@pytest.fixture(scope="module")
def stages(world, tmp_path_factory):
    root, quiet = world["root"], (lambda m: None)
    runs = tmp_path_factory.mktemp("runs")
    eco = world["grid"] / ECOREGION_RASTER
    base = train(root, runs / "base", TrainConfig(steps=60, lr=1e-2, eval_every=30, **SMALL), device=DEVICE, log=quiet)
    rep = train(root, runs / "rep", TrainConfig(steps=40, lr=3e-3, eval_every=20, place_lr=1e-2, **SMALL, **FULL),
                device=DEVICE, log=quiet, init=runs / "base", ecoregion_raster=eco)
    cache = runs / "cache"
    build_cache(runs / "rep", root, cache, ["v3"], device=DEVICE, chunk=500, log=quiet)
    spc = train(root, runs / "species", TrainConfig(steps=60, lr=1e-2, eval_every=30, weight_decay_edges=1.0,
                                                    freeze_shared=True, **SMALL, **FULL),
                device=DEVICE, log=quiet, shared_from=runs / "rep", cache_dir=cache, ecoregion_raster=eco)
    return {"runs": runs, "cache": cache, "records": (base, rep, spc)}


def _model(run, world):
    return JointRangeModel.load(run / "model_best.pt", DEVICE, world["root"] / "field")


def test_stages_train_and_report_both_rankings(stages, world):
    base, rep, spc = stages["records"]
    assert [r["step"] for r in rep["log"]] == [20, 40] and [r["step"] for r in spc["log"]] == [30, 60]
    for rec in (rep["log"][-1], spc["log"][-1]):
        for k in ("dev", "test", "aim"):
            assert np.isfinite(rec[f"{k}_auc_median"]) and np.isfinite(rec[f"{k}_noclip_auc_median"])
    assert spc["log"][-1]["dev_noclip_auc_median"] > 0.7, spc["log"][-1]
    m = _model(stages["runs"] / "rep", world)
    assert m.place_dim == 8 and m.has_penalty and m.field is not None and m.field.n_ch == 3
    assert rep["init"].endswith("base") and spc["shared_from"].endswith("rep")


def test_cache_equals_recomputed_features(stages, world):
    root = world["root"]
    m = _model(stages["runs"] / "rep", world)
    st = Standardizer.load(stages["runs"] / "rep" / "norm.npz")
    data = JointData(root)
    restored = data.restored()
    rows = np.load(stages["cache"] / "rows.npy")
    assert np.array_equal(rows, record_rows(data, restored))
    n0 = len(data["pres"])
    assert (rows >= n0).sum() == len(restored["sid"])
    F, P = np.load(stages["cache"] / "F.npy"), np.load(stages["cache"] / "P.npy")
    pick = np.r_[np.random.default_rng(0).choice(np.flatnonzero(rows < n0), 300, replace=False),
                 np.flatnonzero(rows >= n0)]
    j = rows[pick]
    base, ext = j[j < n0], j[j >= n0] - n0
    X = np.concatenate([data.points()[base], restored["X"][ext]])
    ll = np.c_[np.r_[data["lat"][base], restored["lat"][ext]], np.r_[data["lon"][base], restored["lon"][ext]]]
    rc = np.concatenate([data.positions()[base], restored["rc"][ext]])
    with torch.no_grad(), torch.autocast(torch.device(DEVICE).type, dtype=torch.bfloat16):
        f, p = m.shared(torch.from_numpy(st.transform(X)).to(DEVICE),
                        torch.from_numpy(ll.astype(np.float32)).to(DEVICE),
                        torch.from_numpy(rc.astype(np.float32)).to(DEVICE))
    f, p = f.float().cpu().numpy(), p.float().cpu().numpy()
    # equal up to bfloat16 rounding (2^-8 relative), which may differ with the batch the GPU evaluates
    assert np.abs(F[pick].astype(np.float32) - f).max() <= 8e-3 * np.abs(f).max() + 1e-3
    assert np.abs(P[pick].astype(np.float32) - p).max() <= 8e-3 * np.abs(p).max() + 1e-3
    assert np.abs(f).max() > 0 and np.abs(p).max() > 0                       # the pathways are not empty
    sets = plot_sets(data, calibration_sets(data.species()), st, DEVICE, fill=True, field_dir=root / "field")
    cache = FeatureCache(stages["cache"], DEVICE)
    for ps in sets:
        fc, pc = cache.plot_features(ps)
        fl, pl = plot_features(m, ps)
        assert (fc - fl).abs().max() <= 8e-3 * fl.abs().max() + 1e-3
        assert (pc - pl).abs().max() <= 8e-3 * pl.abs().max() + 1e-3


def test_species_stage_keeps_the_shared_networks(stages, world):
    rep = torch.load(stages["runs"] / "rep" / "model_best.pt", weights_only=True)
    spc = torch.load(stages["runs"] / "species" / "model_best.pt", weights_only=True)
    for k, v in rep.items():
        if k in SPECIES_TENSORS:
            if k != "A":
                assert not torch.equal(v, spc[k]), k                          # species parameters refitted
        elif k != "c0":
            assert torch.equal(v, spc[k]), k                                  # shared networks fixed
    # the scores the species stage evaluated (cached features) equal the full model's at the plots
    m = _model(stages["runs"] / "species", world)
    data = JointData(world["root"])
    st = Standardizer.load(stages["runs"] / "species" / "norm.npz")
    cache = FeatureCache(stages["cache"], DEVICE)
    for ps in plot_sets(data, calibration_sets(data.species()), st, DEVICE, fill=True,
                        field_dir=world["root"] / "field"):
        full = plot_scores(m, ps)
        cached = scores_from_features(m, ps, *cache.plot_features(ps))
        assert np.abs(full - cached).max() <= 1e-2 * np.abs(full).max(), np.abs(full - cached).max()
        assert np.abs(full).max() > 1


def test_store_of_the_full_model(stages, world, tmp_path):
    m = _model(stages["runs"] / "species", world)
    st = Standardizer.load(stages["runs"] / "species" / "norm.npz")
    data = JointData(world["root"])
    S = world["S"]
    inv = pd.DataFrame({"wcvp_accepted_name": [t.replace("_", " ") for t in world["tips"]], "native_l3": "AAA"})
    zs = infer_species(m, world["tips"][:S], Tree.read(world["root"] / "tree/natives.dated.nwk"), inv, {"AAA": [1, 2]})
    assert zs.names == ["Sp_new"] and zs.V.shape == (1, 8) and zs.penalty.shape == (1,)
    gi = ST.GridInputs(world["grid"], None, st.names)
    delta = 0.02
    ST.build_store(m, st, data, tmp_path, {"test": gi}, delta, zs, {"blocks": 4, "block": 64, "cells": 3000}, tile=64,
                   device=DEVICE, positions=data.positions(), log=lambda m_: None)
    store = ST.Store(tmp_path, grids={"test": world["grid"]})
    T = store.T
    assert store.has_penalty and T["codes"].shape[1] == m.width + m.place_dim
    X = gi.block(0, H, 0, W)
    ok = np.flatnonzero(np.isfinite(X[:, st.required_columns]).all(1))
    latlon, rc = gi.locations(0, H, 0, W)
    Hc = ST.shared_features(m, st.transform(X[ok]), latlon[ok], rc[ok], DEVICE)
    Wv, b, pen = ST.species_matrix(m)
    Wv = torch.cat([Wv, torch.cat([zs.W, zs.V], 1).float().to(Wv.device)])
    b = torch.cat([b, torch.tensor(zs.b, device=b.device)])
    full = (Hc @ Wv.T + b).cpu().numpy()
    dec = store.scores("test", 0, H, 0, W, np.arange(S + 1)).reshape(S + 1, -1).T[ok]
    err = np.sqrt(((dec - full) ** 2).sum(1))
    assert err.max() <= math.sqrt(m.width + m.place_dim) * delta / 2 * 1.01 + 1e-3, err.max()
    assert np.isclose(T["penalty"][S], float(zs.penalty[0])) and np.allclose(T["penalty"][:S], pen.cpu().numpy())
    eco = store.ecoregions("test").ravel()
    valid = store.valid("test", 0, H, 0, W).ravel()
    for s in (5, 0, S):                                                      # a species with a small area, inferred
        f = store.scores("test", 0, H, 0, W, [s])[0].ravel()
        area = np.isin(eco, [int(v) for v in T["calibration"][s].split()])
        g = f - float(T["penalty"][s]) * ~area
        u = store.suitability("test", 0, H, 0, W, s).ravel()
        assert (u[valid] >= 1).all() and not u[~valid].any()                 # served outside the area too
        assert (u[valid] == 1 + np.searchsorted(T["quantiles"][s], g[valid])).all()
        out = ~area & valid
        assert out.any() == (s != 0)                                         # species 0's area covers the grid
        assert (u[out] <= 1 + np.searchsorted(T["quantiles"][s], f[out])).all()   # lowered, never raised
        assert (store.in_range("test", 0, H, 0, W, s).ravel() == ((g >= T["p5"][s]) & valid)).all()
