"""Joint range model and map store (ranges/joint).

* Newick parsing (pre-order ids, internal labels, a 5,000-deep tree) and the Brownian path matrix;
* KLT identity: f = y . r + o exactly before quantization, and the species codes form an orthonormal frame;
* tile writer -> Store round trip (window, scores, scattered cells, packed valid mask), lossless recode, and stores
  in the earlier all-int16 layout (3-column index);
* end to end on synthetic data: train a few steps, infer a species without records from its relatives, build a store
  over a small synthetic 240 m grid, and check the decoded scores and maps against the full model, through every
  reading path (windows, scattered cells, points by coordinates: score, suitability, range).
"""
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from ranges.joint import store as ST
from ranges.joint.data import JointData, Standardizer
from ranges.joint.model import JointRangeModel
from ranges.joint.reader import ECOREGION_RASTER
from ranges.joint.train import TrainConfig, train
from ranges.joint.tree import Tree, parse_newick, path_matrix
from ranges.joint.zero_shot import ZeroShot, infer_species

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------------------------------- tree
def test_newick_parse_and_path_matrix():
    parent, length, label = parse_newick("[&U]((a:1,b:3)x:2,(c,d:1.5):0.5)r;")
    assert parent.tolist() == [-1, 0, 1, 1, 0, 4, 4]
    assert length.tolist() == [0.0, 2.0, 1.0, 3.0, 0.5, 0.0, 1.5]
    assert label == ["r", "x", "a", "b", "", "c", "d"]
    t = Tree(parent, length, label)
    assert sorted(t.tips_under(1)) == ["a", "b"] and sorted(t.tips_under(0)) == ["a", "b", "c", "d"]
    A = path_matrix(t, ["b", "d"]).to_dense()
    scale = np.mean([2.0, 1.0, 3.0, 0.5, 1.5])
    ref = np.zeros((2, 7), np.float32)
    ref[0, [3, 1]] = np.sqrt(np.array([3.0, 2.0]) / scale)
    ref[1, [6, 4]] = np.sqrt(np.array([1.5, 0.5]) / scale)
    assert np.allclose(A.numpy(), ref)
    # Brownian covariance: <A_b, A_d> = 0 (they share no branch), |A_b|^2 = root-to-tip time / mean branch
    assert math.isclose(float(A[0] @ A[0]), 5.0 / scale, rel_tol=1e-6)
    deep = "(" * 5000 + "t0:1" + "".join(f",t{i + 1}:1):1" for i in range(5000)) + ";"
    p, _, lab = parse_newick(deep)
    assert len(p) == 10001 and lab.count("") == 5000
    with pytest.raises(KeyError):
        path_matrix(t, ["nope"])


# ----------------------------------------------------------------------------------------------- transform
def test_klt_identity():
    g = torch.Generator().manual_seed(0)
    S, d, n = 900, 64, 5000
    W = torch.randn(S, d, generator=g)
    b = torch.randn(S, generator=g)
    H = torch.randn(n, d, generator=g) @ torch.randn(d, d, generator=g)
    mu, T, R, off, ev = ST.klt(H, W, b)
    y = (H.double() - mu.double()) @ T.double()
    f = y @ R.double().T + off.double()
    ref = H.double() @ W.double().T + b.double()
    assert torch.allclose(f, ref, atol=1e-6 * ref.abs().max()), float((f - ref).abs().max())
    assert torch.allclose(R.double().T @ R.double(), torch.eye(d, dtype=torch.float64), atol=1e-4)
    assert (ev[:-1] >= ev[1:] - 1e-6).all()


def _write_store(out, region, q, valid, codes, offs, tile=ST.TILE):
    H, W, d = q.shape
    w = ST.KltWriter(out, region, H, W, d, tile)
    for r0 in range(0, H, tile):
        w.put(r0, q[r0:r0 + tile])
    w.close()
    np.save(out / f"valid_{region}.npy", np.packbits(valid, axis=1))
    (out / "store.json").write_text(json.dumps({"codec": "klt", "tile": tile, "d": d, "valid": "packbits",
                                                "shape": {region: [H, W]}}))
    np.savez(out / "species.npz", codes=codes, offsets=offs)


class _Int16Writer(ST.KltWriter):
    """The tile layout before the int8 split: every stored channel as int16 byte planes, a 3-column index."""

    def encode_tile(self, q):
        c = np.ascontiguousarray(np.moveaxis(q, -1, 0))
        nz = np.flatnonzero(c.reshape(c.shape[0], -1).any(1))
        if not len(nz):
            return b"", 0, 0
        k = int(nz[-1]) + 1
        planes = np.ascontiguousarray(c[:k]).view(np.uint8).reshape(-1, 2).T
        import zstandard
        return zstandard.ZstdCompressor(level=3).compress(np.ascontiguousarray(planes).tobytes()), k, k

    def close(self):
        self.blob.close()
        self.pool.shutdown()
        np.save(self.path, self.index[..., :3])


def test_tile_roundtrip_and_recode(tmp_path):
    rng = np.random.default_rng(1)
    H, W, d = 300, 4000, 40
    q = np.zeros((H, W, d), np.int16)
    q[..., :3] = rng.integers(-3000, 3000, (H, W, 3))                 # 16-bit channels
    q[..., 3:25] = rng.integers(-127, 128, (H, W, 22))                # 8-bit channels
    q[:128, :128] = 0                                                  # an empty tile
    q[128:256, :, 30:] = 0
    valid = rng.random((H, W)) > 0.3
    codes = rng.standard_normal((3, d)).astype(np.float32)
    offs = rng.standard_normal(3).astype(np.float32)
    src = tmp_path / "a"
    src.mkdir()
    _write_store(src, "hawaii", q, valid, codes, offs)
    recoded = tmp_path / "b"
    ST.recode_store(src, recoded, log=lambda m: None)
    legacy, from_legacy = tmp_path / "c", tmp_path / "d"
    legacy.mkdir()
    _write_store(legacy, "hawaii", q, valid, codes, offs)
    w = _Int16Writer(legacy, "hawaii", H, W, d)
    for r0 in range(0, H, ST.TILE):
        w.put(r0, q[r0:r0 + ST.TILE])
    w.close()
    assert np.load(legacy / "index_hawaii.npy").shape[-1] == 3
    ST.recode_store(legacy, from_legacy, log=lambda m: None)            # old layout -> current layout, lossless
    assert np.load(from_legacy / "index_hawaii.npy").shape[-1] == 4
    for out in (src, recoded, legacy, from_legacy):
        st = ST.Store(out, cache_tiles=4)
        for (r0, r1, c0, c1) in [(0, H, 0, W), (5, 290, 100, 401), (127, 129, 127, 129), (250, 300, 500, 517)]:
            g, v = st.window("hawaii", r0, r1, c0, c1)
            k = g.shape[-1]
            assert (g == q[r0:r1, c0:c1, :k]).all() and not q[r0:r1, c0:c1, k:].any()
            assert (v == valid[r0:r1, c0:c1]).all()
        for (r0, r1, c0, c1) in [(0, H, 0, W), (5, 290, 100, 401), (250, 300, 3990, 4000)]:
            f = np.moveaxis(st.scores("hawaii", r0, r1, c0, c1, [2, 0]), 0, -1)
            ref = q[r0:r1, c0:c1].astype(np.float64) @ codes[[2, 0]].T.astype(np.float64) + offs[[2, 0]]
            assert np.allclose(f, ref, rtol=1e-5, atol=1e-2), float(np.abs(f - ref).max())
        rr, cc = rng.integers(-5, H + 5, 4000), rng.integers(-5, W + 5, 4000)
        G, v = st.cells("hawaii", rr, cc)
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        assert (G[ok] == q[rr[ok], cc[ok]]).all() and not G[~ok].any()
        assert (v[ok] == valid[rr[ok], cc[ok]]).all() and not v[~ok].any()


# ------------------------------------------------------------------------------------------ end to end
NAMES = ["v0", "v1", "v2", "v3"]          # v2 heavy-tailed (log-transformed), v3 sometimes missing (flagged)


def _env(rng, n):
    X = rng.standard_normal((n, 4)).astype(np.float32)
    X[:, 2] = np.exp(X[:, 2])
    X[rng.random(n) < 0.2, 3] = np.nan
    return X


def _suitability(X, opt):
    return np.exp(-((X[:, None, :2] - opt[None]) ** 2).sum(-1) / 0.5)          # [n, species]


def _balanced(tips):
    if len(tips) == 1:
        return tips[0]
    h = len(tips) // 2
    return f"({_balanced(tips[:h])}:1.0,{_balanced(tips[h:])}:1.0)"


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """24 species with records + 1 without; niche optima inherited along the tree (Brownian), training points,
    VegBank-like and AIM-like plots, and a 150 x 300-cell grid with 4 ecoregions."""
    root = tmp_path_factory.mktemp("joint")
    rng = np.random.default_rng(0)
    S = 24
    tips = [f"Sp_{i:02d}" for i in range(S)] + ["Sp_new"]
    order = tips[:12] + ["Sp_new"] + tips[12:S]               # Sp_new: sister clade of Sp_12, Sp_13
    (root / "tree").mkdir()
    (root / "tree/natives.dated.nwk").write_text(_balanced(order) + ";")
    # optima: half of the tree on one side of v0, the other half on the other side, plus noise
    opt = np.c_[np.where(np.arange(S) < 12, -1.0, 1.0), np.zeros(S)] + rng.normal(0, 0.4, (S, 2))
    calib = ["1 2 3 4"] * S
    calib[5] = "1 2"
    pd.DataFrame({"species": tips[:S], "calibration_ecoregions": calib}).to_csv(root / "species.csv", index=False)
    Xs, sid, pres = [], [], []
    for s in range(S):
        pool = _env(rng, 4000)
        p = _suitability(pool, opt[s:s + 1])[:, 0]
        pick = rng.choice(len(pool), 80, replace=False, p=p / p.sum())
        bg = _env(rng, 400)
        Xs += [pool[pick], bg]
        sid += [np.full(80, s), np.full(400, s)]
        pres += [np.ones(80, np.int8), np.zeros(400, np.int8)]
    np.save(root / "train_points_cache.npy", np.concatenate(Xs))

    def plots(n, seed):
        r = np.random.default_rng(seed)
        PX = _env(r, n)
        y = r.random((n, S)) < 0.8 * _suitability(PX, opt)
        return dict(plot_X=PX, plot_lat=r.uniform(30, 36, n).astype(np.float32),
                    plot_lon=r.uniform(-110, -100, n).astype(np.float32), plot_eco=r.integers(1, 5, n).astype(np.int32),
                    eval_sid=np.arange(S, dtype=np.int32), eval_y=y.T.astype(np.int8))
    vb = plots(1500, 1)
    np.savez(root / "joint_data.npz", names=np.array(NAMES), sid=np.concatenate(sid).astype(np.int32),
             pres=np.concatenate(pres), **vb, maxent_occ=np.full((S, 1500), np.nan, np.float16))
    np.savez(root / "plots_aim.npz", **plots(800, 2))
    # grid: band-sequential stack + ecoregion raster (ConusStack layout)
    import rasterio
    from rasterio.transform import from_origin
    grid = root / "test240"
    grid.mkdir()
    H, W = 150, 300
    X = _env(rng, H * W)
    X[:2000, :] = np.nan                                       # cells without climate (e.g. a lake)
    np.ascontiguousarray(X.T.reshape(4, H, W)).tofile(grid / "test240_stack.f32")
    (grid / "test240_stack.json").write_text(json.dumps({"variables": NAMES, "shape": [4, H, W], "crs": "EPSG:5070"}))
    eco = np.repeat(np.repeat(rng.integers(1, 5, (6, 6)), 25, 0), 50, 1).astype(np.uint16)
    with rasterio.open(grid / ECOREGION_RASTER, "w", driver="GTiff", height=H, width=W, count=1, dtype="uint16",
                       crs="EPSG:5070", transform=from_origin(-1e6, 1.5e6, 240, 240)) as r:
        r.write(eco, 1)
    return {"root": root, "grid": grid, "S": S, "tips": tips}


@pytest.fixture(scope="module")
def trained(synthetic, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    cfg = TrainConfig(width=16, depth=2, steps=300, batch_species=8, n_presence=16, n_background=64, lr=1e-2,
                      eval_every=100, log_variables=["v2"], flag_variables=["v3"])
    rec = train(synthetic["root"], out, cfg, device=DEVICE, log=lambda m: None)
    return out, rec


def test_training_learns(trained, synthetic):
    out, rec = trained
    assert [r["step"] for r in rec["log"]] == [100, 200, 300]
    last = rec["log"][-1]
    assert last["dev_auc_median"] > 0.75 and last["test_auc_median"] > 0.75 and last["aim_auc_median"] > 0.75, last
    assert last["dev_species"] == synthetic["S"]
    assert json.loads((out / "run.json").read_text())["config"]["width"] == 16
    model = JointRangeModel.load(out / "model.pt")
    assert model.n_species == synthetic["S"] and model.width == 16
    per = pd.read_csv(out / "per_species_vegbank.csv")
    assert set(per.columns) >= {"species", "joint_dev", "joint_test"}


def _zero_shot(model, synthetic):
    inv = pd.DataFrame({"wcvp_accepted_name": [t.replace("_", " ") for t in synthetic["tips"]], "native_l3": "AAA,BBB"})
    return infer_species(model, synthetic["tips"][:synthetic["S"]],
                         Tree.read(synthetic["root"] / "tree/natives.dated.nwk"), inv, {"AAA": [1, 2], "BBB": [3]})


def test_store_end_to_end(trained, synthetic, tmp_path):
    out, _ = trained
    root, S = synthetic["root"], synthetic["S"]
    model = JointRangeModel.load(out / "model_best.pt", DEVICE)
    st = Standardizer.load(out / "norm.npz")
    data = JointData(root)
    zs = _zero_shot(model, synthetic)
    assert zs.names == ["Sp_new"] and zs.calibration == ["1 2 3"]
    # Sp_new joins its closest trained relatives (Sp_12, Sp_13) at its parent node x: its vector is the path sum from
    # the root to x, not the path to Sp_12 and Sp_13's own common ancestor (which lies below x)
    rel = zs.relatives[0]
    assert sorted(synthetic["tips"][r] for r in rel) == ["Sp_12", "Sp_13"]
    tree = Tree.read(root / "tree/natives.dated.nwk")
    scale = tree.length[tree.length > 0].mean()
    x, expect = tree.parent[tree.node_of()["Sp_new"]], torch.zeros(model.width, device=model.z.device)
    while tree.parent[x] >= 0:
        expect += math.sqrt(tree.length[x] / scale) * model.z[x]
        x = tree.parent[x]
    assert torch.allclose(zs.W[0], expect, atol=1e-5)
    A = model.A.to_dense()
    assert not torch.allclose(zs.W[0], (A[rel[0]] * (A[rel] > 0).all(0)) @ model.z, atol=1e-3)
    assert np.isclose(zs.b[0], float(model.b[rel].mean()), atol=1e-6)

    gi = ST.GridInputs(synthetic["grid"], None, st.names)
    delta = 0.02
    ST.build_store(model, st, data, tmp_path, {"test": gi}, delta, zs, {"blocks": 4, "block": 64, "cells": 3000},
                   tile=64, device=DEVICE, log=lambda m: None)
    store = ST.Store(tmp_path, grids={"test": synthetic["grid"]})
    T = store.T
    assert list(T["species"]) == synthetic["tips"] and T["inferred"].tolist() == [False] * S + [True]
    H, W = store.shape("test")
    X = gi.block(0, H, 0, W)
    ok = np.isfinite(X[:, st.required_columns]).all(1)
    assert (store.valid("test", 0, H, 0, W).ravel() == ok).all()
    with torch.no_grad():
        Wv = torch.cat([model.species_vectors().float(), zs.W.float()])
        bv = torch.cat([model.b.float(), torch.tensor(zs.b, device=model.b.device)])
        h = ST.features(model, st.transform(X[ok]), DEVICE)
        full = (h @ Wv.T + bv).cpu().numpy()                              # [land cells, species]
    dec = store.scores("test", 0, H, 0, W, np.arange(S + 1)).reshape(S + 1, -1).T[ok]
    # orthonormal frame: the total squared score error of a cell over all species is its quantization error |e|^2,
    # at most d (delta / 2)^2
    err = np.sqrt(((dec - full) ** 2).sum(1))
    assert err.max() <= math.sqrt(model.width) * delta / 2 * 1.01 + 1e-3, err.max()
    # window and scattered cells read the same field
    q, _ = store.window("test", 10, 140, 20, 290)
    rr, cc = np.meshgrid(np.arange(10, 140), np.arange(20, 290), indexing="ij")
    G, _ = store.cells("test", rr.ravel(), cc.ravel())
    assert (G[:, :q.shape[-1]] == q.reshape(-1, q.shape[-1])).all()
    # served maps: 0 outside the calibration ecoregions / without climate, else the background-quantile rank
    eco = store.ecoregions("test").ravel()
    for s in (0, 5, S):
        m = store.decode("test", 0, H, 0, W, s).ravel()
        inside = np.isin(eco, [int(v) for v in T["calibration"][s].split()]) & ok
        assert not m[~inside].any() and (m[inside] >= 1).all()
        ref = np.zeros(H * W, np.uint8)                            # exactly the rank of the stored score ...
        ref[ok] = 1 + np.searchsorted(T["quantiles"][s], dec[:, s])
        assert (m[inside] == ref[inside]).all()
        ref[ok] = 1 + np.searchsorted(T["quantiles"][s], full[:, s])   # ... within one rank of the full model's
        assert (np.abs(m[inside].astype(int) - ref[inside]) <= 1).mean() > 0.99
    # the reading paths agree: points by grid cell or by coordinates give the window's score, suitability, range
    s = store.index("Sp 05")
    assert s == 5 and store.index("Sp_05") == 5
    with pytest.raises(KeyError):
        store.index("Nope")
    f = store.scores("test", 0, H, 0, W, [s])[0]
    u = store.suitability("test", 0, H, 0, W, s)
    rng_map = store.in_range("test", 0, H, 0, W, s)
    inside = store.inside("test", 0, H, 0, W, s)
    assert (rng_map == ((f >= T["p5"][s]) & inside)).all() and rng_map.any() and not rng_map[~inside].any()
    rr, cc = np.random.default_rng(3).integers(-3, H + 3, 500), np.random.default_rng(4).integers(-3, W + 3, 500)
    pts = store.at("test", rows=rr, cols=cc, species=[s, S])
    on = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    assert np.allclose(pts["score"][on, 0], f[rr[on], cc[on]], rtol=1e-6, atol=1e-5)
    assert (pts["suitability"][on, 0] == u[rr[on], cc[on]]).all() and not pts["suitability"][~on].any()
    assert (pts["range"][on, 0] == rng_map[rr[on], cc[on]]).all() and not pts["range"][~on].any()
    assert (pts["suitability"][on, 1] == store.suitability("test", 0, H, 0, W, S)[rr[on], cc[on]]).all()
    from pyproj import Transformer
    t = store._grid_raster("test")[1]
    x, y = t.c + (cc[on] + 0.5) * t.a, t.f + (rr[on] + 0.5) * t.e          # cell centres in the grid's CRS
    lon, lat = Transformer.from_crs(5070, 4326, always_xy=True).transform(x, y)
    by_coord = store.at("test", lon=lon, lat=lat, species=s)
    assert (by_coord["suitability"][:, 0] == pts["suitability"][on, 0]).all()
    assert (by_coord["range"][:, 0] == pts["range"][on, 0]).all()
    # species table: quantiles of f_s over the species' own background, P5 over its presences
    sid, pres = np.asarray(data["sid"]), np.asarray(data["pres"]).astype(bool)
    Xp = data.points()
    for s in (0, 7):
        with torch.no_grad():
            fs = (ST.features(model, st.transform(Xp[sid == s]), DEVICE) @ Wv[s] + bv[s]).cpu().numpy()
        assert np.allclose(T["quantiles"][s], np.quantile(fs[~pres[sid == s]], np.linspace(0, 1, 256)[1:-1]), atol=1e-4)
        assert np.isclose(T["p5"][s], np.quantile(fs[pres[sid == s]], 0.05), atol=1e-4)


def test_update_zero_shot_matches_rebuild(trained, synthetic, tmp_path):
    """A store built with other vectors for the species without records, then updated, gives those species the
    same maps as a store rebuilt with the new vectors: identical quantiles and P5, scores within the quantization
    bound of the full model; trained species untouched; a second update changes nothing."""
    out, _ = trained
    model = JointRangeModel.load(out / "model_best.pt", DEVICE)
    st = Standardizer.load(out / "norm.npz")
    data = JointData(synthetic["root"])
    zs = _zero_shot(model, synthetic)
    old = ZeroShot(zs.names, zs.W * 0.5 + 0.3, zs.b - 1.0, zs.calibration, zs.relatives)
    gi = ST.GridInputs(synthetic["grid"], None, st.names)
    kw = dict(delta=0.02, sample={"blocks": 4, "block": 64, "cells": 3000}, tile=64, device=DEVICE, log=lambda m: None)
    ST.build_store(model, st, data, tmp_path / "updated", {"test": gi}, zero_shot=old, **kw)
    ST.build_store(model, st, data, tmp_path / "rebuilt", {"test": gi}, zero_shot=zs, **kw)
    before = dict(np.load(tmp_path / "updated/species.npz"))
    assert ST.update_zero_shot(tmp_path / "updated", model, st, data, zs, DEVICE, log=lambda m: None) == 1
    assert ST.update_zero_shot(tmp_path / "updated", model, st, data, zs, DEVICE, log=lambda m: None) == 0
    assert ST.update_zero_shot(tmp_path / "updated", model, st, data, zs, DEVICE, recompute_all=True,
                               log=lambda m: None) == 1
    U, R = (dict(np.load(tmp_path / f"{d}/species.npz")) for d in ("updated", "rebuilt"))
    S = synthetic["S"]
    for k in ("codes", "offsets", "quantiles", "p5"):
        assert np.array_equal(U[k][:S], before[k][:S])                    # trained species untouched
    assert np.array_equal(U["quantiles"][S], R["quantiles"][S]) and U["p5"][S] == R["p5"][S]
    assert not np.allclose(before["quantiles"][S], U["quantiles"][S])
    H, W = 150, 300
    X = gi.block(0, H, 0, W)
    ok = np.isfinite(X[:, st.required_columns]).all(1)
    with torch.no_grad():
        full = (ST.features(model, st.transform(X[ok]), DEVICE) @ zs.W[0].float() + float(zs.b[0])).cpu().numpy()
    for d in ("updated", "rebuilt"):
        f = ST.Store(tmp_path / d).scores("test", 0, H, 0, W, [S])[0].ravel()[ok]
        assert np.abs(f - full).max() <= math.sqrt(model.width) * 0.02 / 2 * 1.01 + 1e-3, (d, np.abs(f - full).max())
