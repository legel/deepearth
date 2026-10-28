"""PLANT as one absolute probability, the same on any block of any site (2026-10-26: "Habitability modeling should
not be relative to some site-specific baseline. It should be as close as possible to a global probability").

The national range model gives each species an intensity at every 240 m cell, lambda = ln 2 exp(g - P5) (g its served
score: f, less the learned penalty outside its calibration ecoregions; P5 its range threshold), and its habitability
probability there, p = 1 - exp(-lambda) (planting.py _hab: 0.5 at the range edge). The palette read it at the block's
centre cell only (palette.py: suitability[0, i]), so every block carried one national constant and a block whose centre
fell below P5 read no probability at all (two blocks of a bay-edge site). Here it is a field over the whole site: ln lambda at
every 240 m cell covering the site and a ring, the cells without national climate (open water, a bay edge) filled
from their neighbours, and interpolated bilinearly to each 1 m cell, so the national term never steps at a block line.

E3DT's G says where inside a 240 m cell the habitat concentrates, but its level carries the store's own conditioning (its
pooled readout and site vector: 9 nats between neighbours at UC), so G is taken relative to its own 240 m cell, one store
and a fixed national unit. The trait windows are the palette's (palette.logwin, the trait ontology's windows), read per
1 m cell from the soil stage's own layers (soil/soil.simf), never the parcel's centre value: light (the cell's ground
light), soil (pH and texture), water (the cell's soil moisture from the Simulator's hydrology, else its SSURGO drainage),
own crown and conspecific competition, all logs at most 0:

    cloglog P_s(q) = ln lambda_s(q) + (G_s(q) - log mean exp of G_s over cell(q)) + L + W + soil + logA + logC

No site, block, palette rank or scene mean enters it: the same inputs give the same number anywhere.

Writes each block's design/score_u8.bin as round(100 P) (255 where the block scored nothing), habitable at P >= 0.5,
points_top from P, and design/probability.json, the flag the page reads (score_u8 is P in percent; multiply nothing).

    python -m deepearth_site.prob_pass plan.json      (plan as deepearth_site.site_pass's; mode "owned" or "blend")
"""
from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .site_pass import Block, gs, lse

LN_LN2 = math.log(math.log(2.0))
FLOOR = -60.0                          # ln lambda where a species is not served (P = 0 to 26 digits)
HAB_P = 0.5                            # habitable: P at least the range edge's (js/habitability.js HABITABLE_P)
TOP = 25
RING = 3                               # 240 m cells read past the blocks: a coast's gap fills from the same land either side


def read_simf(path) -> dict:
    """{field: float32 [ny * nx]} of a one-frame surface SIMF v2 (soil.write_simf)."""
    b = Path(path).read_bytes()
    if b[:4] != b"SIMF":
        raise ValueError(f"{path}: not SIMF")
    _ver, _frames, nx, ny, nz, nf, _ = struct.unpack_from("<7I", b, 4)
    o = 4 + 28 + 8 + 24 + 8
    names = [b[o + 16 * i:o + 16 * i + 8].rstrip(b"\0").decode() for i in range(nf)]
    o += 16 * nf + 4 * (nz + 1) + 8
    k = nz * ny * nx
    return {n: np.frombuffer(b, np.float32, k, o + 4 * i * k) for i, n in enumerate(names)}


def texture_class(sand, clay):
    """palette.texture_class over arrays (NaN where either is unknown)."""
    t = np.select([clay >= 40, clay >= 27, sand >= 85, sand >= 52], [4.0, 3.0, 0.0, 1.0], 2.0).astype(np.float32)
    return np.where(np.isfinite(sand) & np.isfinite(clay), t, np.nan).astype(np.float32)


def window(v, lo, hi, w):
    """palette.logwin of cell values v [n] against species windows lo, hi [S, 1]: [S, n], 0 where v or the trait is unknown
    (the palette's rule for a condition not held)."""
    from .palette import logwin
    with np.errstate(invalid="ignore"):
        x = logwin(v[None].astype(np.float32), lo, hi, np.float32(w)).astype(np.float32)
    return np.where(np.isfinite(x), x, np.float32(0.0))


# ---------------------------------------------------------------------------------------- site conditions (2026-10-28)
# simulation docs/PLANT_SITE_CONDITIONS.md: the soil survey's salinity, flooding and seasonal water, each held against a
# trait the posteriors already carry and marginalized over it, log sum_k P(trait = k) f_k(cell), so an uncertain species
# is penalized by its chance of being intolerant. Every term is at most 0 and is 0 where its cell value is unknown.
CONDITIONS = ("salt", "flood", "seasonal", "aquatic")
AQUATIC_MSAT = 10.0                    # months saturated near the surface: ground an obligate aquatic can hold
OPEN_WATER = 253                       # the moisture grid's open-water code
SALT_TOP_LOG2 = (1.0, 2.0, 3.0)        # USDA PLANTS salinity tolerance tops: none 2, low 4, moderate 8 dS/m; high none
SALT_W = 0.5                           # log2 EC per nat: each doubling past a class top costs 2 nats
EC_FLOOR = 0.25                        # dS/m: the log scale's floor (fresh ground reads it)
TIDAL_EC = 8.0                         # NWI intertidal ground without a survey EC: the moderate to high boundary
FLOOD_TOP = (0.5, 1.0, 3.0)            # months saturated each flood tolerance class survives (Whitlow and Harris 1979)
FLOOD_W = (0.5, 0.375, 0.25)           # months per nat by root depth: a 30 cm water table saturates more of a deep root
WATER_W = 0.5                          # the annual water window's width, on the trait's 0..5 scale (2.5 x moisture)


def switches(plan: dict | None = None) -> dict:
    """{term: on} from plan["conditions"] (a list), else DEEPEARTH_SITE_CONDITIONS ("salt,flood,seasonal", "" for
    none), else all on. A term on with no layer under it still changes nothing."""
    import os
    v = (plan or {}).get("conditions")
    if v is None:
        v = os.environ.get("DEEPEARTH_SITE_CONDITIONS", ",".join(CONDITIONS))
    if isinstance(v, str):
        v = [s.strip() for s in v.split(",")]
    return {k: k in v for k in CONDITIONS}


def hinge_mix(P, x, tops, w):
    """log sum_k P[:, k] exp(-max(0, x - tops[k]) / w) + P[:, -1] (the top class has no limit): [S, n].
    0 below every class top, linear decay in nats past each: the threshold response (Maas and Hoffman 1977), where a
    logistic window would already cost 0.13 to 0.69 nats inside the class (a 4 to 11 point P change on fresh ground).
    P [S, K + 1]; x [n]; w a scalar or [S, 1]. Rows of P with no posterior read 0."""
    acc = np.zeros((P.shape[0], len(x)), np.float32) + np.nan_to_num(P[:, -1:])
    for k, t in enumerate(tops):
        acc += np.nan_to_num(P[:, k:k + 1]) * np.exp(-np.maximum(0.0, x[None] - t) / w).astype(np.float32)
    out = np.log(np.maximum(acc, 1e-30)).astype(np.float32)
    return np.where(np.isfinite(P).all(1, keepdims=True), out, np.float32(0.0))


def salinity_ec(cond: dict) -> np.ndarray:
    """EC per cell, intertidal ground without a survey value read at TIDAL_EC; NaN unknown."""
    ec = cond.get("ec")
    tid = cond.get("tidal")
    n = len(next(v for k, v in cond.items() if k != "provenance"))
    ec = np.full(n, np.nan, np.float32) if ec is None else ec.astype(np.float32)
    if tid is not None:
        ec = np.where(~np.isfinite(ec) & (tid == 1), np.float32(TIDAL_EC), ec)
    return ec


def months_saturated(cond: dict) -> np.ndarray:
    """Months a year saturated near the surface, raised by the survey's classes: at least 1 under frequent flooding or
    frequent ponding (standing water in more than half of all years), 12 under very frequent flooding; NaN unknown."""
    n = len(next(v for k, v in cond.items() if k != "provenance"))
    nan = np.full(n, np.nan, np.float32)
    m, ff, pf = (cond.get(k, nan) for k in ("msat", "ffreq", "pfreq"))
    known = np.isfinite(m) | np.isfinite(ff) | np.isfinite(pf)
    with np.errstate(invalid="ignore"):
        m = np.where(np.isfinite(m), m, 0.0)
        m = np.where((ff >= 4) | (pf == 3), np.maximum(m, 1.0), m)
        m = np.where(ff == 5, 12.0, m)
    return np.where(known, m, np.nan).astype(np.float32)


def aquatic_allowed(code, cond) -> np.ndarray:
    """bool [n]: where an obligately aquatic species (free-floating, submerged, rooted floating-leaved) can live: open
    water (the moisture grid's 253, NWI subtidal) or ground with the water table within 30 cm 10 months a year or more.
    Emergent wetland plants are not gated: they belong on wet ground."""
    w = np.asarray(code) == OPEN_WATER
    if cond is not None:
        if "tidal" in cond:
            w = w | (cond["tidal"] == 2)
        w = w | (np.nan_to_num(months_saturated(cond), nan=0.0) >= AQUATIC_MSAT)
    return w


def aquatic_gate(conv, plantable, rows, allowed):
    """Habitable as the cell is [S, n] bool. conv: P >= HAB_P where the block scored the cell, made from the gated P
    (aquatic_zero). Open water is not plantable ground for other plants, but it is these
    plants' ground, so their habitable cells are where allowed holds rather than where the ground is plantable."""
    hab = conv & plantable[None]
    if len(rows):
        r = np.asarray(rows)
        hab[r] = conv[r] & allowed[None]
    return hab


def aquatic_zero(P, rows, allowed):
    """P with the aquatic rows set to 0 off `allowed` (in place)."""
    if len(rows):
        P[np.ix_(np.asarray(rows), np.nonzero(~allowed)[0])] = 0.0
    return P


def condition_terms(cond, moist, tv, cat, on) -> tuple[dict, dict]:
    """The new terms for one block: ({"salt", "flood": [S, n]} for each term on with a layer, {"water": [S, n] where
    the seasonal layers hold, NaN elsewhere}), and each term's cells {term: bool [n]}. cond: soil_conditions' fields;
    moist [n] the annual moisture (0..2); tv(key) [S, 1] a continuous trait; cat {trait: P [S, K]} by species."""
    terms, cells = {}, {}
    if cond is None:
        return terms, cells
    S = tv("water_lo").shape[0]
    nanP = lambda K: np.full((S, K), np.nan, np.float32)                                         # noqa: E731
    root = cat.get("root_depth", nanP(3))
    root_ok = np.isfinite(root).all(1, keepdims=True)
    if on.get("salt") and ("ec" in cond or "tidal" in cond) and "salt_tolerance" in cat:
        ec = salinity_ec(cond)
        k = np.isfinite(ec)
        x = np.log2(np.maximum(np.where(k, ec, EC_FLOOR), EC_FLOOR)).astype(np.float32)
        terms["salt"] = np.where(k[None], hinge_mix(cat["salt_tolerance"], x, SALT_TOP_LOG2, SALT_W), np.float32(0.0))
        cells["salt"] = k
    if on.get("flood") and any(f in cond for f in ("msat", "ffreq", "pfreq")) and "flood_tolerance" in cat:
        m = months_saturated(cond)
        k = np.isfinite(m)
        w = np.where(root_ok, np.nan_to_num(root) @ np.asarray(FLOOD_W, np.float32)[:, None], np.float32(FLOOD_W[1]))
        terms["flood"] = np.where(k[None], hinge_mix(cat["flood_tolerance"], np.where(k, m, 0.0).astype(np.float32),
                                                      FLOOD_TOP, w.astype(np.float32)), np.float32(0.0))
        cells["flood"] = k
    if on.get("seasonal") and "mdry" in cond and "mwet" in cond:
        dry, wet = cond["mdry"], cond["mwet"]
        k = np.isfinite(dry) & np.isfinite(wet)
        mid = np.where(np.isfinite(moist), moist, dry)
        # the dry value a root reaches: the top 5 cm's swing for a shallow root, the root zone's annual value for a deep
        R = np.where(root_ok, np.nan_to_num(root), np.array([[0.0, 1.0, 0.0]], np.float32))
        d = R[:, :1] * dry[None] + R[:, 1:2] * (0.5 * (mid + dry))[None] + R[:, 2:] * mid[None]
        lo, hi = tv("water_lo"), tv("water_hi")
        with np.errstate(invalid="ignore"):
            x = (-np.logaddexp(0, -(2.5 * d - lo) / WATER_W) - np.logaddexp(0, -(hi - 2.5 * wet[None]) / WATER_W))
        x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
        terms["water"] = np.where(k[None], x, np.float32(np.nan))
        cells["seasonal"] = k
    return terms, cells


def intensity(rows, cols, species) -> tuple[np.ndarray, np.ndarray, dict]:
    """(ln lambda [n, len(species)] at the national 240 m cells rows/cols, -inf where the species is not served there;
    climate [n], True where the cell has national climate; has_penalty {set: bool})."""
    from .palette import _store
    rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
    out = np.full((len(rows), len(species)), -np.inf, np.float32)
    clim = np.zeros(len(rows), bool)
    pen = {}
    for kind in ("native", "adapted"):
        js = [j for j, (st, i) in enumerate(species) if (st or "native") == kind and int(i) >= 0]
        if not js:
            continue
        S = _store(kind)
        pen[kind] = bool(S.has_penalty)
        ids = [int(species[j][1]) for j in js]
        H, Wd = S.shape("conus")
        on = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < Wd)
        f, valid = S.cell_scores("conus", np.where(on, rows, 0), np.where(on, cols, 0), ids)
        valid = np.asarray(valid, bool) & on
        clim |= valid
        eco = np.zeros(len(rows), np.int64)
        eco[on] = S.ecoregions("conus")[rows[on], cols[on]]
        for a, (j, s) in enumerate(zip(js, ids)):
            g, served = S.served(f[:, a], s, np.isin(eco, S.calibration(s)), valid)
            out[:, j] = np.where(served, LN_LN2 + g - float(S.T["p5"][s]), -np.inf)
    return out, clim, pen


def fill_gaps(V: np.ndarray, known: np.ndarray, max_iter: int = 64) -> tuple[np.ndarray, int]:
    """V [h, w, S] ln lambda, known [h, w] cells with national climate: each other cell takes the log mean exp of its
    known (or already filled) 8 neighbours, ring by ring inward from the coast, the same rule everywhere. A known cell is
    never changed (a species unserved there, under a hard calibration rule, stays unserved). Returns (V, cells filled)."""
    V = V.copy(); K = known.copy(); h, w = K.shape; n0 = int((~K).sum())
    for _ in range(max_iter):
        if K.all():
            break
        P = np.pad(np.where(K[..., None], V, -np.inf), ((1, 1), (1, 1), (0, 0)), constant_values=-np.inf)
        Kp = np.pad(K, 1)
        acc = np.full(V.shape, -np.inf, np.float64); cnt = np.zeros((h, w), np.int64)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == dj == 0:
                    continue
                acc = np.logaddexp(acc, P[1 + di:1 + di + h, 1 + dj:1 + dj + w])
                cnt += Kp[1 + di:1 + di + h, 1 + dj:1 + dj + w]
        new = ~K & (cnt > 0)
        if not new.any():
            break
        V[new] = (acc[new] - np.log(cnt[new])[:, None]).astype(V.dtype)
        K = K | new
    return V, n0 - int((~K).sum())


def bilinear(V: np.ndarray, fr: np.ndarray, fc: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """V [h, w, S] sampled at fractional cell positions (fr, fc) in cell units, cell (i, j)'s centre at (i + .5, j + .5):
    [len(cols), n] for the species columns `cols`."""
    y, x = fr - 0.5, fc - 0.5
    i0 = np.clip(np.floor(y).astype(np.int64), 0, V.shape[0] - 2); j0 = np.clip(np.floor(x).astype(np.int64), 0, V.shape[1] - 2)
    ty = np.clip(y - i0, 0, 1).astype(np.float32)[None]; tx = np.clip(x - j0, 0, 1).astype(np.float32)[None]
    Vs = V[:, :, cols]
    g = lambda a, b: Vs[a, b].T                                                                      # noqa: E731
    return ((1 - ty) * ((1 - tx) * g(i0, j0) + tx * g(i0, j0 + 1)) + ty * ((1 - tx) * g(i0 + 1, j0) + tx * g(i0 + 1, j0 + 1)))


def national_xy(site, nx: int, H: float, crs_wkt: str) -> tuple[np.ndarray, np.ndarray]:
    """Projected national-grid coordinates (m) of a block's 1 m cell centres, row 0 north, as the design grid lays them."""
    from pyproj import Transformer
    cc = -H + (np.arange(nx) + 0.5)
    GX, GY = np.meshgrid(cc, cc[::-1])
    a, B, ep, _ = site._enu
    X = a + GX.ravel()[:, None] * B[0] + GY.ravel()[:, None] * B[1]
    t1, _ = site._tf
    lon, lat, _, _ = t1.transform(X[:, 0], X[:, 1], X[:, 2], np.full(GX.size, ep))
    return Transformer.from_crs(6319, crs_wkt, always_xy=True).transform(lon, lat)


def grid_affine(x, y, nx):
    """The affine map (row, col) -> (x, y) of a block's cells (exact to millimetres over a block), and its inverse."""
    rc = np.array([[0, 0], [0, nx - 1], [nx - 1, 0]], np.float64)
    k = [0, nx - 1, (nx - 1) * nx]
    A = np.c_[rc, np.ones(3)]
    M = np.linalg.solve(A, np.c_[np.asarray(x)[k], np.asarray(y)[k]])          # [3, 2]: rows r, c, 1
    Minv = np.linalg.inv(M[:2])
    return M, (lambda X, Y: (np.c_[X, Y] - M[2]) @ Minv)


def run(plan: dict) -> dict:
    from . import job
    from .palette import _store, aquatic_species, cat_traits, traits
    job.fetch_mirror(plan.get("mirror", "soilplant/v1"), ("national",))
    work = Path(tempfile.mkdtemp(prefix="prob_pass."))
    owner, mode = plan["owner"], plan.get("mode", "owned")
    blocks = [Block(b, owner, mode, work) for b in plan["blocks"]]
    names = sorted({n for b in blocks for n in b.names})
    gi = {n: i for i, n in enumerate(names)}
    S = len(names)
    sp = {}
    for b in blocks:
        for e in json.loads((b.d / "species.json").read_text())["species"]:
            sp.setdefault(e["species"], (e.get("set") or "native", e["id"]))
    Sn = _store("native")
    _, t, crs = Sn._grid_raster("conus")
    for b in blocks:
        b.X, b.Y = national_xy(b.site, b.nx, b.site.half, crs)
        b.fr, b.fc = (t.f - np.asarray(b.Y)) / -t.e, (np.asarray(b.X) - t.c) / t.a
    # "isolate": each block on its own, as the producer's machine sees it (no other block's cells); else the site whole
    fields = []
    for grp in ([[b] for b in blocks] if plan.get("isolate") else [blocks]):
        # every 240 m cell's own reference of G, gathered over the blocks that hold its 1 m cells
        acc, cnt = {}, {}
        for b in grp:
            G = b.absolute()
            idx = np.array([gi[n] for n in b.names])
            for k, st in enumerate(b.stores):
                core = b.W[k] >= 0.999
                if not core.any():
                    continue
                v = np.full(S, -np.inf); v[idx] = lse(G[:, core], 1)
                c = np.zeros(S); c[idx] = core.sum()
                acc[st] = np.logaddexp(acc.get(st, np.full(S, -np.inf)), v); cnt[st] = cnt.get(st, np.zeros(S)) + c
            np.save(b.d / "G.npy", G.astype(np.float16))
        ref = {st: np.where(cnt[st] > 0, acc[st] - np.log(np.maximum(cnt[st], 1)), 0.0) for st in acc}
        # the national field: every 240 m cell under the blocks and RING more, gaps filled, for every species
        fr_all = np.concatenate([b.fr for b in grp]); fc_all = np.concatenate([b.fc for b in grp])
        r0, c0 = int(np.floor(fr_all.min() - 0.5)) - RING, int(np.floor(fc_all.min() - 0.5)) - RING
        r1, c1 = int(np.floor(fr_all.max() - 0.5)) + 2 + RING, int(np.floor(fc_all.max() - 0.5)) + 2 + RING
        RR, CC = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
        lam, clim, pen = intensity(RR.ravel(), CC.ravel(), [sp.get(n, ("native", -1)) for n in names])
        V0 = lam.reshape(RR.shape + (S,))
        V, nfill = fill_gaps(V0, clim.reshape(RR.shape))
        V = np.where(np.isfinite(V), V, FLOOR).astype(np.float32)
        for b in grp:
            b.ref, b.V, b.r0, b.c0 = ref, V, r0, c0
        fields.append({"cells": int(RR.size), "rows": [r0, r1], "cols": [c0, c1], "without_climate": int((~clim).sum()),
                       "filled": nfill, "has_penalty": pen,
                       "unserved_with_climate_share": round(float((~np.isfinite(V0[clim.reshape(RR.shape)])).mean()), 4)})
    field = fields[0] if len(fields) == 1 else {"isolated": len(fields), "cells": sum(f["cells"] for f in fields),
                                                "without_climate": sum(f["without_climate"] for f in fields)}
    print(json.dumps({"field": field}), flush=True)
    tidx, cid, T = traits()
    on = switches(plan)
    cat = cat_traits() if any(on[k] for k in ("salt", "flood", "seasonal")) else {}
    report = {n: gi[n] for n in plan.get("report", []) if n in gi}
    dst_root = plan["dst"].rstrip("/")

    def finish(b):
        idx = np.array([gi[n] for n in b.names])
        S_b = len(b.names)
        G = np.load(b.d / "G.npy").astype(np.float32)
        micro = G.copy()
        R = np.stack([b.ref.get(st, np.zeros(S))[idx] for st in b.stores]).astype(np.float32)
        for k in range(len(b.stores)):
            if (b.W[k] > 0).any():
                micro -= b.W[k][None] * R[k][:, None]
        del G
        lam = np.empty((S_b, b.n), np.float32)
        for a0 in range(0, S_b, 128):
            lam[a0:a0 + 128] = bilinear(b.V, b.fr - b.r0, b.fc - b.c0, idx[a0:a0 + 128])
        for f in ("comp_light.f16", "comp_logA.f16", "comp_logC.f16", "score_u8.bin", "plantable_u8.bin"):
            try:
                gs("cp", f"{b.src}/design/{f}", str(b.d / f))
            except subprocess.CalledProcessError:
                if f != "plantable_u8.bin":
                    raise
        gs("cp", f"{b.src}/soil/soil.simf", str(b.d / "soil.simf"))
        m = lambda f: np.fromfile(b.d / f, np.float16).reshape(S_b, b.n).astype(np.float32)       # noqa: E731
        soil = read_simf(b.d / "soil.simf")
        jj = [tidx.get(n) for n in b.names]
        tv = lambda k: np.array([T[j, cid.index(k)] if j is not None else np.nan for j in jj], np.float32)[:, None]  # noqa: E731
        moist = soil.get("moist", np.full(b.n, np.nan, np.float32))
        terms = {"ln_lambda": lam, "micro": micro, "light": m("comp_light.f16"),
                 "water": np.where(np.isfinite(moist)[None], window(2.5 * moist, tv("water_lo"), tv("water_hi"), 0.5),
                                   window(soil["drain"], tv("drainage_lo"), tv("drainage_hi"), 0.15)),
                 "soil": window(soil["ph"], tv("ph_lo"), tv("ph_hi"), 0.3)
                 + window(texture_class(soil["sand"], soil["clay"]), tv("texture_lo"), tv("texture_hi"), 0.3),
                 "logA": m("comp_logA.f16"), "logC": m("comp_logC.f16")}
        # the site conditions, read from the scene's own sims_v1 copy (as the moisture grid is); a scene without the
        # folder reads None and every array above is unchanged, so P is byte-identical to the pass before them
        from .site import soil_conditions
        cc_ = -b.site.half + (np.arange(b.nx) + 0.5)
        GX_, GY_ = np.meshgrid(cc_, cc_[::-1])
        cond = soil_conditions(b.site, GX_.ravel(), GY_.ravel()) if any(on.values()) else None
        catb = {k: np.array([P_[j] if j is not None else np.full(P_.shape[1], np.nan, np.float32) for j in jj],
                            np.float32) for k, P_ in cat.items()} if cond is not None else {}
        new, ccells = condition_terms(cond, moist, tv, catb, on)
        if "water" in new:
            terms["water"] = np.where(np.isfinite(new["water"]), new["water"], terms["water"])
        for k in ("salt", "flood"):
            if k in new:
                terms[k] = new[k]
        del new
        x = np.zeros((S_b, b.n), np.float32)
        for v in terms.values():
            x += v
        P = -np.expm1(-np.exp(np.minimum(x, 20.0)))
        # obligate aquatics (WCVP hydrophytes) only on open water or ground saturated 10 months a year or more
        aqs = aquatic_species() if on.get("aquatic") else frozenset()
        aq_rows = [i for i, n in enumerate(b.names) if n in aqs]
        allowed = None
        if aq_rows:
            from .site import moisture_codes
            allowed = aquatic_allowed(moisture_codes(b.site, GX_.ravel(), GY_.ravel()), cond)
            P = aquatic_zero(P, aq_rows, allowed)
        old = np.fromfile(b.d / "score_u8.bin", np.uint8).reshape(S_b, b.n)
        prob = np.where(old > 100, 255, np.round(100 * P)).astype(np.uint8)
        pf = b.d / "plantable_u8.bin"
        plantable = np.fromfile(pf, np.uint8).astype(bool) if pf.is_file() else np.ones(b.n, bool)
        conv = (P >= HAB_P) & (old <= 100)
        sp_, _ = b.site.scene_points()
        H = b.site.half
        ix = np.clip((sp_[:, 0] + H).astype(int), 0, b.nx - 1); iy = np.clip((H - sp_[:, 1]).astype(int), 0, b.nx - 1)
        cell = iy * b.nx + ix
        u, inv = np.unique(cell, return_inverse=True)
        pv = np.where(prob[:, u] > 100, 0, prob[:, u]).T.astype(np.int16)          # [cells with returns, S_b]
        k_ = min(TOP, S_b)
        tp = np.argpartition(-pv, k_ - 1, 1)[:, :k_]
        tp = np.take_along_axis(tp, np.argsort(-np.take_along_axis(pv, tp, 1), 1, kind="stable"), 1)
        tpv = np.take_along_axis(pv, tp, 1).astype(np.uint8)
        out = {"score_u8.bin": prob, "habitable_if_converted_u8.bin": conv.astype(np.uint8),
               "habitable_u8.bin": aquatic_gate(conv, plantable, aq_rows, allowed).astype(np.uint8),
               "points_top_id.u16": tp.astype(np.uint16)[inv], "points_top_u8.u8": tpv[inv]}
        dst = f"{dst_root}/{b.id}/design"
        for f, arr in out.items():
            arr.tofile(b.d / f"out_{f}")
            gs("cp", str(b.d / f"out_{f}"), f"{dst}/{f}")
        land_ = plantable & (old[0] <= 100)
        c0_ = (b.nx // 2) * b.nx + b.nx // 2                                       # the block's centre cell
        site_p = (np.percentile(np.where(prob[:, land_] > 100, 0, prob[:, land_]), 90, axis=1) if land_.any()
                  else np.zeros(S_b))
        nat_p = -np.expm1(-np.exp(np.minimum(lam[:, c0_].astype(np.float64), 20.0)))
        flag = {"score": "probability", "unit": "percent", "formula": "cloglog P = ln lambda + micro + light + water + soil + logA + logC",
                "habitable_p": HAB_P, "national": "bilinear ln lambda over the site's 240 m cells, climate gaps filled from neighbours",
                "producer": "deepearth_site.prob_pass",
                # per plant: site_p, P (percent) of the best tenth of the block's plantable ground (the card's site
                # value); national_p, the national field's p at the block's centre (0..1; the Distinctive ranking's tf)
                "species": {n: {"site_p": int(round(float(site_p[i]))), "national_p": round(float(nat_p[i]), 4)}
                            for i, n in enumerate(b.names)}}
        land_c = plantable & (old[0] <= 100)
        shares = {k: round(float(v[land_c].mean()), 3) if land_c.any() else None for k, v in ccells.items()}
        if ccells:                             # only where a term was made: a scene without the layers keeps its bytes
            flag["formula"] = flag["formula"] + "".join(f" + {k}" for k in ("salt", "flood") if k in ccells) + (
                "; water seasonal where the seasonal layers hold" if "seasonal" in ccells else "")
            flag["conditions"] = {"terms": sorted(ccells), "share_of_plantable": shares,
                                  "layers": cond["provenance"]["layers"], "made": cond["provenance"]["made"],
                                  "method": "log sum_k P(trait = k) f_k(cell): salt EC against salt tolerance (class tops "
                                            "2, 4, 8 dS/m), months saturated against flood tolerance (0.5, 1, 3 "
                                            "months, width by root depth), dry season against water_lo and wet "
                                            "season against water_hi"}
        if aq_rows:                            # only where a gated species is in the palette
            flag["aquatic"] = {"species": [b.names[i] for i in aq_rows],
                               "allowed_share_of_grid": round(float(allowed.mean()), 4),
                               "rule": "obligately aquatic (WCVP life form a hydrophyte: free-floating, submerged or "
                                       "rooted floating-leaved): P = 0 except on open water (moisture code 253, NWI "
                                       "subtidal) or where the water table is within 30 cm 10 months a year or more"}
        (b.d / "probability.json").write_text(json.dumps(flag, indent=1))
        gs("cp", str(b.d / "probability.json"), f"{dst}/probability.json")
        got = {}
        land = plantable & (old[0] <= 100)
        for n_ in report:
            if n_ in b.names:
                i = b.names.index(n_)
                got[n_] = {"score_median_before": int(np.median(old[i][land])) if land.any() else None,
                           "prob_median_after": int(np.median(prob[i][land])) if land.any() else None,
                           "prob_p90_after": int(np.percentile(prob[i][land], 90)) if land.any() else None,
                           "terms_median": {k: round(float(np.median(v[i][land])), 2) for k, v in terms.items()} if land.any() else None,
                           "hydrology_share": round(float(np.isfinite(moist)[land].mean()), 3) if land.any() else None}
                if ccells:
                    got[n_]["conditions_share"] = shares
        np.save(b.d / "prob.npy", prob); np.save(b.d / "old.npy", old); np.save(b.d / "land.npy", land)
        for f in b.d.iterdir():
            if f.suffix != ".npy" or f.name == "G.npy":
                f.unlink()
        b.idx = idx
        print(json.dumps({b.id: {"plantable_share": round(float(land.mean()), 3),
                                 **({"conditions_share": shares} if ccells else {}), **got}}), flush=True)
        return got

    with ThreadPoolExecutor(int(plan.get("workers", 3))) as ex:
        res = dict(zip([b.id for b in blocks], ex.map(finish, blocks)))
    edges = edge_jumps(blocks, gi, report)
    print(json.dumps({"edges": edges}), flush=True)
    return {"field": field, "blocks": res, "edges": edges}


def edge_jumps(blocks, gi, report) -> dict:
    """|dP| (and |d score| before) between each block's outer ring of cells and the cell of a neighbouring block at the
    same place (within 1 m), against the same block's next ring inward (the 1 m change inside a block): medians over
    cell pairs plantable on both sides, for each reported species and over every species either side gives P > 0."""
    aff = {b.id: grid_affine(b.X, b.Y, b.nx) for b in blocks}
    load = lambda b, f: np.load(b.d / f, mmap_mode="r")                                             # noqa: E731
    out = {"pairs": {}}
    agg = {"after": [], "before": [], "inner_after": [], "inner_before": []}
    agg_r = {n: {k: [] for k in agg} for n in report}
    for A in blocks:
        nx = A.nx
        ring = np.r_[np.arange(nx), (nx - 1) * nx + np.arange(nx), np.arange(nx) * nx, np.arange(nx) * nx + nx - 1]
        rr, cc = ring // nx, ring % nx
        inner = np.clip(rr + (rr == 0) - (rr == nx - 1), 0, nx - 1) * nx + np.clip(cc + (cc == 0) - (cc == nx - 1), 0, nx - 1)
        XA, YA = np.asarray(A.X)[ring], np.asarray(A.Y)[ring]
        pA, oA, lA = load(A, "prob.npy"), load(A, "old.npy"), load(A, "land.npy")
        for B in blocks:
            if B is A:
                continue
            q = aff[B.id][1](XA, YA)                     # in B's cells (1 m): abutting blocks' edge cells are 1 m apart
            qi = np.clip(np.round(q), 0, B.nx - 1).astype(np.int64)
            ok = np.hypot(*(q - qi).T) <= 1.25
            if not ok.any():
                continue
            ia, ib = ring[ok], qi[ok, 0] * B.nx + qi[ok, 1]
            pB, oB, lB = load(B, "prob.npy"), load(B, "old.npy"), load(B, "land.npy")
            both = lA[ia] & lB[ib]
            ia, ib, inn = ia[both], ib[both], inner[ok][both]
            if not len(ia):
                continue
            common = sorted(set(A.idx) & set(B.idx))
            sa = [list(A.idx).index(g) for g in common]; sb = [list(B.idx).index(g) for g in common]
            PA, PB, OA, OB = (np.asarray(pA[sa][:, ia], np.int16), np.asarray(pB[sb][:, ib], np.int16),
                              np.asarray(oA[sa][:, ia], np.int16), np.asarray(oB[sb][:, ib], np.int16))
            PI, OI = np.asarray(pA[sa][:, inn], np.int16), np.asarray(oA[sa][:, inn], np.int16)
            live = (PA > 0) | (PB > 0)
            rec = {"cells": int(len(ia)), "after": float(np.median(np.abs(PA - PB)[live])) if live.any() else 0.0,
                   "before": float(np.median(np.abs(OA - OB)))}
            agg["after"].append(np.abs(PA - PB)[live]); agg["before"].append(np.abs(OA - OB).ravel())
            agg["inner_after"].append(np.abs(PA - PI)[(PA > 0) | (PI > 0)]); agg["inner_before"].append(np.abs(OA - OI).ravel())
            for n_, g in report.items():
                if g in common:
                    j = common.index(g)
                    rec[n_] = {"after": float(np.median(np.abs(PA[j] - PB[j]))), "before": float(np.median(np.abs(OA[j] - OB[j]))),
                               "inner_after": float(np.median(np.abs(PA[j] - PI[j]))), "inner_before": float(np.median(np.abs(OA[j] - OI[j])))}
                    for k, v in (("after", PA[j] - PB[j]), ("before", OA[j] - OB[j]), ("inner_after", PA[j] - PI[j]), ("inner_before", OA[j] - OI[j])):
                        agg_r[n_][k].append(np.abs(v))
            out["pairs"][f"{A.id}|{B.id}"] = rec
    med = lambda L: (round(float(np.median(np.concatenate(L))), 2) if L and sum(len(x) for x in L) else None)   # noqa: E731
    out["all_species"] = {k: med(v) for k, v in agg.items()}
    out["species"] = {n: {k: med(v) for k, v in d.items()} for n, d in agg_r.items()}
    return out


PRODUCED = ("score_u8.bin", "habitable_u8.bin", "habitable_if_converted_u8.bin", "points_top_id.u16", "points_top_u8.u8",
            "probability.json")


def apply_local(owner: str, scene: str, site_json: Path, stores: list, out: Path, mirror: str) -> dict:
    """The producer's own step (deepearth_site.job, before publication): the scene just built made absolute in place,
    out/design's score, habitable and points_top replaced by P and probability.json added. The scene is on its own here,
    so its 240 m cells' references and the national field are its own (a strided order's site pass can join them)."""
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix="prob_local."))
    res = run({"owner": owner, "dst": str(tmp), "mirror": mirror, "workers": 1,
               "blocks": [{"id": "local", "scene": scene, "src": str(out), "stores": list(stores),
                           "site_json": str(site_json), "scene_ready": True}]})
    for f in PRODUCED:
        shutil.copy(tmp / "local" / "design" / f, out / "design" / f)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"field": res["field"]}


if __name__ == "__main__":
    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text()))))
