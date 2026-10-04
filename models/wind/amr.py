"""Adaptive cells for the wind solver: a 2:1-balanced octree over the production grid's own cells.

The production grid (`domain.Grid`, its dx, its levels, its floor and its top) is the finest level, so a layout whose
every leaf is a single cell IS that grid, and every scene field on a leaf is the volume- or area-weighted sum of the
grid's own. A leaf at level l is an aligned block of 2^l x 2^l x 2^l of the grid's cells (LMAX = 3: 1, 2, 4 and 8 m
cells on the 1 m production grid). The refinement criterion comes from the data: the grid's cells within `shell` of a
surface the survey defines (the ground and its cut cells, every face between a solid and the air: roofs and walls, and
the crowns' edges, where canopy drag meets open air) stay single cells; beyond it the cells double after `band` cells
of each size. No leaf coarser than a single cell touches a solid, so every coarse leaf is open air or canopy, and
blocks wholly inside the ground or a building carry nothing and are dropped: the buried cells cost nothing.

Balance: two leaves that share a face, an edge or a corner differ by at most one level (Popinet 2003, J Comput Phys
190: 572; Burstedde, Wilcox and Ghattas 2011, SIAM J Sci Comput 33: 1103). It is built bottom-up: a block may be one
leaf at level l only if the criterion allows it, all its children could be leaves at l - 1, and so could every child of
its 26 neighbors. A coarse level then never forces a finer one, so one pass over the levels is the balanced tree.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

LMAX = 3
"""The coarsest leaf is 2^LMAX cells a side: 8 m on the production grid's 1 m."""


def _min2(a: np.ndarray) -> np.ndarray:
    """The minimum over aligned 2 x 2 x 2 blocks."""
    nz, ny, nx = a.shape
    return a.reshape(nz // 2, 2, ny // 2, 2, nx // 2, 2).min(axis=(1, 3, 5))


def _max2(a: np.ndarray) -> np.ndarray:
    nz, ny, nx = a.shape
    return a.reshape(nz // 2, 2, ny // 2, 2, nx // 2, 2).max(axis=(1, 3, 5))


def _up(a: np.ndarray, f: int) -> np.ndarray:
    """Each cell repeated f times along every axis."""
    return a if f == 1 else a.repeat(f, 0).repeat(f, 1).repeat(f, 2)


def _min26(a: np.ndarray) -> np.ndarray:
    """The minimum over each cell's 3 x 3 x 3 neighborhood; beyond the domain counts as True (no constraint)."""
    from scipy.ndimage import minimum_filter
    return minimum_filter(a, size=3, mode="constant", cval=True)


def _touch6(a: np.ndarray) -> np.ndarray:
    """Cells with a face neighbor in `a`."""
    out = np.zeros_like(a)
    out[1:] |= a[:-1]
    out[:-1] |= a[1:]
    out[:, 1:] |= a[:, :-1]
    out[:, :-1] |= a[:, 1:]
    out[..., 1:] |= a[..., :-1]
    out[..., :-1] |= a[..., 1:]
    return out


@dataclass
class Criterion:
    """Where the cells may grow.

    Attributes:
        shell: Cells within this distance [grid cells] of a surface stay single cells inside the survey.
        shell_out: The same beyond the survey (and `margin` around it), where the only surface is the smoothed ground.
        band: Cells of each coarser size before the next doubling.
        margin: Grid cells around the survey's columns that count as inside it.
        crowns: "edge" refines the crowns' edges (canopy against open air); "volume" every canopy cell.
        levels: Heights above the ground [m] the product publishes; with `levels_band`, cells within that distance of
            one of them over the display disc stay single cells.
        levels_band: See `levels`; None leaves the published heights to the surfaces' shells.
        disc: Display radius [m] about the scene origin, for `levels` and `core_m`.
        core_m: With it, every cell within `disc` + `core_m` of the origin and at most `core_h` over the bare earth
            stays a single cell: the published product's own volume at the grid's own cells.
        core_level: The core's cells: 0 the grid's own, 1 twice them (a coarser product traded for cost).
        core_h: See `core_m`; None is the tallest published read (the highest level, or the tallest crown or roof in
            the core, where the per-point reads may stand) plus its 2 m clearance and 2 cells.
        lmax: The coarsest level.
    """

    shell: float = 3.0
    shell_out: float = 1.0
    band: float = 2.0
    margin: float = 8.0
    crowns: str = "edge"
    levels: Sequence[float] = (4.0, 5.0, 10.0, 25.0)
    levels_band: Optional[float] = None
    disc: Optional[float] = None
    core_m: Optional[float] = None
    core_h: Optional[float] = 30.0
    core_level: int = 0
    lmax: int = LMAX

    def label(self) -> str:
        s = f"shell {self.shell:g}/{self.shell_out:g} band {self.band:g} margin {self.margin:g} crowns {self.crowns}"
        return s + (f" levels±{self.levels_band:g}" if self.levels_band is not None else "")


def surfaces(scene, crowns: str = "edge") -> Dict[str, np.ndarray]:
    """The cells the criterion measures distance from, by kind: open cells beside a solid (`wall`, ground and buildings
    alike), the ground's cut cells (`cut`), and canopy cells beside open air (`crown`, or every canopy cell)."""
    solid = np.asarray(scene.solid, bool)
    fluid = ~solid
    wall = fluid & _touch6(solid)
    wall[0] |= fluid[0]                       # the floor is ground: its open cells carry the wall stress
    cut = fluid & (np.asarray(scene.cut_open) < 1) if getattr(scene, "cut_open", None) is not None else np.zeros_like(fluid)
    canopy = fluid & (np.asarray(scene.sink) > 0)
    crown = canopy if crowns == "volume" else canopy & _touch6(fluid & ~canopy)
    return {"wall": wall, "cut": cut, "crown": crown}


def required(scene, crit: Criterion, measured: Optional[np.ndarray] = None) -> np.ndarray:
    """The coarsest level each grid cell allows, (nz, ny, nx) int8, before balance.

    Open cells: 0 within `shell` (`shell_out` beyond the survey) of a surface, then 1 for `band` cells of 2, 2 for
    `band` cells of 4, and so on, distances measured in grid cells (an upper level thicker than dx is nearer than it
    counts, so it errs fine). Solids: 0 beside an open cell, the coarsest elsewhere (wholly solid blocks are dropped)."""
    from scipy.ndimage import binary_dilation, distance_transform_edt

    g = scene.grid
    surf = surfaces(scene, crit.crowns)
    s = surf["wall"] | surf["cut"] | surf["crown"]
    d = distance_transform_edt(~s).astype(np.float32)
    solid = np.asarray(scene.solid, bool)
    inside = np.ones((g.ny, g.nx), bool) if measured is None else np.asarray(measured, bool)
    if crit.margin > 0 and measured is not None:
        r = int(np.ceil(crit.margin))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        inside = binary_dilation(inside, structure=(yy ** 2 + xx ** 2) <= crit.margin ** 2)
    shell = np.where(inside, np.float32(crit.shell), np.float32(crit.shell_out))[None]
    lev = np.zeros(d.shape, np.int8)
    edge, width = shell, np.float32(crit.band * 2.0)
    for l in range(1, crit.lmax + 1):
        lev += (d > edge).astype(np.int8)
        edge = edge + width
        width = width * 2
    if crit.core_m is not None and crit.disc is not None and scene.terrain is not None:
        x, y = np.meshgrid(scene.origin[0] + g.xc, scene.origin[1] + g.yc)
        core = np.hypot(x, y) <= crit.disc + crit.core_m
        h = g.zc[:, None, None] - np.asarray(scene.terrain, float)[None]
        top = crit.core_h
        if top is None:      # per column: the highest level, or its own crown or roof (where a per-point read may stand)
            from scipy.ndimage import maximum_filter
            tall = np.asarray(scene.canopy_top if scene.canopy_top is not None else scene.top, float) - np.asarray(scene.terrain, float)
            top = (np.maximum(max(crit.levels), maximum_filter(tall, size=5)) + 2.0 + 2.0 * g.dx)[None]
        sel = core[None] & (h <= top)
        lev[sel] = np.minimum(lev[sel], np.int8(crit.core_level))
    if crit.levels_band is not None and crit.disc is not None and scene.terrain is not None:
        x, y = np.meshgrid(scene.origin[0] + g.xc, scene.origin[1] + g.yc)
        disc = np.hypot(x, y) <= crit.disc
        h = g.zc[:, None, None] - np.asarray(scene.terrain, float)[None]
        near = np.zeros(d.shape, bool)
        for z in crit.levels:
            near |= np.abs(h - z) <= crit.levels_band
        lev[near & disc[None]] = 0
    lev[surf["cut"]] = 0                                   # the ground's own cut cells, whatever the shell beyond the survey
    lev[surf["wall"]] = 0
    lev[solid] = crit.lmax
    lev[solid & _touch6(~solid)] = 0
    return lev


def balance(req: np.ndarray, lmax: int = LMAX) -> np.ndarray:
    """The leaf level of each grid cell, (nz, ny, nx) int8: the coarsest the requirement and 2:1 balance allow.

    ok[l][B]: block B of 2^l cells may be one leaf (level l or coarser above it). Bottom-up: B qualifies when every
    cell in it allows level l, every child qualified at l - 1, and every child of each of its 26 neighbors did."""
    shape = req.shape
    assert all(n % (1 << lmax) == 0 for n in shape), f"grid {shape} is not a multiple of {1 << lmax} cells"
    lo = req
    ok = [np.ones(shape, bool)]
    for l in range(1, lmax + 1):
        lo = _min2(lo)
        children = _min2(ok[-1])
        ok.append((lo >= l) & children & _min26(children))
    leaf = np.zeros(shape, np.int8)
    for l in range(1, lmax + 1):
        leaf += _up(ok[l], 1 << l).astype(np.int8)
    return leaf


def count(leaf: np.ndarray, solid: np.ndarray, lmax: int = LMAX, bricks: Sequence[int] = (4, 8)) -> Dict[str, object]:
    """Leaves per level that the solver keeps (any open cell; a single solid cell beside an open one), and the cells a
    brick layout of b^3 leaves would hold at each level."""
    solid = np.asarray(solid, bool)
    keep0 = (~solid) | _touch6(~solid)
    out, total = {"per_level": {}}, 0
    blocks_open = ~solid
    for l in range(lmax + 1):
        f = 1 << l
        if l:
            blocks_open = _max2(blocks_open)
        at = leaf == l
        if l == 0:
            n = int((at & keep0).sum())
            mask = at & keep0
        else:
            lead = at[::f, ::f, ::f]                         # the leaf's own anchor cell
            mask = lead & blocks_open
            n = int(mask.sum())
        out["per_level"][f"{f} m"] = n
        total += n
        for b in bricks:
            nz, ny, nx = mask.shape
            if nz % b or ny % b or nx % b:
                pad = [(0, (-s) % b) for s in mask.shape]
                m = np.pad(mask, pad)
            else:
                m = mask
            nz, ny, nx = m.shape
            nb = int(m.reshape(nz // b, b, ny // b, b, nx // b, b).any(axis=(1, 3, 5)).sum())
            out.setdefault(f"brick{b}_cells", 0)
            out[f"brick{b}_cells"] += nb * b ** 3
    out["cells"] = total
    out["dense"] = int(leaf.size)
    out["dense_open"] = int((~solid).sum())
    out["ratio"] = round(leaf.size / max(total, 1), 2)
    out["ratio_open"] = round(out["dense_open"] / max(total, 1), 2)
    for b in bricks:
        out[f"brick{b}_ratio"] = round(leaf.size / max(out[f"brick{b}_cells"], 1), 2)
    return out


def check_balance(leaf: np.ndarray) -> int:
    """Pairs of 26-neighboring grid cells whose leaves differ by more than one level (0 for a balanced tree)."""
    from scipy.ndimage import maximum_filter, minimum_filter
    hi = maximum_filter(leaf, size=3, mode="nearest")
    lo = minimum_filter(leaf, size=3, mode="nearest")
    return int(((hi - leaf) > 1).sum() + ((leaf - lo) > 1).sum())


# ── The layout: leaves, their ids and the faces between them ────────────────────────────────────────────────────────


def _spread_bits(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint64)
    out = np.zeros_like(v)
    for b in range(12):
        out |= ((v >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b)
    return out


def morton(k: np.ndarray, j: np.ndarray, i: np.ndarray) -> np.ndarray:
    """The Z-order key of each (k, j, i): leaves near in space lie near in memory, so a gather reads cached lines."""
    return _spread_bits(i) | (_spread_bits(j) << np.uint64(1)) | (_spread_bits(k) << np.uint64(2))


@dataclass
class Layout:
    """The kept leaves, in Z-order.

    Attributes:
        shape: The grid's (nz, ny, nx).
        level: (N,) each leaf's level; it spans 2^level grid cells a side.
        anchor: (N, 3) its lowest grid cell (k, j, i).
        owner: (nz, ny, nx) int32, the leaf each grid cell lies in, -1 where nothing is kept (wholly solid blocks, and
            solid cells with no open face neighbor).
    """

    shape: tuple
    level: np.ndarray
    anchor: np.ndarray
    owner: np.ndarray

    @property
    def n(self) -> int:
        return int(self.level.size)

    @property
    def size(self) -> np.ndarray:
        return (1 << self.level.astype(np.int64))


def layout(leaf: np.ndarray, solid: np.ndarray, lmax: int = LMAX) -> Layout:
    """The leaves `balance` chose that the solver keeps, Z-ordered, with each grid cell's owner."""
    solid = np.asarray(solid, bool)
    keep0 = (~solid) | _touch6(~solid)
    open_l = ~solid
    parts = []
    for l in range(lmax + 1):
        f = 1 << l
        if l:
            open_l = _max2(open_l)
        if l == 0:
            k, j, i = np.nonzero((leaf == 0) & keep0)
        else:
            kk, jj, ii = np.nonzero((leaf[::f, ::f, ::f] == l) & open_l)
            k, j, i = kk * f, jj * f, ii * f
        parts.append((np.full(k.size, l, np.int8), np.stack([k, j, i], 1).astype(np.int32)))
    level = np.concatenate([p[0] for p in parts])
    anchor = np.concatenate([p[1] for p in parts])
    order = np.argsort(morton(anchor[:, 0], anchor[:, 1], anchor[:, 2]), kind="stable")
    level, anchor = level[order], anchor[order]
    ids = np.empty(level.size, np.int32)
    ids[order] = np.arange(level.size, dtype=np.int32)          # the id each part's entry now carries
    owner = np.full(leaf.shape, -1, np.int32)
    start = 0
    for l, (lv, an) in enumerate(parts):
        f, n = 1 << l, lv.size
        mine = ids[start:start + n]
        start += n
        if l == 0:
            owner[an[:, 0], an[:, 1], an[:, 2]] = mine
        else:
            blk = np.full(tuple(s // f for s in leaf.shape), -1, np.int32)
            blk[an[:, 0] // f, an[:, 1] // f, an[:, 2] // f] = mine
            owner = np.where(leaf == l, _up(blk, f), owner)
    return Layout(shape=tuple(leaf.shape), level=level, anchor=anchor, owner=owner)


def uniform_layout(shape: tuple, solid: Optional[np.ndarray] = None) -> Layout:
    """Every grid cell its own leaf (the dense grid itself), in the dense grid's own order, solids included: the layout
    the operators are tested on against the dense solver."""
    nz, ny, nx = shape
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    owner = np.arange(nz * ny * nx, dtype=np.int32).reshape(shape)
    return Layout(shape=tuple(shape), level=np.zeros(owner.size, np.int8),
                  anchor=np.stack([k.ravel(), j.ravel(), i.ravel()], 1).astype(np.int32), owner=owner)


@dataclass
class FaceSet:
    """The faces between leaves along one axis, low leaf `a` to high leaf `b`, sorted by (a, b), with the grid faces
    each is the union of: `inv` maps every grid face (`pos`, the low grid cell's (k, j, i)) to its leaf face."""

    axis: int
    a: "torch.Tensor"
    b: "torch.Tensor"
    inv: "torch.Tensor"
    pos: "torch.Tensor"

    @property
    def n(self) -> int:
        return int(self.a.numel())

    def sum(self, values: "torch.Tensor") -> "torch.Tensor":
        """Per leaf face, the sum of a per-grid-face quantity given at `pos`."""
        import torch
        out = torch.zeros(self.n, dtype=values.dtype, device=values.device)
        return out.index_add_(0, self.inv, values)


def face_sets(owner, device=None) -> List[FaceSet]:
    """The faces between distinct kept leaves along z, y and x (axes 0, 1, 2 of the grid)."""
    import torch
    o = torch.as_tensor(owner, device=device)
    n = int(o.max()) + 1
    out = []
    for axis in range(3):
        m_ = o.shape[axis]
        lo, hi = o.narrow(axis, 0, m_ - 1), o.narrow(axis, 1, m_ - 1)
        m = (lo != hi) & (lo >= 0) & (hi >= 0)
        pos = m.nonzero()
        a, b = lo[m].long(), hi[m].long()
        del m
        key, inv = torch.unique(a * n + b, return_inverse=True)
        out.append(FaceSet(axis=axis, a=key // n, b=key % n, inv=inv, pos=pos))
        del lo, hi, a, b
    return out


# ── A-posteriori estimators on a solved field (to check the geometric criterion against) ─────────────────────────────


def wavelet_required(q, fluid, zeta, lmax: int = LMAX):
    """The coarsest level each grid cell allows by the wavelet estimate of `q` (C, nz, ny, nx) [torch], as Basilisk's
    adapt_wavelet: at each level, restrict by the volume average, prolong back by cell-centerd trilinear interpolation
    from the 2^3 nearest coarse cells, and split every coarse block where a child's detail exceeds `zeta` (per component,
    (C,) or a scalar) (van Hooft, Popinet, van Heerwaarden, van der Linden, de Roode and van de Wiel 2018, Boundary-Layer
    Meteorol 167: 421, section 2.2; Popinet 2015, J Comput Phys 302: 336). Blocks touching a solid, or beside one at
    their level, are left to the geometric criterion (their detail is the wall's, not the flow's). Returns int8."""
    import torch
    import torch.nn.functional as F
    zeta = torch.as_tensor(zeta, dtype=q.dtype, device=q.device).reshape(-1, 1, 1, 1)
    req = torch.full(q.shape[1:], lmax, dtype=torch.int8, device=q.device)
    f, m = q, fluid.to(q.dtype)[None]
    for l in range(1, lmax + 1):
        c = F.avg_pool3d(f[None], 2)[0]
        mc = -F.max_pool3d(-m[None], 2)[0]                          # 1 where the coarse block is all open
        safe = -F.max_pool3d(-mc[None], 3, 1, 1)[0]                  # and every coarse neighbor is too
        p = F.interpolate(c[None], scale_factor=2, mode="trilinear", align_corners=False)[0]
        w = ((f - p).abs() / zeta).amax(0, keepdim=True)
        bad = (F.max_pool3d(w[None], 2)[0] > 1) & (safe > 0)
        up = bad[0].repeat_interleave(1 << l, 0).repeat_interleave(1 << l, 1).repeat_interleave(1 << l, 2)
        req = torch.where(up, torch.minimum(req, torch.tensor(l - 1, dtype=torch.int8, device=q.device)), req)
        f, m = c, mc
    return req


def gradient_required(q, spacing, threshold, lmax: int = LMAX):
    """The coarsest level each cell allows by a gradient tagger, as AMR-Wind's GradientMagRefinement and
    VorticityMagRefinement (Sharma et al. 2024, Wind Energy 27: 225; amr-wind/utilities/tagging): a cell of size
    2^l needs splitting while |grad q| 2^l exceeds `threshold`, i.e. while q changes by more than the threshold across
    one cell. `q` (nz, ny, nx), `spacing` the grid's (zc, yc, xc) coordinates. Returns int8."""
    import torch
    g = torch.gradient(q, spacing=spacing)
    mag = torch.sqrt(sum(gi * gi for gi in g))
    lev = torch.floor(torch.log2(threshold / mag.clamp(min=1e-30))).clamp(0, lmax)
    return lev.to(torch.int8)


def tiled(lay: Layout, tile: tuple = (4, 4, 4)) -> Layout:
    """`lay` reordered for dense-speed kernels: the single cells in tiles of `tile` grid cells (tiles in Z-order, each
    tile's cells in (k, j, i) order, a tile's slots that hold no kept cell kept as phantom leaves: no volume, no face,
    solid), then the coarse leaves in Z-order. A kernel then reads a tile as a dense block and its halo through the
    tile neighbor table (`Layout.tile_nb`), as NanoVDB's leaf nodes (Museth 2021) and AMReX's boxes do."""
    tz, ty, tx = tile
    nz, ny, nx = lay.shape
    assert nz % tz == 0 and ny % ty == 0 and nx % tx == 0
    single = lay.level == 0
    an = lay.anchor
    tk = np.stack([an[single, 0] // tz, an[single, 1] // ty, an[single, 2] // tx], 1)
    tkey = (tk[:, 0].astype(np.int64) * (ny // ty) + tk[:, 1]) * (nx // tx) + tk[:, 2]
    tiles = np.unique(tkey)
    tc = np.stack([tiles // ((ny // ty) * (nx // tx)), (tiles // (nx // tx)) % (ny // ty), tiles % (nx // tx)], 1)
    order = np.argsort(morton(tc[:, 0], tc[:, 1], tc[:, 2]), kind="stable")
    tiles, tc = tiles[order], tc[order]
    T, per = tiles.size, tz * ty * tx
    sorted_keys = np.sort(tiles)
    zrank = np.empty(T, np.int64)
    zrank[np.searchsorted(sorted_keys, tiles)] = np.arange(T)
    tile_rank = zrank[np.searchsorted(sorted_keys, tkey)]
    local = ((an[single, 0] % tz) * ty + an[single, 1] % ty) * tx + an[single, 2] % tx
    new_single = tile_rank * per + local
    coarse = np.nonzero(~single)[0]
    nn = T * per + coarse.size
    new_id = np.empty(lay.n, np.int64)
    new_id[np.nonzero(single)[0]] = new_single
    new_id[coarse] = T * per + np.arange(coarse.size)
    level = np.zeros(nn, np.int8)
    level[new_id] = lay.level
    anchor = np.zeros((nn, 3), np.int32)
    s = np.arange(T * per)
    lk = s % per
    tr = s // per
    anchor[:T * per, 0] = tc[tr, 0] * tz + lk // (ty * tx)
    anchor[:T * per, 1] = tc[tr, 1] * ty + (lk // tx) % ty
    anchor[:T * per, 2] = tc[tr, 2] * tx + lk % tx
    anchor[new_id] = lay.anchor
    phantom = np.ones(nn, bool)
    phantom[new_id] = False
    owner = np.where(lay.owner >= 0, new_id[np.maximum(lay.owner, 0)], -1).astype(np.int32)
    out = Layout(shape=lay.shape, level=level, anchor=anchor, owner=owner)
    out.phantom = phantom
    out.tile = tuple(tile)
    out.n_tiles = T
    # each tile's six neighbors (-z, +z, -y, +y, -x, +x), -1 where none
    pos = {tuple(c): i for i, c in enumerate(tc.tolist())}
    nb = np.full((T, 6), -1, np.int32)
    for d, (dz, dy, dx) in enumerate(((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))):
        for i, (a, b, c) in enumerate(tc.tolist()):
            nb[i, d] = pos.get((a + dz, b + dy, c + dx), -1)
    out.tile_nb = nb
    return out
