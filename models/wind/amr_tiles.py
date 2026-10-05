"""Dense-speed kernels for the single cells of a tiled layout (`amr.tiled`): the cells sit in 4^3 tiles in memory, a
tile's six neighbors in a table, so a kernel finds every neighbor by arithmetic and reads no index per cell, as the
dense grid's fused stencil does; faces that touch a coarse leaf, and the coarse leaves' own rows, stay a small CSR.

The storage is NanoVDB's leaf-node layout (Museth 2021, "NanoVDB: A GPU-friendly and portable VDB data structure for
real-time rendering and simulation", ACM SIGGRAPH 2021 Talks; 8^3 leaves there, 4^3 here because the 1 m shell is
thin) and AMReX's tile-by-tile kernels (Zhang et al. 2019, JOSS 4: 1370, ParallelFor over each box's cells); the kernel
is Triton's (Tillet, Kung and Cox 2019, MAPL), one program to a few tiles. Every kernel is checked against the CSR
product of the same operator (`tests/test_amr.py`)."""

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:          # a CPU-only environment: the CSR path serves
    triton = None

DIRS = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
"""The six neighbor directions in (z, y, x), the order of the tile table and of the coefficient planes."""


if triton is not None:
    @triton.jit
    def _nb(t, loc, z, y, x, valid, nb_ptr, d: tl.constexpr, TZ: tl.constexpr, TY: tl.constexpr,
            TX: tl.constexpr):
        """The neighbor's slot in direction d, and whether it exists."""
        PER: tl.constexpr = TZ * TY * TX
        if d == 0:
            inside = z > 0
            step = -TY * TX
            wrap = (TZ - 1) * TY * TX
        elif d == 1:
            inside = z < TZ - 1
            step = TY * TX
            wrap = -(TZ - 1) * TY * TX
        elif d == 2:
            inside = y > 0
            step = -TX
            wrap = (TY - 1) * TX
        elif d == 3:
            inside = y < TY - 1
            step = TX
            wrap = -(TY - 1) * TX
        elif d == 4:
            inside = x > 0
            step = -1
            wrap = TX - 1
        else:
            inside = x < TX - 1
            step = 1
            wrap = -(TX - 1)
        nbt = tl.load(nb_ptr + t * 6 + d, mask=valid & (inside == 0), other=-1)
        own = t * PER + loc
        src = tl.where(inside, own + step, nbt * PER + loc + wrap)
        ok = valid & (inside | (nbt >= 0))
        return src, ok

    @triton.jit
    def _apply_kernel(x_ptr, x_sch, diag_ptr, c_ptr, nb_ptr, map_ptr, y_ptr, y_srow, y_sch, b_ptr, b_sch, safe_ptr,
                      color_ptr, out_ptr, out_sch, n_tiles, n_slots, color, g_div, d_gs, c_gs, s_gs, MODE: tl.constexpr,
                      BT: tl.constexpr, TZ: tl.constexpr, TY: tl.constexpr, TX: tl.constexpr):
        """Channel program_id(1), of operator group ch // g_div (its diagonal, safe diagonal and coefficients d_gs and c_gs
        apart). MODE 0: out = M x on the tile slots (diag x - sum_d c_d x_d + the rest's row, read
        through `map` where a slot has one). MODE 1: one color of Gauss-Seidel, out = x + where(color, (b - M x) /
        safe, 0)."""
        PER: tl.constexpr = TZ * TY * TX
        pid = tl.program_id(0)
        ch = tl.program_id(1)
        offs = tl.arange(0, BT * PER)
        t = pid * BT + offs // PER
        loc = offs % PER
        valid = t < n_tiles
        z = loc // (TY * TX)
        y = (loc // TX) % TY
        x = loc % TX
        idx = t * PER + loc
        xb = x_ptr + ch * x_sch
        grp = ch // g_div
        xv = tl.load(xb + idx, mask=valid, other=0.0)
        acc = tl.load(diag_ptr + grp * d_gs + idx, mask=valid, other=0.0) * xv
        row = tl.load(map_ptr + idx, mask=valid, other=-1)
        acc = acc + tl.load(y_ptr + row * y_srow + ch * y_sch, mask=valid & (row >= 0), other=0.0)
        for d in tl.static_range(6):
            src, ok = _nb(t, loc, z, y, x, valid, nb_ptr, d, TZ, TY, TX)
            v = tl.load(xb + src, mask=ok, other=0.0)
            c = tl.load(c_ptr + grp * c_gs + d * n_slots + idx, mask=valid, other=0.0)
            acc = acc - c * v
        if MODE == 0:
            tl.store(out_ptr + ch * out_sch + idx, acc, mask=valid)
        else:
            bv = tl.load(b_ptr + ch * b_sch + idx, mask=valid, other=0.0)
            sv = tl.load(safe_ptr + grp * s_gs + idx, mask=valid, other=1.0)
            cv = tl.load(color_ptr + idx, mask=valid, other=-1)
            new = tl.where(cv == color, xv + (bv - acc) / sv, xv)
            tl.store(out_ptr + ch * out_sch + idx, new, mask=valid)


class TileOp:
    """An operator's level-0 rows split for the tile kernels: per tile slot its diagonal and six neighbor
    coefficients (positive: the row reads diag x - sum c x_nb), and a CSR `rest` over the rows that have one: every
    face that touches a coarse leaf (its tile rows first) and the coarse leaves' whole rows (last, in order). Built
    from a `amr_graph.GraphOp` on a tiled layout; every channel of a (C, n) vector in one launch."""

    BT = 4      # tiles a program: 256 lanes

    def __init__(self, plan: "TilePlan", op):
        self.plan, self.op = plan, op
        n_slots = plan.n_slots
        dt = op.diag.dtype
        ab, ba = op.offdiag()                         # matrix entries (negative conductances), per level-0 face
        c = torch.zeros(6 * n_slots, dtype=dt, device=ab.device)
        c[plan.pos_a] = -ab[plan.held]
        c[plan.pos_b] = -ba[plan.held]
        self.c = c
        self.diag = op.diag[:n_slots].contiguous()
        self.rest = plan.rest_matrix(ab, ba, op.diag)
        self.rest_rows = None
        if self.rest.is_cuda:
            import amr_csr
            self.rest_rows = amr_csr.Rows(self.rest)
            self.rest = None                    # its int32 copy is what the kernel reads
        self.symmetric = op.symmetric
        self.n = op.diag.numel()

    def _rest(self, x: torch.Tensor) -> torch.Tensor:
        """The rest's rows times x: (m, C)."""
        if self.rest_rows is not None:          # the row kernel: no cuSPARSE workspace (amr_csr.Rows)
            return self.rest_rows.apply(x).T
        return self.rest @ x.T.contiguous()

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """(C, n) -> (C, n)."""
        x = x.contiguous()
        out = torch.empty_like(x)
        y = self._rest(x)
        p = self.plan
        self._launch(x, y, None, None, None, 0, out, 0)
        out[:, p.n_slots:] = y[p.m_tile:].T
        return out

    def relax(self, safe: torch.Tensor, colors: torch.Tensor, color: int, b: torch.Tensor, x: torch.Tensor):
        """One color of the Gauss-Seidel sweep on (C, n) vectors: the tile rows in one kernel, the coarse rows from
        the rest."""
        x, b = x.contiguous(), b.contiguous()
        p = self.plan
        ns = p.n_slots
        out = torch.empty_like(x)
        y = self._rest(x)
        if color in p.tile_colors(colors):
            self._launch(x, y, b, safe, colors, 1, out, color)
        else:                           # a color only coarse leaves carry (2 and 8 m): the tiles keep their values
            out[:, :ns] = x[:, :ns]
        cm = colors[ns:] == color
        out[:, ns:] = torch.where(cm, x[:, ns:] + (b[:, ns:] - y[p.m_tile:].T) / safe[ns:], x[:, ns:])
        return out

    def _launch(self, x, y, b, safe, colors, mode, out, color, map_=None, g_div=1, groups=False):
        """One launch over every channel: `groups` reads each channel's operator (group ch // g_div) from the
        coefficient, diagonal and safe-diagonal planes stacked by group; else one operator serves all."""
        p = self.plan
        ns = p.n_slots
        grid = (triton.cdiv(p.n_tiles, self.BT), x.shape[0])
        safe_ = safe if safe is not None else self.diag
        _apply_kernel[grid](x, x.stride(0), self.diag, self.c, p.nb, p.rest_map if map_ is None else map_, y,
                            y.stride(0), y.stride(1), b if b is not None else x, (b if b is not None else x).stride(0),
                            safe_, colors if colors is not None else p.dummy_color, out, out.stride(0), p.n_tiles, ns,
                            color, g_div, ns if groups else 0, 6 * ns if groups else 0,
                            safe_.shape[-1] if (groups and safe_.dim() > 1) else 0,
                            MODE=mode, BT=self.BT, TZ=p.tile[0], TY=p.tile[1], TX=p.tile[2])


class GroupTileOp(TileOp):
    """`TileOp` for G operators at once (`amr_graph.GroupOp`: a heading's momentum or k each), on (G R, n) vectors,
    channel ch reading operator ch // R. The rest (faces touching a coarse leaf, the coarse rows) by gather and scatter
    with each group's coefficients, read by the kernel as a whole vector."""

    def __init__(self, plan: "TilePlan", op):
        self.plan, self.op = plan, op
        ns = plan.n_slots
        ab, ba = op.offdiag()                         # (G, F)
        G = ab.shape[0]
        c = torch.zeros(G, 6 * ns, dtype=ab.dtype, device=ab.device)
        c[:, plan.pos_a] = -ab[:, plan.held]
        c[:, plan.pos_b] = -ba[:, plan.held]
        self.c = c.contiguous()
        self.diag = op.diag[:, :ns].contiguous()
        self.r_ab, self.r_ba = ab[:, plan.rest_faces].contiguous(), ba[:, plan.rest_faces].contiguous()
        self.r_diag = op.diag[:, ns:].contiguous()
        self.symmetric = op.symmetric
        self.n, self.G, self.R = op.diag.shape[-1], G, op.R

    def _rest_sub(self, x: torch.Tensor) -> torch.Tensor:
        """(C, m): the rest's rows times x, on the rest's own rows alone (tile rows with a coarse neighbor, then every
        coarse row), so nothing the size of the whole vector is written."""
        p, G = self.plan, self.G
        n = x.shape[-1]
        xg = x.reshape(G, -1, n)
        y = torch.zeros(G, xg.shape[1], p.m, dtype=x.dtype, device=x.device)
        y.index_add_(-1, p.sub_a, self.r_ab[:, None, :] * xg[..., p.rest_b])
        y.index_add_(-1, p.sub_b, self.r_ba[:, None, :] * xg[..., p.rest_a])
        y[..., p.m_tile:] += self.r_diag[:, None, :] * xg[..., p.n_slots:]
        return y.reshape(x.shape[0], p.m)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        y = self._rest_sub(x)
        p = self.plan
        self._launch(x, y.T, None, None, None, 0, out, 0, g_div=self.R, groups=True)
        out[:, p.n_slots:] = y[:, p.m_tile:]
        return out

    def relax(self, safe: torch.Tensor, colors: torch.Tensor, color: int, b: torch.Tensor, x: torch.Tensor):
        """`safe` (G, n): each group's safe diagonal."""
        x, b = x.contiguous(), b.contiguous()
        p = self.plan
        ns = p.n_slots
        out = torch.empty_like(x)
        y = self._rest_sub(x)
        if color in p.tile_colors(colors):
            self._launch(x, y.T, b, safe.contiguous(), colors, 1, out, color, g_div=self.R, groups=True)
        else:
            out[:, :ns] = x[:, :ns]
        cm = colors[ns:] == color
        sg = safe[:, ns:].repeat_interleave(self.R, dim=0)
        out[:, ns:] = torch.where(cm, x[:, ns:] + (b[:, ns:] - y[:, p.m_tile:]) / sg, x[:, ns:])
        return out


class TilePlan:
    """What a tiled layout's level 0 fixes once: which faces the tiles hold (both leaves single cells; the kernel finds
    the neighbor by arithmetic) and where their coefficients go, and the CSR pattern of everything else: the rows with
    a face to a coarse leaf (`rest_map` sends a tile slot to its row) and every coarse leaf's row."""

    def __init__(self, lay, lv0):
        dev = lv0.a.device
        self.tile = lay.tile
        self.n_tiles = lay.n_tiles
        per = self.tile[0] * self.tile[1] * self.tile[2]
        self.n_slots = self.n_tiles * per
        self.nb = torch.as_tensor(lay.tile_nb, device=dev).contiguous()
        self.dummy_color = torch.zeros(1, dtype=torch.int8, device=dev)
        a, b, axis = lv0.a.long(), lv0.b.long(), lv0.axis.long()
        ns = self.n_slots
        held = (a < ns) & (b < ns)
        self.held = held
        # a's neighbor in +axis is b (direction 2 axis + 1), b's in -axis is a (direction 2 axis)
        self.pos_a = ((2 * axis[held] + 1) * ns + a[held]).int()
        self.pos_b = ((2 * axis[held]) * ns + b[held]).int()
        # the rest: faces not held, both directions, and the coarse rows' diagonals, on the rows that carry one
        n = lv0.n
        ra, rb = a[~held], b[~held]
        coarse = torch.arange(ns, n, device=dev)
        rows = torch.cat([ra, rb, coarse])
        cols = torch.cat([rb, ra, coarse])
        tile_rows = torch.unique(rows[rows < ns])
        self.m_tile = int(tile_rows.numel())
        sub = torch.full((n,), -1, dtype=torch.int64, device=dev)
        sub[tile_rows] = torch.arange(self.m_tile, device=dev)
        sub[coarse] = self.m_tile + torch.arange(coarse.numel(), device=dev)
        self.rest_map = sub[:ns].int().contiguous()
        self.sub_a, self.sub_b = sub[ra], sub[rb]               # the rest faces' rows in the rest's own numbering
        m = self.m_tile + coarse.numel()
        srows = sub[rows]
        order = torch.argsort(srows * n + cols)
        self.rest_perm = order.int()
        srows = srows[order]
        self.rest_col = cols[order].to(torch.int32)
        crow = torch.zeros(m + 1, dtype=torch.int64, device=dev)
        crow[1:] = torch.cumsum(torch.bincount(srows, minlength=m), 0)
        self.rest_crow = crow.to(torch.int32)
        self.n, self.m = n, m
        self.coarse = coarse
        self.rest_faces = (~held).nonzero()[:, 0]               # for operators in groups (GroupTileOp)
        self.rest_a, self.rest_b = ra, rb
        self.identity_map = torch.arange(ns, device=dev, dtype=torch.int32)

    def rest_matrix(self, ab, ba, diag):
        import warnings
        held = self.held
        vals = torch.cat([ab[~held], ba[~held], diag[self.coarse]])[self.rest_perm]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return torch.sparse_csr_tensor(self.rest_crow, self.rest_col, vals, size=(self.m, self.n),
                                           check_invariants=False)

    def tile_colors(self, colors: torch.Tensor) -> set:
        """The colors the tile slots carry (single cells: 0 and 1), cached per color array."""
        key = (colors.data_ptr(), colors.numel())
        cache = self.__dict__.setdefault("_colors", {})
        if key not in cache:
            cache[key] = set(torch.unique(colors[:self.n_slots]).tolist())
        return cache[key]
