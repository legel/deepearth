"""Fused face kernels for the octree's stencil phases: convection, diffusion and the velocity gradients, one Triton
program a block of faces, every face's arithmetic in registers and its two leaves' sums by atomic adds, where the eager
path (`amr_model`) runs each of those terms as its own whole-array kernel (the roofline: those
phases ran 18 to 127 times their least bytes on Fisher's grid).

Each kernel computes the eager path's arithmetic in the same order, face by face; only the order of the sums into a
leaf differs (index_add_ on a GPU is an atomic sum too), so the two agree to round-off (tests/test_amr.py). Leading
dimensions (headings, components) are channels: program_id(1), with each array's channel stride."""

from typing import Dict, List, Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:          # a CPU-only environment: the eager path serves
    triton = None

Tensor = torch.Tensor
BLOCK = 1024


def available(q: Tensor) -> bool:
    return triton is not None and q.is_cuda


if triton is not None:
    @triton.jit
    def _side_means_kernel(q_ptr, q_ch, a_ptr, b_ptr, wl_ptr, wh_ptr, lo_ptr, hi_ptr, o_ch, F, BLOCK: tl.constexpr):
        """lo[b] += w_low q[a], hi[a] += w_high q[b]: each leaf's low and high sides' area-weighted means of its
        neighbors' values (`Geo.low_mean`, `high_mean` of q[a] and q[b]), boundary faces apart."""
        pid = tl.program_id(0)
        ch = tl.program_id(1).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < F
        a = tl.load(a_ptr + offs, mask=m, other=0)
        b = tl.load(b_ptr + offs, mask=m, other=0)
        qa = tl.load(q_ptr + ch * q_ch + a, mask=m, other=0.0)
        qb = tl.load(q_ptr + ch * q_ch + b, mask=m, other=0.0)
        wl = tl.load(wl_ptr + offs, mask=m, other=0.0)
        wh = tl.load(wh_ptr + offs, mask=m, other=0.0)
        tl.atomic_add(lo_ptr + ch * o_ch + b, wl * qa, mask=m)
        tl.atomic_add(hi_ptr + ch * o_ch + a, wh * qb, mask=m)

    @triton.jit
    def _face_pair(q_ptr, q_ch, ch, a, b, offs, m, d_ptr, d_ch, side_ptr, HAS_D: tl.constexpr):
        """q at each face's two leaves, the coarser corrected to the finer's line (`Stencil.face_values`)."""
        qa = tl.load(q_ptr + ch * q_ch + a, mask=m, other=0.0)
        qb = tl.load(q_ptr + ch * q_ch + b, mask=m, other=0.0)
        if HAS_D:
            d = tl.load(d_ptr + ch * d_ch + offs, mask=m, other=0.0)
            side = tl.load(side_ptr + offs, mask=m, other=0)
            qa = qa + tl.where(side == 1, d, 0.0)
            qb = qb + tl.where(side == 2, d, 0.0)
        return qa, qb

    @triton.jit
    def _convect_kernel(q_ptr, q_ch, a_ptr, b_ptr, f_ptr, f_ch, d_ptr, d_ch, side_ptr, lo_ptr, hi_ptr, out_ptr, F,
                        HAS_D: tl.constexpr, BLOCK: tl.constexpr):
        """out[a] += F q_f, out[b] -= F q_f, q_f the upwind van Leer MUSCL value (`solver._muscl`), the far value the
        upwind leaf's own upwind side mean (lo or hi, `_side_means_kernel`)."""
        pid = tl.program_id(0)
        ch = tl.program_id(1).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < F
        a = tl.load(a_ptr + offs, mask=m, other=0)
        b = tl.load(b_ptr + offs, mask=m, other=0)
        qa, qb = _face_pair(q_ptr, q_ch, ch, a, b, offs, m, d_ptr, d_ch, side_ptr, HAS_D)
        f = tl.load(f_ptr + ch * f_ch + offs, mask=m, other=0.0)
        pos = f > 0
        far = tl.where(pos, tl.load(lo_ptr + ch * q_ch + a, mask=m, other=0.0),
                       tl.load(hi_ptr + ch * q_ch + b, mask=m, other=0.0))
        up = tl.where(pos, qa, qb)
        down = tl.where(pos, qb, qa)
        jump = down - up
        r = (up - far) / tl.where(jump == 0, 1e-30, jump)
        ar = tl.abs(r)
        face = up + 0.5 * ((r + ar) / (1.0 + ar)) * jump
        fl = f * face
        tl.atomic_add(out_ptr + ch * q_ch + a, fl, mask=m)
        tl.atomic_add(out_ptr + ch * q_ch + b, -fl, mask=m)

    @triton.jit
    def _diffuse_kernel(q_ptr, q_ch, a_ptr, b_ptr, c_ptr, c_ch, d_ptr, d_ch, side_ptr, out_ptr, F,
                        HAS_D: tl.constexpr, BLOCK: tl.constexpr):
        """out[a] += c (qa - qb), out[b] -= c (qa - qb)."""
        pid = tl.program_id(0)
        ch = tl.program_id(1).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < F
        a = tl.load(a_ptr + offs, mask=m, other=0)
        b = tl.load(b_ptr + offs, mask=m, other=0)
        qa, qb = _face_pair(q_ptr, q_ch, ch, a, b, offs, m, d_ptr, d_ch, side_ptr, HAS_D)
        fl = tl.load(c_ptr + ch * c_ch + offs, mask=m, other=0.0) * (qa - qb)
        tl.atomic_add(out_ptr + ch * q_ch + a, fl, mask=m)
        tl.atomic_add(out_ptr + ch * q_ch + b, -fl, mask=m)

    @triton.jit
    def _neighbors_kernel(q_ptr, q_ch, a_ptr, b_ptr, wl_ptr, wh_ptr, d_ptr, d_ch, side_ptr, lo_ptr, hi_ptr, F,
                          HAS_D: tl.constexpr, BLOCK: tl.constexpr):
        """lo[b] += wn_low qa, hi[a] += wn_high qb with the corrected face values (`Stencil.neighbor_values`)."""
        pid = tl.program_id(0)
        ch = tl.program_id(1).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < F
        a = tl.load(a_ptr + offs, mask=m, other=0)
        b = tl.load(b_ptr + offs, mask=m, other=0)
        qa, qb = _face_pair(q_ptr, q_ch, ch, a, b, offs, m, d_ptr, d_ch, side_ptr, HAS_D)
        wl = tl.load(wl_ptr + offs, mask=m, other=0.0)
        wh = tl.load(wh_ptr + offs, mask=m, other=0.0)
        tl.atomic_add(lo_ptr + ch * q_ch + b, wl * qa, mask=m)
        tl.atomic_add(hi_ptr + ch * q_ch + a, wh * qb, mask=m)


def _flat(q: Tensor) -> Tensor:
    """(..., n) as (C, n), contiguous."""
    return q.reshape(-1, q.shape[-1]).contiguous()


def _sides(st, ax: int) -> Tensor:
    """Per face of `ax`: 1 where the a leaf is the coarser of a coarse-fine pair, 2 where the b leaf is, else 0."""
    key = ("fused_side", ax)
    if key not in st._cache:
        f = st.gr.faces[ax]
        side = torch.zeros(f.n, dtype=torch.int8, device=f.a.device)
        if st.cf_idx[ax] is not None and st.cf_idx[ax].numel():
            side[st.cf_idx[ax]] = torch.where(st.cf_coarse_a[ax], 1, 2).to(torch.int8)
        st._cache[key] = side
    return st._cache[key]


def _delta_args(st, ax: int, delta, C: int, F: int, like: Tensor):
    if delta is None or delta[ax] is None:
        return like, 0, like, False
    d = delta[ax].reshape(C, F)
    if not d.is_contiguous():
        d = d.contiguous()
    return d, d.stride(0), _sides(st, ax), True


def side_means(geo, ax: int, q: Tensor):
    """(low mean, high mean) per leaf of q (..., n) over its interior faces of `ax` (the boundary faces' terms the
    caller adds), as `Geo.low_mean(ax, q[a], ...)`, `Geo.high_mean(ax, q[b], ...)`."""
    qf = _flat(q)
    C, n = qf.shape
    lo, hi = torch.zeros_like(qf), torch.zeros_like(qf)
    F = geo.a[ax].numel()
    if F:
        _side_means_kernel[(triton.cdiv(F, BLOCK), C)](qf, qf.stride(0), geo.a[ax], geo.b[ax], geo.w_low[ax],
                                                       geo.w_high[ax], lo, hi, lo.stride(0), F, BLOCK=BLOCK)
    return lo.reshape(q.shape), hi.reshape(q.shape)


def convection(model, geo, q: Tensor, F, Fb, ghosts: Dict[str, Tensor], delta=None) -> Tensor:
    """`AmrModel.convection` with each axis's interior faces in two kernels (side means, then the fluxes); the
    boundary faces (a few planes) as the eager path computes them."""
    from solver import _muscl
    lim = model.cfg.limiter
    qf = _flat(q)
    C, n = qf.shape
    out = torch.zeros_like(qf)
    for ax in range(3):
        a, b = geo.a[ax], geo.b[ax]
        nf = a.numel()
        lo_s, hi_s = geo.lo_side(ax), geo.hi_side(ax)
        lo, hi = side_means(geo, ax, qf)
        for s, arr in ((lo_s, lo), (hi_s, hi)):               # the boundary faces' share of each side mean
            arr.index_add_(-1, geo.bcell[s], geo.wb[s] * ghosts[s].reshape(C, -1))
        if nf:
            f = F[ax].reshape(C, nf)
            f = f if f.is_contiguous() else f.contiguous()
            d, d_ch, side, has_d = _delta_args(model.st, ax, delta, C, nf, qf)
            _convect_kernel[(triton.cdiv(nf, BLOCK), C)](qf, qf.stride(0), a, b, f, f.stride(0), d, d_ch, side, lo,
                                                         hi, out, nf, HAS_D=has_d, BLOCK=BLOCK)
        for s, high in ((lo_s, False), (hi_s, True)):
            cell, gv, fb = geo.bcell[s], ghosts[s].reshape(C, -1), Fb[s].reshape(C, -1)
            qc = qf[..., cell]
            if high:
                face = torch.where(fb > 0, _muscl(lo[..., cell], qc, gv, lim), gv)
                out.index_add_(-1, cell, fb * face)
            else:
                face = torch.where(fb > 0, gv, _muscl(hi[..., cell], qc, gv, lim))
                out.index_add_(-1, cell, -fb * face)
    return out.reshape(q.shape)


def diffusion(model, geo, c, cb, q: Tensor, delta=None) -> Tensor:
    """`AmrModel.apply_diffusion`: each axis's interior faces in one kernel, the boundary faces as the eager path."""
    from amr_graph import SIDES
    qf = _flat(q)
    C, n = qf.shape
    out = torch.zeros_like(qf)
    for ax in range(3):
        nf = geo.a[ax].numel()
        if not nf:
            continue
        cc = c[ax].reshape(C, nf) if c[ax].numel() == C * nf else c[ax].expand(C, nf)
        cc = cc if cc.is_contiguous() else cc.contiguous()
        d, d_ch, side, has_d = _delta_args(model.st, ax, delta, C, nf, qf)
        _diffuse_kernel[(triton.cdiv(nf, BLOCK), C)](qf, qf.stride(0), geo.a[ax], geo.b[ax], cc, cc.stride(0), d,
                                                     d_ch, side, out, nf, HAS_D=has_d, BLOCK=BLOCK)
    for s in SIDES:
        cell = geo.bcell[s]
        out.index_add_(-1, cell, cb[s].reshape(C, -1) * qf[..., cell])
    return out.reshape(q.shape)


def neighbor_values(st, ax: int, q: Tensor, delta=None):
    """`Stencil.neighbor_values`: (low, high) neighbor means of q (..., n) along `ax`, in one kernel."""
    c = st.cast(q.dtype)
    qf = _flat(q)
    C, n = qf.shape
    f = st.gr.faces[ax]
    lo, hi = torch.zeros_like(qf), torch.zeros_like(qf)
    nf = f.a.numel()
    if nf:
        d, d_ch, side, has_d = _delta_args(st, ax, delta, C, nf, qf)
        _neighbors_kernel[(triton.cdiv(nf, BLOCK), C)](qf, qf.stride(0), f.a, f.b, c["wn_low"][ax], c["wn_high"][ax],
                                                       d, d_ch, side, lo, hi, nf, HAS_D=has_d, BLOCK=BLOCK)
    return lo.reshape(q.shape), hi.reshape(q.shape)


def grads(st, u: Tensor, deltas: List) -> List[List[Tensor]]:
    """g[c][ax] = d u_c / d(z, y, x)[ax] for u (H, 3, n), as `Stencil.grad`, each axis's three components in one
    kernel (the components are channels; each its own correction)."""
    coef = st.cast(u.dtype)["coef"]
    out = [[None] * 3 for _ in range(3)]
    H, _, n = u.shape
    for ax in range(3):
        dl = None
        if all(d is not None and d[ax] is not None for d in deltas):
            dl = [None, None, None]
            dl[ax] = torch.stack([deltas[c][ax] for c in range(3)], 1)          # (H, 3, F)
        elif any(d is not None and d[ax] is not None for d in deltas):          # mixed: one component at a time
            for c in range(3):
                lo, hi = neighbor_values(st, ax, u[:, c], deltas[c])
                a, b, cc = coef[ax]
                out[c][ax] = a * lo + b * u[:, c] + cc * hi
            continue
        lo, hi = neighbor_values(st, ax, u, dl)
        a, b, cc = coef[ax]
        for c in range(3):
            out[c][ax] = a * lo[:, c] + b * u[:, c] + cc * hi[:, c]
    return out


if triton is not None:
    @triton.jit
    def _g(lo_ptr, hi_ptr, u_ptr, ca_ptr, cb_ptr, cc_ptr, ch3, c: tl.constexpr, n, i, m):
        """d u_c / d axis at leaf i, the axis's (lo, hi, coef) planes given: a lo + b u + c hi (`Stencil.grad`)."""
        lo = tl.load(lo_ptr + (ch3 + c) * n + i, mask=m, other=0.0)
        hi = tl.load(hi_ptr + (ch3 + c) * n + i, mask=m, other=0.0)
        uc = tl.load(u_ptr + (ch3 + c) * n + i, mask=m, other=0.0)
        a = tl.load(ca_ptr + i, mask=m, other=0.0)
        b = tl.load(cb_ptr + i, mask=m, other=0.0)
        cc = tl.load(cc_ptr + i, mask=m, other=0.0)
        return a * lo + b * uc + cc * hi

    @triton.jit
    def _strain_kernel(u_ptr, lo0, hi0, lo1, hi1, lo2, hi2, a0, b0, c0, a1, b1, c1, a2, b2, c2, out_ptr, n,
                       BLOCK: tl.constexpr):
        """`AmrModel.strain_rest` at each leaf from its nine gradients, heading program_id(1)."""
        pid = tl.program_id(0)
        h = tl.program_id(1).to(tl.int64)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        ch3 = h * 3
        g02 = _g(lo2, hi2, u_ptr, a2, b2, c2, ch3, 0, n, i, m)
        g11 = _g(lo1, hi1, u_ptr, a1, b1, c1, ch3, 1, n, i, m)
        g20 = _g(lo0, hi0, u_ptr, a0, b0, c0, ch3, 2, n, i, m)
        out = 2 * (g02 * g02 + g11 * g11 + g20 * g20)
        g01 = _g(lo1, hi1, u_ptr, a1, b1, c1, ch3, 0, n, i, m)
        g12 = _g(lo2, hi2, u_ptr, a2, b2, c2, ch3, 1, n, i, m)
        s = g01 + g12
        out = out + s * s
        wx = _g(lo2, hi2, u_ptr, a2, b2, c2, ch3, 2, n, i, m)
        g00 = _g(lo0, hi0, u_ptr, a0, b0, c0, ch3, 0, n, i, m)
        out = out + wx * wx + 2 * g00 * wx
        wy = _g(lo1, hi1, u_ptr, a1, b1, c1, ch3, 2, n, i, m)
        g10 = _g(lo0, hi0, u_ptr, a0, b0, c0, ch3, 1, n, i, m)
        out = out + wy * wy + 2 * g10 * wy
        tl.store(out_ptr + h * n + i, out, mask=m)


def strain(st, u: Tensor, deltas: List) -> Tensor:
    """`AmrModel.strain_rest` (H, n) for u (H, 3, n): each axis's neighbor means of the three components in one kernel
    (`grads`' first half), then every leaf's nine gradients and the strain in one."""
    coef = st.cast(u.dtype)["coef"]
    uc = u.contiguous()
    H, _, n = uc.shape
    sides = []
    for ax in range(3):
        dl = None
        if deltas[0] is not None and deltas[0][ax] is not None:
            dl = [None, None, None]
            dl[ax] = torch.stack([deltas[c][ax] for c in range(3)], 1)
        lo, hi = neighbor_values(st, ax, uc, dl)
        sides += [lo.contiguous(), hi.contiguous()]
    out = torch.empty(H, n, dtype=u.dtype, device=u.device)
    cf = [x.contiguous() for abc in coef for x in abc]
    _strain_kernel[(triton.cdiv(n, BLOCK), H)](uc, *sides, *cf, out, n, BLOCK=BLOCK)
    return out
