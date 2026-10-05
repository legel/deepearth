"""The storm's sub-step as five hand-fused Triton kernels in place of the twenty-two torch.compile makes of `_substep`.

Fisher Museum's storm (662 x 662 cells, L4, 2026-10-04): 22 kernels a sub-step, 0.102 ms of GPU time, most of them
launch-bound reductions and copies; the RTX PRO 6000 ran it only 1.7x faster than the L4 on 6x the bandwidth. Here:

    _peak   the deepest face per program, x and y apart       reads h, z, zmax
    _dt     the CFL step from the programs' maxima            one program
    _faces  every face's new discharge in place, edges too   reads h, z, zmax, n2, q; writes q
    _depth  each cell's new depth, the mass terms' shares  reads q, h, rain share, source, mask; writes h
    _close  the mass terms from the shares, the clock         one program

Each cell and face is the same arithmetic as the compiled sub-step, operation for operation (inductor's own code for
`_substep`, read from TORCH_LOGS=output_code on torch 2.9.1), its multiply-adds contracted as inductor's are. Measured
on production's storms against the compiled sub-step, L4, 2026-10-04: Fisher Museum's 30 h (422,208 sub-steps) and
Berkeley Oxford and Hearst's 72 h (1,502,178), every frame of depth, u and v, the peak depth, the sub-step count and
every mass term the same to the bit; solve 51.7 to 41.4 s and 176.7 to 126.0 s. Only the configuration production's
storms run is fused (face CFL, split soil, float32, no gauge or watershed probe, a power-of-two cell, MIN_CELLS and
up); `usable` says so and anything else runs the compiled sub-step.
"""
import math
import os
from typing import Optional

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from physics import FROUDE_CAP, G, MANNING_EXP, MIN_DEPTH

BLOCK = 1024
FMA = os.environ.get("HYDRO_FUSED_FMA", "both")
"""Which kernels contract multiply-adds into FMAs: both, as LLVM contracts inductor's at production sizes (Fisher and
Berkeley to the bit). "faces", "depth" and "none" are lab switches (HYDRO_FUSED_FMA)."""
SRC_FMA = os.environ.get("HYDRO_FUSED_SRC_FMA", "1") != "0"
"""The rim source's multiply-add as one FMA in `_depth` (HYDRO_FUSED_SRC_FMA=0 is a lab switch)."""
MIN_CELLS = 24_000
"""The smallest grid whose compiled sub-step the kernels match to the bit: below it inductor folds the depth into its
reduction and contracts differently (97 x 83 differed in the last bit of 1,308 cells; 160 x 150 and up matched, L4,
2026-10-04). A smaller storm runs compiled; it is quick either way."""
WRITTEN_FOR = {"G": 9.81, "MANNING_EXP": 7.0 / 3.0, "MIN_DEPTH": 1e-4, "FROUDE_CAP": 0.9}
"""The physics constants the kernels carry as literals, as inductor writes them; a change to any runs the compiled
sub-step until the kernels are written again."""


def usable(grid, kern, dtype: torch.dtype, device: torch.device, block: int) -> Optional[str]:
    """None when the fused sub-step runs this storm, else why not."""
    held = {"G": G, "MANNING_EXP": MANNING_EXP, "MIN_DEPTH": MIN_DEPTH, "FROUDE_CAP": FROUDE_CAP}
    if held != WRITTEN_FOR:
        return f"physics constants {held} are not the kernels' {WRITTEN_FOR}"
    if device.type != "cuda":
        return "not on CUDA"
    if dtype != torch.float32:
        return "float64 storm"
    if not kern.split:
        return "soil inside the sub-step"
    if grid.gauge is not None or grid.ws_sx is not None:
        return "a gauge or watershed probe"
    if getattr(kern, "cell_cfl", False):
        return "the cell CFL (inductor fuses that graph otherwise, and production takes the face CFL)"
    if grid.z.numel() < MIN_CELLS:
        return f"{grid.z.numel()} cells, under the {MIN_CELLS} at which inductor fuses the sub-step as the kernels copy"
    if math.frexp(kern.dx)[0] != 0.5:
        return f"dx {kern.dx} is not a power of two"      # 1 / dx is exact, as inductor's multiply by it assumes
    return None


@triton.jit
def _hf(z_ptr, h_ptr, zm_ptr, hi, lo, f, m):
    """A face's flow depth, with the surfaces either side: (hf, eta high side, eta low side)."""
    ehi = tl.load(z_ptr + hi, m, other=0.0) + tl.load(h_ptr + hi, m, other=0.0)
    elo = tl.load(z_ptr + lo, m, other=0.0) + tl.load(h_ptr + lo, m, other=0.0)
    hf = triton_helpers.maximum(triton_helpers.maximum(ehi, elo) - tl.load(zm_ptr + f, m, other=0.0), 0.0)
    return hf, ehi, elo


@triton.jit
def _peak(h_ptr, z_ptr, zx_ptr, zy_ptr, part_ptr, R: tl.constexpr, C: tl.constexpr, NPX: tl.constexpr,
          BLOCK: tl.constexpr):
    """Each program's deepest interior face: the first NPX programs over the x faces, the rest over the y faces."""
    pid = tl.program_id(0)
    if pid < NPX:
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < R * (C - 1)
        r = i // (C - 1)
        c = i % (C - 1)
        hf, ehi, elo = _hf(z_ptr, h_ptr, zx_ptr, r * C + c + 1, r * C + c, i, m)
    else:
        i = (pid - NPX) * BLOCK + tl.arange(0, BLOCK)
        m = i < (R - 1) * C
        hf, ehi, elo = _hf(z_ptr, h_ptr, zy_ptr, i + C, i, i, m)
    tl.store(part_ptr + pid, triton_helpers.max2(tl.where(m, hf, float("-inf")), 0))


@triton.jit
def _dt(part_ptr, t_end_ptr, t_ptr, src_dt_ptr, dt_ptr, NP: tl.constexpr, AD: tl.constexpr, DT_S: tl.constexpr,
        BLOCK: tl.constexpr):
    acc = tl.full([BLOCK], float("-inf"), tl.float32)
    for off in range(0, NP, BLOCK):
        j = off + tl.arange(0, BLOCK)
        acc = triton_helpers.maximum(acc, tl.load(part_ptr + j, j < NP, other=float("-inf")))
    peak = triton_helpers.max2(acc, 0).to(tl.float64)
    # _cfl_dt as inductor compiles it: (1 / sqrt(g peak)) * (alpha dx)
    wave = (1 / libdevice.sqrt(peak * tl.full([], 9.81, tl.float64))) * tl.full([], AD, tl.float64)
    dts = tl.full([], DT_S, tl.float64)
    dt = tl.where(peak > tl.full([], 0.0001, tl.float64), triton_helpers.minimum(wave, dts), dts)
    dt = triton_helpers.minimum(dt, tl.load(t_end_ptr) - tl.load(t_ptr))
    dt = triton_helpers.maximum(dt, tl.full([], 0.0, tl.float64))
    dt = triton_helpers.minimum(dt, tl.load(src_dt_ptr))
    tl.store(dt_ptr, dt)


@triton.jit
def _flux(qold, hf, ehi, elo, n2, dt, INV_DX: tl.constexpr):
    """One interior face's discharge after `_face_flux`, in inductor's order: the momentum update, the wet test and
    the Froude cap."""
    num = qold - (((hf * 9.81) * dt.to(tl.float32)) * (ehi - elo)) * INV_DX
    den = ((((dt * tl.full([], 9.81, tl.float64)).to(tl.float32) * n2) * tl.abs(qold))
           / (libdevice.pow(hf, 2.3333333333333335) + 1e-10)) + 1.0
    q = tl.where(hf > 0.0001, num / den, 0.0)
    cap = (hf * 0.9) * libdevice.sqrt(triton_helpers.maximum(hf, 0.0001) * 9.81)
    return triton_helpers.minimum(triton_helpers.maximum(q, -cap), cap)


@triton.jit
def _faces(h_ptr, z_ptr, zx_ptr, zy_ptr, n2x_ptr, n2y_ptr, qx_ptr, qy_ptr, w_ptr, e_ptr, n_ptr, s_ptr, dt_ptr, we_ptr,
           R: tl.constexpr, C: tl.constexpr, NPX: tl.constexpr, INV_DX: tl.constexpr, BLOCK: tl.constexpr):
    """Every interior face's new discharge, in place (each reads only its own old one); the face beside an edge also
    writes the edge face, from its new value and the prescribed inflow, as `_edge` does, and the west and east edges
    once more side by side in `we` for the closing sums (a column of qx is strided)."""
    pid = tl.program_id(0)
    dt = tl.load(dt_ptr)
    if pid < NPX:
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < R * (C - 1)
        r = i // (C - 1)
        c = i % (C - 1)                                         # the face between cells c and c + 1, qx[r, c + 1]
        hf, ehi, elo = _hf(z_ptr, h_ptr, zx_ptr, r * C + c + 1, r * C + c, i, m)
        f = r * (C + 1) + c + 1
        q = _flux(tl.load(qx_ptr + f, m, other=0.0), hf, ehi, elo, tl.load(n2x_ptr + i, m, other=0.0), dt, INV_DX)
        tl.store(qx_ptr + f, q, m)
        w = tl.load(w_ptr + r, m & (c == 0), other=0.0)
        qw = tl.where(w != 0.0, w, triton_helpers.minimum(q, 0.0))
        tl.store(qx_ptr + (f - 1), qw, m & (c == 0))
        tl.store(we_ptr + r, qw, m & (c == 0))
        e = -tl.load(e_ptr + r, m & (c == C - 2), other=0.0)
        qe = tl.where(e != 0.0, e, triton_helpers.maximum(q, 0.0))
        tl.store(qx_ptr + (f + 1), qe, m & (c == C - 2))
        tl.store(we_ptr + R + r, qe, m & (c == C - 2))
    else:
        i = (pid - NPX) * BLOCK + tl.arange(0, BLOCK)
        m = i < (R - 1) * C
        r = i // C                                              # the face between rows r and r + 1, qy[r + 1, c]
        c = i % C
        hf, ehi, elo = _hf(z_ptr, h_ptr, zy_ptr, i + C, i, i, m)
        f = i + C
        q = _flux(tl.load(qy_ptr + f, m, other=0.0), hf, ehi, elo, tl.load(n2y_ptr + i, m, other=0.0), dt, INV_DX)
        tl.store(qy_ptr + f, q, m)
        nn = tl.load(n_ptr + c, m & (r == 0), other=0.0)
        tl.store(qy_ptr + c, tl.where(nn != 0.0, nn, triton_helpers.minimum(q, 0.0)), m & (r == 0))
        ss = -tl.load(s_ptr + c, m & (r == R - 2), other=0.0)
        tl.store(qy_ptr + (f + C), tl.where(ss != 0.0, ss, triton_helpers.maximum(q, 0.0)), m & (r == R - 2))


@triton.jit
def _depth(h_ptr, qx_ptr, qy_ptr, rain_ptr, vf_ptr, src_ptr, inv_ptr, dt_ptr, part_ptr, SRC_FMA: tl.constexpr,
           R: tl.constexpr, C: tl.constexpr, NP: tl.constexpr, INV_DX: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < R * C
    r = i // C
    c = i % C
    # every load first and every operation after, in the order inductor's depth kernel takes them
    dt = tl.load(dt_ptr)
    rain = tl.load(rain_ptr)
    qa = tl.load(qx_ptr + (r * (C + 1) + c), m, other=0.0)
    qb = tl.load(qx_ptr + (r * (C + 1) + c + 1), m, other=0.0)
    qc = tl.load(qy_ptr + i, m, other=0.0)
    qd = tl.load(qy_ptr + (i + C), m, other=0.0)
    h0 = tl.load(h_ptr + i, m, other=0.0)
    vf = tl.load(vf_ptr + i, m, other=0.0)
    src = tl.load(src_ptr + i, m, other=0.0)
    inv = tl.load(inv_ptr + i, m, other=0).to(tl.int1)
    dtx = (dt * tl.full([], INV_DX, tl.float64)).to(tl.float32)
    t31 = qa - qb
    t33 = t31 + qc
    t35 = t33 - qd
    t36 = dtx * t35
    t41 = (dt * rain).to(tl.float32) * vf
    t42 = t36 + t41
    if SRC_FMA:
        t46 = tl.fma(dt.to(tl.float32), src, t42)
    else:
        t46 = t42 + dt.to(tl.float32) * src
    wet = h0 + t46
    created = triton_helpers.minimum(wet, 0.0)
    h = wet - created
    spilled = tl.where(inv, h, 0.0)
    tl.store(h_ptr + i, h - spilled, m)
    tl.store(part_ptr + pid, tl.sum(tl.where(m, spilled.to(tl.float64), 0.0), 0))
    tl.store(part_ptr + NP + pid, tl.sum(tl.where(m, created.to(tl.float64), 0.0), 0))


@triton.jit
def _close(we_ptr, qy_ptr, part_ptr, src_sum_ptr, dt_ptr, t_ptr, acc_ptr, R: tl.constexpr, C: tl.constexpr,
           NP: tl.constexpr, DX: tl.constexpr, BLOCK: tl.constexpr):
    """The sub-step's mass terms (the edges out and in, the spilled and created shares) and the clock, in one program:
    every edge read contiguously, the west and east from `we`."""
    s1 = tl.zeros([BLOCK], tl.float64)
    s2 = tl.zeros([BLOCK], tl.float64)
    s6 = tl.zeros([BLOCK], tl.float64)
    s7 = tl.zeros([BLOCK], tl.float64)
    for off in range(0, R, BLOCK):
        e = off + tl.arange(0, BLOCK)
        me = e < R
        west = tl.load(we_ptr + e, me, other=0.0)
        east = tl.load(we_ptr + R + e, me, other=0.0)
        s1 += tl.where(me, triton_helpers.maximum(-west, 0.0).to(tl.float64), 0.0)
        s2 += tl.where(me, triton_helpers.maximum(east, 0.0).to(tl.float64), 0.0)
        s6 += tl.where(me, triton_helpers.maximum(west, 0.0).to(tl.float64), 0.0)
        s7 += tl.where(me, triton_helpers.maximum(-east, 0.0).to(tl.float64), 0.0)
    s3 = tl.zeros([BLOCK], tl.float64)
    s4 = tl.zeros([BLOCK], tl.float64)
    s8 = tl.zeros([BLOCK], tl.float64)
    s9 = tl.zeros([BLOCK], tl.float64)
    for off in range(0, C, BLOCK):
        e = off + tl.arange(0, BLOCK)
        me = e < C
        north = tl.load(qy_ptr + e, me, other=0.0)
        south = tl.load(qy_ptr + R * C + e, me, other=0.0)
        s3 += tl.where(me, triton_helpers.maximum(-north, 0.0).to(tl.float64), 0.0)
        s4 += tl.where(me, triton_helpers.maximum(south, 0.0).to(tl.float64), 0.0)
        s8 += tl.where(me, triton_helpers.maximum(north, 0.0).to(tl.float64), 0.0)
        s9 += tl.where(me, triton_helpers.maximum(-south, 0.0).to(tl.float64), 0.0)
    sp = tl.zeros([BLOCK], tl.float64)
    cr = tl.zeros([BLOCK], tl.float64)
    for off in range(0, NP, BLOCK):
        j = off + tl.arange(0, BLOCK)
        sp += tl.load(part_ptr + j, j < NP, other=0.0)
        cr += tl.load(part_ptr + NP + j, j < NP, other=0.0)
    dt = tl.load(dt_ptr)
    dxf = tl.full([], DX, tl.float64)
    leaving = (((((tl.sum(s1, 0) + tl.sum(s2, 0)) + tl.sum(s3, 0)) + tl.sum(s4, 0))
                + (tl.sum(sp, 0) * dxf) / triton_helpers.maximum(dt, tl.full([], 1e-30, tl.float64))) * dxf)
    entering = (((((tl.sum(s6, 0) + tl.sum(s7, 0)) + tl.sum(s8, 0)) + tl.sum(s9, 0)) + tl.load(src_sum_ptr) * dxf) * dxf)
    tl.store(acc_ptr + 0, tl.load(acc_ptr + 0) + (leaving * dt + 0.0))
    tl.store(acc_ptr + 1, tl.load(acc_ptr + 1) + (entering * dt + 0.0))
    tl.store(acc_ptr + 6, tl.load(acc_ptr + 6) + (dt > 0.0).to(tl.float64))
    tl.store(acc_ptr + 7, tl.load(acc_ptr + 7) - tl.sum(cr, 0))
    tl.store(t_ptr, tl.load(t_ptr) + dt)


class Step:
    """`_substep(s, g, f, kern)` for one storm, its scratch made once so a CUDA graph can capture it."""

    def __init__(self, state, grid, forcing, kern, cfl_cell: bool = False):
        rows, cols = grid.z.shape
        dev = grid.z.device
        self.R, self.C = rows, cols
        self.npx = triton.cdiv(rows * (cols - 1), BLOCK)
        self.np_faces = self.npx + triton.cdiv((rows - 1) * cols, BLOCK)
        self.np_cells = triton.cdiv(rows * cols, BLOCK)
        self.peak = torch.empty(self.np_faces, dtype=torch.float32, device=dev)
        self.part = torch.empty(2 * self.np_cells, dtype=torch.float64, device=dev)
        self.we = torch.zeros(2 * rows, dtype=torch.float32, device=dev)
        self.dt = torch.zeros((), dtype=torch.float64, device=dev)
        self.src_sum = torch.zeros((), dtype=torch.float64, device=dev)
        full = lambda v, shape: v if torch.is_tensor(v) else torch.full(shape, float(v), dtype=torch.float32, device=dev)
        self.n2x = full(grid.n2_x, (rows, cols - 1)).contiguous()
        self.n2y = full(grid.n2_y, (rows - 1, cols)).contiguous()
        self.inv = grid.invalid.to(torch.int8).contiguous()
        self.ad, self.dt_s, self.dx = float(kern.alpha * kern.dx), float(kern.dt_s), float(kern.dx)
        self.inv_dx = 1.0 / float(kern.dx)
        self.forcing_changed(forcing)

    def forcing_changed(self, forcing) -> None:
        """The interval's rim source total, which `_substep` summed every sub-step; outside any graph."""
        self.src_sum.copy_(forcing.source.sum(dtype=torch.float64))

    def __call__(self, s, g, f, kern) -> None:
        R, C = self.R, self.C
        _peak[(self.np_faces,)](s.h, g.z, g.zmax_x, g.zmax_y, self.peak, R, C, self.npx, BLOCK)
        _dt[(1,)](self.peak, f.t_end, s.t, f.source_dt, self.dt, self.np_faces, self.ad, self.dt_s, BLOCK)
        _faces[(self.np_faces,)](s.h, g.z, g.zmax_x, g.zmax_y, self.n2x, self.n2y, s.qx, s.qy, f.west, f.east,
                                 f.north, f.south, self.dt, self.we, R, C, self.npx, self.inv_dx, BLOCK,
                                 enable_fp_fusion=FMA in ("faces", "both"))
        _depth[(self.np_cells,)](s.h, s.qx, s.qy, f.rain, g.valid_f, f.source, self.inv, self.dt, self.part, SRC_FMA,
                                 R, C, self.np_cells, self.inv_dx, BLOCK, enable_fp_fusion=FMA in ("depth", "both"))
        _close[(1,)](self.we, s.qy, self.part, self.src_sum, self.dt, s.t, s.acc, R, C, self.np_cells, self.dx, BLOCK)
