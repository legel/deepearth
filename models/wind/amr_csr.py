"""A row kernel for the multigrid's coarse levels: M x and one Gauss-Seidel color in one Triton launch, each lane a row
walking its own entries (CSR, the rows' longest in each block bounding the walk), where the eager path ran a color as a
whole-level product by gather and two scatters, a subtraction, a division and a masked add: eight launches and as many
passes over the level, four colors a sweep (the roofline: the solves at 4 to 8 times their least
bytes on Fisher's grid). The rows are each level's own (`amr_graph.MGLevel`); the pattern is made once a level, the
values once a multigrid, from each face's two entries (`GraphOp.offdiag`).

The arithmetic is the eager product's, the sum over a row in the row's own order (the eager sums are atomic, in no
order), so the two agree to round-off (tests/test_amr.py)."""

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:          # a CPU-only environment: the eager path serves
    triton = None

Tensor = torch.Tensor
BLOCK = 256


if triton is not None:
    @triton.jit
    def _row_kernel(x_ptr, x_ch, rowptr_ptr, col_ptr, val_ptr, val_g, diag_ptr, d_g, kmax_ptr, b_ptr, b_ch, safe_ptr,
                    s_g, color_ptr, out_ptr, o_ch, n, color, g_div, MODE: tl.constexpr, BLOCK: tl.constexpr):
        """Channel program_id(1), of operator group ch // g_div. MODE 0: out = M x. MODE 1: out = x + (b - M x) / safe
        on the rows of `color`, x elsewhere."""
        pid = tl.program_id(0)
        ch = tl.program_id(1).to(tl.int64)
        grp = ch // g_div
        r = pid * BLOCK + tl.arange(0, BLOCK)
        m = r < n
        start = tl.load(rowptr_ptr + r, mask=m, other=0)
        end = tl.load(rowptr_ptr + r + 1, mask=m, other=0)
        xv = tl.load(x_ptr + ch * x_ch + r, mask=m, other=0.0)
        acc = tl.load(diag_ptr + grp * d_g + r, mask=m, other=0.0) * xv
        kmax = tl.load(kmax_ptr + pid)
        for k in range(0, kmax):
            e = start + k
            ok = m & (e < end)
            c = tl.load(col_ptr + e, mask=ok, other=0)
            v = tl.load(val_ptr + grp * val_g + e, mask=ok, other=0.0)
            acc += v * tl.load(x_ptr + ch * x_ch + c, mask=ok, other=0.0)
        if MODE == 0:
            tl.store(out_ptr + ch * o_ch + r, acc, mask=m)
        else:
            bv = tl.load(b_ptr + ch * b_ch + r, mask=m, other=0.0)
            sv = tl.load(safe_ptr + grp * s_g + r, mask=m, other=1.0)
            cv = tl.load(color_ptr + r, mask=m, other=-1)
            tl.store(out_ptr + ch * o_ch + r, tl.where(cv == color, xv + (bv - acc) / sv, xv), mask=m)


def available(op) -> bool:
    return triton is not None and op.diag.is_cuda


def pattern(lv):
    """The level's rows as CSR, cached on it: row pointer, columns, each entry's source in cat([ab, ba]) (row a reads
    ab, row b reads ba), and each block's longest row."""
    got = getattr(lv, "_csr_rows", None)
    if got is not None:
        return got
    a, b = lv.a.long(), lv.b.long()
    n, nf = lv.n, a.numel()
    dev = a.device
    rows = torch.cat([a, b])
    cols = torch.cat([b, a])
    src = torch.arange(2 * nf, device=dev)
    order = torch.argsort(rows * n + cols, stable=True)
    rows, cols, src = rows[order], cols[order], src[order]
    count = torch.bincount(rows, minlength=n)
    rowptr = torch.zeros(n + 1, dtype=torch.int64, device=dev)
    rowptr[1:] = torch.cumsum(count, 0)
    nb = triton.cdiv(n, BLOCK)
    pad = torch.zeros(nb * BLOCK, dtype=count.dtype, device=dev)
    pad[:n] = count
    kmax = pad.reshape(nb, BLOCK).amax(1)
    got = lv._csr_rows = (rowptr.int(), cols.int(), src.int(), kmax.int())
    return got


class CsrOp:
    """One coarse level's operator (`GraphOp`, or `GroupOp` with G groups of R channels) for the row kernel."""

    def __init__(self, op):
        self.op = op
        lv = op.level
        self.rowptr, self.col, src, self.kmax = pattern(lv)
        ab, ba = op.offdiag()
        grouped = op.diag.dim() > 1
        self.G = op.diag.shape[0] if grouped else 1
        self.R = getattr(op, "R", 1) if grouped else None
        self.val = torch.cat([ab.reshape(self.G, -1), ba.reshape(self.G, -1)], -1)[:, src].contiguous()
        self.diag = op.diag.reshape(self.G, -1).contiguous()
        self.n = self.diag.shape[-1]

    def _g_div(self, C: int) -> int:
        return self.R if self.R is not None else C

    def apply(self, x: Tensor) -> Tensor:
        """(C, n) -> (C, n)."""
        x = x.contiguous()
        out = torch.empty_like(x)
        C = x.shape[0]
        _row_kernel[(triton.cdiv(self.n, BLOCK), C)](x, x.stride(0), self.rowptr, self.col, self.val,
                                                     self.val.stride(0), self.diag, self.diag.stride(0), self.kmax, x,
                                                     x.stride(0), self.diag, self.diag.stride(0), self.rowptr, out,
                                                     out.stride(0), self.n, 0, self._g_div(C), MODE=0, BLOCK=BLOCK)
        return out

    def relax(self, safe: Tensor, colors: Tensor, color: int, b: Tensor, x: Tensor) -> Tensor:
        """One color of the Gauss-Seidel sweep, `safe` (n,) or (G, n)."""
        x, b = x.contiguous(), b.contiguous()
        s = safe.reshape(-1, self.n).contiguous()
        out = torch.empty_like(x)
        C = x.shape[0]
        _row_kernel[(triton.cdiv(self.n, BLOCK), C)](x, x.stride(0), self.rowptr, self.col, self.val,
                                                     self.val.stride(0), self.diag, self.diag.stride(0), self.kmax, b,
                                                     b.stride(0), s, s.stride(0) if s.shape[0] > 1 else 0, colors,
                                                     out, out.stride(0), self.n, color, self._g_div(C), MODE=1,
                                                     BLOCK=BLOCK)
        return out


class Rows:
    """A fixed CSR matrix (torch sparse CSR, on the GPU) multiplied by the row kernel: no cuSPARSE workspace, which
    for a matrix the size of the level (the projection's coarse-fine correction, n x n with few rows filled) held
    more of the card than the product's operands."""

    def __init__(self, A: Tensor):

        self.n = A.shape[0]
        self.rowptr = A.crow_indices().int().contiguous()
        self.col = A.col_indices().int().contiguous()
        self.val = A.values().contiguous()
        count = (self.rowptr[1:] - self.rowptr[:-1]).long()
        nb = triton.cdiv(self.n, BLOCK)
        pad = torch.zeros(nb * BLOCK, dtype=count.dtype, device=count.device)
        pad[:self.n] = count
        self.kmax = pad.reshape(nb, BLOCK).amax(1).int()
        self.zero = torch.zeros(self.n, dtype=self.val.dtype, device=self.val.device)

    @property
    def dtype(self):
        return self.val.dtype

    def apply(self, x: Tensor) -> Tensor:
        """(C, n) -> (C, rows)."""
        x = x.contiguous()
        C = x.shape[0]
        out = torch.empty(C, self.n, dtype=x.dtype, device=x.device)
        _row_kernel[(triton.cdiv(self.n, BLOCK), C)](x, x.stride(0), self.rowptr, self.col, self.val, 0, self.zero, 0,
                                                     self.kmax, x, x.stride(0), self.zero, 0, self.rowptr, out,
                                                     out.stride(0), self.n, 0, C, MODE=0, BLOCK=BLOCK)
        return out
