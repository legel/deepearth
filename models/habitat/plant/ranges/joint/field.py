"""The landscape around a location: Entropy3D perceptive fields over the raw 240 m environmental field.

Why. The environment network reads the predictors at a location's own cell. What a plant meets there also depends on
the surroundings: whether the cell lies on a ridge or in a valley, at the foot of a mountain range or in the middle of
a basin, next to a coastline or a lake, at the edge of a desert. The field pathway lets the model read those
surroundings directly from the raw data, at scales it learns.

Data. A pyramid of the 240 m grid (``FieldPyramid``): 12 channels by default, fine terrain (elevation, slope,
northness, eastness, topographic position) and climate (lapse-rate-corrected annual mean temperature, warmest-month
maximum, coldest-month minimum; WorldClim temperature seasonality, annual precipitation, precipitation seasonality,
driest-quarter precipitation). Precipitation amounts enter as log(1 + value). Each channel is standardized by its
mean and standard deviation over the grid's valid cells and stored as int8 in steps of 1/16 standard deviation
(-127..127, i.e. +-7.9 SD; -128 = missing). Level k is the mean of 2^k x 2^k level-0 cells, missing cells left out.

Perceptive fields. The design follows DeepEarth's Entropy4D perceptive fields (encoders with a learnable position,
extent and shape in space and time) in their static, purely spatial case, here called Entropy3D. Around a location
x there is a centre field (the values at x) and R rings at learnable radii r_j (initially 1, 2, 4, ..., 128 cells:
0.24 to 31 km). Ring j is read at A angles theta_a = 2 pi a / A (a = 0 north, clockwise through east) by bilinear
interpolation on the pyramid level whose cell is about r_j / 2 (blending the two levels around log2 r_j - 1, so the
read moves smoothly with r_j and the radii receive gradients). Missing samples (sea, outside the grid) are left out.
Per ring and channel c, with d_a = v_c(theta_a) - v_c(x) the difference from the centre value and n the number of
valid samples, the ring is summarized by its circular harmonics:

    a_0 = (1/n) sum_a d_a                                        the radial profile: higher or lower than x, on average
    a_m = (2/n) sum_a d_a cos(m theta_a),  b_m = (2/n) sum_a d_a sin(m theta_a)        m = 1..M (M = 3)
    |(a_m, b_m)|                                                  the strength of order m, whatever its direction

Order 1 is a gradient across the ring (e.g. uphill to the west), order 2 an axis (a ridge or valley running through
x), order 3 a three-fold pattern. The cos/sin pairs are aligned to north (they know the direction: a south-facing
slope differs from a north-facing one); their magnitudes do not change when the landscape is rotated about the
vertical. With A = 8 angles the sums are exact for orders up to 3. Each ring also carries its missing share
(1 - n / A, averaged over channels: how much of the ring is sea or off the grid).

Interaction and pooling. The centre (its values and missing flags) and each ring become tokens of width d_c (two-layer
networks), plus a learned embedding per ring scale. Self-attention blocks let every token read the others (a valley
inside a plateau differs from one inside a plain). Learned-query attention pools the tokens into C(x), and a two-layer
network G maps C(x) to the width of the environment features; G's last layer starts at zero, so a model gains the
field without changing its scores at the start. The species score stays linear in the species vector:
f_s(x) = <h(x) + G(C(x)), w_s> + ... (model.py).

Computation. ``ring_sums`` evaluates every ring of every location in one fused CUDA kernel (kernels/ring_harmonics.cu,
compiled from source on first use; one warp per location and ring, one lane per angle, the radius gradient
analytic); on a CPU the PyTorch reference ``ring_sums_reference`` computes the same sums (equal to float32 rounding;
tests/test_field.py). Training holds the kernel's small outputs and recomputes only the token network in the
backward pass (gradient checkpointing).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

MISSING = -128                       # int8 code of a missing value
STEP = 1.0 / 16.0                    # int8 step in standard deviations


# ------------------------------------------------------------------------------------------------------- pyramid
def pool2(x: torch.Tensor) -> torch.Tensor:
    """Mean over 2 x 2 cells of a [H, W] tensor, NaN = missing and left out (odd edges keep their own cells)."""
    ok = torch.isfinite(x)
    s = F.avg_pool2d(torch.where(ok, x, 0.0)[None, None], 2, ceil_mode=True, divisor_override=1)[0, 0]
    n = F.avg_pool2d(ok.float()[None, None], 2, ceil_mode=True, divisor_override=1)[0, 0]
    return torch.where(n > 0, s / n.clamp(min=1), torch.full_like(s, float("nan")))


def build_pyramid(stack: str | Path, channels: Sequence[str], out: str | Path, levels: int = 8,
                  log_channels: Sequence[str] = ("bio_12", "bio_17"), device: str = "cpu", log=print) -> dict:
    """Write the field pyramid of a band-sequential float32 grid stack (``<stack>.f32`` with its ``.json``: variables,
    shape [bands, H, W], transform, crs; e.g. the CONUS 240 m fine stack) into ``out``: ``field_L<k>.npy`` int8
    [C, H_k, W_k] and ``field.json`` (channels, mean, sd, transform, crs, levels, shapes, step). Channels whose name
    ends with one of ``log_channels`` enter as log(1 + value)."""
    stack, out = Path(stack), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(stack.with_suffix(".json").read_text())
    C0, H, W = meta["shape"]
    src = np.memmap(stack, dtype=np.float32, mode="r", shape=(C0, H, W))
    dev = torch.device(device)
    shapes = [(H, W)]
    for _ in range(1, levels):
        h, w = shapes[-1]
        shapes.append(((h + 1) // 2, (w + 1) // 2))
    fields = [np.lib.format.open_memmap(out / f"field_L{k}.npy", mode="w+", dtype=np.int8,
                                        shape=(len(channels), *shapes[k])) for k in range(levels)]
    mean, sd = [], []
    for c, name in enumerate(channels):
        x = torch.from_numpy(np.array(src[meta["variables"].index(name)])).to(dev)
        x = torch.where(torch.isfinite(x) & (x > -1e30), x, torch.full_like(x, float("nan")))
        if name.endswith(tuple(log_channels)):
            x = torch.log1p(x.clamp(min=0))
        v = x[torch.isfinite(x)].double()
        mu, s = float(v.mean()), float(v.std()) + 1e-6
        mean.append(mu)
        sd.append(s)
        z = (x - mu) / s
        for k in range(levels):
            q = torch.round(z / STEP).clamp(-127, 127)
            fields[k][c] = torch.where(torch.isfinite(q), q, torch.full_like(q, MISSING)).to(torch.int8).cpu().numpy()
            if k + 1 < levels:
                z = pool2(z)
        del x, z, v
        log(f"{name}: mean {mu:.4g}, sd {s:.4g}")
    for f in fields:
        f.flush()
    info = {"channels": list(channels), "mean": mean, "sd": sd, "transform": meta["transform"], "crs": meta["crs"],
            "levels": levels, "shapes": shapes, "step": STEP,
            "log": [n for n in channels if n.endswith(tuple(log_channels))]}
    (out / "field.json").write_text(json.dumps(info, indent=1))
    return info


class FieldPyramid:
    """A field pyramid in memory, pixel-major: level k is int8 [H_k, W_k, C] (a cell's channels are contiguous, so a
    bilinear corner is one memory read). ``levels`` live on ``device``; the kernel needs them on the GPU."""

    def __init__(self, directory: str | Path, device: str | torch.device = "cpu"):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "field.json").read_text())
        self.channels = list(self.meta["channels"])
        self.transform = list(self.meta["transform"])
        self.levels = [torch.from_numpy(np.ascontiguousarray(
            np.load(self.directory / f"field_L{k}.npy", mmap_mode="r").transpose(1, 2, 0))).to(device)
            for k in range(int(self.meta["levels"]))]

    @classmethod
    def from_levels(cls, levels: Sequence[torch.Tensor], channels: Sequence[str]) -> "FieldPyramid":
        """A pyramid from int8 [H_k, W_k, C] tensors (tests, synthetic fields)."""
        p = cls.__new__(cls)
        p.directory, p.meta, p.channels, p.transform = None, {}, list(channels), None
        p.levels = [torch.as_tensor(v, dtype=torch.int8).contiguous() for v in levels]
        return p

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def device(self) -> torch.device:
        return self.levels[0].device

    def to(self, device) -> "FieldPyramid":
        self.levels = [v.to(device) for v in self.levels]
        return self


# ------------------------------------------------------------------------------------------------- ring sums
def bilinear(level: torch.Tensor, r: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Values (standard deviations) [..., C] and summed weight of the valid corners [..., C] at fractional cell
    coordinates of one pyramid level (cell centres at +0.5). Missing corners are left out and the remaining weights
    renormalized; a weight near 1 means all four corners exist."""
    H, W, C = level.shape
    y, x = r - 0.5, c - 0.5
    y0, x0 = torch.floor(y), torch.floor(x)
    wy, wx = y - y0, x - x0
    flat = level.view(-1, C)
    out = wsum = 0.0
    for dy, fy in ((0, 1 - wy), (1, wy)):
        for dx, fx in ((0, 1 - wx), (1, wx)):
            yy, xx = (y0 + dy).long(), (x0 + dx).long()
            inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            v = flat[(yy.clamp(0, H - 1) * W + xx.clamp(0, W - 1)).reshape(-1)].reshape(*yy.shape, C)
            ok = inside[..., None] & (v != MISSING)
            w = (fy * fx)[..., None] * ok
            out = out + w * torch.where(ok, v.to(w.dtype) * STEP, torch.zeros((), dtype=w.dtype, device=v.device))
            wsum = wsum + w
    return out / wsum.clamp(min=1e-6), wsum


def ring_sums_reference(levels: Sequence[torch.Tensor], rc: torch.Tensor, radii: torch.Tensor, v_centre: torch.Tensor,
                        angles: int, orders: int) -> torch.Tensor:
    """The ring harmonic sums in PyTorch, differentiable in ``radii``: [B, R, C, 2M + 2] = (S0, Sc_1..M, Ss_1..M, n)
    per location, ring and channel (definitions in kernels/ring_harmonics.cu). ``rc`` [B, 2] fractional level-0
    (row, column), ``radii`` [R] in level-0 cells, ``v_centre`` [B, C]."""
    th = torch.arange(angles, dtype=rc.dtype, device=rc.device) * (2 * math.pi / angles)
    m = torch.arange(1, orders + 1, dtype=rc.dtype, device=rc.device)[:, None] * th[None]
    cos_mt, sin_mt = torch.cos(m), torch.sin(m)                                     # [M, A]
    n_lev = len(levels)
    out = []
    for j in range(radii.shape[0]):
        rj = radii[j]
        lev = (torch.log2(rj) - 1).clamp(0, n_lev - 1)                             # cell ~ radius / 2
        k0 = int(torch.floor(lev).item())
        k1 = min(k0 + 1, n_lev - 1)
        t = lev - k0
        r = rc[:, None, 0] - rj * torch.cos(th)[None]                              # north = decreasing row
        c = rc[:, None, 1] + rj * torch.sin(th)[None]
        v, w = bilinear(levels[k0], r / 2 ** k0, c / 2 ** k0)                      # [B, A, C]
        if k1 != k0:
            v1, w1 = bilinear(levels[k1], r / 2 ** k1, c / 2 ** k1)
            v, w = (1 - t) * v + t * v1, (1 - t) * w + t * w1
        ok = (w > 0.5).float()
        d = (v - v_centre[:, None]) * ok                                           # [B, A, C]
        out.append(torch.cat([d.sum(1)[..., None], torch.einsum("bac,ma->bcm", d, cos_mt),
                              torch.einsum("bac,ma->bcm", d, sin_mt), ok.sum(1)[..., None]], -1))
    return torch.stack(out, 1)


_EXT = None


def _extension():
    """The compiled kernel (kernels/ring_harmonics.cu), built with the PyTorch C++ extension loader on first use and
    cached in kernels/build."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        d = Path(__file__).resolve().parent / "kernels"
        (d / "build").mkdir(exist_ok=True)
        _EXT = load(name="ring_harmonics_ext", sources=[str(d / "ring_harmonics.cu")], build_directory=str(d / "build"),
                    extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    return _EXT


class RingSums(torch.autograd.Function):
    """The fused kernel as a differentiable function of the radii (the field values are data)."""

    @staticmethod
    def forward(ctx, radii, rc, v_centre, levels, angles, orders):
        rc, radii = rc.float().contiguous(), radii.float().contiguous()
        ctx.save_for_backward(radii, rc)
        ctx.levels, ctx.angles, ctx.orders = levels, angles, orders
        return _extension().forward(levels, rc, radii, v_centre.float().contiguous(), angles, orders)

    @staticmethod
    def backward(ctx, grad):
        radii, rc = ctx.saved_tensors
        g = _extension().backward(ctx.levels, rc, radii, grad.float(), ctx.angles, ctx.orders)
        return g, None, None, None, None, None


def ring_sums(levels: Sequence[torch.Tensor], rc: torch.Tensor, radii: torch.Tensor, v_centre: torch.Tensor,
              angles: int, orders: int) -> torch.Tensor:
    """[B, R, C, 2M + 2] ring harmonic sums: the CUDA kernel for tensors on a GPU, the PyTorch reference otherwise."""
    with torch.autocast(rc.device.type, enabled=False):
        if rc.is_cuda:
            return RingSums.apply(radii.float(), rc.float(), v_centre.float(), list(levels), angles, orders)
        return ring_sums_reference(levels, rc.float(), radii.float(), v_centre.float(), angles, orders)


# ------------------------------------------------------------------------------------------------- the module
class AttentionBlock(nn.Module):
    """Pre-norm self-attention block over a token sequence: x + MHA(LN(x)), then + MLP(LN(x)) with a 4x hidden width
    and GELU (the latent block of DeepEarth's fusion model, core/fusion.py, with its default settings)."""

    def __init__(self, d: int, heads: int, mult: int = 4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Linear(mult * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.n1(x)
        x = x + self.attn(a, a, a, need_weights=False)[0]
        return x + self.ffn(self.n2(x))


class HarmonicField(nn.Module):
    """G(C(x)) [B, d] from fractional level-0 grid positions rc [B, 2] (row, column) of the field pyramid (module
    docstring). The pyramid is data, not a parameter: ``attach`` it before use."""

    def __init__(self, d: int, n_channels: int, d_c: int = 64, heads: int = 4, layers: int = 2,
                 radii_cells: Sequence[float] = (1, 2, 4, 8, 16, 32, 64, 128), angles: int = 8, orders: int = 3):
        super().__init__()
        if not 1 <= angles <= 32:
            raise ValueError("angles: 1 to 32 (one GPU lane per angle)")
        self.n_ch, self.A, self.M, self.heads, self.d_c = n_channels, angles, orders, heads, d_c
        self.log_r = nn.Parameter(torch.log(torch.tensor(radii_cells, dtype=torch.float32)))
        th = torch.arange(angles, dtype=torch.float32) * (2 * math.pi / angles)
        m = torch.arange(1, orders + 1, dtype=torch.float32)[:, None] * th[None]
        # sample directions and harmonic bases (kept with the weights: they define the angles a checkpoint used)
        self.register_buffer("cos_t", torch.cos(th))
        self.register_buffer("sin_t", torch.sin(th))
        self.register_buffer("cos_mt", torch.cos(m))
        self.register_buffer("sin_mt", torch.sin(m))
        C = n_channels
        self.ring_in = nn.Sequential(nn.Linear(C * (1 + 3 * orders) + 1, d_c), nn.SiLU(), nn.Linear(d_c, d_c))
        self.centre_in = nn.Sequential(nn.Linear(2 * C, d_c), nn.SiLU(), nn.Linear(d_c, d_c))
        self.scale_emb = nn.Parameter(torch.randn(len(radii_cells) + 1, d_c) * 0.02)
        self.blocks = nn.ModuleList([AttentionBlock(d_c, heads) for _ in range(layers)])
        self.query = nn.Parameter(torch.randn(heads, d_c) * 0.02)
        self.key = nn.Linear(d_c, d_c)
        self.out = nn.Sequential(nn.Linear(heads * d_c, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        self.pyramid: FieldPyramid | None = None

    @staticmethod
    def spec_from_state(state: dict, prefix: str = "field.") -> dict:
        """Constructor arguments (but ``d``) of a saved field: every size is read off its tensors."""
        d_c, f_ring = state[prefix + "ring_in.0.weight"].shape
        C = state[prefix + "centre_in.0.weight"].shape[1] // 2
        layers = len({k[len(prefix + "blocks."):].split(".")[0] for k in state if k.startswith(prefix + "blocks.")})
        return dict(n_channels=C, d_c=int(d_c), heads=int(state[prefix + "query"].shape[0]), layers=layers,
                    radii_cells=tuple(torch.exp(state[prefix + "log_r"]).tolist()),
                    angles=int(state[prefix + "cos_t"].shape[0]), orders=int((f_ring - 1 - C) // (3 * C)))

    def attach(self, pyramid: FieldPyramid) -> "HarmonicField":
        if pyramid.n_channels != self.n_ch:
            raise ValueError(f"the pyramid has {pyramid.n_channels} channels, the field was built for {self.n_ch}")
        self.pyramid = pyramid
        return self

    @property
    def radii(self) -> torch.Tensor:
        """Ring radii in level-0 cells."""
        return torch.exp(self.log_r)

    def centre(self, rc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Centre values [B, C] and their validity (0/1) [B, C] at level 0 (no parameters)."""
        with torch.no_grad():
            v, w = bilinear(self.pyramid.levels[0], rc[:, 0].float(), rc[:, 1].float())
        return v, (w > 0.5).float()

    def sums(self, rc: torch.Tensor, v_centre: torch.Tensor) -> torch.Tensor:
        return ring_sums(self.pyramid.levels, rc, self.radii, v_centre, self.A, self.M)

    def ring_features(self, S: torch.Tensor) -> torch.Tensor:
        """Ring token inputs [B, R, C (1 + 3M) + 1] from the sums S [B, R, C, 2M + 2]: per channel a_0, then the
        a_m, the b_m and the magnitudes of every channel and order, then the ring's missing share."""
        B, R, C, M = S.shape[0], S.shape[1], self.n_ch, self.M
        n_valid = S[..., -1]
        n = n_valid.clamp(min=1)
        a0 = S[..., 0] / n
        ac = S[..., 1:M + 1] * (2 / n)[..., None]
        as_ = S[..., M + 1:2 * M + 1] * (2 / n)[..., None]
        mag = torch.sqrt(ac * ac + as_ * as_ + 1e-8)
        missing = 1 - (n_valid / self.A).mean(-1)
        return torch.cat([a0, ac.reshape(B, R, C * M), as_.reshape(B, R, C * M), mag.reshape(B, R, C * M),
                          missing[..., None]], -1)

    def head(self, v_centre: torch.Tensor, ok_centre: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Tokens -> self-attention -> pooled C(x) -> G(C(x)) [B, d]."""
        B = S.shape[0]
        centre = self.centre_in(torch.cat([v_centre * ok_centre, 1 - ok_centre], 1))[:, None]
        tok = torch.cat([centre, self.ring_in(self.ring_features(S))], 1) + self.scale_emb[None]   # [B, 1 + R, d_c]
        for blk in self.blocks:
            tok = blk(tok)
        att = torch.einsum("btc,hc->bht", self.key(tok), self.query) / math.sqrt(self.d_c)
        a = torch.softmax(att.float(), -1).to(tok.dtype)
        return self.out(torch.einsum("bht,btc->bhc", a, tok).reshape(B, -1))

    def forward(self, rc: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        """G(C(x)) [B, d]. Rows are processed in chunks (the fused attention launches one GPU block per row and head,
        which must stay within one launch grid). With gradients, the kernel's sums are kept and the token network of
        each chunk is recomputed in the backward pass, so memory stays bounded by one chunk's activations."""
        if self.pyramid is None:
            raise RuntimeError("no field pyramid attached (HarmonicField.attach)")
        outs = []
        for i in range(0, len(rc), chunk):
            r = rc[i:i + chunk]
            v_c, ok_c = self.centre(r)
            S = self.sums(r, v_c)
            if torch.is_grad_enabled():
                outs.append(torch.utils.checkpoint.checkpoint(self.head, v_c, ok_c, S, use_reentrant=False))
            else:
                outs.append(self.head(v_c, ok_c, S))
        return torch.cat(outs) if outs else torch.zeros(0, self.out[-1].out_features, device=rc.device)
