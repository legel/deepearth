"""FluxFill network: a transformer with factorised time and variable attention over day-tokens, a per-half-hour
Student-t head (v0, marginal) and a per-token flow-matching head (v1, joint samples by masked generation).

Tokens (docs/MODEL.md section 2): one per variable and day, holding 48 standardised values and 48 visible flags,
embedded by a shared linear map; a token with no visible value is replaced by a learned mask vector. Imagery adds one
row of day-tokens (Clay CLS, SCL fractions). Every token adds a variable embedding, a provenance embedding (tower,
reanalysis, reanalysis-land, imagery), a day-of-window embedding, day-of-year harmonics and a site-location encoding.

Encoder (section 3): blocks alternate time attention (each variable over its 180 days, rotary position encoding of the
day index; Su et al. 2021, arXiv:2104.09864) and variable attention (each day across its variables and its imagery
token). Absent variables (not measured in the window) and dropped modalities are excluded as keys of variable
attention, so they contribute nothing. Factorised (axial) attention follows Ho et al. 2019 (arXiv:1912.12180) and
its use for multivariate series in Zhang and Yan 2023 (Crossformer, ICLR).

Heads:
- `StudentTHead`: location, scale and degrees of freedom (> 2) per half-hour, trained on the negative log-likelihood.
  A heteroscedastic Student-t likelihood follows Kendall and Gal 2017 (NeurIPS) for learned aleatoric scale and Lange,
  Little and Taylor 1989 (J. Am. Stat. Assoc. 84:881) for the t model of heavy-tailed errors.
- `FlowHead`: a small MLP with adaptive layer norm conditioned on the token's encoder state, which transports
  Gaussian noise to the token's 48 values by conditional flow matching (Lipman et al. 2023, ICLR, arXiv:2210.02747;
  Liu et al. 2023 rectified flow, arXiv:2209.03003). It replaces the denoising-diffusion head of masked
  autoregressive generation with continuous tokens (Li et al. 2024, arXiv:2406.11838), whose MLP-with-adaLN design
  (after Peebles and Xie 2023, DiT) it follows.
"""

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import IS_FLUX, PROV_OF_VAR, STEPS_PER_DAY, V_BASE, V_TAB, VARS, WINDOW_DAYS
from imagery import CLS_DIM, N_META

K = STEPS_PER_DAY


def option_module(build, zero_prefix: Optional[str] = None):
    """Build an optional module without changing the base model's initialisation: it is constructed under a saved
    and restored random state (so the parameters built after it draw the same numbers as without it) and its
    parameters are set to zero (so the network computes the same function as the base at initialisation). An A/B arm
    with an option then starts from exactly the base weights. `zero_prefix`: zero only the parameters whose name
    starts with it (the output projection of a deeper module, whose hidden layers keep their random initialisation
    so that they can learn; the output is still zero at initialisation)."""
    with torch.random.fork_rng(devices=[]):
        m = build()
    for name, p in m.named_parameters():
        if zero_prefix is None or name.startswith(zero_prefix):
            nn.init.zeros_(p)
    return m


class SinrPlace(nn.Module):
    """Location encoder of SINR (Cole et al. 2023, "Spatial implicit neural representations for global-scale species
    mapping", ICML), as ported in the DeepEarth habitat model (models/habitat/plant/ranges/joint/place.py, SinrPlace):
    sin/cos of longitude and latitude, Linear(n_in, hidden) + ReLU, `blocks` residual blocks
    (Linear-ReLU-Linear-ReLU with a skip connection) and a zero-initialised projection to d.

    Additions for FluxFill: elevation (km) as a fifth input, and an optional ladder of frequencies (NeRF positional
    encoding, Mildenhall et al. 2020; Fourier features, Tancik et al. 2020): sin and cos of 2^k pi lon / 180 and
    2^k pi lat / 90 for k = 0..n_freq. n_freq = 0 is SINR's input. A site without a location (all-zero site
    features, the "unknown location" encoding) gets a zero output."""

    def __init__(self, d: int, n_freq: int = 0, hidden: int = 256, blocks: int = 4):
        super().__init__()
        self.n_freq = n_freq
        self.inp = nn.Sequential(nn.Linear(4 * (n_freq + 1) + 1, hidden), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                                   nn.ReLU()) for _ in range(blocks)])
        self.out = nn.Linear(hidden, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def encode(self, site: torch.Tensor) -> torch.Tensor:
        """(B, 4) unit-sphere xyz and elevation (data.site_features) -> (B, 4 (n_freq + 1) + 1)."""
        x, y, z, elev = site.float().unbind(-1)
        lat = torch.asin(z.clamp(-1.0, 1.0))                     # radians; pi lat / 90 in degrees is 2 lat here
        lon = torch.atan2(y, x)                                   # radians; pi lon / 180 in degrees is lon here
        f = [elev[:, None]]
        for k in range(self.n_freq + 1):
            w = 2.0 ** k
            f += [torch.sin(w * lon)[:, None], torch.cos(w * lon)[:, None], torch.sin(2 * w * lat)[:, None],
                  torch.cos(2 * w * lat)[:, None]]
        return torch.cat(f, -1)

    def forward(self, site: torch.Tensor) -> torch.Tensor:
        h = self.inp(self.encode(site))
        for blk in self.blocks:
            h = h + blk(h)
        known = (site != 0).any(-1, keepdim=True)
        return torch.where(known, self.out(h), torch.zeros_like(h[:, :1]).expand(-1, self.out.out_features))


def rope_tables(n: int, dim: int, base: float = 10000.0, device=None):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    f = torch.outer(torch.arange(n, device=device).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    """x (..., n, dim); rotate pairs (x[2i], x[2i+1])."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, key_ok: Optional[torch.Tensor] = None, rope=None):
        n, L, d = x.shape
        q, k, v = self.qkv(x).view(n, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        mask = None if key_ok is None else key_ok[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(n, L, d))


class Block(nn.Module):
    """Pre-norm attention + MLP over one axis of the (B, R, D, d) token grid."""

    def __init__(self, d: int, heads: int, axis: str, mlp: int = 4, dropout: float = 0.0):
        super().__init__()
        self.axis = axis
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, row_ok, rope):
        B, R, D, d = x.shape
        if self.axis == "time":
            y = self.attn(self.n1(x).reshape(B * R, D, d), rope=rope).view(B, R, D, d)
        else:
            xt = self.n1(x).transpose(1, 2).reshape(B * D, R, d)
            ok = row_ok[:, None, :].expand(B, D, R).reshape(B * D, R)
            y = self.attn(xt, key_ok=ok).view(B, D, R, d).transpose(1, 2)
        x = x + self.drop(y)
        return x + self.drop(self.mlp(self.n2(x)))


class FluxFillNet(nn.Module):
    def __init__(self, d: int = 192, depth: int = 6, heads: int = 6, mlp: int = 4, dropout: float = 0.0,
                 flow_width: int = 256, flow_depth: int = 3, use_imagery: bool = True, memory: bool = False,
                 mem_residual: bool = False, img_nll: bool = False, components: int = 1, local_base: bool = False,
                 gated_base: bool = False, gate_scale: bool = False, edge_continue: bool = False,
                 memory_skew: bool = False, driver_anomaly: bool = False, memory_skew_fluxes_only: bool = False,
                 scale_anchor: bool = False, year_anomaly: bool = False, sinr_place: int = -1,
                 sinr_replace: bool = False, scale_anchor_fch4: bool = False):
        super().__init__()
        # img_nll: the imagery head is a diagonal Gaussian (mean and log-scale per Clay dimension) trained on its
        # negative log-likelihood, a proper score; otherwise a point head trained on MSE (v0, kept for old checkpoints).
        self.img_nll = img_nll
        self.d, self.use_imagery, self.memory = d, use_imagery, memory
        # mem_residual: both heads predict relative to the site-memory mean (standardised), where memory exists, so
        # the default prediction is the site's own climatological level and the encoder learns the deviation from it.
        self.mem_residual = mem_residual and memory
        # local_base: where the visible values give a local mean diurnal course (the same half-hour on the days within
        # +/- LOCAL_DAYS, at least LOCAL_MIN of them), the residual base is that course; the site memory mean elsewhere.
        self.local_base = local_base and self.mem_residual
        # gated_base (E20): a two-component Student-t per half-hour, component 0 anchored at the local mean diurnal
        # course, component 1 at the memory mean, each with its own residual, scale and df; the mixture weight (the
        # gate) comes from the encoder state plus a linear term on the local evidence (count of visible local values,
        # log distance to the nearest visible value, memory and local availability). No local course: memory only.
        self.gated_base = gated_base and self.mem_residual
        if self.gated_base:
            components, self.local_base = 2, False
            self.gate_in = nn.Linear(4, 1)
        # gate_scale (E21): the same per-half-hour evidence also scales each component, scale_k * exp(linear_k(f)), so a
        # component's spread can depend on whether it acts alone (no local course) or beside the other one; zero-init.
        self.gate_scale = gate_scale and self.gated_base
        # edge_continue (E24): the local component's anchor continues the anomaly of the nearest visible step,
        # course(t) + A_edge * rho_v ** d(t), rho_v = sigmoid(edge_logit_v) per variable, initialised at 0.5 (the
        # lag-1 autocorrelation of NEE anomalies measured on visible data).
        self.edge_continue = edge_continue and self.gated_base
        if self.edge_continue:
            self.edge_logit = nn.Parameter(torch.zeros(V_TAB))
        # driver_anomaly (E25): both anchors add sum_k beta_k x anomaly_k(t) over the ERA5 drivers SW_IN, TA, VPD, where
        # anomaly_k is the driver minus its own local course (visible values only) and beta (per token, from the encoder
        # state) is zero-initialised, so the head starts identical and learns each site's day-to-day light, temperature
        # and VPD response.
        self.driver_anomaly = driver_anomaly and self.gated_base
        if self.driver_anomaly:
            self.driver_rows = [VARS.index(k) for k in ("ERA5_SW_IN", "ERA5_TA", "ERA5_VPD")]
            self.beta_out = option_module(lambda: nn.Linear(d, len(self.driver_rows)))
        # scale_anchor (E27): both components' scales are relative to the site memory's standard deviation at that day
        # of year and half-hour, scale_k * (mem_s + SCALE_FLOOR) where memory exists (scale_k * (1 + SCALE_FLOOR)
        # where it does not), as mem_residual makes the location relative to the memory mean; no parameters.
        self.scale_anchor = scale_anchor and self.gated_base
        # scale_anchor_fch4 (E33): the same, on the FCH4 row only, so a low-flux methane site gets widths on its own
        # scale (its memory spread) instead of the population's; no parameters.
        self.scale_anchor_fch4 = scale_anchor_fch4 and self.gated_base and "FCH4" in VARS
        if self.scale_anchor_fch4:
            self.register_buffer("fch4_row", torch.tensor(np.array(VARS) == "FCH4"), persistent=False)
        # year_anomaly (E28): the memory component's anchor adds this window's departure from the site's other years,
        # mem + c_near,v * A_near + c_win,v * A_win, where A is the mean of (visible value - memory mean) at the same
        # half-hour over the window's visible days, weighted exp(-|day distance| / YEAR_ANOM_DAYS) (A_near) or
        # uniformly (A_win); c is zero-initialised per variable, so the arm starts identical to the base. It carries
        # the year's state (a crop after harvest, a different crop, a disturbed stand) into long gaps.
        self.year_anomaly = year_anomaly and self.gated_base
        if self.year_anomaly:
            self.yanom_coef = nn.Parameter(torch.zeros(2, V_TAB))
        if self.gate_scale:
            self.scale_in = nn.Linear(4, 2)
            nn.init.zeros_(self.scale_in.weight); nn.init.zeros_(self.scale_in.bias)
        # memory_skew (E23): the skewness of the site memory as a fourth memory channel (scaled by 1/5, its clip).
        self.memory_skew = (memory_skew or memory_skew_fluxes_only) and memory
        # E23b: the skewness enters only the flux rows (NEE, LE, H, FCH4); zero elsewhere
        self.register_buffer("skew_rows", torch.tensor(IS_FLUX if memory_skew_fluxes_only else np.ones(V_TAB, bool)),
                             persistent=False)
        self.tab_in = nn.Linear((5 if memory else 2) * K, d)
        if self.memory_skew:     # its own projection, zero at initialisation, built without drawing random numbers
            self.skew_in = option_module(lambda: nn.Linear(K, d, bias=False))
        self.mask_tab = nn.Parameter(torch.zeros(d))
        self.img_norm = nn.LayerNorm(CLS_DIM, elementwise_affine=False)
        self.img_in = nn.Linear(CLS_DIM, d)
        self.img_meta = nn.Linear(N_META, d)
        self.mask_img = nn.Parameter(torch.zeros(d))
        # Variable embeddings: the base rows and the imagery row (var_emb, as in every checkpoint); with the extended
        # channel set (E30, data.CHANNELS) the extra rows get their own table, built by option_module (zero, and
        # without changing the random numbers of the base weights), placed between the base rows and imagery.
        self.var_emb = nn.Embedding(V_BASE + 1, d)
        if V_TAB > V_BASE:
            self.extra_emb = option_module(lambda: nn.Embedding(V_TAB - V_BASE, d))
        self.prov_emb = nn.Embedding(4, d)                  # tower, reanalysis, reanalysis-land, imagery
        self.day_emb = nn.Embedding(WINDOW_DAYS, d)
        self.doy_in = nn.Linear(4, d)
        self.site_in = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        # sinr_place (E29): SinrPlace with sinr_place frequency octaves (0: SINR's sin/cos input) added to the location
        # term (site_in) or, with sinr_replace, in its place; output projection zero at initialisation.
        self.sinr_place, self.sinr_replace = sinr_place, sinr_replace and sinr_place >= 0
        if sinr_place >= 0:
            self.place = option_module(lambda: SinrPlace(d, n_freq=sinr_place), zero_prefix="out.")
        self.register_buffer("prov", torch.tensor(list(PROV_OF_VAR) + [3]), persistent=False)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(Block(d, heads, "time", mlp, dropout))
            self.blocks.append(Block(d, heads, "var", mlp, dropout))
        self.norm = nn.LayerNorm(d)
        self.head_t = StudentTHead(d, components, spread=not gated_base)
        self.head_img = nn.Linear(d, (2 if img_nll else 1) * CLS_DIM)
        self.head_flow = FlowHead(d, flow_width, flow_depth)
        nn.init.normal_(self.mask_tab, std=0.02)
        nn.init.normal_(self.mask_img, std=0.02)
        cos, sin = rope_tables(WINDOW_DAYS, d // heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def encode(self, b: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Token states (B, V + 1, D, d); row V is imagery."""
        z, obs = b["z"], b["obs"].float()
        B = z.shape[0]
        parts = [z * obs, obs]
        if self.memory:                                                           # site memory channels
            mh = b["mem_h"].float()
            parts += [b["mem_m"] * mh, b["mem_s"] * mh, mh]
        tab = self.tab_in(torch.cat(parts, -1))                                   # (B, V, D, d)
        if self.memory_skew:
            tab = tab + self.skew_in(b["mem_k"] * b["mem_h"].float() * self.skew_rows.view(1, -1, 1, 1).float() / 5.0)
        empty = (obs.sum(-1, keepdim=True) == 0)
        # A token with no visible value: the learned mask vector, plus its memory channels when memory is on (the value
        # channels of such a token are zero, so tab - bias is exactly the memory part of the projection).
        tab = torch.where(empty, self.mask_tab.to(tab.dtype) + tab - self.tab_in.bias.to(tab.dtype), tab)
        io = b["img_obs"][..., None]
        img = self.img_in(self.img_norm(b["img_cls"])) + self.img_meta(b["img_meta"])
        img = torch.where(io, img, self.mask_img.to(img.dtype))[:, None]          # (B, 1, D, d)
        x = torch.cat([tab, img], 1)
        R = x.shape[1]
        x = x + self.var_table()[None, :, None] + self.prov_emb(self.prov)[None, :, None]
        x = x + self.day_emb.weight[None, None] + self.doy_in(b["doy"])[:, None]
        if not self.sinr_replace:
            x = x + self.site_in(b["site"])[:, None, None]
        if self.sinr_place >= 0:
            x = x + self.place(b["site"]).to(x.dtype)[:, None, None]
        img_ok = b["img_present"].reshape(B, 1) & torch.tensor(self.use_imagery, device=z.device)
        row_ok = torch.cat([b["present"], img_ok], 1)
        row_ok = row_ok | ~row_ok.any(1, keepdim=True)
        rope = (self.rope_cos, self.rope_sin)
        for blk in self.blocks:
            x = blk(x, row_ok, rope)
        return self.norm(x)

    def var_table(self) -> torch.Tensor:
        """(V_TAB + 1, d) embeddings of the rows in order: base variables, extra variables (E30), imagery."""
        w = self.var_emb.weight
        if V_TAB == V_BASE:
            return w
        return torch.cat([w[:V_BASE], self.extra_emb.weight, w[V_BASE:]], 0)

    def offset(self, b) -> torch.Tensor:
        """(B, V, D, 48) residual base, standardised: the local mean diurnal course (local_base) where it exists, else
        the site-memory mean where memory exists, else 0; zero without mem_residual."""
        if not self.mem_residual:
            return torch.zeros_like(b["z"])
        base = b["mem_m"] * b["mem_h"].float()
        if self.local_base:
            mean, ok = local_diurnal_course(b["z"], b["obs"])
            base = torch.where(ok, mean, base)
        return base

    def gated_params(self, b, params, h=None):
        loc, scale, df, logw = params
        mem = b["mem_m"] * b["mem_h"].float()
        lmean, ok, cnt = local_course_stats(b["z"], b["obs"])
        dist = nearest_visible_distance(b["obs"])
        feats = torch.stack([cnt / (2 * LOCAL_DAYS), torch.log1p(dist) / math.log(WINDOW_DAYS * K), b["mem_h"].float(),
                             ok.float()], -1)
        diff = (logw[..., 0] - logw[..., 1]) + self.gate_in(feats.to(logw.dtype))[..., 0].float()
        diff = torch.where(ok, diff, torch.full_like(diff, -30.0))          # no local course: memory component only
        logw = torch.log_softmax(torch.stack([diff, torch.zeros_like(diff)], -1), -1)
        local = lmean
        if self.edge_continue:
            A, dist_e = edge_anomaly(b["z"], b["obs"], lmean, ok)
            rho = torch.sigmoid(self.edge_logit).view(1, -1, 1, 1)
            local = lmean + A * torch.exp(dist_e * torch.log(rho))
        if self.year_anomaly:
            an, an_ok = year_anomalies(b["z"], b["obs"], b["mem_m"], b["mem_h"])
            c = self.yanom_coef.view(2, 1, -1, 1, 1)
            mem = torch.where(b["mem_h"], mem + c[0] * an[0] + c[1] * an[1], mem)
        base = torch.stack([torch.where(ok, local, mem), mem], -1)
        if self.driver_anomaly and h is not None:
            dr = self.driver_rows
            danom = torch.where(b["obs"][:, dr] & ok[:, dr], b["z"][:, dr] - lmean[:, dr], torch.zeros_like(lmean[:, dr]))
            beta = self.beta_out(h[:, :V_TAB]).float()                                   # (B, V, D, 3)
            shift = (beta[:, :, :, :, None] * danom[:, None].permute(0, 1, 3, 2, 4).float()).sum(3)   # (B, V, D, 48)
            base = base + shift[..., None]
        if self.gate_scale:
            scale = scale * torch.exp(self.scale_in(feats.to(scale.dtype)).float().clamp(-5.0, 5.0))
        if self.scale_anchor:
            s = torch.where(b["mem_h"], b["mem_s"], torch.ones_like(b["mem_s"])) + SCALE_FLOOR
            scale = scale * s[..., None].to(scale.dtype)
        if self.scale_anchor_fch4:
            s = torch.where(b["mem_h"], b["mem_s"], torch.ones_like(b["mem_s"])) + SCALE_FLOOR
            s = torch.where(self.fch4_row.view(1, -1, 1, 1), s, torch.ones_like(s))
            scale = scale * s[..., None].to(scale.dtype)
        return loc + base, scale, df, logw

    def forward(self, b):
        h = self.encode(b)
        params = self.head_t(h[:, :V_TAB])
        if self.gated_base:
            return h, self.gated_params(b, params, h)
        off = self.offset(b)
        loc = params[0] + (off[..., None] if params[0].dim() == off.dim() + 1 else off)
        return h, (loc,) + tuple(params[1:])


LOCAL_DAYS, LOCAL_MIN = 3, 2
SCALE_FLOOR = 0.02          # standardised units (0.02 population standard deviations)
YEAR_ANOM_DAYS = 10.0       # e-folding distance of the near-window anomaly (E28)


def year_anomalies(z, obs, mem_m, mem_h, tau: float = YEAR_ANOM_DAYS):
    """(2, B, V, D, 48) mean departure of the visible values from the memory mean at the same half-hour, over the
    window's days weighted exp(-|d - d'| / tau) (index 0) or uniformly (index 1), and (B, V, D, 48) where any
    visible value entered. Only visible values with memory enter (z * obs), so no hidden value is read."""
    D = z.shape[2]
    o = (obs & mem_h).float()
    a = (z - mem_m) * o
    dd = torch.arange(D, device=z.device, dtype=torch.float32)
    W = torch.stack([torch.exp(-(dd[:, None] - dd[None, :]).abs() / tau), torch.ones(D, D, device=z.device)])
    num = torch.einsum("wde,bvek->wbvdk", W, a.float())
    den = torch.einsum("wde,bvek->wbvdk", W, o)
    ok = den[1] > 0
    return torch.where(den > 0, num / den.clamp_min(1e-6), torch.zeros_like(num)), ok


def local_diurnal_course(z, obs, days: int = LOCAL_DAYS, min_count: int = LOCAL_MIN):
    mean, ok, _ = local_course_stats(z, obs, days, min_count)
    return mean, ok


def nearest_visible_distance(obs):
    """(B, V, D, 48) distance in half-hours from each step to the nearest visible step of the same variable in the
    window (the window length where the variable has none)."""
    B, V, D, K_ = obs.shape
    T = D * K_
    o = obs.reshape(B, V, T)
    t = torch.arange(T, device=obs.device).expand(B, V, T)
    big = torch.full_like(t, -10 * T)
    last = torch.cummax(torch.where(o, t, big), -1).values
    nxt = (T - 1) - torch.cummax(torch.where(o.flip(-1), t, big), -1).values.flip(-1)
    nxt = torch.where(nxt > 2 * T, torch.full_like(nxt, 10 * T), nxt)
    d = torch.minimum(t - last, nxt - t).clamp(0, T)
    return d.reshape(B, V, D, K_).float()


def edge_anomaly(z, obs, course, ok):
    """Anomaly at the nearest visible step of each variable and the distance to it, both (B, V, D, 48).

    The anomaly of a visible step is its value minus the local course there (0 where that step has no local course);
    every step takes the anomaly of its nearest visible step (the earlier one on a tie) and the distance in
    half-hours. Only visible values enter, so no hidden value is read."""
    B, V, D, K_ = z.shape
    T = D * K_
    o, zf, cf, kf = (x.reshape(B, V, T) for x in (obs, z, course, ok))
    an = torch.where(o & kf, zf - cf, torch.zeros_like(zf))
    t = torch.arange(T, device=z.device).expand(B, V, T)
    big = 10 * T
    last = torch.cummax(torch.where(o, t, torch.full_like(t, -big)), -1).values
    nxt = (T - 1) - torch.cummax(torch.where(o.flip(-1), t, torch.full_like(t, -big)), -1).values.flip(-1)
    d_last = torch.where(last >= 0, t - last, torch.full_like(t, big))
    d_next = torch.where(nxt <= T - 1, nxt - t, torch.full_like(t, big))
    use_last = d_last <= d_next
    e = torch.where(use_last, last, nxt).clamp(0, T - 1)
    dist = torch.minimum(d_last, d_next)
    A = torch.gather(an, -1, e)
    A = torch.where(dist < big, A, torch.zeros_like(A))
    return A.reshape(B, V, D, K_), dist.clamp(max=T).float().reshape(B, V, D, K_)


def local_course_stats(z, obs, days: int = LOCAL_DAYS, min_count: int = LOCAL_MIN):
    """Mean of the visible values at the same half-hour on the days within +/- `days` (the day itself excluded), and
    where at least `min_count` of them exist. z, obs: (B, V, D, 48); returns (mean, ok) of the same shape. Uses only
    visible values (z * obs), so it never reads a hidden value."""
    o = obs.float()
    zo = z * o
    S, C = torch.zeros_like(zo), torch.zeros_like(o)
    D = z.shape[2]
    for k in range(1, days + 1):
        if k >= D:
            break
        S[:, :, k:] += zo[:, :, :-k]; C[:, :, k:] += o[:, :, :-k]          # day d - k
        S[:, :, :-k] += zo[:, :, k:]; C[:, :, :-k] += o[:, :, k:]          # day d + k
    ok = C >= min_count
    return torch.where(ok, S / C.clamp_min(1.0), torch.zeros_like(S)), ok, C


class StudentTHead(nn.Module):
    """(B, V, D, d) -> location, scale, degrees of freedom, each (B, V, D, 48).

    With `components` = M > 1, a mixture of M Student-t per half-hour (a mixture density network, Bishop 1994,
    NCRG/94/004): location, scale and df gain a trailing component axis (B, V, D, 48, M) and a fourth tensor holds
    the mixture log-weights. A mixture can be skewed and multimodal, and its median can differ from its mean."""

    def __init__(self, d: int, components: int = 1, spread: bool = True):
        super().__init__()
        self.m, self.spread = components, spread
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, (3 if components == 1 else 4 * components) * K))

    def forward(self, h):
        if self.m == 1:
            o = self.mlp(h).float().unflatten(-1, (3, K))
            loc = o[..., 0, :]
            scale = F.softplus(o[..., 1, :]) + 1e-3
            df = 2.0 + 1e-2 + F.softplus(o[..., 2, :] + 2.0)
            return loc, scale, df
        o = self.mlp(h).float().unflatten(-1, (4, K, self.m))
        # spread the initial component locations so the components do not start identical
        spread = torch.linspace(-0.5, 0.5, self.m, device=o.device) if self.spread else 0.0
        loc = o[..., 0, :, :] + spread
        scale = F.softplus(o[..., 1, :, :]) + 1e-3
        df = 2.0 + 1e-2 + F.softplus(o[..., 2, :, :] + 2.0)
        logw = torch.log_softmax(o[..., 3, :, :], -1)
        return loc, scale, df, logw


def marginal_nll(y, params):
    """Negative log-likelihood of y (..., 48) under a Student-t or a Student-t mixture (head output)."""
    if len(params) == 3:
        return student_t_nll(y, *params)
    loc, scale, df, logw = params
    return -torch.logsumexp(logw - student_t_nll(y[..., None], loc, scale, df), -1)


def mixture_cdf(params, y):
    """Exact predictive CDF F(y) (numpy) of a Student-t head (loc, scale, df) or a mixture head (loc, scale, df,
    logw; component axis last). A central interval of level q contains y exactly when (1 - q) / 2 <= F(y) <= (1 + q) / 2,
    so coverage and the PIT follow from F(y) without sampling."""
    from scipy import stats as sst
    y = np.asarray(y, np.float64)
    if len(params) == 3:
        loc, scale, df = (np.asarray(u, np.float64) for u in params)
        return sst.t.cdf((y - loc) / scale, df)
    loc, scale, df, logw = (np.asarray(u, np.float64) for u in params)
    return (np.exp(logw) * sst.t.cdf((y[..., None] - loc) / scale, df)).sum(-1)


def marginal_mean(params):
    if len(params) == 3:
        return params[0]
    loc, _, _, logw = params
    return (logw.exp() * loc).sum(-1)


def sample_marginal_params(params, n: int):
    """n independent draws per half-hour from the head output, (n, ...)."""
    if len(params) == 3:
        return sample_student_t(*params, n)
    loc, scale, df, logw = params
    c = torch.distributions.Categorical(logits=logw).sample((n,))            # (n, ..., 48)
    g = lambda x: torch.gather(x.expand((n,) + x.shape), -1, c[..., None])[..., 0]
    return sample_student_t(g(loc), g(scale), g(df), 1)[0]


def imagery_scores(net, h_img, target):
    """Scores of imagery reconstruction. h_img (N, d) token states, target (N, 1024) layer-normalised Clay CLS.
    Returns nll (diagonal Gaussian, mean over dimensions; None for a point head), mse and cosine of the mean."""
    o = net.head_img(h_img).float()
    mu = o[:, :CLS_DIM]
    nll = None
    if net.img_nll:
        log_s = o[:, CLS_DIM:].clamp(-7.0, 5.0)
        nll = (0.5 * math.log(2 * math.pi) + log_s + 0.5 * ((target - mu) * torch.exp(-log_s)) ** 2).mean()
    mse = ((mu - target) ** 2).mean()
    cos = F.cosine_similarity(mu, target, dim=-1).mean()
    return nll, mse, cos


def student_t_nll(y, loc, scale, df):
    z = (y - loc) / scale
    return (torch.lgamma(df / 2) - torch.lgamma((df + 1) / 2) + 0.5 * torch.log(df * math.pi) + torch.log(scale)
            + 0.5 * (df + 1) * torch.log1p(z * z / df))


def sample_student_t(loc, scale, df, n: int, generator=None):
    """n independent draws per element: loc + scale * N / sqrt(Chi2(df) / df)."""
    shape = (n,) + tuple(loc.shape)
    g = torch.randn(shape, device=loc.device, generator=generator)
    chi2 = torch.distributions.Chi2(df.expand(shape)).sample()
    return loc + scale * g / torch.sqrt(chi2 / df)


# ------------------------------------------------------------------------------------------------- flow head (v1)

def timestep_embedding(t, dim: int):
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    a = t.float()[:, None] * 1000.0 * f[None]
    return torch.cat([a.cos(), a.sin()], -1)


class AdaLNBlock(nn.Module):
    def __init__(self, w: int):
        super().__init__()
        self.norm = nn.LayerNorm(w, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(w, 3 * w))
        nn.init.zeros_(self.mod[1].weight); nn.init.zeros_(self.mod[1].bias)

    def forward(self, x, c):
        shift, scale, gate = self.mod(c).chunk(3, -1)
        return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class FlowHead(nn.Module):
    """Velocity field v(x_t, t | token state, visible values) for one day-token's 48 values."""

    def __init__(self, d: int, w: int = 256, depth: int = 3):
        super().__init__()
        self.x_in = nn.Linear(3 * K, w)
        self.c_in = nn.Linear(d, w)
        self.t_in = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
        self.blocks = nn.ModuleList([AdaLNBlock(w) for _ in range(depth)])
        self.norm = nn.LayerNorm(w, elementwise_affine=False)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(w, 2 * w))
        self.out = nn.Linear(w, K)
        self.w = w
        nn.init.zeros_(self.mod[1].weight); nn.init.zeros_(self.mod[1].bias)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, xt, t, cond, zvis, obs):
        """xt, zvis, obs (N, 48); t (N,); cond (N, d)."""
        x = self.x_in(torch.cat([xt, zvis * obs, obs], -1))
        c = self.c_in(cond) + self.t_in(timestep_embedding(t, self.w))
        for blk in self.blocks:
            x = blk(x, c)
        shift, scale = self.mod(c).chunk(2, -1)
        return self.out(self.norm(x) * (1 + scale) + shift)

    def loss(self, cond, x1, tmask, zvis, obs, reps: int = 4, rows=None):
        """Conditional flow-matching loss on the target elements of N tokens, each repeated `reps` times with its
        own time and noise (as Li et al. 2024 repeat the diffusion loss). rows (N,): variable of each token, for the
        per-variable mean (E46); None weighs every element equally."""
        cond, x1, tmask, zvis, obs = (u.repeat(reps, 1) for u in (cond, x1, tmask, zvis, obs))
        t = torch.rand(x1.shape[0], device=x1.device)
        x0 = torch.randn_like(x1)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        v = self.forward(xt, t, cond, zvis, obs).float()
        err = ((v - (x1 - x0)) ** 2) * tmask
        if rows is not None:
            r = rows.repeat(reps)
            s = torch.zeros(V_TAB, device=err.device).index_add(0, r, err.sum(-1))
            n = torch.zeros(V_TAB, device=err.device).index_add(0, r, tmask.sum(-1))
            return (s / n.clamp_min(1)).sum() / V_TAB
        return err.sum() / tmask.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, cond, zvis, obs, steps: int = 24, generator=None):
        """Midpoint ODE integration from t = 0 (noise) to t = 1."""
        x = torch.randn(cond.shape[0], K, device=cond.device, generator=generator)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            k1 = self.forward(x, t, cond, zvis, obs).float()
            xm = x + 0.5 * dt * k1
            k2 = self.forward(xm, t + 0.5 * dt, cond, zvis, obs).float()
            x = x + dt * k2
        return x
