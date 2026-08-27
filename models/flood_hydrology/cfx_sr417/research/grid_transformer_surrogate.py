#!/usr/bin/env python3
"""GridTransformerSurrogate — a minimal FloodSformer-family model: fully-convolutional
autoencoder (spatial compression) + a small cross-attention Transformer (temporal prediction,
conditioned on a scalar forcing input). See build_grid_surrogate_dataset_site3.py's docstring
for why this exists and what it's being compared against.

Deliberately minimal, not a reproduction of FloodSformer's full architecture (no GAN loss on the
autoencoder, no VPTR-specific attention variant) — the point of this experiment is the
SCALE/ARCHITECTURE comparison against our own mesh-GNN result, not chasing FloodSformer's own
reported accuracy. Honest proof-of-concept, consistent with this project's stated ethos of real,
proportionate implementations rather than oversold ones.

Resolution-agnostic by construction: every layer is either a strided conv (kernel-size-only,
works at any input size) or attention over a token count derived from the actual input's shape
at forward time (no fixed-size learned position table). This is required for the experiment
itself — the same trained weights must run, unmodified, on training-resolution grids AND on
site3's true full-resolution grid for the inference-cost benchmark.

Downsample factor: 5 stride-2 conv layers -> 32x. Callers must pad H, W up to a multiple of 32
before calling forward() (see `pad_to_multiple`/`crop_to` below).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DOWNSAMPLE = 32   # 2**5


def pad_to_multiple(x, multiple=DOWNSAMPLE):
    """Zero-pad the last two dims of x up to the next multiple of `multiple`. Returns
    (padded, (orig_h, orig_w)) so the caller can crop back with crop_to()."""
    h, w = x.shape[-2], x.shape[-1]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    return x, (h, w)


def crop_to(x, hw):
    h, w = hw
    return x[..., :h, :w]


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, transpose=False):
        super().__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(8, c_out), c_out)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvAutoencoder(nn.Module):
    """5-layer strided-conv encoder/decoder. 1-channel depth frame <-> [C, H/32, W/32] latent."""

    def __init__(self, channels=(16, 32, 64, 64, 64)):
        super().__init__()
        enc = []
        c_prev = 1
        for c in channels:
            enc.append(ConvBlock(c_prev, c))
            c_prev = c
        self.encoder = nn.Sequential(*enc)
        self.latent_channels = channels[-1]

        dec = []
        rev = list(reversed(channels))
        for i in range(len(rev) - 1):
            dec.append(ConvBlock(rev[i], rev[i + 1], transpose=True))
        dec.append(nn.ConvTranspose2d(rev[-1], 1, kernel_size=4, stride=2, padding=1))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return F.relu(self.decoder(z))   # depth >= 0


def sinusoidal_pos_embed_2d(h, w, dim, device):
    """Standard 2D sinusoidal position embedding, computed at CALL time from the actual latent
    grid shape — resolution-agnostic (no learned table tied to one H, W)."""
    assert dim % 4 == 0
    d4 = dim // 4
    y = torch.arange(h, device=device).float()
    x = torch.arange(w, device=device).float()
    div = torch.exp(torch.arange(0, d4, device=device).float() * (-math.log(10000.0) / d4))
    pe_y = torch.cat([torch.sin(y[:, None] * div), torch.cos(y[:, None] * div)], dim=1)  # [h, d4*2]
    pe_x = torch.cat([torch.sin(x[:, None] * div), torch.cos(x[:, None] * div)], dim=1)  # [w, d4*2]
    pe = torch.cat([
        pe_y[:, None, :].expand(h, w, d4 * 2),
        pe_x[None, :, :].expand(h, w, d4 * 2),
    ], dim=-1)   # [h, w, dim]
    return pe.reshape(h * w, dim)


class GridTransformerSurrogate(nn.Module):
    """Autoencoder + a single cross-attention block: a "predict the next latent frame" query
    (initialized from the most recent context frame's own latent, i.e. residual prediction)
    attends over K past latent frames' tokens (each carrying a time-index + 2D spatial position
    embedding). A scalar forcing value (design-storm rain rate, mm/hr, log1p-scaled) is added to
    the query tokens — FloodSformer's own "inflow discharge" cross-attention conditioning,
    adapted to this solver's actual boundary forcing (rain rate, not inflow discharge — this
    project's grid solver has no inflow boundary condition; see the watershed-mismatch
    entries)."""

    def __init__(self, n_context=4, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.ae = ConvAutoencoder(channels=(16, 32, d_model, d_model, d_model))
        assert self.ae.latent_channels == d_model
        self.n_context = n_context
        self.d_model = d_model

        self.time_embed = nn.Embedding(n_context, d_model)
        self.forcing_mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model,
            dropout=0.0, batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, past_frames, forcing_mm_hr):
        """past_frames: [B, K, 1, H, W] (already padded to a multiple of 32).
        forcing_mm_hr: [B] scalar rain rate driving the transition to the predicted frame.
        Returns: predicted_frame [B, 1, H, W] (same padded size), next_latent [B, C, H', W']."""
        B, K, _, H, W = past_frames.shape
        device = past_frames.device
        assert K == self.n_context

        # Encode every context frame -> latents [B, K, C, H', W']
        flat = past_frames.reshape(B * K, 1, H, W)
        z = self.ae.encode(flat)
        _, C, Hp, Wp = z.shape
        z = z.reshape(B, K, C, Hp, Wp)

        pos = sinusoidal_pos_embed_2d(Hp, Wp, C, device)          # [H'*W', C]
        n_tok = Hp * Wp

        # Context tokens: every (time, space) pair, + time embedding + spatial position.
        ctx = z.permute(0, 1, 3, 4, 2).reshape(B, K, n_tok, C)     # [B, K, HW, C]
        t_idx = torch.arange(K, device=device)
        ctx = ctx + self.time_embed(t_idx)[None, :, None, :] + pos[None, None, :, :]
        ctx = ctx.reshape(B, K * n_tok, C)

        # Query: last context frame's own latent tokens (residual init) + position + forcing.
        last_latent = z[:, -1]                                    # [B, C, H', W']
        query = last_latent.permute(0, 2, 3, 1).reshape(B, n_tok, C)
        force = self.forcing_mlp(torch.log1p(forcing_mm_hr).reshape(B, 1, 1))  # [B,1,C] -> broadcast
        query = query + pos[None, :, :] + force

        out = self.decoder(tgt=query, memory=ctx)                 # [B, HW, C]
        delta = out.reshape(B, Hp, Wp, C).permute(0, 3, 1, 2)      # [B, C, H', W']
        next_latent = last_latent + delta                         # residual prediction

        pred = self.ae.decode(next_latent)                        # [B, 1, H, W]
        return pred, next_latent
