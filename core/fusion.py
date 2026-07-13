"""DeepEarth: a config-driven model of spatio-temporally covarying variables.

Given whichever variables are observed at a location plus those at nearby places/times, it infers the rest,
trained by masked reconstruction so any variable predicts any other. Space-time enters via Earth4D through two
channels: an absolute (coarse regional memory) and a relative (neighbor-offset, transferring across place/time).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepearth.encoders.spacetime.earth4d import Earth4D
from deepearth.encoders.biological.phylogenomic import SpeciesGraph


@dataclass
class Variable:
    """One variable DeepEarth models at each observation.

    kind: "continuous" (vector, cosine-reconstructed) or "categorical" (class label, probability-reconstructed).
    dim/num_classes: widths. reconstruct: a target (False = input-only). neighbor: also carried from neighbors.
    """
    name: str
    kind: str
    dim: int = 0
    num_classes: int = 0
    reconstruct: bool = True
    neighbor: bool = False


class SpaceTimeField(nn.Module):
    """Encode each neighbor's space-time offset from the query via Earth4D in relative mode."""

    def __init__(self, d_model: int, window: Sequence[float], levels: int = 24, reference_latitude_deg: float = 0.0,
                 finest: Sequence[float] = (0.1, 0.1, 1.0, 0.042), log2_hashmap_size: int = 22):
        super().__init__()
        # relative-only: skip absolute projections; the relative encoder carries high-frequency local structure
        self.earth4d = Earth4D(verbose=False, enable_relative=True, enable_absolute=False,
                               relative_window=tuple(window), relative_finest=tuple(finest),
                               relative_levels=levels, relative_log2_hashmap_size=log2_hashmap_size)
        self.proj = nn.Sequential(nn.Linear(self.earth4d.relative_output_dim, d_model), nn.GELU(),
                                  nn.Linear(d_model, d_model))
        self.m_per_deg = 111_320.0
        self.m_per_deg_lon = 111_320.0 * math.cos(math.radians(reference_latitude_deg))

    def forward(self, query_coords: torch.Tensor, neighbor_coords: torch.Tensor) -> torch.Tensor:
        delta = neighbor_coords - query_coords.unsqueeze(1)
        offset = torch.stack([delta[..., 0] * self.m_per_deg, delta[..., 1] * self.m_per_deg_lon,
                              delta[..., 2], delta[..., 3]], dim=-1)
        return self.proj(self.earth4d.encode_relative(offset))


class ManifoldField(nn.Module):
    """Encode each neighbor's own position within a vector subspace, such as an evolutionary manifold."""

    def __init__(self, d_model: int, dim: int, hidden: int = 256):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, d_model))

    def forward(self, neighbor_positions: torch.Tensor) -> torch.Tensor:
        return self.encode(neighbor_positions)


class SmoothGeoField(nn.Module):
    """Random Fourier Features over (lat, lon, elev, time): a SMOOTH, continuous geo encoding that GENERALIZES to
    held-out locations, unlike the memorizing hash grid whose fine cells are untrained at unseen points. A transferable
    low-frequency spatial prior aimed at A1 (species-from-geography under spatial holdout). Complements, not replaces,
    the absolute hash memory. (Rahimi & Recht RFF / Tancik Fourier-features; the SatCLIP idea of a generalizing geo prior.)
    """

    def __init__(self, d_model: int, per_scale: int = 32, sigmas: Sequence[float] = (1.0, 4.0, 16.0, 64.0)):
        super().__init__()
        # HIERARCHICAL sigma-bank (Hybrid-SDM / Fourier-features): each scale is a fixed Gaussian projection giving a
        # shift-invariant RBF kernel at that bandwidth; low sigma = broad/transferable, high = fine. Multi-scale coverage
        # coarse->fine without per-cell parameters, so it generalizes to held-out locations by construction.
        B = torch.cat([torch.randn(4, per_scale) * s for s in sigmas], dim=1)   # [4, per_scale*n_scales]
        self.register_buffer("B", B)
        self.register_buffer("coord_scale", torch.tensor([1 / 90.0, 1 / 180.0, 1 / 3000.0, 1 / 60.0]))  # lat°, lon°, elev m, time
        n_features = 2 * per_scale * len(sigmas)                               # cos+sin over every scale's directions
        self.proj = nn.Sequential(nn.Linear(n_features, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = (coords * self.coord_scale) @ self.B * (2.0 * math.pi)             # [B, per_scale*n_scales]
        return self.proj(torch.cat([torch.cos(x), torch.sin(x)], dim=-1))       # [B, d_model]


class NeighborContext(nn.Module):
    """One token per (neighbor, subspace) = subspace encoding + neighbor feature projections + a subspace marker.

    space_time: kwargs for SpaceTimeField; manifolds: {name: dim} vector subspaces; feature_dims: {name: dim} features.
    """

    def __init__(self, d_model: int, space_time: dict, manifolds: Dict[str, int] | None = None,
                 feature_dims: Dict[str, int] | None = None):
        super().__init__()
        self.d_model = d_model
        self.space_time = SpaceTimeField(d_model, **space_time)
        self.manifolds = nn.ModuleDict({name: ManifoldField(d_model, dim) for name, dim in (manifolds or {}).items()})
        self.features = nn.ModuleDict({name: nn.Linear(dim, d_model) for name, dim in (feature_dims or {}).items()})
        self.field_marker = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(d_model) * 0.02) for name in ["space_time", *(manifolds or {})]})

    def forward(self, query_coords: torch.Tensor, neighbor_coords: torch.Tensor,
                manifold_positions: Dict[str, torch.Tensor] | None = None,
                neighbor_features: Dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        B, K = neighbor_coords.shape[0], neighbor_coords.shape[1]
        features = query_coords.new_zeros(B, K, self.d_model)
        for name, val in (neighbor_features or {}).items():
            features = features + self.features[name](val)
        tokens = [self.space_time(query_coords, neighbor_coords) + features + self.field_marker["space_time"]]
        for name, field in self.manifolds.items():
            tokens.append(field(manifold_positions[name]) + features + self.field_marker[name])
        return torch.cat(tokens, dim=1)


class DeepEarth(nn.Module):
    """Config-driven model of spatio-temporally covarying variables (see module docstring).

    variables: the modeled variables. d_model/n_latents/n_layers/n_heads: latent-attention backbone.
    relative_window: neighbor-offset half-extent per axis (m, time). manifolds: extra vector subspaces {name: dim}.
    """

    # Train-only: blank the neighbor community on a fraction of rows so the model learns universal->biology induction rather than leaning on a community crutch absent at eval.
    COMMUNITY_DROPOUT = 0.5

    def __init__(
        self,
        variables: Sequence[Variable],
        *,
        d_model: int = 256,
        n_latents: int = 24,
        n_layers: int = 4,
        n_heads: int = 8,
        relative_window: Sequence[float] = (2500.0, 2500.0, 300.0, 180.0),
        relative_finest: Sequence[float] = (0.1, 0.1, 1.0, 0.042),
        relative_log2_hashmap_size: int = 22,
        manifolds: Optional[Dict[str, int]] = None,
        capacity: int = 16,
        reference_latitude_deg: float = 0.0,
        species_variable: Optional[str] = None,
        species_embedding: Optional[torch.Tensor] = None,
        species_layers: int = 2,
        species_heads: int = 4,
        species_top_k: Optional[int] = None,
        species_flex: bool = False,
        species_operator: str = "ou-attention",
        species_tree: Optional[dict] = None,
        species_text: Optional[torch.Tensor] = None,
        compile_processor: bool = False,
        rounds: int = 1,
        write_back: bool = True,
        revise: bool = False,
        round_loss: str = "final",
        learned_mask: Optional[bool] = None,
        feedback_detach: bool = False,
        flex_attention: bool = False,
        decoder_hidden: Optional[int] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        contrastive_weight: float = 0.0,
        contrastive_vars: Optional[Sequence[str]] = None,
        smooth_geo: bool = False,
        smooth_geo_sigmas: Optional[Sequence[float]] = None,
        smooth_geo_per_scale: int = 32,
        n_pollinators: int = 0,
        pollinator_distance: Optional[torch.Tensor] = None,
        pollinator_text: Optional[torch.Tensor] = None,
        pollinator_top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.loss_weights = loss_weights or {}
        self.contrastive_weight = contrastive_weight
        self.contrastive_vars = set(contrastive_vars or ())
        self.variables = list(variables)
        self.names = [v.name for v in self.variables]
        self.d_model = d_model

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        # Interface decoder factory (science.md rule 23: rich interface decoders reading from the latent field).
        # decoder_hidden=None -> a single Linear (lean); set -> a 1-hidden-layer MLP for richer reconstruction.
        def _dec(out_dim):
            return (nn.Sequential(nn.Linear(d_model, decoder_hidden), nn.GELU(), nn.Linear(decoder_hidden, out_dim))
                    if decoder_hidden else nn.Linear(d_model, out_dim))
        for v in self.variables:
            if v.kind == "continuous":
                self.encoders[v.name] = nn.Linear(v.dim, d_model)
                if v.reconstruct:
                    self.decoders[v.name] = _dec(v.dim)
            elif v.kind == "categorical":
                if v.name == species_variable and species_embedding is not None:
                    continue          # species tokens and logits use the refined species graph, not these dead heads
                self.encoders[v.name] = nn.Embedding(v.num_classes, d_model)
                if v.reconstruct:
                    self.decoders[v.name] = _dec(v.num_classes)
            else:
                raise ValueError(f"unknown variable kind {v.kind!r} for {v.name!r}")
        self.type_emb = nn.Parameter(torch.randn(len(self.variables), d_model) * 0.02)
        # A dedicated always-present space-time token, so a query revealing no variable keeps its position (variable tokens are zeroed by the present-mask).
        self.position_token = nn.Parameter(torch.randn(d_model) * 0.02)
        # Normalize content and position to matched unit scale before adding (token-embedding + PE practice): both stay legible and the absolute-encoder gradient stays bounded.
        self.tok_norm = nn.LayerNorm(d_model)
        self.pos_norm = nn.LayerNorm(d_model)
        self.decode_query = nn.Parameter(torch.randn(len(self.variables), d_model) * 0.02)

        # Absolute location memory: coarse regional/long-period memorization (~200M, 20% of Earth4D); fine structure lives in the relative encoder.
        self.absolute_encoder = Earth4D(verbose=False, spatial_levels=18, temporal_levels=18,
                                        spatial_log2_hashmap_size=20, temporal_log2_hashmap_size=20,
                                        freq_log_scale_init=-2.5)   # start coarse (~1 km finest); learned from there
        # Project Earth4D's [xyz | xyt|yzt|xzt] as separate spatial/spatiotemporal channels; each variable learns a
        # softmax prior over which it reads, so time-invariant modalities can shut time out while vision keeps it.
        self.absolute_proj_s = nn.Sequential(nn.Linear(self.absolute_encoder.spatial_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.absolute_proj_t = nn.Sequential(nn.Linear(self.absolute_encoder.spatiotemporal_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        gate0 = torch.zeros(len(self.variables), 2); gate0[:, 0] = 2.0   # init ~0.88 spatial / 0.12 temporal
        self.pos_channel_gate = nn.Parameter(gate0)
        # Smooth transferable geo prior (RFF): added to the memorizing hash position -> generalizes to held-out regions (A1).
        self.smooth_geo = SmoothGeoField(
            d_model, per_scale=smooth_geo_per_scale,
            sigmas=tuple(smooth_geo_sigmas) if smooth_geo_sigmas else (1.0, 4.0, 16.0, 64.0),
        ) if smooth_geo else None
        # neighbor context over coordinate subspaces: space-time, plus any vector manifolds (e.g. biological)
        neighbor_dims = {v.name: (v.dim if v.kind == "continuous" else d_model)
                         for v in self.variables if v.neighbor}
        self.neighbor_emb = nn.ModuleDict(
            {v.name: nn.Embedding(v.num_classes, d_model) for v in self.variables if v.neighbor and v.kind == "categorical"})
        self.neighbors = NeighborContext(
            d_model, space_time=dict(window=relative_window, levels=18, finest=relative_finest,
                                     log2_hashmap_size=relative_log2_hashmap_size,
                                     reference_latitude_deg=reference_latitude_deg),
            manifolds=manifolds, feature_dims=neighbor_dims)

        # species graph: refine identity through phylogenetic-neighbor attention. "tree" propagates over the dated phylogeny; "ou-attention" biases attention with an embedding-derived distance.
        self.species_variable = species_variable
        if species_variable is not None and species_embedding is not None:
            if species_operator == "tree":
                assert species_tree is not None, "species_operator='tree' needs the parsed tree (source.tree)"
                self.species_graph = SpeciesGraph(species_embedding.shape[0], d_model, operator="tree",
                                                  tree=species_tree, n_layers=species_layers, species_text=species_text)
            else:
                distance = SpeciesGraph.distance_from_embedding(species_embedding)
                self.species_graph = SpeciesGraph(species_embedding.shape[0], d_model, distance,
                                                  n_heads=species_heads, n_layers=species_layers,
                                                  top_k=species_top_k, flex=species_flex, species_text=species_text)
        else:
            self.species_graph = None
        self._refined_species = None

        # Optional modules (scale-mixing, diffusion, experience, inductive) live on the research branch; kept inert here so the forward's guards take the no-op path.
        self.scale_mixer = None
        self.diffusion_heads = nn.ModuleDict()
        self.experience = None
        self._memory_key = None
        self._memory_features = None
        self.inductive = None

        # latent-attention backbone
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.read = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.q_norm = nn.LayerNorm(d_model); self.kv_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                                                batch_first=True, norm_first=True) for _ in range(n_layers)])

        # Iterative joint-field denoising: refine the state tokens over K rounds (read -> latent self-attn -> write back
        # each variable's belief through its interface decoder). rounds=1 with learned_mask off is the single-shot model.
        self.rounds = rounds
        self.write_back = write_back
        self.revise = revise
        self.round_loss = round_loss
        self.learned_mask = (rounds > 1) if learned_mask is None else learned_mask
        self.feedback_detach = feedback_detach
        self.flex_attention = flex_attention
        nv = len(self.variables)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)      # a masked slot's placeholder (vs the zero vector)
        self.update_gate = nn.Parameter(torch.zeros(nv))                 # sigmoid=0.5: gain re-injecting a masked belief
        self.revise_rate = nn.Parameter(torch.full((nv,), -3.0))        # sigmoid~0.05: how far an observed token revises
        self.read_gate = nn.Parameter(torch.ones(d_model))              # per-dim gate on each round's latent read (identity at init)
        self._round_stack = None
        # Community-distribution head (MADE joint-dist as a jointly-trained benchmark head): a SEPARATE readout trained
        # on a DETACHED latent toward the LOCAL community distribution. Created LAST so it consumes no init RNG ahead of
        # the backbone -> the shared model initializes bit-identically to the no-head champion (zero regression on
        # B1-B19/B7 by construction), while supplying the community distribution for B20-22/B39/B40 at test.
        self.comm_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)) \
            if (species_variable is not None and species_embedding is not None) else None
        # Pollinator-distribution head (plant->pollinator interaction, MADE joint-dist benchmark head): a detached readout
        # from the species-pooled latent into a learned pollinator-vocab basis, trained toward the plant's local GloBI
        # pollinator distribution. Created LAST (zero init-RNG shift) -> supplies B41/B51-B54 with no backbone regression.
        self.poll_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)) if n_pollinators > 0 else None
        self.poll_emb = nn.Parameter(torch.randn(n_pollinators, d_model) * 0.02) if n_pollinators > 0 else None
        # Cross-tree interaction (rule 27): decode the plant->pollinator interaction against a SECOND, separately
        # phylo-refined pollinator species graph (its own tree), so an observed interaction propagates to BOTH sides'
        # relatives. Falls back to the free poll_emb table until the pollinator tree is wired.
        self.pollinator_graph = SpeciesGraph(n_pollinators, d_model, pollinator_distance, top_k=pollinator_top_k,
                                             species_text=pollinator_text) \
            if (n_pollinators > 0 and pollinator_distance is not None) else None
        # Ecophysiology head (B34, jointly-trained benchmark head on a DETACHED latent): predict a species' peak
        # fire-season live fuel moisture from its phylo-refined representation. Created last (no init-RNG shift).
        self.lfmc_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)) \
            if species_variable is not None else None
        # Symbiosis head (B42, jointly-trained on a DETACHED pooled latent): predict a plant's mycorrhizal type
        # (AM/EcM/ErM/OM/NM, FungalRoot) from its representation — does the niche predict the plant-fungal symbiosis?
        self.myco_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 5)) \
            if species_variable is not None else None
        # Phenology head (B26, jointly-trained on a DETACHED pooled latent): predict whether an observation is flowering
        # from its (space-time-conditioned) representation. Per-observation label from PhenoVision. Created last.
        self.flower_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)) \
            if species_variable is not None else None
        if compile_processor:
            self._refine = torch.compile(self._refine)

    # ---------------------------------------------------------------- tokens
    def _variable_token(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if name == self.species_variable and self._refined_species is not None:
            return self._refined_species[value.clamp(min=0)]
        v = self.variables[self.names.index(name)]
        return self.encoders[name](value if v.kind == "continuous" else value.clamp(min=0))

    def context(self, query_coords: torch.Tensor, neighbor_coords: torch.Tensor,
                manifold_positions: Optional[Dict[str, torch.Tensor]] = None,
                neighbor_values: Optional[Dict[str, torch.Tensor]] = None,
                batch_indices: Optional[torch.Tensor] = None) -> dict:
        """Encode the query's space-time position and its neighbor tokens.

        query_coords [B,4]=(lat,lon,elev,time); neighbor_coords [B,K,4]; manifold_positions {name: [B,K,dim]};
        neighbor_values {name: [B,K,...]}. Returns {"position": [B,d], "tokens": [B, subspaces*K, d]}; pass as ``context``.
        """
        # Community dropout (train only): blank the neighbor community on a fraction of rows, leaving only space-time offset geometry (matches how benchmarks query at eval).
        if self.training and self.COMMUNITY_DROPOUT > 0.0 and (neighbor_values or manifold_positions):
            Bd = query_coords.shape[0]
            keep = (torch.rand(Bd, device=query_coords.device) >= self.COMMUNITY_DROPOUT)
            if neighbor_values:
                neighbor_values = {n: v * keep.view(Bd, *([1] * (v.dim() - 1))).to(v.dtype)
                                   for n, v in neighbor_values.items()}
            if manifold_positions:
                manifold_positions = {n: p * keep.view(Bd, *([1] * (p.dim() - 1))).to(p.dtype)
                                      for n, p in manifold_positions.items()}
        # Sparse-hash path: read the absolute encoder from precomputed indices as a detached leaf so its hash trains through sparse Adam; the leaf grad is captured for the sparse step (plus dy_dx for the resolution gradient).
        if getattr(self, "_sparse_hash", False) and batch_indices is not None:
            flat = self.read_absolute_leaf(batch_indices)
        else:
            flat = self.absolute_encoder(query_coords)
        pos_s, pos_t = self._project_position(flat)
        if self.smooth_geo is not None:
            pos_s = pos_s + self.smooth_geo(query_coords)     # a smooth transferable geo prior is spatial-only
        feats = {name: (self.neighbor_emb[name](val) if name in self.neighbor_emb else val)
                 for name, val in (neighbor_values or {}).items()}
        tokens = self.neighbors(query_coords, neighbor_coords, manifold_positions, feats)
        return {"position_s": pos_s, "position_t": pos_t, "position": pos_s + pos_t, "tokens": tokens}

    def context_from_flat(self, flat: torch.Tensor, query_coords: torch.Tensor, neighbor_coords: torch.Tensor,
                          manifold_positions: Optional[Dict[str, torch.Tensor]] = None,
                          neighbor_values: Optional[Dict[str, torch.Tensor]] = None) -> dict:
        """Same as :meth:`context` but the absolute encoding is supplied as an already-read leaf ``flat``, keeping the precompute+detach out of the compiled region."""
        pos_s, pos_t = self._project_position(flat)
        if self.smooth_geo is not None:
            pos_s = pos_s + self.smooth_geo(query_coords)
        feats = {name: (self.neighbor_emb[name](val) if name in self.neighbor_emb else val)
                 for name, val in (neighbor_values or {}).items()}
        tokens = self.neighbors(query_coords, neighbor_coords, manifold_positions, feats)
        return {"position_s": pos_s, "position_t": pos_t, "position": pos_s + pos_t, "tokens": tokens}

    def _project_position(self, flat: torch.Tensor):
        """Project Earth4D's [xyz | xyt|yzt|xzt] output into (spatial, spatiotemporal) d_model channels separately,
        so downstream fusion can route time-invariant vs time-varying position per variable (see pos_channel_gate)."""
        s = self.absolute_encoder.spatial_dim
        return self.absolute_proj_s(flat[..., :s]), self.absolute_proj_t(flat[..., s:])

    def set_memory(self, key: torch.Tensor, features: Dict[str, torch.Tensor]) -> None:
        """Install the experience-replay memory bank (a key and per-anchor features), refreshed between epochs."""
        self._memory_key = key
        self._memory_features = features

    def enable_sparse_hash(self, coords: torch.Tensor, lr: float = 3e-4, weight_decay: float = 3e-4) -> None:
        """Precompute the absolute encoder over a fixed coordinate set and route it through sparse Adam (each batch reads few entries). Then pass ``batch_indices`` to :meth:`context` and call :meth:`sparse_hash_step` after backward."""
        self.absolute_encoder.precompute(coords)
        e = self.absolute_encoder
        self._abs_encs = [e.xyz_encoder, e.xyt_encoder, e.yzt_encoder, e.xzt_encoder]
        for en in self._abs_encs:
            en.init_sparse_adam(lr=lr, weight_decay=weight_decay)
        self._abs_odims = [en.num_levels * en.level_dim for en in self._abs_encs]
        self._abs_L = e.xyz_encoder.num_levels
        self._abs_F = e.features_per_level
        self._abs_dydx = None            # per-sub-encoder dy_dx captured by read_absolute_leaf (for the resolution grad)
        self._abs_inputs = None          # per-sub-encoder normalized inputs used
        self._sparse_hash = True

    def read_absolute_leaf(self, batch_indices: torch.Tensor) -> torch.Tensor:
        """Read the absolute encoder from precomputed indices as a detached leaf (requires_grad) and stash the per-
        sub-encoder dy_dx + inputs so :meth:`sparse_hash_step` can form the per_level_scale (resolution) gradient. Keep
        this out of any compiled region (it launches the eager hash kernel)."""
        flat, dydx, inputs = self.absolute_encoder.forward_precomputed(batch_indices, return_dydx=True)
        self._abs_dydx = dydx
        self._abs_inputs = inputs
        leaf = flat.detach().requires_grad_(True)
        self._abs_leaf = (leaf, batch_indices)
        return leaf

    def absolute_hash_params(self):
        """The absolute-encoder embeddings, optimized by sparse Adam and so excluded from the main optimizer."""
        return [en.embeddings for en in self._abs_encs]

    def set_sparse_lr(self, lr: float) -> None:
        for en in self._abs_encs:
            en.set_adam_lr(lr)

    def sparse_hash_step(self, flat: torch.Tensor = None, bidx: torch.Tensor = None) -> None:
        """Apply the sparse Adam update to the absolute encoder from the leaf gradient; call after ``loss.backward()``. Pass the leaf (compiled path) or omit to use the one captured in :meth:`context`; the accumulation buffer is cleared first. Also forms the per_level_scale (resolution) gradient from the captured dy_dx and routes it to the main optimizer, so resolution trains through the precompute."""
        if flat is None:
            flat, bidx = self._abs_leaf
        g = flat.grad
        off = 0
        for i, (en, d) in enumerate(zip(self._abs_encs, self._abs_odims)):
            en._adam_grad_buffer.zero_()
            g_e = g[:, off:off + d].contiguous()
            en.accumulate_grad(g_e, bidx)         # sparse embedding grad (touched entries only)
            # per_level_scale (resolution) gradient, same formula as the standard backward, from the captured dy_dx.
            if self._abs_dydx is not None and en.per_level_scale.requires_grad:
                B = g_e.shape[0]; L = en.num_levels; D = en.input_dim; C = en.level_dim
                gb = g_e.view(B, L, C)
                dd = self._abs_dydx[i].view(B, L, D, C)
                inp = self._abs_inputs[i]
                contrib = (torch.einsum('blc,bldc->bld', gb.float(), dd.float()) * inp.float().unsqueeze(1)).sum(0)  # [L,D]
                scale = torch.exp2(en.per_level_scale.view(L, D).float()) * en.base_resolution.view(1, D).float() - 1.0
                grad_pls = (0.6931471805599453 * (scale + 1.0) / scale.clamp_min(1e-6) * contrib).to(en.per_level_scale.dtype)
                if en.per_level_scale.grad is None:
                    en.per_level_scale.grad = grad_pls
                else:
                    en.per_level_scale.grad.add_(grad_pls)
            off += d
        for en in self._abs_encs:
            en.adam_step(bidx)                    # sparse Adam on the embeddings (touched entries only)
            en.transfer_index_logits_grad()       # route learned-probing grad to the main optimizer
        self._abs_dydx = None; self._abs_inputs = None   # consumed; avoid stale reuse next step

    def encode(self, values: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor], context: dict) -> torch.Tensor:
        """Build the refinable per-variable state tokens (masked slots carry a placeholder) + the fixed context tokens,
        then refine the latents against them over K rounds. Each variable token carries the query position."""
        if self.species_graph is not None:
            # ablation (rule 27 / benchmark families): _ablate_species -> use the UN-refined seed (graph off) so a
            # benchmark scored with vs without refinement isolates the phylogenomic contribution.
            self._refined_species = self.species_graph._seed() if getattr(self, "_ablate_species", False) \
                else self.species_graph()                    # refine all species once per forward
        if getattr(self, "pollinator_graph", None) is not None:
            self._refined_pollinators = self.pollinator_graph()   # refine all pollinators once per forward (rule 27 basis)
        pos_s, pos_t = context["position_s"], context["position_t"]                          # [B,d] each
        w = torch.softmax(self.pos_channel_gate, dim=-1)                                     # [V,2] per-variable prior
        pos_v = w[:, 0].view(1, -1, 1) * pos_s.unsqueeze(1) + w[:, 1].view(1, -1, 1) * pos_t.unsqueeze(1)   # [B,V,d]
        pres = torch.stack([present[n] for n in self.names], dim=1)                          # [B,V] bool
        val = torch.stack([self._variable_token(n, values[n]) for n in self.names], dim=1)   # [B,V,d] value embeddings
        content = torch.where(pres[..., None], val, self.mask_token) if self.learned_mask else val
        T = self.tok_norm(content + self.type_emb) + self.pos_norm(pos_v)                    # [B,V,d] per-variable position
        if not self.learned_mask:
            T = T * pres[..., None].to(T.dtype)              # single-shot behavior: a masked slot is the zero vector
        ctx = [(context["position"] + self.position_token).unsqueeze(1), context["tokens"]]  # always-present position (combined) + neighbor tokens
        if context.get("cls_tokens") is not None: ctx.append(context["cls_tokens"])
        if context.get("experience") is not None: ctx.append(context["experience"])
        return self._refine(T, torch.cat(ctx, dim=1), pres, val, pos_v)

    def _refine(self, T: torch.Tensor, C: torch.Tensor, present: torch.Tensor, value_emb: torch.Tensor,
                pos: torch.Tensor) -> torch.Tensor:
        """K rounds of joint-field denoising: the latents read the state+context tokens then attend among themselves;
        each round writes every variable's belief back into its state token, so masked variables are inducted jointly
        and observed ones may be revised. rounds=1 (no write-back) is the single-shot Processor."""
        z = self.latents.unsqueeze(0).expand(T.shape[0], -1, -1)
        gate = self.read_gate.view(1, 1, -1)
        stack = [] if self.round_loss == "all" else None
        for k in range(self.rounds):
            # The context C (neighbors, position) is fixed across rounds, so read it only in round 0; later rounds
            # refine against the updated variable states T alone (~5x fewer keys), which the latents already carry C from.
            kv = self.kv_norm(torch.cat([T, C], dim=1) if k == 0 else T)
            z = z + gate * self.read(self.q_norm(z), kv, kv)[0]
            for blk in self.blocks:
                z = blk(z)
            if stack is not None:
                stack.append(z)
            if self.write_back and k < self.rounds - 1:
                T = self._interface_update(z, value_emb, present, pos)
        if stack is not None:
            self._round_stack = torch.stack(stack, 0)        # [K,B,n_lat,d] (K fixed -> graph-safe)
        return z

    def _pooled_all(self, z: torch.Tensor) -> torch.Tensor:
        """Vectorized per-variable attention-pooling: every variable reads the latents through its own query in one
        batched op -> [B,V,d] (replaces the Python loop of tiny GEMMs)."""
        scores = torch.einsum("bld,vd->blv", z, self.decode_query) / (self.d_model ** 0.5)   # [B,L,V]
        w = torch.softmax(scores, dim=1)                                                      # over latents
        return torch.einsum("blv,bld->bvd", w, z)                                             # [B,V,d]

    def _reencode(self, name: str, pooled: torch.Tensor) -> torch.Tensor:
        """A per-variable interface backbone: predict the variable from its latent belief, then re-embed the prediction
        as a proper token (same space the encoders produce), so the write-back injects structure the read can use. For
        the species variable this routes the inferred posterior through the phylo-refined table -> a phylogenetically-
        conditioned prior; for categoricals, the class-mixture embedding; for continuous, decode->re-encode."""
        if name == self.species_variable and self._refined_species is not None:
            return torch.softmax(pooled @ self._refined_species.t(), dim=-1) @ self._refined_species
        if name not in self.decoders:
            return pooled
        v = self.variables[self.names.index(name)]
        if v.kind == "categorical":
            return torch.softmax(self.decoders[name](pooled), dim=-1) @ self.encoders[name].weight
        return self.encoders[name](self.decoders[name](pooled))

    def _interface_update(self, z: torch.Tensor, value_emb: torch.Tensor, present: torch.Tensor,
                          pos: torch.Tensor) -> torch.Tensor:
        """Re-inject round-k beliefs as the next round's state tokens: each variable reads the latents through its own
        query, is decoded+re-embedded into token space by its interface backbone, then masked slots carry the gated
        belief while observed slots keep their value (optionally revised). Flows to the latents only through the
        O(N*n_lat) read, so no O(N^2); each variable keeps its own query/decoder so marginals are not collapsed."""
        P = self._pooled_all(z)                                                              # [B,V,d] vectorized pooling
        E = torch.stack([self._reencode(n, P[:, i]) for i, n in enumerate(self.names)], dim=1)   # per-variable re-embed
        g = torch.sigmoid(self.update_gate).view(1, -1, 1)
        if self.revise:
            r = torch.sigmoid(self.revise_rate).view(1, -1, 1)
            obs = (1.0 - r) * value_emb + r * E
        else:
            obs = value_emb
        content = torch.where(present[..., None], obs, g * E)
        T = self.tok_norm(content + self.type_emb) + self.pos_norm(pos)   # pos is per-variable [B,V,d]
        return T.detach() if self.feedback_detach else T

    def _pooled(self, latents: torch.Tensor, name: str) -> torch.Tensor:
        """Attention-weighted pooling of the latents into one vector for reading variable ``name``."""
        i = self.names.index(name)
        w = torch.softmax((latents @ self.decode_query[i]) / (self.d_model ** 0.5), dim=-1)
        return torch.einsum("bl,bld->bd", w, latents)

    def decode(self, latents: torch.Tensor, name: str) -> torch.Tensor:
        """Read one variable back from the latents. The species variable reads against the refined species states; a diffusion variable is sampled from its head."""
        pooled = self._pooled(latents, name)
        if name == self.species_variable and self._refined_species is not None:
            return pooled @ self._refined_species.t()
        if name in self.diffusion_heads:
            return self.diffusion_heads[name].sample(pooled)
        return self.decoders[name](pooled)

    def decode_field(self, latents: torch.Tensor, query_pos: torch.Tensor,
                     names: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
        """Dense-field decode (Senseiver-style): read EVERY variable at each of G dense query positions from the latents
        that encode the sparse observations. ``query_pos`` [B,G,d] = ``absolute_proj(Earth4D(grid_coords))``. Each
        (position, variable) query = the position + that variable's decode-query, cross-attends the latents (O(G*V*L),
        linear in the grid via the bottleneck), then goes through the variable's head. Returns {name: [B,G,...]}. This
        turns the model into a dense forecaster: encode observations -> query a whole space-time volume."""
        names = list(names) if names is not None else [v.name for v in self.variables if v.reconstruct]
        q = self.pos_norm(query_pos).unsqueeze(2) + self.decode_query.view(1, 1, -1, self.d_model)   # [B,G,V,d]
        w = torch.softmax(torch.einsum("bgvd,bld->bgvl", q, latents) / (self.d_model ** 0.5), dim=-1)  # [B,G,V,L]
        read = torch.einsum("bgvl,bld->bgvd", w, latents)                                             # [B,G,V,d]
        out = {}
        for name in names:
            r = read[:, :, self.names.index(name)]                                                    # [B,G,d]
            if name == self.species_variable and self._refined_species is not None:
                out[name] = r @ self._refined_species.t()
            elif name in self.decoders:
                out[name] = self.decoders[name](r)
        return out

    def query_field(self, values, present, context, grid_coords: torch.Tensor,
                    names: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
        """End-to-end dense field: encode the sparse observations into latents, then decode every variable across the
        dense space-time grid ``grid_coords`` [B,G,4]. The single query the model trains on is the G=1 special case."""
        z = self.encode(values, present, context)
        pos_s, pos_t = self._project_position(self.absolute_encoder(grid_coords.reshape(-1, grid_coords.shape[-1])))
        query_pos = (pos_s + pos_t).view(grid_coords.shape[0], grid_coords.shape[1], self.d_model)
        return self.decode_field(z, query_pos, names)

    @torch.no_grad()
    def marginal_fidelity(self, values: Dict[str, torch.Tensor], observed: Dict[str, torch.Tensor],
                          context: dict) -> Dict[str, Dict[str, float]]:
        """Pluralism probe (science.md rule 23): hide ONLY each variable in turn and measure its own decode fidelity at
        K=1 vs K=rounds. Pluralism is conserved iff a variable's marginal does not degrade as the joint coupling (K)
        rises. Off the training path; a monitor, not a loss."""
        was, r0 = self.training, self.rounds
        self.eval()
        out: Dict[str, Dict[str, float]] = {}
        for v in self.variables:
            if not v.reconstruct:
                continue
            present = {n: observed[n].clone() for n in self.names}
            present[v.name] = torch.zeros_like(observed[v.name])           # hide only this variable
            w = observed[v.name].float()
            res = {}
            for k in sorted({1, r0}):
                self.rounds = k
                pred = self.decode(self.encode(values, present, context), v.name)
                fid = 1.0 - self._reconstruction_error(v.name, pred, values[v.name])   # cosine (cont) / 1-CE/logC (cat)
                res[f"K{k}"] = float((fid * w).sum() / w.sum().clamp_min(1.0))
            out[v.name] = res
        self.rounds = r0
        if was:
            self.train()
        return out

    # ---------------------------------------------------------------- training / inference
    def _reconstruction_error(self, name: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        v = self.variables[self.names.index(name)]
        if v.kind == "categorical":
            # Normalize by log(num_classes) so every categorical term shares the ~[0,1] scale of the continuous cosine terms (a wide identity head otherwise dominates the shared gradient).
            return F.cross_entropy(pred, target, reduction="none") / math.log(max(int(v.num_classes), 2))
        return 1.0 - F.cosine_similarity(pred, target, dim=-1)

    def reconstruction_loss(self, values: Dict[str, torch.Tensor], observed: Dict[str, torch.Tensor],
                            context: dict, hide_prob: float = 0.35) -> torch.Tensor:
        """One training step: reveal each variable with prob ``1 - hide_prob`` and reconstruct every hidden-but-observed variable, at one fixed shape."""
        B = len(next(iter(observed.values()))); dev = self.type_emb.device
        present = {n: (torch.rand(B, device=dev) > hide_prob) & observed[n] for n in self.names}
        # Fully blank a fraction of queries so the model must reconstruct from bare space-time + neighbors, training the position->variable pathway (else the absolute channel stays inert at inference).
        blank = torch.rand(B, device=dev) < 0.15
        for n in self.names:
            present[n] = present[n] & ~blank
        return self.masked_loss(values, observed, present, context)

    def masked_loss(self, values: Dict[str, torch.Tensor], observed: Dict[str, torch.Tensor],
                    present: Dict[str, torch.Tensor], context: dict) -> torch.Tensor:
        """Reconstruction loss for a fixed reveal mask (no randomness), so it can be compiled/CUDA-graphed with the random masking left outside."""
        z = self.encode(values, present, context)
        if self.round_loss == "all" and self._round_stack is not None:
            zs = self._round_stack                           # [K,B,n_lat,d]: deep-supervise every round, same targets
            loss = sum(self._decode_loss(zs[k], values, observed, present) for k in range(zs.shape[0])) / zs.shape[0]
        else:
            loss = self._decode_loss(z, values, observed, present)
        if self.inductive is not None:                       # auxiliary: name embedding -> evolutionary position
            loss = loss + 0.1 * self.inductive.loss(self._species_text, self._species_e1)
        # Rule 25: mask a fraction of species' seeds and reconstruct their refined embedding from phylogenetic relatives
        # (self-distillation toward the full-info refinement) -> the model can place a species of uncertain tree position.
        if self.training and getattr(self, "_phylo_mask_weight", 0.0) > 0 and self._refined_species is not None:
            m = torch.rand(self._refined_species.shape[0], device=z.device) < 0.15
            if m.any():
                loss = loss + self._phylo_mask_weight * F.mse_loss(self.species_graph(mask=m)[m], self._refined_species[m].detach())
        return loss

    def _decode_loss(self, z: torch.Tensor, values: Dict[str, torch.Tensor], observed: Dict[str, torch.Tensor],
                     present: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-variable reconstruction error over the hidden-but-observed targets, decoded from latents ``z``."""
        loss, n_terms = z.new_zeros(()), 0
        for v in self.variables:
            if not v.reconstruct:
                continue
            w = ((~present[v.name]) & observed[v.name]).to(z.dtype)
            if v.name in self.diffusion_heads:
                err = self.diffusion_heads[v.name].loss(values[v.name], self._pooled(z, v.name), reduce=False)
            else:
                pred = self.decode(z, v.name)
                err = self._reconstruction_error(v.name, pred, values[v.name])
                # Cross-modal contrastive (JEPA / rule 16): the predicted continuous embedding must retrieve its own
                # target against the batch (InfoNCE) -- a global signal point-wise cosine cannot give.
                if self.contrastive_weight > 0.0 and v.name in self.contrastive_vars and v.kind == "continuous":
                    pn = F.normalize(pred.float(), dim=-1); tn = F.normalize(values[v.name].float(), dim=-1)
                    logits = pn @ tn.t() / 0.05    # temperature (sweeping 0.07->0.05: sharper, more discriminative retrieval)
                    loss = loss + self.contrastive_weight * F.cross_entropy(
                        logits, torch.arange(logits.shape[0], device=logits.device))
            # Per-variable loss weight (science.md rule 18): focus reconstruction where benchmarks have headroom.
            loss = loss + self.loss_weights.get(v.name, 1.0) * (err * w).sum() / w.sum().clamp_min(1.0)
            # Distribution matching (MADE joint-dist) via the SEPARATE community head on a DETACHED latent: trains only
            # comm_head toward the local community distribution, with zero gradient into the shared representation ->
            # guaranteed no regression on identity/phylo/traits, while the head supplies B20-22/B39/B40 at test.
            if getattr(self, "_sdist_weight", 0.0) > 0 and v.name == self.species_variable \
                    and "_sdist_idx" in values and getattr(self, "comm_head", None) is not None:
                comm = (self.comm_head(self._pooled(z, v.name).detach()) @ self._refined_species.detach().t()).float()
                sidx = values["_sdist_idx"].clamp(0, comm.shape[1] - 1)   # -1 padding -> 0 (its freq is 0, harmless)
                tgt = torch.zeros_like(comm).scatter_add_(1, sidx, values["_sdist_frq"].float())
                kl = -(tgt * F.log_softmax(comm, -1)).sum(-1)            # soft cross-entropy toward the local distribution
                loss = loss + self._sdist_weight * (kl * w).sum() / w.sum().clamp_min(1.0)
            # Pollinator distribution matching via the SEPARATE detached poll_head toward the plant's GloBI pollinator
            # distribution (zero gradient into the shared representation). Enables B41/B51-B54, no backbone regression.
            if getattr(self, "_poll_weight", 0.0) > 0 and v.name == self.species_variable \
                    and "_poll_idx" in values and getattr(self, "poll_head", None) is not None:
                # detach the PLANT latent (protect the shared backbone) but NOT the pollinator basis: the interaction
                # loss trains poll_head + the pollinator graph, so partners propagate to a pollinator's relatives (rule 27)
                pl = (self.poll_head(self._pooled(z, v.name).detach()) @ self._pollinator_basis().t()).float()
                pidx = values["_poll_idx"].clamp(0, pl.shape[1] - 1)
                ptg = torch.zeros_like(pl).scatter_add_(1, pidx, values["_poll_frq"].float())
                pv = values["_poll_valid"].float()                       # only plants with known pollinators contribute
                pkl = -(ptg * F.log_softmax(pl, -1)).sum(-1) * pv
                loss = loss + self._poll_weight * (pkl * w).sum() / (pv * w).sum().clamp_min(1.0)
            # Ecophysiology (B34): detached head predicts log live-fuel-moisture toward the species' value (protects backbone)
            if getattr(self, "_lfmc_weight", 0.0) > 0 and v.name == self.species_variable \
                    and "_lfmc" in values and getattr(self, "lfmc_head", None) is not None:
                pred = self.lfmc_head(self._pooled(z, v.name).detach()).squeeze(-1).float()
                tgt = torch.log(values["_lfmc"].clamp_min(1.0)); lv = values["_lfmc_valid"].float()
                loss = loss + self._lfmc_weight * ((pred - tgt) ** 2 * lv).sum() / lv.sum().clamp_min(1.0)
            # Symbiosis (B42): detached head predicts the mycorrhizal type (cross-entropy toward the FungalRoot label)
            if getattr(self, "_myco_weight", 0.0) > 0 and v.name == self.species_variable \
                    and "_myco" in values and getattr(self, "myco_head", None) is not None:
                logit = self.myco_head(self._pooled(z, v.name).detach())
                mv = values["_myco_valid"].float()
                ce = F.cross_entropy(logit, values["_myco"].clamp_min(0), reduction="none")
                loss = loss + self._myco_weight * (ce * mv).sum() / mv.sum().clamp_min(1.0)
            # Phenology (B26): detached head predicts flowering (BCE) toward the per-observation PhenoVision label
            if getattr(self, "_flower_weight", 0.0) > 0 and v.name == self.species_variable \
                    and "_flower" in values and getattr(self, "flower_head", None) is not None:
                logit = self.flower_head(self._pooled(z, v.name).detach()).squeeze(-1).float()
                fv = values["_flower_valid"].float()
                bce = F.binary_cross_entropy_with_logits(logit, values["_flower"].float(), reduction="none")
                loss = loss + self._flower_weight * (bce * fv).sum() / fv.sum().clamp_min(1.0)
            n_terms += 1
        return loss / max(n_terms, 1)

    def _pollinator_basis(self) -> torch.Tensor:
        """Pollinator output embeddings for the interaction head (rule 27): phylo-refined by the pollinator species
        graph if wired (cached per forward), else the free table. The refined basis is what makes a predicted pollinator
        lift its phylogenetic relatives, and training propagate to them."""
        rp = getattr(self, "_refined_pollinators", None)
        return rp if rp is not None else self.poll_emb

    @torch.no_grad()
    def infer(self, values: Dict[str, torch.Tensor], given: Sequence[str], targets: Sequence[str],
              context: dict, observed: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Predict ``targets`` from the variables in ``given`` (plus space-time + neighbor context). A given variable
        is revealed only where actually ``observed`` (else a missing value would enter as a spurious zero token)."""
        B = context["position"].shape[0]
        dev = self.type_emb.device
        present = {n: torch.zeros(B, dtype=torch.bool, device=dev) for n in self.names}
        for n in given:
            present[n] = observed[n] if (observed is not None and n in observed) \
                else torch.ones(B, dtype=torch.bool, device=dev)
        z = self.encode(values, present, context)
        out = {}
        for t in targets:
            if t == "community":                                                 # env-conditioned community distribution
                out[t] = (self.comm_head(self._pooled(z, self.species_variable)) @ self._refined_species.t()) if getattr(self, "comm_head", None) is not None \
                    else self.decode(z, self.species_variable)                   # fallback (no head): the identity posterior
            elif t == "pollinator":                                              # plant -> pollinator interaction (rule 27)
                out[t] = self.poll_head(self._pooled(z, self.species_variable)) @ self._pollinator_basis().t()
            elif t == "lfmc":                                                    # species -> live fuel moisture (B34)
                out[t] = self.lfmc_head(self._pooled(z, self.species_variable)).squeeze(-1).exp()
            elif t == "myco":                                                    # species -> mycorrhizal type logits (B42)
                out[t] = self.myco_head(self._pooled(z, self.species_variable))
            elif t == "flower":                                                  # observation -> flowering probability (B26)
                out[t] = torch.sigmoid(self.flower_head(self._pooled(z, self.species_variable)).squeeze(-1))
            else:
                out[t] = self.decode(z, t)
        return out
