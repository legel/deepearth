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
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F

from deepearth.autoresearch.main.editable_files.encoders.earth4d import Earth4D, SmoothGeoField
from deepearth.autoresearch.main.editable_files.encoders.phylogenomic import SpeciesGraph


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


class NeighborContext(nn.Module):
    """One token per (neighbor, subspace) = subspace encoding + neighbor feature projections + a subspace marker.

    space_time: kwargs for SpaceTimeField; manifolds: {name: dim} vector subspaces; feature_dims: {name: dim} features.
    """

    def __init__(self, d_model: int, space_time: dict, manifolds: Dict[str, int] | None = None,
                 feature_dims: Dict[str, int] | None = None, neighbor_op: str = "add"):
        super().__init__()
        self.d_model = d_model
        self.space_time = SpaceTimeField(d_model, **space_time)
        self.manifolds = nn.ModuleDict({name: ManifoldField(d_model, dim) for name, dim in (manifolds or {}).items()})
        self.features = nn.ModuleDict({name: nn.Linear(dim, d_model) for name, dim in (feature_dims or {}).items()})
        self.field_marker = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(d_model) * 0.02) for name in ["space_time", *(manifolds or {})]})
        # neighbor_op: how a neighbor's VALUE (features) binds with its POSITION encoding. 'add'=bare sum (champion).
        self.neighbor_op = neighbor_op
        if neighbor_op == "film":       # position FiLM-modulated by the neighbor's value
            self.film_g = nn.Linear(d_model, d_model); self.film_b = nn.Linear(d_model, d_model)
        elif neighbor_op == "gate":     # neighbor value gated by where it is
            self.feat_gate = nn.Linear(d_model, d_model)
        elif neighbor_op == "bind":     # explicit Hadamard value x position binding term
            self.bind_proj = nn.Linear(d_model, d_model)

    def _combine(self, pos: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        op = self.neighbor_op
        if op == "add":  return pos + features
        if op == "film": return pos * (1.0 + self.film_g(features)) + self.film_b(features)
        if op == "gate": return pos + features * torch.sigmoid(self.feat_gate(pos))
        if op == "bind": return pos + features + self.bind_proj(pos * features)
        raise ValueError(f"unknown neighbor_op {op}")

    def forward(self, query_coords: torch.Tensor, neighbor_coords: torch.Tensor,
                manifold_positions: Dict[str, torch.Tensor] | None = None,
                neighbor_features: Dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        B, K = neighbor_coords.shape[0], neighbor_coords.shape[1]
        features = query_coords.new_zeros(B, K, self.d_model)
        for name, val in (neighbor_features or {}).items():
            features = features + self.features[name](val)
        tokens = [self._combine(self.space_time(query_coords, neighbor_coords), features) + self.field_marker["space_time"]]
        for name, field in self.manifolds.items():
            tokens.append(self._combine(field(manifold_positions[name]), features) + self.field_marker[name])
        return torch.cat(tokens, dim=1)


class RMSNorm(nn.Module):
    """RMSNorm (Zhang & Sennrich 2019): scale-only, no mean-centering — a genuinely different normalizer than
    LayerNorm, cheaper and with different gradient dynamics. An architecture lever, not a knob."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class GLUFFN(nn.Module):
    """Gated-linear-unit FFN (SwiGLU / GeGLU, Shazeer 2020): a value branch multiplicatively gated by a learned
    activation branch — a different information-routing structure than the MLP's single activation path."""

    def __init__(self, d: int, hidden: int, act: str = "swish"):
        super().__init__()
        self.gate = nn.Linear(d, hidden); self.up = nn.Linear(d, hidden); self.down = nn.Linear(hidden, d)
        self.act = F.silu if act == "swish" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))


class LatentBlock(nn.Module):
    """Configurable pre-norm latent self-attention block — the ARCHITECTURE-variant replacement for
    ``nn.TransformerEncoderLayer``. norm in {ln, rms}; ffn in {mlp, swiglu, geglu}. The ``torch`` block stays the
    loop default (champion reproduces byte-identically); this activates only when a config selects a variant."""

    def __init__(self, d_model: int, n_heads: int, ffn: str = "mlp", norm: str = "ln", mult: int = 4):
        super().__init__()
        Norm = (lambda d: RMSNorm(d)) if norm == "rms" else (lambda d: nn.LayerNorm(d))
        self.n1, self.n2 = Norm(d_model), Norm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        h = mult * d_model
        if ffn == "swiglu":
            self.ffn = GLUFFN(d_model, h, "swish")
        elif ffn == "geglu":
            self.ffn = GLUFFN(d_model, h, "gelu")
        else:
            self.ffn = nn.Sequential(nn.Linear(d_model, h), nn.GELU(), nn.Linear(h, d_model))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:   # drop-in for TransformerEncoderLayer(x)
        a = self.n1(x); x = x + self.attn(a, a, a, need_weights=False)[0]
        return x + self.ffn(self.n2(x))


RETRIEVAL_TEMPERATURE = 0.05    # matches the training InfoNCE; scoring must not use a different one


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
        absolute_log2_hashmap_size: int = 20,   # dominates total params: 4 hash tables x levels x 2^size x 2 feats
        absolute_levels: int = 18,
        species_variable: Optional[str] = None,
        species_embedding: Optional[torch.Tensor] = None,
        species_layers: int = 2,
        species_heads: int = 4,
        species_top_k: Optional[int] = None,
        species_flex: bool = False,
        species_operator: str = "ou-attention",
        species_tree: Optional[dict] = None,
        species_tip_row: Optional[torch.Tensor] = None,   # (latent-clade) species-local vocab index of each in-tree tip, in tree-tip order
        species_mask_posterior: Optional[str] = None,
        species_distance: Optional[tuple] = None,   # (dated[m,m], model_idx[m]): real dated patristic for tree-covered species -> replaces the embedding shadow (ou-attention)
        species_text: Optional[torch.Tensor] = None,
        species_family: Optional[torch.Tensor] = None,
        family_env_expert: bool = False,
        family_env_vars: Optional[Sequence[str]] = None,
        family_alphaearth_expert: bool = False,
        family_env_residual: bool = False,
        ecological_family_map: bool = False,
        orthogonal_blank_hidden: int = 0,
        task_occupancy_experts: bool = False,
        task_niche_prior: bool = False,
        niche_coord_mean: Optional[torch.Tensor] = None,
        niche_coord_scale: Optional[torch.Tensor] = None,
        niche_ae_mean: Optional[torch.Tensor] = None,
        niche_ae_scale: Optional[torch.Tensor] = None,
        compile_processor: bool = False,
        rounds: int = 1,
        write_back: bool = True,
        revise: bool = False,
        round_loss: str = "final",
        learned_mask: Optional[bool] = None,
        feedback_detach: bool = False,
        flex_attention: bool = False,
        decoder_hidden: Optional[int] = None,
        mod_encoder: str = "linear",
        block_ffn: str = "torch",
        block_norm: str = "ln",
        read_depth: int = 1,
        read_op: str = "mha",
        neighbor_op: str = "add",
        token_op: str = "add",
        read_cond: bool = False,
        joint_decode: bool = False,
        grad_checkpoint: bool = False,
        diffusion: bool = False,
        loss_weights: Optional[Dict[str, float]] = None,
        contrastive_weight: float = 0.0,
        contrastive_vars: Optional[Sequence[str]] = None,
        smooth_geo: bool = False,
        smooth_geo_sigmas: Optional[Sequence[float]] = None,
        smooth_geo_per_scale: int = 32,
        alphaearth_geo: bool = False,
        n_pollinators: int = 0,
        pollinator_distance: Optional[torch.Tensor] = None,
        pollinator_text: Optional[torch.Tensor] = None,
        pollinator_top_k: Optional[int] = None,
        poll_species_idx: Optional[torch.Tensor] = None,
        poll_species_frq: Optional[torch.Tensor] = None,
        poll_species_mixture: float = 0.0,
        poll_species_all_masked: bool = False,
        phylo_head_routing: bool = False,
        species_trait_recon: bool = False,
        continuous_calibration: bool = False,
    ) -> None:
        super().__init__()
        self.phylo_head_routing = phylo_head_routing   # route the phylo-refined species embedding into the trait heads
        self.species_trait_recon = species_trait_recon # LCA fine-tuning: reconstruct phylo-conserved traits FROM the refined species embedding (tree learns trait structure)
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
        def _enc(in_dim):   # [Ensue] mod_encoder toggle: deeper/normalized modality tokenizer (default = bare Linear)
            if mod_encoder == "mlp2":
                return nn.Sequential(nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            if mod_encoder == "mlp2ln":
                return nn.Sequential(nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
            if mod_encoder == "prenorm":       # per-modality INPUT LayerNorm before projection (standardize raw feature scales across modalities)
                return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model))
            if mod_encoder == "prenormmlp2":    # input-normalized 2-layer tokenizer (scale-fix + capacity)
                return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            return nn.Linear(in_dim, d_model)
        for v in self.variables:
            if v.kind == "continuous":
                self.encoders[v.name] = _enc(v.dim)
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
        # Calibration of the continuous heads. The continuous reconstruction loss is a mean-centered cosine,
        # which is invariant to per-dimension scale and offset, so a head is never asked to predict in the
        # target's units -- it only has to point the right way. For the z-scored variables that is a real
        # deficit and the Gaussian charges for it: topo, hydro and phenology are scored WORSE than the constant
        # that predicts each channel's mean. These per-dimension affines are fitted by that same Gaussian on a
        # DETACHED prediction, so they turn a direction into a calibrated value without touching the
        # representation. Directional variables are scored by retrieval and are left alone (see _decode_loss).
        #
        # ones/zeros draw no RNG, so with the flag on the rest of the model still initializes bit-identically
        # to the default path: a gain that lands here cannot be an initialization re-roll.
        self.continuous_calibration = continuous_calibration
        self.cal_gain = nn.ParameterDict()
        self.cal_bias = nn.ParameterDict()
        if continuous_calibration:
            for v in self.variables:
                if v.kind == "continuous" and v.reconstruct and v.name in self.decoders:
                    self.cal_gain[v.name] = nn.Parameter(torch.ones(v.dim))
                    self.cal_bias[v.name] = nn.Parameter(torch.zeros(v.dim))

        self.type_emb = nn.Parameter(torch.randn(len(self.variables), d_model) * 0.02)
        # A dedicated always-present space-time token, so a query revealing no variable keeps its position (variable tokens are zeroed by the present-mask).
        self.position_token = nn.Parameter(torch.randn(d_model) * 0.02)
        # Normalize content and position to matched unit scale before adding (token-embedding + PE practice): both stay legible and the absolute-encoder gradient stays bounded.
        self.tok_norm = nn.LayerNorm(d_model)
        self.pos_norm = nn.LayerNorm(d_model)
        self.decode_query = nn.Parameter(torch.randn(len(self.variables), d_model) * 0.02)

        # Absolute location memory: coarse regional/long-period memorization (~200M, 20% of Earth4D); fine structure lives in the relative encoder.
        self.absolute_encoder = Earth4D(verbose=False, spatial_levels=absolute_levels, temporal_levels=absolute_levels,
                                        spatial_log2_hashmap_size=absolute_log2_hashmap_size,
                                        temporal_log2_hashmap_size=absolute_log2_hashmap_size,
                                        freq_log_scale_init=-2.5)   # start coarse (~1 km finest); learned from there
        # Project Earth4D's [xyz | xyt|yzt|xzt] as separate spatial/spatiotemporal channels; each variable learns a
        # softmax prior over which it reads, so time-invariant modalities can shut time out while vision keeps it.
        self.absolute_proj_s = nn.Sequential(nn.Linear(self.absolute_encoder.spatial_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.absolute_proj_t = nn.Sequential(nn.Linear(self.absolute_encoder.spatiotemporal_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        gate0 = torch.zeros(len(self.variables), 2); gate0[:, 0] = 2.0   # init ~0.88 spatial / 0.12 temporal
        self.pos_channel_gate = nn.Parameter(gate0)
        # Smooth transferable geo prior (RFF): added to the memorizing hash position -> generalizes to held-out regions.
        self.smooth_geo = SmoothGeoField(
            d_model, per_scale=smooth_geo_per_scale,
            sigmas=tuple(smooth_geo_sigmas) if smooth_geo_sigmas else (1.0, 4.0, 16.0, 64.0),
        ) if smooth_geo else None
        # AlphaEarth (Google Satellite Embedding V1, 64d) as a SatCLIP-style LEARNED geo prior: projected and added to the
        # spatial position channel that EVERY head reads (like smooth_geo), NOT a reconstruction variable competing for head capacity.
        self.alphaearth_geo = nn.Sequential(nn.Linear(64, d_model), nn.GELU(), nn.Linear(d_model, d_model)) if alphaearth_geo else None
        # neighbor context over coordinate subspaces: space-time, plus any vector manifolds (e.g. biological)
        neighbor_dims = {v.name: (v.dim if v.kind == "continuous" else d_model)
                         for v in self.variables if v.neighbor}
        self.neighbor_emb = nn.ModuleDict(
            {v.name: nn.Embedding(v.num_classes, d_model) for v in self.variables if v.neighbor and v.kind == "categorical"})
        self.neighbors = NeighborContext(
            d_model, space_time=dict(window=relative_window, levels=18, finest=relative_finest,
                                     log2_hashmap_size=relative_log2_hashmap_size,
                                     reference_latitude_deg=reference_latitude_deg),
            manifolds=manifolds, feature_dims=neighbor_dims, neighbor_op=neighbor_op)

        # species graph: refine identity through phylogenetic-neighbor attention. "tree" propagates over the dated phylogeny; "ou-attention" biases attention with an embedding-derived distance.
        self.species_variable = species_variable
        self.register_buffer("species_family", species_family)
        self.family_env_vars = tuple(family_env_vars or ())
        self.family_env_residual = family_env_residual
        self.ecological_family_map = ecological_family_map
        self.family_count = int(species_family.max()) + 1 if species_family is not None else 0
        if species_variable is not None and species_embedding is not None:
            if species_operator == "tree":
                assert species_tree is not None, "species_operator='tree' needs the parsed tree (source.tree)"
                self.species_graph = SpeciesGraph(species_embedding.shape[0], d_model, operator="tree",
                                                  tree=species_tree, n_layers=species_layers, species_text=species_text)
            elif species_operator == "latent-clade":                                 # rule 29: exact tree-GP refinement + out-of-tree clade cross-attention
                assert species_tree is not None and species_tip_row is not None, \
                    "species_operator='latent-clade' needs source.lca_tree + source.lca_tip_row"
                self.species_graph = SpeciesGraph(species_embedding.shape[0], d_model, operator="latent-clade",
                                                  tree=species_tree, tip_row=species_tip_row, n_heads=species_heads,
                                                  n_layers=species_layers, species_text=species_text,
                                                  mask_posterior=species_mask_posterior)
            else:
                distance = SpeciesGraph.distance_from_embedding(species_embedding)   # BioCLIP-embedding shadow (kept for inductively-placed species not on the dated tree)
                if species_distance is not None:                                     # overwrite the tree-covered block with the REAL dated patristic (rules 7-12), rescaled to the shadow's scale
                    dated, midx = species_distance
                    blk = dated / (dated[~torch.eye(len(dated), dtype=torch.bool, device=dated.device)].mean() + 1e-9) \
                                * distance[midx][:, midx][~torch.eye(len(midx), dtype=torch.bool, device=distance.device)].mean()
                    blk = 0.5 * (blk + blk.t()); blk.fill_diagonal_(0.0)
                    distance = distance.clone(); distance[midx.unsqueeze(1), midx.unsqueeze(0)] = blk
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
        self.n_heads = n_heads
        self.read_op = read_op
        # READ-operator variants (architecture lever). 'mha'=champion-identical stock cross-attention.
        # 'slot': competitive slot-attention (softmax over LATENTS, not keys) -> latents partition/specialize the field.
        # 'typed': separate cross-attn over variable-state vs neighbor-context, per-latent gated -> respects key type.
        # 'crossself': latents co-attend to the field AND each other in one softmax -> dissolves read/process split.
        if read_op == "slot":
            self.slot_q = nn.Linear(d_model, d_model); self.slot_k = nn.Linear(d_model, d_model)
            self.slot_v = nn.Linear(d_model, d_model); self.slot_o = nn.Linear(d_model, d_model)
        elif read_op == "typed":
            self.read_T = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.read_C = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.type_gate = nn.Linear(d_model, 2)
        elif read_op == "crossself":
            self.read_cs = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.token_op = token_op        # value x position binding for the query's own variable tokens
        if token_op in ("bind", "filmbind"):
            self.tok_bind_proj = nn.Linear(d_model, d_model)
        if token_op in ("film", "filmbind"):
            self.tok_film_g = nn.Linear(d_model, d_model); self.tok_film_b = nn.Linear(d_model, d_model)
        self.read_cond = read_cond      # location-aware read: FiLM the read query by the query's GLOBAL position
        if read_cond:
            self.read_film_g = nn.Linear(d_model, d_model); self.read_film_b = nn.Linear(d_model, d_model)
        self.joint_decode = joint_decode  # cross-variable joint decoder: variables attend to each other before decoding
        self.grad_checkpoint = grad_checkpoint  # recompute read+block activations in backward (memory<->compute)
        self.diffusion = diffusion  # rule-22: masked states start as noise, denoised over rounds
        if joint_decode:
            self.joint_block = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, batch_first=True, norm_first=True)
        self.q_norm = nn.LayerNorm(d_model); self.kv_norm = nn.LayerNorm(d_model)
        # Latent self-attention blocks. Default block_ffn="torch" = nn.TransformerEncoderLayer (champion-identical);
        # an ARCHITECTURE variant swaps in the configurable pre-norm LatentBlock (ffn: mlp/swiglu/geglu, norm: ln/rms).
        def _block():
            if block_ffn == "torch":
                return nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, batch_first=True, norm_first=True)
            return LatentBlock(d_model, n_heads, ffn=block_ffn, norm=block_norm)
        self.blocks = nn.ModuleList([_block() for _ in range(n_layers)])
        # Fusion depth (architecture lever): re-read the fixed context between latent blocks instead of once up front,
        # so the latents keep pulling from the neighbor/position context as they process. read_depth=1 (default) -> no
        # extra reads -> champion-identical (empty ModuleList consumes no init RNG).
        self.extra_reads = nn.ModuleList([nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                                          for _ in range(max(0, read_depth - 1))])

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
        # Community-distribution head (MADE joint-dist): a SEPARATE readout trained on a DETACHED latent toward the
        # LOCAL community distribution, so no gradient reaches the shared backbone. Created LAST so it consumes no init
        # RNG ahead of the backbone -> the shared model initializes bit-identically with or without this head.
        self.comm_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)) \
            if (species_variable is not None and species_embedding is not None) else None
        # Pollinator-distribution head (plant->pollinator interaction, MADE joint-dist): a DETACHED readout from the
        # species-pooled latent into a learned pollinator-vocab basis, trained toward the plant's local GloBI pollinator
        # distribution. Created LAST so it consumes no init RNG ahead of the backbone.
        self.poll_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)) if n_pollinators > 0 else None
        self.poll_emb = nn.Parameter(torch.randn(n_pollinators, d_model) * 0.02) if n_pollinators > 0 else None
        # Cross-tree interaction (rule 27): decode the plant->pollinator interaction against a SECOND, separately
        # phylo-refined pollinator species graph (its own tree), so an observed interaction propagates to BOTH sides'
        # relatives. Falls back to the free poll_emb table until the pollinator tree is wired.
        self.pollinator_graph = SpeciesGraph(n_pollinators, d_model, pollinator_distance, top_k=pollinator_top_k,
                                             species_text=pollinator_text) \
            if (n_pollinators > 0 and pollinator_distance is not None) else None
        # Phylo-conserved trait heads (B34/B42/B26) on a DETACHED latent. With phylo_head_routing the head reads the
        # pooled latent CONCATENATED with the expected phylo-refined species embedding, so the species graph's
        # conserved signal (relatives' trait values) reaches the head -- and its graph-gain (B57/B58/B62) is real.
        _hd = 2 * d_model if self.phylo_head_routing else d_model
        # Ecophysiology (B34): predict a species' peak fire-season live fuel moisture from its phylo-refined representation.
        self.lfmc_head = nn.Sequential(nn.Linear(_hd, d_model), nn.GELU(), nn.Linear(d_model, 1)) \
            if species_variable is not None else None
        # Symbiosis (B42): predict a plant's mycorrhizal type (AM/EcM/ErM/OM/NM, FungalRoot label).
        self.myco_head = nn.Sequential(nn.Linear(_hd, d_model), nn.GELU(), nn.Linear(d_model, 5)) \
            if species_variable is not None else None
        # Phenology (B26): predict whether a (space-time-conditioned) observation is flowering (PhenoVision label).
        self.flower_head = nn.Sequential(nn.Linear(_hd, d_model), nn.GELU(), nn.Linear(d_model, 1)) \
            if species_variable is not None else None
        # LCA fine-tuning (rule 25 extended to traits): a head reconstructs a species' phylo-conserved trait FROM its
        # refined embedding. Trained NON-detached (shapes the tree) with phylo masking, so a masked species' trait is
        # imputed from relatives through the tree -- learned nearest-neighbor-on-the-dated-tree, which must beat raw NN.
        # Per-species labels (_species_myco/_valid) are attached post-construction from the data source.
        if self.species_trait_recon and species_variable is not None:
            self.species_myco_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 5))
            self._species_myco = None; self._species_myco_valid = None; self._train_species = None
        self.family_env_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True) \
            if family_env_expert and self.family_count else None
        if self.family_env_attn is not None:
            self.family_env_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.family_env_head = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, self.family_count))
            if family_env_residual:
                nn.init.zeros_(self.family_env_head[-1].weight)
                nn.init.zeros_(self.family_env_head[-1].bias)
        self.family_ae_head = None
        if family_alphaearth_expert and self.family_count:
            # This private readout must not re-roll the shared model or its training masks.
            with torch.random.fork_rng(devices=[]):
                self.family_ae_head = nn.Sequential(
                    nn.LayerNorm(64), nn.Linear(64, d_model), nn.GELU(),
                    nn.Linear(d_model, self.family_count))
        self.occupancy_experts = None
        if task_occupancy_experts and self.family_ae_head is not None:
            with torch.random.fork_rng(devices=[]):
                width = 2 * d_model
                outputs = {"identity": d_model, "community": d_model, "family": self.family_count}
                if self.poll_emb is not None:
                    outputs["pollinator"] = d_model
                self.occupancy_experts = nn.ModuleDict({
                    name: nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d_model), nn.GELU(),
                                        nn.Linear(d_model, out))
                    for name, out in outputs.items()
                })
            for head in self.occupancy_experts.values():
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        self.niche_trunk = None
        self.niche_experts = None
        if task_niche_prior and self.family_ae_head is not None:
            if any(x is None for x in (niche_coord_mean, niche_coord_scale,
                                       niche_ae_mean, niche_ae_scale)):
                raise ValueError("task_niche_prior requires training-split feature statistics")
            self.register_buffer("niche_coord_mean", niche_coord_mean.float())
            self.register_buffer("niche_coord_scale", niche_coord_scale.float().clamp_min(1e-4))
            self.register_buffer("niche_ae_mean", niche_ae_mean.float())
            self.register_buffer("niche_ae_scale", niche_ae_scale.float().clamp_min(1e-4))
            with torch.random.fork_rng(devices=[]):
                self.niche_trunk = nn.Sequential(
                    nn.LayerNorm(116), nn.Linear(116, 512), nn.GELU(),
                    nn.Linear(512, 256), nn.GELU())
                outputs = {"family": self.family_count, "identity": d_model,
                           "community": d_model}
                if self.poll_emb is not None:
                    outputs["pollinator"] = d_model
                self.niche_experts = nn.ModuleDict({
                    name: nn.Linear(256, size) for name, size in outputs.items()
                })
        self.blank_adapters = nn.ModuleDict()
        self.blank_family = None
        if orthogonal_blank_hidden > 0 and self.family_count:
            with torch.random.fork_rng(devices=[]):
                for route in ("species", "community"):
                    self.blank_adapters[route] = nn.Sequential(
                        nn.LayerNorm(d_model), nn.Linear(d_model, orthogonal_blank_hidden), nn.GELU(),
                        nn.Linear(orthogonal_blank_hidden, d_model))
                self.blank_family = nn.Linear(d_model, self.family_count)
            for adapter in self.blank_adapters.values():
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
            nn.init.zeros_(self.blank_family.weight)
            nn.init.zeros_(self.blank_family.bias)
        self.poll_species_mixture = float(poll_species_mixture)
        self.poll_species_all_masked = bool(poll_species_all_masked)
        if self.poll_species_mixture > 0:
            if poll_species_idx is None or poll_species_frq is None:
                raise ValueError("poll_species_mixture requires the species-pollinator table")
            self.register_buffer("poll_species_idx", poll_species_idx.long(), persistent=False)
            self.register_buffer("poll_species_frq", poll_species_frq.float(), persistent=False)
        else:
            self.poll_species_idx = None
            self.poll_species_frq = None
        if compile_processor:
            self._refine = torch.compile(self._refine)

    # ---------------------------------------------------------------- tokens
    def _token_combine(self, base: torch.Tensor, posn: torch.Tensor) -> torch.Tensor:
        op = self.token_op
        if op == "add":  return base + posn
        if op == "bind": return base + posn + self.tok_bind_proj(base * posn)
        if op == "film": return base * (1.0 + self.tok_film_g(posn)) + self.tok_film_b(posn)
        if op == "filmbind": return base * (1.0 + self.tok_film_g(posn)) + self.tok_film_b(posn) + self.tok_bind_proj(base * posn)
        raise ValueError(f"unknown token_op {op}")

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
            raw = self.read_absolute_leaf(batch_indices)
            flat = self.absolute_encoder.transform_precomputed(raw, query_coords)
        else:
            flat = self.absolute_encoder(query_coords)
        pos_s, pos_t = self._project_position(flat)
        if self.smooth_geo is not None:
            pos_s = pos_s + self.smooth_geo(query_coords)     # a smooth transferable geo prior is spatial-only
        feats = {name: (self.neighbor_emb[name](val) if name in self.neighbor_emb else val)
                 for name, val in (neighbor_values or {}).items()}
        tokens = self.neighbors(query_coords, neighbor_coords, manifold_positions, feats)
        return {"position_s": pos_s, "position_t": pos_t, "position": pos_s + pos_t,
                "tokens": tokens, "coords": query_coords.detach()}

    def context_from_flat(self, flat: torch.Tensor, query_coords: torch.Tensor, neighbor_coords: torch.Tensor,
                          manifold_positions: Optional[Dict[str, torch.Tensor]] = None,
                          neighbor_values: Optional[Dict[str, torch.Tensor]] = None) -> dict:
        """Same as :meth:`context` but the raw absolute hash encoding is supplied as an already-read leaf.

        The learned spatial-front and temporal-flow transformations remain inside the differentiable region;
        only the sparse hash-table read itself stays outside compilation.
        """
        flat = self.absolute_encoder.transform_precomputed(flat, query_coords)
        pos_s, pos_t = self._project_position(flat)
        if self.smooth_geo is not None:
            pos_s = pos_s + self.smooth_geo(query_coords)
        feats = {name: (self.neighbor_emb[name](val) if name in self.neighbor_emb else val)
                 for name, val in (neighbor_values or {}).items()}
        tokens = self.neighbors(query_coords, neighbor_coords, manifold_positions, feats)
        return {"position_s": pos_s, "position_t": pos_t, "position": pos_s + pos_t,
                "tokens": tokens, "coords": query_coords.detach()}

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
                floor = (1.0 - torch.log2(en.base_resolution)).view(1, -1)
                pls = en.per_level_scale.view(L, D).float().clamp_min(floor).clamp_max(20.0)
                scale = torch.exp2(pls) * en.base_resolution.view(1, D).float() - 1.0
                grad_pls = 0.6931471805599453 * (scale + 1.0) / scale * contrib
                grad_pls = torch.nan_to_num(grad_pls, nan=0.0, posinf=0.0, neginf=0.0).to(en.per_level_scale.dtype)
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
        if self.alphaearth_geo is not None and "alphaearth" in values:                       # SatCLIP-style geo prior: enrich the spatial position seen by every head
            _ae = self.alphaearth_geo(values["alphaearth"])
            pos_s = pos_s + _ae
            context = {**context, "position": context["position"] + _ae}
        w = torch.softmax(self.pos_channel_gate, dim=-1)                                     # [V,2] per-variable prior
        pos_v = w[:, 0].view(1, -1, 1) * pos_s.unsqueeze(1) + w[:, 1].view(1, -1, 1) * pos_t.unsqueeze(1)   # [B,V,d]
        pres = torch.stack([present[n] for n in self.names], dim=1)                          # [B,V] bool
        val = torch.stack([self._variable_token(n, values[n]) for n in self.names], dim=1)   # [B,V,d] value embeddings
        if self.diffusion:                                   # rule-22: masked slots begin as noise (round-0 state)
            content = torch.where(pres[..., None], val, torch.randn_like(val))
        else:
            content = torch.where(pres[..., None], val, self.mask_token) if self.learned_mask else val
        T = self._token_combine(self.tok_norm(content + self.type_emb), self.pos_norm(pos_v))  # [B,V,d] value x position
        if not self.learned_mask:
            T = T * pres[..., None].to(T.dtype)              # single-shot behavior: a masked slot is the zero vector
        ctx = [(context["position"] + self.position_token).unsqueeze(1), context["tokens"]]  # always-present position (combined) + neighbor tokens
        if context.get("cls_tokens") is not None: ctx.append(context["cls_tokens"])
        if context.get("experience") is not None: ctx.append(context["experience"])
        return self._refine(T, torch.cat(ctx, dim=1), pres, val, pos_v, gpos=context["position"])

    def _read_fn(self, q: torch.Tensor, kv: torch.Tensor, V: int) -> torch.Tensor:
        """The Perceiver READ. 'mha' is the stock cross-attention; variants restructure how latents pull the field."""
        op = self.read_op
        if op == "mha":
            return self.read(q, kv, kv)[0]
        if op == "crossself":
            keys = torch.cat([q, kv], dim=1)          # latents attend to themselves + the field jointly
            return self.read_cs(q, keys, keys)[0]
        if op == "typed":
            Tk = kv[:, :V]; Ck = kv[:, V:]
            rT = self.read_T(q, Tk, Tk)[0]
            if Ck.shape[1] == 0:                       # later rounds carry only variable-state tokens
                return rT
            rC = self.read_C(q, Ck, Ck)[0]
            g = torch.softmax(self.type_gate(q), dim=-1)
            return g[..., :1] * rT + g[..., 1:] * rC
        if op == "slot":
            B, L, d = q.shape; H = self.n_heads; dh = d // H; M = kv.shape[1]
            Q = self.slot_q(q).view(B, L, H, dh).transpose(1, 2)
            K = self.slot_k(kv).view(B, M, H, dh).transpose(1, 2)
            Vv = self.slot_v(kv).view(B, M, H, dh).transpose(1, 2)
            logits = torch.matmul(Q, K.transpose(-1, -2)) / (dh ** 0.5)   # [B,H,L,M]
            attn = torch.softmax(logits, dim=2)                          # latents COMPETE for each input token
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)        # per-slot weighted mean over inputs
            out = torch.matmul(attn, Vv).transpose(1, 2).reshape(B, L, d)
            return self.slot_o(out)
        raise ValueError(f"unknown read_op {op}")

    def _refine(self, T: torch.Tensor, C: torch.Tensor, present: torch.Tensor, value_emb: torch.Tensor,
                pos: torch.Tensor, gpos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """K rounds of joint-field denoising: the latents read the state+context tokens then attend among themselves;
        each round writes every variable's belief back into its state token, so masked variables are inducted jointly
        and observed ones may be revised. rounds=1 (no write-back) is the single-shot Processor."""
        z = self.latents.unsqueeze(0).expand(T.shape[0], -1, -1)
        gate = self.read_gate.view(1, 1, -1)
        stack = [] if self.round_loss == "all" else None
        for k in range(self.rounds):
            # The context C (neighbors, position) is fixed across rounds, so read it only in round 0; later rounds
            # refine against the updated variable states T alone (~5x fewer keys), which the latents already carry C from.
            V = T.shape[1]
            kv = self.kv_norm(torch.cat([T, C], dim=1) if k == 0 else T)
            q = self.q_norm(z)
            if self.read_cond and gpos is not None:
                gp = gpos.unsqueeze(1)
                q = q * (1.0 + self.read_film_g(gp)) + self.read_film_b(gp)
            r = torch.utils.checkpoint.checkpoint(self._read_fn, q, kv, V, use_reentrant=False) \
                if self.grad_checkpoint else self._read_fn(q, kv, V)
            z = z + gate * r
            for i, blk in enumerate(self.blocks):
                z = torch.utils.checkpoint.checkpoint(blk, z, use_reentrant=False) if self.grad_checkpoint else blk(z)
                if i < len(self.extra_reads):     # deeper fusion: re-read the context between blocks (read_depth>1)
                    z = z + gate * self.extra_reads[i](self.q_norm(z), kv, kv)[0]
            if stack is not None:
                stack.append(z)
            if self.write_back and k < self.rounds - 1:
                T = self._interface_update(z, value_emb, present, pos, k)
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
                          pos: torch.Tensor, k: int = 0) -> torch.Tensor:
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
        if self.diffusion:                                   # denoise: noise into masked slots decays to 0 by the final round
            sigma = max(0.0, 1.0 - (k + 1) / max(1, self.rounds - 1))
            masked = g * E + sigma * torch.randn_like(E)
        else:
            masked = g * E
        content = torch.where(present[..., None], obs, masked)
        T = self._token_combine(self.tok_norm(content + self.type_emb), self.pos_norm(pos))   # value x position
        return T.detach() if self.feedback_detach else T

    def _pooled(self, latents: torch.Tensor, name: str) -> torch.Tensor:
        """Attention-weighted pooling of the latents into one vector for reading variable ``name``."""
        i = self.names.index(name)
        w = torch.softmax((latents @ self.decode_query[i]) / (self.d_model ** 0.5), dim=-1)
        return torch.einsum("bl,bld->bd", w, latents)

    def _head_in(self, z: torch.Tensor, name: str, detach: bool = False) -> torch.Tensor:
        """Input to a phylo-conserved trait head. Baseline: the pooled latent. With phylo_head_routing: the pooled
        latent concatenated with the expected phylo-refined species embedding (posterior over the refined species
        table), so the graph's conserved signal reaches the head and the ablation (seed vs refined) moves the output."""
        pooled = self._pooled(z, name)
        if not self.phylo_head_routing or self._refined_species is None:
            return pooled.detach() if detach else pooled
        sp = self._reencode(self.species_variable, pooled)      # expected phylo-refined species embedding
        return torch.cat([pooled.detach(), sp.detach()], -1) if detach else torch.cat([pooled, sp], -1)

    @staticmethod
    def _frozen_head(module: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        for layer in module:
            if isinstance(layer, nn.Linear):
                x = F.linear(x, layer.weight.detach(),
                             None if layer.bias is None else layer.bias.detach())
            elif isinstance(layer, nn.GELU):
                x = F.gelu(x, approximate=layer.approximate)
            else:
                raise TypeError(f"unsupported frozen head layer: {type(layer).__name__}")
        return x

    def _blank_route(self, z: torch.Tensor, route: str) -> torch.Tensor:
        pooled = self._pooled(z, self.species_variable).detach()
        return pooled + self.blank_adapters[route](pooled)

    def _blank_species(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        corrected = self._blank_route(z, "species")
        species = corrected @ self._refined_species.detach().t()
        return corrected, species, self.blank_family(corrected)

    def _orthogonal_blank_loss(self, z: torch.Tensor, values: Dict[str, torch.Tensor],
                               observed: Dict[str, torch.Tensor], blank: torch.Tensor) -> torch.Tensor:
        _, species, family_residual = self._blank_species(z.detach())
        family = self._family_alphaearth_logits(values).detach() + family_residual
        species = self._factor_family_mass(species, family)
        target = values[self.species_variable]
        valid = (blank & observed[self.species_variable] & observed["alphaearth"]).float()
        species_loss = 0.5 * (
            (F.cross_entropy(species.float(), target, reduction="none") * valid).sum()
            / valid.sum().clamp_min(1.0) / math.log(species.shape[-1])
            + (F.cross_entropy(family.float(), self.species_family[target], reduction="none") * valid).sum()
            / valid.sum().clamp_min(1.0) / math.log(max(self.family_count, 2)))
        terms = [species_loss]
        if self._sdist_weight > 0 and "_sdist_idx" in values:
            corrected = self._blank_route(z, "community")
            logits = self._frozen_head(self.comm_head, corrected) @ self._refined_species.detach().t()
            idx = values["_sdist_idx"].clamp(0, logits.shape[1] - 1)
            dist = torch.zeros_like(logits, dtype=torch.float32).scatter_add_(
                1, idx, values["_sdist_frq"].float())
            terms.append((-(dist * F.log_softmax(logits.float(), -1)).sum(-1) * blank).sum()
                         / blank.sum().clamp_min(1) / math.log(logits.shape[-1]))
        return sum(terms)

    def _calibrated(self, name: str, pred: torch.Tensor) -> torch.Tensor:
        """Apply the continuous head's fitted per-dimension affine. Inert (and byte-identical) when the flag is off:
        the ParameterDict is empty, so the membership test is the whole cost."""
        if name in self.cal_gain:
            return pred * self.cal_gain[name] + self.cal_bias[name]
        return pred

    def decode(self, latents: torch.Tensor, name: str, calibrated: bool = True) -> torch.Tensor:
        """Read one variable back from the latents. The species variable reads against the refined species states; a diffusion variable is sampled from its head.

        ``calibrated=False`` returns the bare head output, which is what the mean-centered cosine training loss
        reads -- keeping the representation's gradient identical to the uncalibrated model."""
        if self.joint_decode:
            Pall = self._pooled_all(latents)                 # [B,V,d]
            Pall = Pall + self.joint_block(Pall)             # variables attend to each other (joint field)
            pooled = Pall[:, self.names.index(name)]
        else:
            pooled = self._pooled(latents, name)
        if name == self.species_variable and self._refined_species is not None:
            return self._pooled(latents, name) @ self._refined_species.t()
        pooled = self._pooled(latents, name)
        if name in self.diffusion_heads:
            return self.diffusion_heads[name].sample(pooled)
        out = self.decoders[name](pooled)
        return self._calibrated(name, out) if calibrated else out

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
    @staticmethod
    def _directional(ref: torch.Tensor) -> bool:
        """Are these targets L2-normalized, i.e. points on a sphere whose magnitude carries nothing? Decides which
        likelihood scores the variable, and therefore whether a magnitude calibration has anything to calibrate."""
        return bool(ref.numel()) and bool(((ref.norm(dim=-1) - 1.0).abs() < 1e-3).all())

    def _reconstruction_error(self, name: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        v = self.variables[self.names.index(name)]
        if v.kind == "categorical":
            # Normalize by log(num_classes) so every categorical term shares the ~[0,1] scale of the continuous cosine terms (a wide identity head otherwise dominates the shared gradient).
            return F.cross_entropy(pred, target, reduction="none") / math.log(max(int(v.num_classes), 2))
        # ANOMALY (mean-centered) cosine — matches the eval metric. Raw cosine has a gameable floor for shared-mean
        # embeddings: the decoder can satisfy it by predicting the MEAN direction and never learn the obs-specific
        # signal. Centering removes that floor, forcing the obs-specific reconstruction. Mean is over OBSERVED rows
        # (unobserved targets are 0 vectors).
        obs = target.norm(dim=-1) > 1e-6
        mu = (target[obs].mean(0, keepdim=True) if obs.any() else target.mean(0, keepdim=True)).detach()
        return 1.0 - F.cosine_similarity(pred - mu, target - mu, dim=-1)

    def _family_env_logits(self, values: Dict[str, torch.Tensor], observed: Dict[str, torch.Tensor],
                           context: dict, alphaearth_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Private early expert over environment tokens; the shared fusion path is detached."""
        tokens, valid = [], []
        for name in self.family_env_vars:
            i = self.names.index(name)
            token = self.tok_norm(self._variable_token(name, values[name]) + self.type_emb[i]).detach()
            tokens.append(token); valid.append(observed[name])
        if self.family_env_residual and self.family_ae_head is not None and "alphaearth" in values:
            tokens.append(self.family_ae_head[:-1](values["alphaearth"].detach()).detach())
            ae_valid = observed.get("alphaearth", alphaearth_valid)
            if ae_valid is None:
                ae_valid = torch.ones(values["alphaearth"].shape[0], dtype=torch.bool,
                                      device=values["alphaearth"].device)
            valid.append(ae_valid)
        tokens.append((context["position"] + self.position_token).detach())
        valid.append(torch.ones_like(valid[0]))
        kv = torch.stack(tokens, 1)
        mask = torch.stack(valid, 1)
        q = self.family_env_query.expand(kv.shape[0], -1, -1)
        pooled = self.family_env_attn(q, kv, kv, key_padding_mask=~mask, need_weights=False)[0][:, 0]
        return self.family_env_head(pooled)

    def _family_conditioned_logits(self, values: Dict[str, torch.Tensor], observed: Dict[str, torch.Tensor],
                                   context: dict, alphaearth_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = self._family_env_logits(values, observed, context, alphaearth_valid)
        if not self.family_env_residual or self.family_ae_head is None or "alphaearth" not in values:
            return residual
        valid = observed.get("alphaearth", alphaearth_valid)
        if valid is None:
            valid = torch.ones(values["alphaearth"].shape[0], dtype=torch.bool,
                               device=values["alphaearth"].device)
        prior = self._family_alphaearth_logits(values).detach()
        return torch.where(valid[:, None], prior + residual, residual)

    def _family_alphaearth_logits(self, values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Private habitat-to-family occupancy posterior from frozen AlphaEarth features."""
        return self.family_ae_head(values["alphaearth"].detach())

    def _occupancy_feature(self, values: Dict[str, torch.Tensor], observed: Dict[str, torch.Tensor],
                           context: dict) -> torch.Tensor:
        alphaearth = values["alphaearth"].detach() * observed["alphaearth"][:, None]
        with torch.no_grad():
            habitat = self.family_ae_head[:-1](alphaearth)
        return torch.cat((habitat, context["position"].detach()), -1)

    def _niche_feature(self, values: Dict[str, torch.Tensor], observed: Dict[str, torch.Tensor],
                       context: dict) -> torch.Tensor:
        coords = (context["coords"] - self.niche_coord_mean) / self.niche_coord_scale
        geo = [coords]
        for frequency in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            geo.extend((torch.sin(math.pi * frequency * coords),
                        torch.cos(math.pi * frequency * coords)))
        alphaearth = (values["alphaearth"].detach() - self.niche_ae_mean) / self.niche_ae_scale
        alphaearth = alphaearth * observed["alphaearth"][:, None]
        return self.niche_trunk(torch.cat((alphaearth, *geo), -1))

    def occupancy_expert_loss(self, values: Dict[str, torch.Tensor], observed: Dict[str, torch.Tensor],
                              context: dict) -> Optional[torch.Tensor]:
        if self.occupancy_experts is None:
            return None
        feature = self._occupancy_feature(values, observed, context)
        target = values[self.species_variable]
        valid = (observed[self.species_variable] & observed["alphaearth"]).float()
        base_family = self._family_conditioned_logits(
            values, observed, context, observed.get("alphaearth")).detach()
        family = base_family + self.occupancy_experts["family"](feature)
        niche = self._niche_feature(values, observed, context) if self.niche_trunk is not None else None
        if niche is not None:
            family = family + self.niche_experts["family"](niche)
        identity = self.occupancy_experts["identity"](feature)
        if niche is not None:
            identity = identity + self.niche_experts["identity"](niche)
        species = identity @ self._refined_species.detach().to(identity.dtype).t()
        species = self._factor_family_mass(species, family)
        losses = [
            (F.cross_entropy(species.float(), target, reduction="none") * valid).sum()
            / valid.sum().clamp_min(1.0) / math.log(species.shape[-1]),
            (F.cross_entropy(family.float(), self.species_family[target], reduction="none") * valid).sum()
            / valid.sum().clamp_min(1.0) / math.log(max(self.family_count, 2)),
        ]
        if "_sdist_idx" in values:
            community = self.occupancy_experts["community"](feature)
            if niche is not None:
                community = community + self.niche_experts["community"](niche)
            logits = community @ self._refined_species.detach().to(community.dtype).t()
            idx = values["_sdist_idx"].clamp(0, logits.shape[-1] - 1)
            dist = torch.zeros_like(logits, dtype=torch.float32).scatter_add_(
                1, idx, values["_sdist_frq"].float())
            keep = dist.sum(-1) > 0
            if keep.any():
                losses.append(-(dist[keep] * F.log_softmax(logits[keep].float(), -1)).sum(-1).mean()
                              / math.log(logits.shape[-1]))
        if "pollinator" in self.occupancy_experts and "_poll_idx" in values:
            pollinator = self.occupancy_experts["pollinator"](feature)
            if niche is not None and "pollinator" in self.niche_experts:
                pollinator = pollinator + self.niche_experts["pollinator"](niche)
            logits = pollinator @ self._pollinator_basis().detach().to(pollinator.dtype).t()
            idx = values["_poll_idx"].clamp(0, logits.shape[-1] - 1)
            dist = torch.zeros_like(logits, dtype=torch.float32).scatter_add_(
                1, idx, values["_poll_frq"].float())
            keep = values["_poll_valid"].bool()
            if keep.any():
                losses.append(-(dist[keep] * F.log_softmax(logits[keep].float(), -1)).sum(-1).mean()
                              / math.log(logits.shape[-1]))
        return sum(losses) / len(losses)

    def _factor_family_mass(self, species_logits: torch.Tensor, family_logits: torch.Tensor) -> torch.Tensor:
        prob = species_logits.float().softmax(-1)
        mass = prob.new_zeros(prob.shape[0], self.family_count)
        mass.scatter_add_(1, self.species_family.expand(prob.shape[0], -1), prob)
        correction = F.log_softmax(family_logits.float(), -1) - mass.clamp_min(1e-8).log()
        return species_logits + correction[:, self.species_family].to(species_logits.dtype)

    def _hierarchical_family_map(self, species_logits: torch.Tensor) -> torch.Tensor:
        """Promote the best species from the family with the most posterior mass."""
        logits = species_logits.float()
        family = self.species_family.expand(len(logits), -1)
        family_mass = logits.new_zeros(len(logits), self.family_count)
        family_mass.scatter_add_(1, family, logits.softmax(-1))
        winning = family_mass.argmax(-1)
        eligible = self.species_family[None] == winning[:, None]
        selected = logits.masked_fill(~eligible, -torch.inf).argmax(-1)
        top = logits.amax(-1).to(species_logits.dtype)
        return species_logits.scatter(1, selected[:, None], top[:, None] + 1e-4)

    def reconstruction_loss(self, values: Dict[str, torch.Tensor], observed: Dict[str, torch.Tensor],
                            context: dict, hide_prob: float = 0.35) -> torch.Tensor:
        """One training step: reveal each variable with prob ``1 - hide_prob`` and reconstruct every hidden-but-observed variable, at one fixed shape."""
        B = len(next(iter(observed.values()))); dev = self.type_emb.device
        present = {n: (torch.rand(B, device=dev) > hide_prob) & observed[n] for n in self.names}
        # Fully blank a fraction of queries so the model must reconstruct from bare space-time + neighbors, training the position->variable pathway (else the absolute channel stays inert at inference).
        blank = torch.rand(B, device=dev) < 0.15
        for n in self.names:
            present[n] = present[n] & ~blank
        loss = self.masked_loss(values, observed, present, context)
        if self.family_env_attn is not None:
            valid = observed[self.species_variable].to(loss.dtype)
            target = self.species_family[values[self.species_variable]]
            err = F.cross_entropy(self._family_conditioned_logits(values, observed, context), target, reduction="none")
            loss = loss + (err * valid).sum() / valid.sum().clamp_min(1.0) / math.log(max(self.family_count, 2))
        if self.family_ae_head is not None:
            valid = (observed[self.species_variable] & observed["alphaearth"]).to(loss.dtype)
            target = self.species_family[values[self.species_variable]]
            err = F.cross_entropy(self._family_alphaearth_logits(values), target, reduction="none")
            loss = loss + (err * valid).sum() / valid.sum().clamp_min(1.0) / math.log(max(self.family_count, 2))
        if self.blank_adapters:
            loss = loss + self._orthogonal_blank_loss(self._blank_query_z, values, observed, blank)
            self._blank_query_z = None
        return loss

    def masked_loss(self, values: Dict[str, torch.Tensor], observed: Dict[str, torch.Tensor],
                    present: Dict[str, torch.Tensor], context: dict) -> torch.Tensor:
        """Reconstruction loss for a fixed reveal mask (no randomness), so it can be compiled/CUDA-graphed with the random masking left outside."""
        z = self.encode(values, present, context)
        if self.blank_adapters:
            self._blank_query_z = z.detach()
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
                loss = loss + self._phylo_mask_weight * self.species_graph.masked_reconstruction_loss(
                    m, self._refined_species.detach(), metric="mse")
        # LCA fine-tuning: predict each masked species' mycorrhizal type from its RELATIVE-reconstructed embedding, so the
        # tree learns trait phylogenetic structure and imputes it for held-out clades (must beat BioCLIP nearest-neighbor).
        if self.training and getattr(self, "species_trait_recon", False) and getattr(self, "_species_myco", None) is not None:
            v = self._species_myco_valid & self._train_species                  # train species with myco labels (no held-out leak)
            if v.any():                                                          # DETACHED default (rule 31) reads the tree's phylo locality without reshaping it toward myco; trait_recon_detach=False + small weight tests gentle backprop (rule-31 tradeoff study).
                emb = self._refined_species[v] if not getattr(self, "_trait_recon_detach", True) else self._refined_species[v].detach()
                loss = loss + getattr(self, "_trait_recon_weight", 1.0) * F.cross_entropy(self.species_myco_head(emb), self._species_myco[v].clamp_min(0))
        # Pollinator phylo self-distillation (mirrors rule 25 for INTERACTIONS): a species' pollinator distribution
        # predicted from its PHYLO-RELATIVES (masked seed) must match its full-info prediction -> trains the model to
        # transfer interactions across the phylogeny. Diagnostic (2026-07-15): plant-phylo oracle reaches recall 0.216
        # but the model only 0.037 (B55) -> the signal is present and unexploited. Default weight 0 = champion-identical.
        if self.training and getattr(self, "_poll_phylo_weight", 0.0) > 0 and self._refined_species is not None \
                and getattr(self, "poll_head", None) is not None:
            m = torch.rand(self._refined_species.shape[0], device=z.device) < 0.15
            if m.any():
                basis = self._pollinator_basis().detach().t()                                        # [d, n_poll]
                full = F.softmax((self.poll_head(self._refined_species[m].detach()) @ basis).float(), dim=-1)   # target
                rel = F.log_softmax((self.poll_head(self.species_graph(mask=m)[m]) @ basis).float(), dim=-1)    # from relatives
                loss = loss + self._poll_phylo_weight * F.kl_div(rel, full, reduction="batchmean")
        return loss

    @torch.no_grad()

    def calibrate_nats(self, targets: Dict[str, torch.Tensor]) -> None:
        """Freeze the reference statistics val_bpb scores against, once, from a fixed reference draw.

        Without this the likelihood is estimated from whatever batch is in hand: the Gaussian variance moves with
        the sample and the retrieval negatives move with batch composition, so the absolute score depends on which
        examples happened to be drawn. Call once before scoring; every batch then uses the same reference.
        """
        self._nats_ref = {}
        for name, t in targets.items():
            t = t.float()
            obs = t.norm(dim=-1) > 1e-6
            ref = t[obs] if obs.any() else t
            if not ref.numel():
                continue
            if bool(((ref.norm(dim=-1) - 1.0).abs() < 1e-3).all()):
                self._nats_ref[name] = ("directional", F.normalize(ref, dim=-1))    # fixed retrieval bank
            else:
                self._nats_ref[name] = ("gaussian", ref.var(0, unbiased=False).clamp_min(1e-6))
        self._nats_floor = self._retrieval_floors()

    @torch.no_grad()
    def _retrieval_floors(self) -> Dict[str, float]:
        """The score a PERFECT predictor gets on each directional variable. Diagnostic only -- this
        changes no loss and no reported val_bpb.

        The bank is drawn with replacement from the test index, and several variables are per-species
        rather than per-observation (`phylo` is `self.phylo[cls]`), so the same row appears many times.
        Identical rows split the softmax mass, so predicting the target exactly still costs ~log(m)
        nats for multiplicity m. Measured: the phylo bank holds 925 unique species across 4096 rows,
        mean multiplicity 14.4, floor 2.11 nats against a chance of 8.32 -- a quarter of the apparent
        range is unreachable, which is why phylo reads as the worst variable in the decomposition.

        Computed by scoring the bank against itself: the perfect prediction for row i IS row i.
        """
        floors: Dict[str, float] = {}
        for name, (kind, stat) in getattr(self, "_nats_ref", {}).items():
            if kind != "directional":
                continue
            bank = stat
            total, n = 0.0, bank.shape[0]
            for i in range(0, n, 512):                       # chunked: the full n x n logit matrix is large
                chunk = bank[i:i + 512]
                sim = chunk @ bank.t()
                gold = sim.argmax(-1)
                total += float(F.cross_entropy(sim / RETRIEVAL_TEMPERATURE, gold, reduction="sum"))
            floors[name] = total / max(n, 1)
        return floors

    def retrieval_floors(self) -> Dict[str, float]:
        """Per-variable irreducible floor in nats, or {} before `calibrate_nats`. Report it beside the
        decomposition: a variable's headroom is its bits MINUS this, not its bits."""
        return dict(getattr(self, "_nats_floor", {}))

    def _reconstruction_nats(self, name: str, pred: torch.Tensor, target: torch.Tensor):
        """Held-out reconstruction in NATS, with the dimension count. Measurement only -- never the training loss.

        `_reconstruction_error` is deliberately rescaled (categorical CE divided by log C, continuous as a cosine
        distance) so the shared gradient stays balanced. Those are not log-likelihoods, so val_bpb computes its own.

        Reference statistics come from `calibrate_nats`, frozen once, never from the batch in hand -- otherwise the
        Gaussian variance and the retrieval negatives both move with whichever examples were sampled.

        Categorical: cross-entropy in nats; chance is log(num_classes).
        Directional (L2-normalized targets): retrieval against the frozen bank; chance is log(bank size).
        Continuous: diagonal Gaussian NLL against the frozen per-dimension variance. A differential entropy, so it
        is not zero-based -- predicting the reference mean scores the fitted Gaussian's entropy, and a perfect
        predictor scores that minus 0.5 nats per dimension. Only differences are meaningful.
        """
        v = self.variables[self.names.index(name)]
        if v.kind == "categorical":
            return F.cross_entropy(pred.float(), target, reduction="none"), 1
        t = target.float()
        ref_kind, ref_stat = getattr(self, "_nats_ref", {}).get(name, (None, None))
        if ref_kind is None:
            raise RuntimeError(f"call calibrate_nats() before scoring {name}: reference statistics are not frozen")
        if ref_kind == "directional":
            bank = ref_stat.to(t.device)                       # FIXED negatives, not the batch's own rows
            logits = (F.normalize(pred.float(), dim=-1) @ bank.t()) / RETRIEVAL_TEMPERATURE
            gold = (F.normalize(t, dim=-1) @ bank.t()).argmax(-1)     # each target's own row in the frozen bank
            return F.cross_entropy(logits, gold, reduction="none"), 1
        var = ref_stat.to(t.device)                            # FROZEN per-dimension variance
        nll = 0.5 * (((pred.float() - t) ** 2) / var + torch.log(2.0 * math.pi * var)).sum(-1)
        return nll, int(t.shape[-1])

    def variable_losses(self, z: torch.Tensor, values: Dict[str, torch.Tensor],
                        observed: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor]) -> Dict[str, tuple]:
        """Per-variable reconstruction error as ``{name: (summed_error, n_targets)}`` over the hidden-but-observed
        targets. The decomposition of val_bpb.

        Reconstruction terms only: the contrastive, sdist and pollinator auxiliaries are training regularizers on
        detached or global signals, not per-variable held-out reconstruction, so they are excluded. Unweighted --
        `loss_weights` steers training, it does not change how many bits a variable actually costs.
        """
        out = {}
        for v in self.variables:
            if not v.reconstruct:
                continue
            w = ((~present[v.name]) & observed[v.name]).to(z.dtype)
            n = int(w.sum())
            if not n:
                continue
            pred = (self._pooled(z, v.name) if v.name in self.diffusion_heads else self.decode(z, v.name))
            if v.name in self.diffusion_heads:
                # A diffusion head samples rather than scoring a density. Silently omitting it would let val_bpb
                # improve while that modality regressed -- fail instead of dropping it.
                raise NotImplementedError(
                    f"{v.name} is scored by a diffusion head; val_bpb has no likelihood for it. "
                    "Give the head a log-density or exclude the variable from the objective explicitly.")
            nats, per_row = self._reconstruction_nats(v.name, pred, values[v.name])
            out[v.name] = (float((nats * w).sum()), n * per_row)
        return out

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
                pred = self.decode(z, v.name, calibrated=False)
                err = self._reconstruction_error(v.name, pred, values[v.name])
                # Fit the head's calibration affine by the same standardized Gaussian the objective scores, on a
                # DETACHED prediction: the gradient reaches cal_gain/cal_bias and nothing else, so the representation
                # trains exactly as it does with the flag off. Isolates "the head has no scale" from "the
                # representation is wrong" -- the two ways a continuous variable can be expensive in bits.
                if v.name in self.cal_gain:
                    tgt = values[v.name].float()
                    seen = tgt.norm(dim=-1) > 1e-6
                    ref = tgt[seen] if seen.any() else tgt
                    # Only the Gaussian-scored variables. A directional target's magnitude is fixed at 1 and carries
                    # no information, so there is nothing to calibrate; its affine is never given a gradient, stays
                    # at exactly ones/zeros, and leaves that head's output bit-identical.
                    if not self._directional(ref):
                        var = ref.var(0, unbiased=False).clamp_min(1e-6).detach()
                        cal = self._calibrated(v.name, pred.detach().float())
                        z2 = (((cal - tgt) ** 2) / var).mean(-1)  # per-dimension; 1.0 = predicting the target mean
                        loss = loss + (z2 * w).sum() / w.sum().clamp_min(1.0)
                # Cross-modal contrastive (JEPA / rule 16): the predicted continuous embedding must retrieve its own
                # target against the batch (InfoNCE) -- a global signal point-wise cosine cannot give.
                if self.contrastive_weight > 0.0 and v.name in self.contrastive_vars and v.kind == "continuous":
                    pn = F.normalize(pred.float(), dim=-1); tn = F.normalize(values[v.name].float(), dim=-1)
                    logits = pn @ tn.t() / 0.05    # InfoNCE temperature
                    loss = loss + self.contrastive_weight * F.cross_entropy(
                        logits, torch.arange(logits.shape[0], device=logits.device))
            # Per-variable loss weight (science.md rule 18): focus reconstruction where benchmarks have headroom.
            loss = loss + self.loss_weights.get(v.name, 1.0) * (err * w).sum() / w.sum().clamp_min(1.0)
            # Distribution matching (MADE joint-dist) via the SEPARATE community head on a DETACHED latent: trains only
            # comm_head toward the local community distribution, with zero gradient into the shared representation.
            if getattr(self, "_sdist_weight", 0.0) > 0 and v.name == self.species_variable \
                    and "_sdist_idx" in values and getattr(self, "comm_head", None) is not None:
                _cp = self._pooled(z, v.name)                                   # [Ensue] comm_attached: let the community loss shape the backbone
                _cp = _cp if getattr(self, "_comm_attached", False) else _cp.detach()
                comm = (self.comm_head(_cp) @ self._refined_species.detach().t()).float()
                sidx = values["_sdist_idx"].clamp(0, comm.shape[1] - 1)   # -1 padding -> 0 (its freq is 0, harmless)
                tgt = torch.zeros_like(comm).scatter_add_(1, sidx, values["_sdist_frq"].float())
                kl = -(tgt * F.log_softmax(comm, -1)).sum(-1)            # soft cross-entropy toward the local distribution
                loss = loss + self._sdist_weight * (kl * w).sum() / w.sum().clamp_min(1.0)
            # Pollinator distribution matching via the SEPARATE detached poll_head toward the plant's GloBI pollinator
            # distribution (zero gradient into the shared representation).
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
                pred = self.lfmc_head(self._head_in(z, v.name, detach=True)).squeeze(-1).float()
                tgt = torch.log(values["_lfmc"].clamp_min(1.0)); lv = values["_lfmc_valid"].float()
                loss = loss + self._lfmc_weight * ((pred - tgt) ** 2 * lv).sum() / lv.sum().clamp_min(1.0)
            # Symbiosis (B42): detached head predicts the mycorrhizal type (cross-entropy toward the FungalRoot label)
            if getattr(self, "_myco_weight", 0.0) > 0 and v.name == self.species_variable \
                    and "_myco" in values and getattr(self, "myco_head", None) is not None:
                logit = self.myco_head(self._head_in(z, v.name, detach=True))
                mv = values["_myco_valid"].float()
                ce = F.cross_entropy(logit, values["_myco"].clamp_min(0), reduction="none")
                loss = loss + self._myco_weight * (ce * mv).sum() / mv.sum().clamp_min(1.0)
            # Phenology (B26): detached head predicts flowering (BCE) toward the per-observation PhenoVision label
            if getattr(self, "_flower_weight", 0.0) > 0 and v.name == self.species_variable \
                    and "_flower" in values and getattr(self, "flower_head", None) is not None:
                logit = self.flower_head(self._head_in(z, v.name, detach=True)).squeeze(-1).float()
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

    def _pollinator_species_posterior(self, species_logits: torch.Tensor) -> torch.Tensor:
        k = min(64, species_logits.shape[-1])
        weight, species = F.softmax(species_logits.float(), -1).topk(k, -1)
        idx = self.poll_species_idx[species].clamp(0, self.poll_emb.shape[0] - 1)
        mass = weight[:, :, None] * self.poll_species_frq[species]
        mixture = species_logits.new_zeros(species_logits.shape[0], self.poll_emb.shape[0], dtype=torch.float32)
        mixture.scatter_add_(1, idx.flatten(1), mass.flatten(1))
        mixture = mixture / mixture.sum(-1, keepdim=True).clamp_min(1e-8)
        return mixture.clamp_min(1e-8).log()

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
        context_z = None
        if self.blank_adapters and "community" in targets:
            context_present = {n: torch.zeros_like(present[n]) for n in self.names}
            context_z = self.encode(values, context_present, context)
        ae_valid = observed.get("alphaearth") if observed is not None else None
        family_logits = self._family_conditioned_logits(values, present, context, ae_valid) \
            if self.family_env_attn is not None and tuple(given) == self.family_env_vars else None
        family_valid = None
        if self.family_ae_head is not None and not given:
            family_logits = self._family_alphaearth_logits(values)
            family_valid = observed.get("alphaearth") if observed is not None else None
        universal = tuple(given) == self.family_env_vars
        occupancy = None
        niche = None
        if self.occupancy_experts is not None and (not given or universal):
            occupancy_observed = observed or {
                "alphaearth": torch.ones(B, dtype=torch.bool, device=dev),
            }
            occupancy = self._occupancy_feature(values, occupancy_observed, context)
            if self.niche_trunk is not None:
                niche = self._niche_feature(values, occupancy_observed, context)
            if family_logits is not None:
                family_logits = family_logits + self.occupancy_experts["family"](occupancy)
                if niche is not None:
                    family_logits = family_logits + self.niche_experts["family"](niche)
        blank_species = self._blank_species(z) if self.blank_adapters and not given else None
        if blank_species is not None and family_logits is not None:
            family_logits = family_logits + blank_species[2]
        poll_species = None
        if self.poll_species_idx is not None and (
                not given or universal or (self.poll_species_all_masked and self.species_variable not in given)):
            species_logits = blank_species[1] if blank_species is not None else self.decode(z, self.species_variable)
            if occupancy is not None:
                species_logits = species_logits + self.occupancy_experts["identity"](occupancy) @ self._refined_species.t()
            if niche is not None:
                species_logits = species_logits + self.niche_experts["identity"](niche) @ self._refined_species.t()
            if family_logits is not None:
                factored = self._factor_family_mass(species_logits, family_logits)
                species_logits = torch.where(family_valid[:, None], factored, species_logits) \
                    if family_valid is not None else factored
            poll_species = self._pollinator_species_posterior(species_logits)
        out = {}
        for t in targets:
            if t == "community":                                                 # env-conditioned community distribution
                pooled = self._pooled(z, self.species_variable)
                if getattr(self, "comm_head", None) is not None:
                    logits = self.comm_head(pooled) @ self._refined_species.t()
                    if context_z is not None:
                        base = self.comm_head(self._pooled(context_z, self.species_variable))
                        corrected = self.comm_head(self._blank_route(context_z, "community"))
                        logits = logits + (corrected - base) @ self._refined_species.t()
                    out[t] = logits
                else:
                    out[t] = self.decode(z, self.species_variable)                # fallback: identity posterior
                if occupancy is not None and universal:
                    out[t] = out[t] + self.occupancy_experts["community"](occupancy) @ self._refined_species.t()
                if niche is not None and universal:
                    out[t] = out[t] + self.niche_experts["community"](niche) @ self._refined_species.t()
            elif t == "pollinator":                                              # plant -> pollinator interaction (rule 27)
                out[t] = self.poll_head(self._pooled(z, self.species_variable)) @ self._pollinator_basis().t()
                if occupancy is not None and "pollinator" in self.occupancy_experts:
                    out[t] = out[t] + self.occupancy_experts["pollinator"](occupancy) @ self._pollinator_basis().t()
                if niche is not None and "pollinator" in self.niche_experts:
                    out[t] = out[t] + self.niche_experts["pollinator"](niche) @ self._pollinator_basis().t()
                if poll_species is not None:
                    alpha = self.poll_species_mixture
                    out[t] = poll_species if alpha >= 1.0 else torch.logaddexp(
                        F.log_softmax(out[t].float(), -1) + math.log1p(-alpha),
                        poll_species + math.log(alpha))
            elif t == "lfmc":                                                    # species -> live fuel moisture (B34)
                out[t] = self.lfmc_head(self._head_in(z, self.species_variable)).squeeze(-1).exp()
            elif t == "myco":                                                    # species -> mycorrhizal type logits (B42)
                out[t] = self.myco_head(self._head_in(z, self.species_variable))
            elif t == "flower":                                                  # observation -> flowering probability (B26)
                out[t] = torch.sigmoid(self.flower_head(self._head_in(z, self.species_variable)).squeeze(-1))
            else:
                out[t] = blank_species[1] if blank_species is not None and t == self.species_variable \
                    else self.decode(z, t)
                if t == self.species_variable and occupancy is not None:
                    out[t] = out[t] + self.occupancy_experts["identity"](occupancy) @ self._refined_species.t()
                if t == self.species_variable and niche is not None:
                    out[t] = out[t] + self.niche_experts["identity"](niche) @ self._refined_species.t()
                if t == self.species_variable and family_logits is not None:
                    factored = self._factor_family_mass(out[t], family_logits)
                    out[t] = torch.where(family_valid[:, None], factored, out[t]) \
                        if family_valid is not None else factored
                if t == self.species_variable and self.ecological_family_map and (not given or universal):
                    out[t] = self._hierarchical_family_map(out[t])
        return out
