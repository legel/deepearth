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
        compile_processor: bool = False,
    ) -> None:
        super().__init__()
        self.variables = list(variables)
        self.names = [v.name for v in self.variables]
        self.d_model = d_model

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for v in self.variables:
            if v.kind == "continuous":
                self.encoders[v.name] = nn.Linear(v.dim, d_model)
                if v.reconstruct:
                    self.decoders[v.name] = nn.Linear(d_model, v.dim)
            elif v.kind == "categorical":
                if v.name == species_variable and species_embedding is not None:
                    continue          # species tokens and logits use the refined species graph, not these dead heads
                self.encoders[v.name] = nn.Embedding(v.num_classes, d_model)
                if v.reconstruct:
                    self.decoders[v.name] = nn.Linear(d_model, v.num_classes)
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
        self.absolute_proj = nn.Sequential(nn.Linear(self.absolute_encoder.output_dim, d_model), nn.GELU(),
                                           nn.Linear(d_model, d_model))
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
                                                  tree=species_tree, n_layers=species_layers)
            else:
                distance = SpeciesGraph.distance_from_embedding(species_embedding)
                self.species_graph = SpeciesGraph(species_embedding.shape[0], d_model, distance,
                                                  n_heads=species_heads, n_layers=species_layers,
                                                  top_k=species_top_k, flex=species_flex)
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
        if compile_processor:
            self._process = torch.compile(self._process)

    # ---------------------------------------------------------------- tokens
    def _variable_token(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if name == self.species_variable and self._refined_species is not None:
            return self._refined_species[value.clamp(min=0)]
        v = self.variables[self.names.index(name)]
        return self.encoders[name](value if v.kind == "continuous" else value.clamp(min=0))

    def context(self, query_coords: torch.Tensor, neighbor_coords: torch.Tensor,
                manifold_positions: Optional[Dict[str, torch.Tensor]] = None,
                neighbor_values: Optional[Dict[str, torch.Tensor]] = None) -> dict:
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
        position = self.absolute_proj(self.absolute_encoder(query_coords))
        feats = {name: (self.neighbor_emb[name](val) if name in self.neighbor_emb else val)
                 for name, val in (neighbor_values or {}).items()}
        tokens = self.neighbors(query_coords, neighbor_coords, manifold_positions, feats)
        return {"position": position, "tokens": tokens}

    def set_memory(self, key: torch.Tensor, features: Dict[str, torch.Tensor]) -> None:
        """Install the experience-replay memory bank (a key and per-anchor features), refreshed between epochs."""
        self._memory_key = key
        self._memory_features = features

    def encode(self, values: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor], context: dict) -> torch.Tensor:
        """Fuse the present variable tokens and neighbor tokens into the latents. Every variable token carries the query position; tokens are scaled by the present-mask, then the latents read across them and attend among themselves."""
        if self.species_graph is not None:
            self._refined_species = self.species_graph()     # refine all species once per forward
        B = len(next(iter(present.values())))
        toks = []
        for i, name in enumerate(self.names):
            t = self.tok_norm(self._variable_token(name, values[name]) + self.type_emb[i]) + self.pos_norm(context["position"])
            toks.append((t * present[name][:, None].to(t.dtype)).unsqueeze(1))
        toks.append((context["position"] + self.position_token).unsqueeze(1))   # always present: survives full masking
        toks.append(context["tokens"])
        if context.get("cls_tokens") is not None:
            toks.append(context["cls_tokens"])
        if context.get("experience") is not None:
            toks.append(context["experience"])
        return self._process(torch.cat(toks, dim=1))

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        """Latent-attention Processor: the latents read the token set, then attend among themselves. Pure PyTorch, so it compiles cleanly while the Earth4D hash kernel stays eager."""
        z = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        z = z + self.read(self.q_norm(z), self.kv_norm(x), self.kv_norm(x))[0]
        for blk in self.blocks:
            z = blk(z)
        return z

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
        loss, n_terms = z.new_zeros(()), 0
        for v in self.variables:
            if not v.reconstruct:
                continue
            w = ((~present[v.name]) & observed[v.name]).to(z.dtype)
            if v.name in self.diffusion_heads:
                err = self.diffusion_heads[v.name].loss(values[v.name], self._pooled(z, v.name), reduce=False)
            else:
                err = self._reconstruction_error(v.name, self.decode(z, v.name), values[v.name])
            loss = loss + (err * w).sum() / w.sum().clamp_min(1.0)
            n_terms += 1
        loss = loss / max(n_terms, 1)
        if self.inductive is not None:                       # auxiliary: name embedding -> evolutionary position
            loss = loss + 0.1 * self.inductive.loss(self._species_text, self._species_e1)
        return loss

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
        return {t: self.decode(z, t) for t in targets}
