"""DeepEarth: a general model of spatio-temporally covarying variables.

At any point in space and time the world presents a set of covarying variables — measurements and derived
representations of light, matter, life, and climate. None is privileged over another; all covary through shared
physical and historical structure. DeepEarth models that covariation directly: given whichever variables are
observed at a location, together with the variables observed at nearby places and times, it infers the variables
that are not observed. It learns by repeatedly hiding a random subset of the variables and reconstructing them
from the rest, so that at inference any variable can be predicted from any others.

Space and time enter through two channels, from a single encoder (:class:`Earth4D`):
  * an absolute channel — a coarse, smoothly varying memory of *where and when* an observation sits (regional
    structure a model can legitimately memorize);
  * a relative channel — an encoding of the *offset* to each neighboring observation, which depends only on the
    relative configuration and therefore transfers structure learned at one place and time to all others.

The model is config-driven: the set of variables (their names, whether continuous or categorical, their widths,
whether they are reconstruction targets, and whether they are also carried from neighbors) is passed in, not
hard-coded. A small set of learnable latent vectors attends over the variable-length set of present variable
tokens plus the space-time context tokens, then attends among itself; each variable is read back from the latents.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepearth.encoders.spacetime.earth4d import Earth4D
from deepearth.core.context import NeighborContext
from deepearth.encoders.biological.phylogenomic import SpeciesGraph


@dataclass
class Variable:
    """One variable DeepEarth models at each observation.

    Args:
        name: unique variable name.
        kind: ``"continuous"`` (a vector; reconstructed by directional/cosine agreement) or ``"categorical"``
            (a class label; reconstructed as a probability over classes).
        dim: vector width for continuous variables.
        num_classes: number of classes for categorical variables.
        reconstruct: whether the variable is a reconstruction target (set ``False`` for input-only conditioning).
        neighbor: whether the variable is also carried from spatial-temporal neighbors into the context.
    """
    name: str
    kind: str
    dim: int = 0
    num_classes: int = 0
    reconstruct: bool = True
    neighbor: bool = False


class DeepEarth(nn.Module):
    """Config-driven model of spatio-temporally covarying variables (see module docstring).

    Args:
        variables: the variables to model.
        d_model, n_latents, n_layers, n_heads: latent-attention backbone width, latent count, depth, heads.
        relative_window: half-extent of the neighbor-offset window per axis, in metres and time units; size it to
            how far neighbors reach (see :meth:`Earth4D.fit_relative_window`).
        manifolds: additional vector subspaces treated like space and time, as ``{name: dim}`` (e.g. a biological
            manifold, ``{"biological": 2048}``).
        capacity: resolution of the space-time encoders (number of hash levels); higher resolves finer structure.
        reference_latitude_deg: latitude used when converting geographic offsets to metres.
    """

    # Community dropout: EVERY point-prediction benchmark queries with geometry-only neighbors (space-time offsets,
    # NO neighbor identity/vision/phylo -- see demos.batch_context "geometry" mode). Training, however, always feeds
    # the full neighbor community, so the model learns to read biology off the community crutch and then collapses to
    # baseline at eval when the crutch is gone (the persistent zeros A1/A5/A6/Q5/Q7). Dropping the community biotic
    # signal on a fraction of training rows removes that train/eval mismatch and forces genuine universal->biology
    # induction. Train-split only; the geometry (space-time offset) neighbors are kept.
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
        # A dedicated, always-present space-time token. Position is otherwise fused only onto variable tokens, which
        # are zeroed by the present-mask, so a query that reveals no variable (predict from bare place-time) loses its
        # position entirely and the absolute Earth4D channel goes inert at inference. This token always carries it.
        self.position_token = nn.Parameter(torch.randn(d_model) * 0.02)
        # exp9 (information flow): position is correctly integrated onto every modality token (that IS the positional
        # encoder), but the absolute encoding (norm ~9) was swamping the variable content (norm ~0.3-5), so tokens
        # carried "where" not "what" and the absolute encoder's gradient ran away. Fix WITHOUT weakening position:
        # normalize content and position to matched (unit) scale before adding — standard token-embedding + PE
        # practice — so both are legible AND the norm's backward bounds the runaway absolute-encoder gradient.
        self.tok_norm = nn.LayerNorm(d_model)
        self.pos_norm = nn.LayerNorm(d_model)
        self.decode_query = nn.Parameter(torch.randn(len(self.variables), d_model) * 0.02)

        # Absolute location memory: broad regional / long-period memorization of where and when an observation sits
        # (coarse -- the fine, high-frequency structure lives in the relative encoder). Sized to ~200M (20% of Earth4D).
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

        # species graph: refine the identity representation through phylogenetic-neighbor attention, so an
        # observation of one species informs its relatives and identity is read from the refined representation
        # The operator refines the species states over evolutionary structure: ``tree`` propagates over the actual
        # dated phylogeny (topology + branch lengths, ancestral state at internal nodes); ``ou-attention`` biases
        # attention with a distance matrix (here derived from the embedding, a lossy shadow of the tree).
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

        # Optional modules — scale-mixing, diffusion, experience, inductive placement — are integrated but were never
        # validated in the champion (off by default); they live on the deepcal-research branch. This build keeps them
        # inert so the forward's ``is not None`` / empty-dict guards take their default (no-op) path.
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
                neighbor_values: Optional[Dict[str, torch.Tensor]] = None,
                batch_indices: Optional[torch.Tensor] = None) -> dict:
        """Encode the query's space-time position and its neighbor tokens.

        Args:
            query_coords: ``[B, 4]`` = (latitude, longitude, elevation, time).
            neighbor_coords: ``[B, K, 4]`` for each observation's K neighbors.
            manifold_positions: subspace name -> neighbor positions ``[B, K, dim]`` in that vector subspace.
            neighbor_values: variable name -> neighbor values ``[B, K, ...]`` to project in.
        Returns ``{"position": [B, d_model], "tokens": [B, subspaces * K, d_model]}``; the position is fused onto
        every variable token, and the tokens are one per (neighbor, subspace). Pass the whole dict as ``context``
        to :meth:`reconstruction_loss` / :meth:`infer`.
        """
        # Community dropout (train only): on a fraction of rows, blank the neighbor community — zero the neighbor
        # biotic feature values and the biological (phylo) manifold positions — leaving only the space-time offset
        # geometry. This matches how every point-prediction benchmark queries the model at eval and trains genuine
        # universal->biology induction instead of leaning on a community crutch it never gets at inference.
        if self.training and self.COMMUNITY_DROPOUT > 0.0 and (neighbor_values or manifold_positions):
            Bd = query_coords.shape[0]
            keep = (torch.rand(Bd, device=query_coords.device) >= self.COMMUNITY_DROPOUT)
            if neighbor_values:
                neighbor_values = {n: v * keep.view(Bd, *([1] * (v.dim() - 1))).to(v.dtype)
                                   for n, v in neighbor_values.items()}
            if manifold_positions:
                manifold_positions = {n: p * keep.view(Bd, *([1] * (p.dim() - 1))).to(p.dtype)
                                      for n, p in manifold_positions.items()}
        # Sparse-hash path: read the absolute encoder from its precomputed indices as a detached leaf, so the hash
        # trains through the sparse Adam optimizer (which only touches the entries this batch reads) rather than a
        # full step over the whole table. The leaf's gradient is captured for accumulate_grad/adam_step.
        if getattr(self, "_sparse_hash", False) and batch_indices is not None:
            flat = self.absolute_encoder.forward_precomputed(batch_indices).detach().requires_grad_(True)
            self._abs_leaf = (flat, batch_indices)
            position = self.absolute_proj(flat)
        else:
            position = self.absolute_proj(self.absolute_encoder(query_coords))
        feats = {name: (self.neighbor_emb[name](val) if name in self.neighbor_emb else val)
                 for name, val in (neighbor_values or {}).items()}
        tokens = self.neighbors(query_coords, neighbor_coords, manifold_positions, feats)
        return {"position": position, "tokens": tokens}

    def context_from_flat(self, flat: torch.Tensor, query_coords: torch.Tensor, neighbor_coords: torch.Tensor,
                          manifold_positions: Optional[Dict[str, torch.Tensor]] = None,
                          neighbor_values: Optional[Dict[str, torch.Tensor]] = None) -> dict:
        """Same as :meth:`context`, but the absolute encoding is supplied as an already-read leaf ``flat`` (from
        ``forward_precomputed``) instead of computed here. This keeps the precompute+detach out of the compiled
        region, so the rest of the step still graph-captures while the hash trains through sparse Adam."""
        position = self.absolute_proj(flat)
        feats = {name: (self.neighbor_emb[name](val) if name in self.neighbor_emb else val)
                 for name, val in (neighbor_values or {}).items()}
        tokens = self.neighbors(query_coords, neighbor_coords, manifold_positions, feats)
        return {"position": position, "tokens": tokens}

    def set_memory(self, key: torch.Tensor, features: Dict[str, torch.Tensor]) -> None:
        """Install the experience-replay memory bank (a key and per-anchor features), refreshed between epochs."""
        self._memory_key = key
        self._memory_features = features

    def enable_sparse_hash(self, coords: torch.Tensor, lr: float = 3e-4, weight_decay: float = 3e-4) -> None:
        """Precompute the absolute encoder for a fixed coordinate set and route it through sparse Adam. The hash
        table dominates the parameters, but each batch only reads a few entries, so a full optimizer step is wasteful.
        After this, pass ``batch_indices`` (row indices into ``coords``) to :meth:`context`, and call
        :meth:`sparse_hash_step` after ``backward``. The absolute embeddings then update outside the main optimizer."""
        self.absolute_encoder.precompute(coords)
        e = self.absolute_encoder
        self._abs_encs = [e.xyz_encoder, e.xyt_encoder, e.yzt_encoder, e.xzt_encoder]
        for en in self._abs_encs:
            en.init_sparse_adam(lr=lr, weight_decay=weight_decay)
        self._abs_odims = [en.num_levels * en.level_dim for en in self._abs_encs]
        self._abs_L = e.xyz_encoder.num_levels
        self._abs_F = e.features_per_level
        self._sparse_hash = True

    def absolute_hash_params(self):
        """The absolute-encoder embeddings, optimized by sparse Adam and so excluded from the main optimizer."""
        return [en.embeddings for en in self._abs_encs]

    def set_sparse_lr(self, lr: float) -> None:
        for en in self._abs_encs:
            en.set_adam_lr(lr)

    def sparse_hash_step(self, flat: torch.Tensor = None, bidx: torch.Tensor = None) -> None:
        """Apply the sparse Adam update to the absolute encoder from the leaf gradient. Call after ``loss.backward()``.
        Pass the leaf explicitly (compiled path) or omit to use the one captured in :meth:`context` (eager path). The
        accumulation buffer must be cleared first: ``adam_step`` does not clear it."""
        if flat is None:
            flat, bidx = self._abs_leaf
        g = flat.grad
        off = 0
        for en, d in zip(self._abs_encs, self._abs_odims):
            en._adam_grad_buffer.zero_()
            en.accumulate_grad(g[:, off:off + d].contiguous(), bidx)
            off += d
        for en in self._abs_encs:
            en.adam_step(bidx)

    def encode(self, values: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor], context: dict) -> torch.Tensor:
        """Fuse the present variable tokens and the neighbor tokens into the latents.

        Every variable token carries the query's space-time position, so each token entering the latent attention
        knows where and when it sits. A token is scaled by its present-mask; the latents then read across the full
        token set and attend among themselves. The step holds one fixed shape, so it runs as a single fused
        schedule and can be graph-captured.
        """
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
        """The latent-attention Processor: the latents read the token set, then attend among themselves. Pure
        PyTorch (no custom CUDA), so it compiles cleanly while the Earth4D hash kernel stays eager."""
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
        """Read one variable back from the latents. The species variable is read against the refined species
        representations; a diffusion variable is sampled from its head conditioned on the pooled latent."""
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
            # Normalize by log(num_classes): a 2141-way identity head reaches CE ~21 at init and otherwise consumes
            # ~60% of the shared-backbone gradient, starving the high-dimensional continuous modalities. Dividing by
            # log(K) puts every categorical term on the same ~[0,1] scale as the continuous cosine terms.
            return F.cross_entropy(pred, target, reduction="none") / math.log(max(int(v.num_classes), 2))
        return 1.0 - F.cosine_similarity(pred, target, dim=-1)

    def reconstruction_loss(self, values: Dict[str, torch.Tensor], observed: Dict[str, torch.Tensor],
                            context: dict, hide_prob: float = 0.35) -> torch.Tensor:
        """One training step: reveal each variable as an input with probability ``1 - hide_prob`` and reconstruct
        every hidden-but-observed variable. The per-variable error runs across the full batch and is weighted by
        the hidden-observed mask, keeping the step at one fixed shape."""
        B = len(next(iter(observed.values()))); dev = self.type_emb.device
        present = {n: (torch.rand(B, device=dev) > hide_prob) & observed[n] for n in self.names}
        # For a fraction of queries, hide EVERY variable, so the model must reconstruct from bare space-time position
        # and the neighbor community alone. Without this the model never trains the position->variable pathway (a
        # revealed modality always carried the answer), and the absolute Earth4D channel stays inert at inference.
        blank = torch.rand(B, device=dev) < 0.15
        for n in self.names:
            present[n] = present[n] & ~blank
        return self.masked_loss(values, observed, present, context)

    def masked_loss(self, values: Dict[str, torch.Tensor], observed: Dict[str, torch.Tensor],
                    present: Dict[str, torch.Tensor], context: dict) -> torch.Tensor:
        """The reconstruction loss for a given reveal mask (no randomness), so it can be compiled and CUDA-graphed
        with the random masking left outside the captured region."""
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
