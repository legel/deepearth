# Scientific Principles of DeepEarth

DeepEarth is a **self-supervised multi-modal deep learning architecture for ecological simulation and optimization**.

The model features two innovations in learning representations:

- **I. Earth4D Space-Time GNN** — learnable global positional encodings for real world observations.
- **II. Phylogenomic Species GNN** — learnable biological encodings, based on evolutionary tree of life.

The model is a **causal forecaster**, e.g. predictor of plant growth and flowering.

DeepEarth is based on the following foundations:

- **I. Vision Encoders: [DINOv3][dinov3], [CLAY v1.5][clay]** — embedding iNaturalist, NAIP, Sentinel-2 imagery.
- **II. Graph Neural Networks: [GraphCast][graphcast], [GenCast][gencast]** — causal spatio-temporal simulators.
- **III. Multi-Modal Fusion: [JEPA][jepa], [PerceiverIO][perceiverio]** — joint probability distribution embedding.

DeepEarth **learns through masked autoencoding**, including by masking and reconstructing *embeddings*. 

## Rules and guidelines for future development

1. **Earth4D must learn the spatio-temporal distributions of all data**, through a causal auto-regressive model
   trained to forecast future states from past states.
2. **Earth4D must learn via absolute & relative coordinate systems:**
   - **A.** Its **absolute encoder** must remember dynamics in absolute spatial and temporal coordinates, e.g. GPS
     and Month/Day/Year HH:MM:SS, similar to a NeRF-like GIS tool.
   - **B.** Its **relative encoder** must input limited context windows, focused on a limited spatial region, going
     back in time, like a physics-inspired 4D LSTM.
3. **Earth4D must train positional encodings that are fused with each token** in each context window during each
   batch, becoming a unifying fabric for all data inputs.
4. **Earth4D must remain at least as fast and compute-optimized as it currently is**, e.g. based on CUDA kernels now
   in production, originally programmed by NVIDIA.
5. **Earth4D must retain large-scale learning capacity;** small models must have no less than 100M parameters, while
   base models should target 1B params.
6. **Earth4D must retain parallelizable memory and compute architectures**, such that several subsets of geography
   and time can be concurrently processed at scale.
7. **The phylogenomic species network must preserve embeddings per species**, which share processing topologically
   according to their evolutionary history.
8. **The phylogenomic species network must be self-supervised**, 100% based on scientifically-derived trees.
9. **The phylogenomic species network must have an embedding representation**, in which new species with unknown
   status in the tree of life can be projected.
10. **The phylogenomic species network must be learnable**, with updates to both species in-context and neighboring
    in-context species, during every batch.
11. Therefore, learning during a batch that only includes observations of Species A must necessarily **update
    embeddings of Species B, C, ..., n-th neighbor**.
12. Therefore, the phylogenomic species network must become **extremely fast** (most likely through optimized CUDA)
    to both gather and update from.
13. **DeepEarth's core multi-modal fusion must input a context window of tokens 1, 2, ..., T.**
14. **Each token *t* must contain** Earth4D space-time positional encoding (if it exists), phylogenomic encoding (if
    it exists), plus modality-specific encoding (e.g. DINOv3 vision embedding of an iNaturalist photo), and
    data-modality type encoding (e.g. low-dimensional `nn.Embedding` learnable by data-modality type, implicitly
    communicating "DINOv3 embedding of iNaturalist photo"). In this example, the token *t*'s Earth4D
    encoding would be revealing "at GPS coordinate (lat, lon), elevation X.X m, timestamp M/D/YY HH:MM:SS", and 
    phylogenomic encoding would be revealing "observation of Species X, which is similar to Species Y, Z, ...
    in dimensions A, B, ...".
15. Similar to **PerceiverIO**, DeepEarth must be capable of simply masking large parts of its tokens, as a method 
    for querying that token. The model must be capable of considering all tokens in the context window, attending 
    to all information in the context window. It must also have **"long-term memory"** it can access. 
16. Mathematically and statistically, DeepEarth must retain a learning representation that is capable of
    **simultaneously modeling the joint distributions of all of these variables**, and how these precisely overlap,
    project, covary, and operate with more or less dependence. Generative multi-modal models must be inspired by
    **Geoffrey Hinton** [[Srivastava & Salakhutdinov, Multimodal Learning with Deep Boltzmann Machines][dbm]].
17. Inspired by **[MADE: Masked Autoencoder for Distribution Estimation][made]**, DeepEarth must be trained as a
    Bayesian probabilistic model, such that the inclusion of additional evidence (e.g. unmasking of information)
    gives rise to the most likely posterior estimation of overall system state.
18. **All available data must be included.** DeepEarth must ingest every prepared modality, not a curated subset. By
    the Bayesian principle of rule 17, additional evidence can only sharpen (or leave unchanged) a posterior — never
    degrade it. Therefore **every modality must lift induction, or at worst not strongly support it**. If adding a
    modality measurably *hurts* any benchmark, that is a **bug** in how the data is integrated (fusion, masking,
    missing-value handling, or loss balance) to be found and fixed — never a reason to drop the modality. We do not
    validate "a better model without certain data"; we make the model correctly use all of it.
19. **Minimize files and tokens.** The whole system must express its function in as few files and tokens as possible,
    without compromising baseline production-quality clarity — self-documenting names and standard abbreviations,
    code that does the talking. Fewer, denser, sharper files always beat more, thinner ones. This is measured over the
    **critical-path surface that is *subject to change*** — every file whose content shapes end-to-end system behavior
    and that an autoresearch agent may edit: all model/encoder/fusion code on the champion path, the config, and the
    `README.md` / `science.md` / `autoresearch.md` documents. It **excludes** the fixed harness that is not subject to
    change — `prepare.py` (downloads + caches data) and `evaluate.py` (runs the benchmarks and computes the final
    score, the immutable ground truth). Each condensing pass must **quantify** that surface (file count + total
    tokens, `.md` included) and drive it down. Every autoresearch agent reads exactly this enumerated surface, no more.
20. **Fixed experiment budget: 10 minutes.** Each autoresearch experiment trains for 10 minutes of wall-clock (startup
    and compilation excluded), then is scored by `evaluate.py`. Report benchmarks at that budget; compare experiments
    at equal time so improvements reflect real efficiency, not just more steps.
21. **Speed is a first-class score lever.** Because the budget is fixed (rule 20), wall-clock throughput converts
    directly into training steps and therefore into `net_score`: any acceleration of the algorithm that does not
    change its per-step mathematics *must* score at least as high, and under the budget, strictly higher. Optimizing
    throughput — CUDA kernels, sparse/fused updates, compilation, memory traffic, batch size — is therefore a prized
    research path, not a mere engineering nicety, and sits alongside the standing speed mandates (rules 4, 12). The
    discipline is that a speedup must be **non-compromising**: bit-identical to the champion per step (verified against
    the exact model, e.g. `hashencoder/test_precompute_exact.py`), so the extra steps are pure upside and never a
    silent approximation traded for pace. A faster champion that ties the slower one at fixed time is a red flag — it
    means the claimed speedup is not real, or a hidden compromise is cancelling it.
22. **Joint decoding is iterative, not one-shot — a multi-modal diffusion.** Reconstruct all variables *together* by
    refining every state over K rounds: each round fuses all states through the shared latent bottleneck, the latents
    self-attend (the joint model, rule 16), then every state re-reads the fused context *and* its own previous state and
    updates. This is joint sampling toward the highest-likelihood configuration of all variables at once (Hinton/DBM;
    diffusion-style fusion of many asynchronous patch-samples). Even observed ("ground-truth") states may be revised as
    evidence accumulates. A **state** is `(Earth4D position · variable-type · value|mask)`; grouping states by variable,
    by token, or by space-time patch are three views of one field.
23. **Conserve the pluralism of variable distributions.** Never collapse a variable's own manifold into the shared
    representation. Each variable keeps its own channel and its own decoder trained to reconstruct its own marginal;
    cross-modal influence flows only through the **O(N·L) latent highway, never O(N²)**. Be rich in **interface
    decoders** that read from and write to the latent field — that is where cross-modal learning happens. Testable
    invariant: as coupling strengthens (K, write-back), per-variable *marginal* fidelity must hold while the *joint*
    likelihood rises.
24. **Model the dense 4D field — measure-everything-everywhere.** Every ecological quantity could in principle be
    measured at every point in space-time (mostly it is "air with trace constituents at levels/directions, described by
    embeddings"). DeepEarth models the whole 4D volume of a region: for every space-time patch it infers the most likely
    embedding of every variable, pinning observed values and inducting the rest, **sampling between sparse observations
    in space and time**. Tie physical-model resolution to matched-resolution deep sensor features (e.g. DINOv3 SAT493M
    32×32 ↔ 300 m NAIP) so dense, always-available inputs inductively map to sparsely-observed targets (species, ground
    vision, phenology, pollination). The goal: a dense, calibrated, forward-in-time field of all variables everywhere.
25. **The phylogenomic embedding is itself a maskable, reconstructable variable.** A species' evolutionary position is
    not a fixed lookup — it is a state in the MADE field (rule 17), so DeepEarth must be able to *mask* a species'
    phylogenomic embedding and reconstruct it from context (its observations, environment, and phylogenetic neighbors).
    This makes the tree soft: a species whose position is uncertain (a new or poorly-placed taxon) gets a posterior
    embedding induced from evidence, and once placed it automatically propagates neighbor updates through the species
    graph (rules 7–12). Training must randomly withhold the phylogenomic embedding for a fraction of species per batch
    and reconstruct it, so the capability is learned, not assumed — the mechanism by which any species, in-tree or not,
    acquires a phylogenomically-consistent representation.
26. **Seed species embeddings from a frozen [BioCLIP 2.5][bioclip2] text prior, adapt only through a small learned probe.**
    Do not initialize per-species embeddings randomly, and do not naively fine-tune a foundation embedding (unconstrained
    fine-tuning provably distorts the pretrained geometry — [LP-FT][lpft]). Instead, freeze the BioCLIP 2.5 **text**
    embedding (`imageomics/bioclip-2.5-vith14`, ViT-H, **1024-d**) of the flattened Linnaean string; its geometry already
    encodes taxonomic *and* ecological structure (validated: same-genus vs cross-genus cosine .649/.359, Δ.29 — sharper
    than BioCLIP-2's .809/.593) and a small feed-forward probe (MLP / attentive probe) maps it into the model's `d_model`
    phylogenomic seed. Because the probe handles the projection, the prior need not share the image-token space (our image
    "bio" tokens remain BioCLIP-2 768-d — re-embedding them at 2.5 is an optional future upgrade). The probe is backprop-tuned by the joint objective, so the model **discovers** phylogenomic structure inside
    the BioCLIP space while preserving it. The seed is then refined by the species graph (rules 7–12). Two invariants:
    (a) the frozen text encoder makes an **unseen** species embeddable by the identical text→probe path (zero-shot
    placement, rule 25); (b) a species' seed is computed **once per unique species per batch** and shared across all its
    tokens — never recomputed per observation.
27. **Interactions carry phylogenetic signal — induce them bidirectionally across two trees.** Related plants share
    pollinators and related pollinators visit related plants (phylogenetically structured bipartite networks): trait
    conservation (rules 7–12, 25) shares evidence *within* a tree, interactions *across* two. Model a plant↔pollinator
    interaction as a bilinear form between two separately phylo-refined representations — the plant's from the plant
    species graph, the pollinator's from a second pollinator graph — each keeping its own distance so within-kingdom
    resolution survives the deep plant–animal split. Each side reading through its own phylogeny makes one observed pair
    (plant A, pollinator B) propagate both ways: the query-side plant graph associates B with A's relatives, the
    output-side pollinator graph associates A with B's relatives, so the interaction head decodes against the *refined*
    pollinator embeddings. The model predicts a species' partners from its relatives' partners; test it held-out by
    hiding a focal species' interactions and recovering them through relatives, plus the graph ablation.
28. **No fuzzy science — every component grounded in the state-of-the-art model of real data.** Crude statistical
    approximations are not acceptable. Each part of the model must be either (a) closely fit to real data with enough
    statistical expressiveness to *generatively simulate* it, or (b) a frozen embedding from a peer-reviewed,
    scientifically published, SOTA domain **foundation model run on real data**. Accept **no unvalidated topological
    assumptions and no unvalidated Bayesian priors anywhere** in the model. Never assume scientific data follows a
    simplistic structure or is fully implied by a grossly simplified heuristic. For each domain: research the latest
    SOTA model, **download it, run it, validate it against real data**, and verify that every dataset, encoder, model,
    and architecture is directly based upon the best published work of scientists.
29. **Refine the species graph through the phylogeny's own internal nodes — the exact tree-GP factorization, not a
    kernel approximation.** Trait evolution on a dated tree is an Ornstein-Uhlenbeck Gaussian process whose covariance
    `Cov(i,j) = (σ²/2α)·e^{-α·d(i,j)}` decays with cophenetic distance (Hansen 1997; Jones & Moriarty 2013). A dense or
    top-k cophenetic-distance attention encodes the right *kernel* but the wrong *operator* (a normalized kernel
    smoother, not the inverse-covariance the GP posterior requires) over a needlessly dense `N×N` structure. The tree is
    a Gaussian Markov random field: conditioning on ancestral states renders the tips conditionally independent, so the
    **internal clade nodes are the *exact* Markov blanket / inducing set** — not a Nyström approximation — and two-pass
    (post-order + pre-order) message passing computes the exact posterior in **O(N)** (Felsenstein 1973; Ho & Ané 2014;
    Ji et al. 2020; Karcher & Ané 2025). This is Multi-head Latent Attention (DeepSeek-V2) applied along the
    *phylogenetic* axis rather than the head axis: the shared low-rank latent is the **ancestral-clade state shared by
    all its descendant species**, and the branch-length OU contraction `e^{-α·ℓ}` is the absorbed up-projection.
    Species **not** in the tree carry no position; they must soft-attach to the refined clade latents by cross-attention
    keyed on the frozen BioCLIP-2.5 prior (the ISAB "distribute" step; rules 25–26), so a novel species acquires a
    phylogenetically-consistent posterior from its relatives — unifying the in-tree and out-of-tree paths in one
    operator (`LatentCladeAttention`). The refinement operator must be this exact, `O(N)`, out-of-tree-complete
    construction; a dense/top-k attention is an interim approximation to be replaced, never the champion.

30. **Report every champion improvement as before->after — always run the benchmarks and use `champion_report.py`.**
    No SoTA champion is committed without `python -m deepearth.autoresearch.champion_report --log <run> --desc
    <result> --save`: the commit headline states the net score BEFORE->AFTER (harmonic mean + arithmetic), the body
    describes what changed / why / how, and an enumerated list reports every benchmark's before->after (delta) with an
    explicit regressions summary — no individual metric may regress. The helper diffs against the committed
    `autoresearch/champion_scores.json` so every improvement is unambiguous, comparable, and reproducible by collaborators.

31. **Heads are DETACHED read-outs by default; the universal self-supervised reconstruction is inviolable.** The core
    learns ONE representation whose dense, always-available capabilities (species, family, vision, environment) are
    never traded for a niche supervised signal -- each head (community, pollinator, lfmc, mycorrhiza, flowering, ...)
    trains on the `.detach()`-ed pooled latent, so none commandeers the core and the net measures the core's capacity.
    A head MAY backprop only if it (a) writes to a dedicated **trait-subspace** -- extra latent bandwidth concatenated
    with, never overwriting, the universal channels (rule 23 for heads); (b) is one of **>=K reliability-weighted
    heads** (sparse traits -> ~0 weight); (c) regresses **no universal metric beyond noise** (the champion-report guard
    is the hard floor). Adopt only when the universal arithmetic holds while the niche family rises. A small coupling
    weight does NOT protect the universal axis (a single myco head cost -0.012 at both w=0.1 and w=1.0) -- only
    subspace isolation does.

32. **Score AND optimize 100% of the benchmark suite -- nothing excluded.** Every benchmark exists to be measured and
    driven up. The net score is the harmonic mean over ALL active benchmarks -- capabilities AND ablation-delta /
    information-gain diagnostics. Metrics not naturally bounded or that can sit near 0 (the deltas) are renormalized
    (logistic, evaluate._net_value) so inclusion NEVER exceeds 1.0, NEVER forms a below-0 well, and is always
    monotonically beneficial to raise (repetitive signal is fine). A champion must carry the WHOLE suite, not a subset.

## References

- **DINOv3** — Siméoni et al., "DINOv3," 2025. arXiv:2508.10104 — <https://arxiv.org/abs/2508.10104>
- **CLAY v1.5** — Clay Foundation Model (open geospatial foundation model) — <https://madewithclay.org> /
  <https://github.com/Clay-foundation/model>
- **GraphCast** — Lam et al., "Learning skillful medium-range global weather forecasting," *Science* 382:1416–1421
  (2023). DOI: [10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336)
- **GenCast** — Price et al., "Probabilistic weather forecasting with machine learning," *Nature* 637:84–90 (2025).
  DOI: [10.1038/s41586-024-08252-9](https://doi.org/10.1038/s41586-024-08252-9)
- **JEPA (I-JEPA)** — Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive
  Architecture," *CVPR* (2023). DOI: [10.1109/CVPR52729.2023.01499](https://doi.org/10.1109/CVPR52729.2023.01499) —
  arXiv:2301.08243
- **Perceiver IO** — Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs & Outputs," *ICLR*
  (2022). arXiv:2107.14795 — <https://arxiv.org/abs/2107.14795>
- **Deep Boltzmann Machines (multimodal, Hinton lineage)** — Srivastava & Salakhutdinov, "Multimodal Learning with
  Deep Boltzmann Machines," *NeurIPS* (2012) / *JMLR* 15:2949–2980 (2014) —
  <https://jmlr.org/papers/v15/srivastava14b.html>
- **MADE** — Germain, Gregor, Murray & Larochelle, "MADE: Masked Autoencoder for Distribution Estimation," *ICML*
  (2015), PMLR 37:881–889. arXiv:1502.03509 — <https://arxiv.org/abs/1502.03509>
- **BioCLIP** — Stevens et al., "BioCLIP: A Vision Foundation Model for the Tree of Life," *CVPR* (2024).
  arXiv:2311.18803 — <https://arxiv.org/abs/2311.18803>
- **BioCLIP 2 / 2.5** — Gu et al., "BioCLIP 2: A Foundation Model for the Tree of Life at Scale" (TreeOfLife-200M),
  *NeurIPS* (2025). arXiv:2505.23883 — <https://arxiv.org/abs/2505.23883>. Rule 26 uses the **2.5** ViT-H checkpoint
  `imageomics/bioclip-2.5-vith14` (1024-d text), a later release on the same lineage.
- **LP-FT** — Kumar et al., "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution," *ICLR*
  (2022). arXiv:2202.10054 — <https://arxiv.org/abs/2202.10054>

[dinov3]: https://arxiv.org/abs/2508.10104
[clay]: https://github.com/Clay-foundation/model
[graphcast]: https://doi.org/10.1126/science.adi2336
[gencast]: https://doi.org/10.1038/s41586-024-08252-9
[jepa]: https://doi.org/10.1109/CVPR52729.2023.01499
[perceiverio]: https://arxiv.org/abs/2107.14795
[dbm]: https://jmlr.org/papers/v15/srivastava14b.html
[made]: https://arxiv.org/abs/1502.03509
[bioclip2]: https://arxiv.org/abs/2505.23883
[lpft]: https://arxiv.org/abs/2202.10054
