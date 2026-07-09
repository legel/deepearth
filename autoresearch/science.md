# DeepEarth — Science

DeepEarth is a **self-supervised multi-modal deep learning architecture designed for ecological simulation and
optimization**.

The latest model (targeting **v1** maturity) features two innovative methods of encoding, processing, and decoding
key data modalities:

- **I. Earth4D SpaceTime GNN** — learnable (*x*, *y*, *z*, *t*) positional encodings that are fused with all
  observed data.
- **II. Phylogenomic Species GNN** — learnable per-species encodings, topologically based on evolution's tree of
  life.

The model is designed to become a **causal forecaster of high-value ecological metrics**, e.g. simulate plant
growth and flowering.

Beyond the innovations above, DeepEarth is being developed based upon the following key foundations:

- **I. Pre-trained Vision Encoders: [DINOv3][dinov3], [CLAY v1.5][clay]** — deep embeddings of iNaturalist, NAIP
  aerial, and Sentinel-2 imagery.
- **II. Graph Neural Networks: [GraphCast][graphcast] + [GenCast][gencast]** — causal spatio-temporal simulation
  methods from Google DeepMind.
- **III. Multi-Modal Joint Embedding: [JEPA][jepa] + [PerceiverIO][perceiverio]** — inductive learning of joint
  probability distributions across all data.

A core principle of DeepEarth is that the model **learns through masked autoencoding**. We see great promise in
following **JEPA** by masking and reconstructing *embeddings*. For example, one core benchmark of DeepEarth is the
capacity to **"imagine" DINOv3 vision embeddings**, given a context window that describes an environment.

The exact architectural specifications, computational design, machine learning techniques, etc. are **flexible**, as
long as certain constraints are met.

## Rules and core principles

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
    communicating "DINOv3 embedding of iNaturalist photo"), whereas in this example, the token *t*'s Earth4D
    encoding would be revealing "at GPS coordinate (lat, lon), elevation X.X m, timestamp M/D/YY HH:MM:SS", and the
    token's phylogenomic encoding would be revealing "observation of Species X, which is similar to Species Y, Z, ...
    in dimensions A, B, ...".
15. Similar to or exactly like **PerceiverIO**, DeepEarth must be capable of simply masking large parts of its
    tokens, as a method for querying that token. The model must be capable of considering all tokens in the context
    window, attending to all information in the context window. It must also have **"long-term memory"** it can
    access.
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

[dinov3]: https://arxiv.org/abs/2508.10104
[clay]: https://github.com/Clay-foundation/model
[graphcast]: https://doi.org/10.1126/science.adi2336
[gencast]: https://doi.org/10.1038/s41586-024-08252-9
[jepa]: https://doi.org/10.1109/CVPR52729.2023.01499
[perceiverio]: https://arxiv.org/abs/2107.14795
[dbm]: https://jmlr.org/papers/v15/srivastava14b.html
[made]: https://arxiv.org/abs/1502.03509
