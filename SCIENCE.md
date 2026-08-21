# DeepEarth scientific contract

DeepEarth is a self-supervised ecological world model. Its state is a fibered, hash-addressed Earth4D mesh; every
measurement writes to that state and every prediction is a query-conditioned read from it.

1. Earth4D learns causal spatial and temporal distributions and supports forecasting.
2. Earth4D represents both absolute planetary coordinates and relative local context.
3. Space-time addresses every mesh cell; position is part of the shared state, not an auxiliary feature.
4. Earth4D remains parallel, sparse, and CUDA-efficient.
5. Held-out human capability, not parameter count, measures model capacity.
6. Geography and time can be partitioned and processed concurrently.
7. Every species retains a distinct phylogenomic state.
8. Phylogenomic structure is learned from scientific trees.
9. Unknown species can be projected into the biological state.
10. Training updates observed species and their phylogenetic neighbors.
11. Biological evidence propagates across related species.
12. Phylogenomic gather, refinement, and update remain sparse and efficient.
13. Fusion consumes a context of addressed multimodal states.
14. A state binds position, modality type, modality value, and biological structure when present.
15. Masking is the query interface; the reader may attend to all available state.
16. The shared representation models the joint distribution of all variables.
17. Additional observed evidence sharpens the posterior rather than replacing it.
18. Every prepared modality participates; harmful evidence routing is a model defect.
19. Keep one production model path and minimize its files, parameters, and code surface.
20. Comparable public runs use exactly 2,291 optimizer steps and seeds 1337 and 1338.
21. Efficiency changes preserve per-step mathematics and report parameters, memory, and runtime.
22. Joint reconstruction iteratively refines a coherent multimodal state.
23. Variables retain distinct fibers; cross-modal influence travels through the shared mesh.
24. The model represents a dense four-dimensional field from sparse observations.
25. Phylogenomic position is maskable and reconstructable.
26. Species states begin from frozen BioCLIP taxonomic priors transformed by a shared learned probe.
27. Plant-pollinator interactions propagate across both phylogenetic structures.
28. Scientific components use validated real data or published foundation representations.
29. Tree refinement uses internal clade states and branch-length-aware message passing.
30. Every promoted result reports the complete before-and-after human-capability scorecard.
31. Task heads must not commandeer or regress the universal world state.
32. Harmonic mean is the primary breadth objective; arithmetic mean is reported alongside it.

`autoresearch/evaluate.py` is the authoritative public measurement implementation. Aggregate scores are comparable
only within the same tagged protocol.
