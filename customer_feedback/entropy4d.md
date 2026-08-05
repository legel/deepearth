# Entropy4D: Mass-Energy Space-Time Perceptive Fields for Causal World Models

*Lance Legel, Ecological Intelligence, Inc.*

---

## Abstract

Entropy4D is a neural architecture for predicting the future configuration of mass-energy in a bounded region of space-time from past observations. The representation is constructed from a configurable population of *perceptive fields*. Each perceptive field is a differentiable encoder, parameterized by a position, orientation, extent, and topology in (_x_, _y_, _z_, _t_) world coordinates. The basis of topologies includes linear, radial, and spherical harmonics; a prior distribution weights them to span the symmetry regimes common to physical data. Each perceptive field is iterated through a sequence of focusing transformations that adapt its geometric and spectral configuration to the entropy of the observations it samples. Following prior brain-inspired world models, Entropy4D minimizes surprise by predicting future states of its internal representations, namely its perceptive field encodings. We give complete mathematical and computational specifications for end-to-end implementation, illustrate the representation on a thrown projectile observed from arbitrary viewpoints, and identify twenty target domains spanning sub-atomic, cellular, ecological, and cosmological scales.

---

## 1. Position

A physical world model takes the configuration of mass-energy in a bounded region of space-time and returns a probability distribution over future configurations of the same region. Brain-inspired world models in particular are designed to minimize surprise by predicting the future state of their encoded representations [1, 2, 3, 4].

Entropy4D pursues this causal prediction objective within remarkable versatility. The model is (i) multi-scale, (ii) multi-modal, (iii) equivariant to spatio-temporal symmetries across physical phenomena, and (iv) supervised by geometric and spectral properties of data. The primitive of Entropy4D is the perceptive field, a generalization of prior 3D/4D encoders [5, 6, 7]. Each perceptive field has differentiable parameters, with both equivariant and non-equivariant encoding channels.

---

## 2. Formalism

### 2.1 Mass-Energy Field and Observations

We measure the physical world. A bounded region of space-time, the hypercube of focus, is parameterized by the half-extents $\Delta x, \Delta y, \Delta z, \Delta t$ around an origin. Within it, an unknown mass-energy field $\mathbb{E}$ is observed as a finite set of measurements

$$\Theta = \{\theta_i := (e_i, x_i, y_i, z_i, t_i)\}_{i=1}^{N},$$

where each $e_i$ encodes the modality-specific content of the measurement (an image patch, a sensor reading, an atomic coordinate) and $(x_i, y_i, z_i, t_i)$ locates it in space-time. Observations are sparse, irregular, heterogeneous, and noisy.

The present is the origin: $t = 0$. Past observations satisfy $t_i \le 0$. The task of an Entropy4D model is to encode $\Theta_{\le 0}$ into a representation from which the field at $t > 0$ can be predicted.

### 2.2 Perceptive Field

A perceptive field is a tuple

$$p = (c, R, s, \phi, \mathcal{T}),$$

where $c \in \mathbb{R}^4$ is its origin in space-time, $R \in SO(3)$ its spatial orientation, $s \in \mathbb{R}^4_{>0}$ its per-axis extent, $\phi$ a vector of internal frequency and phase parameters, and $\mathcal{T}$ a discrete type label drawn from the basis of Section 3. The continuous parameters $(c, R, s, \phi)$ are differentiable. The type $\mathcal{T}$ is fixed at instantiation.

A perceptive field defines an encoding map

$$E_p : \Theta \rightarrow O_p \in \mathbb{R}^D,$$

decomposed as $E_p = \mathrm{Read}_p \circ \mathrm{Aggregate}_p \circ \mathrm{Sample}_p$. The sampler restricts $\Theta$ to observations within the field's space-time support. The aggregator accumulates them into the field's internal representation, with structure determined by $\mathcal{T}$. The reader projects to a fixed-dimension embedding $O_p$.

### 2.3 Population, Focus, and Canonical State

An Entropy4D model is a population of $P$ perceptive fields. Each field is iterated through configurations

$$p^{(0)} \rightarrow p^{(1)} \rightarrow \cdots \rightarrow p^{(K)},$$

with $p^{(0)}$ sampled from a prior and $p^{(K)}$ the *canonical configuration*. Each transition is produced by a differentiable focusing operator $\Phi$ that updates the configuration of each field based on the joint state of all fields:

$$p_i^{(k+1)} = \Phi\big(p_i^{(k)},\ \{p_j^{(k)}, O_{p_j}^{(k)}\}_{j=1}^{P}\big).$$

The canonical embedding of perceptive field $i$ at observation time $t$ is $O_{p_i}(t) := E_{p_i^{(K)}}(\Theta_{\le t})$. The canonical state of the population at time $t$ is

$$O_P(t) := \big(O_{p_1}(t), \ldots, O_{p_P}(t)\big) \in \mathbb{R}^{P \times D},$$

and is the object on which all downstream prediction operates.

---

## 3. The Perceptive Field Basis

The basis of perceptive field types is chosen so that any band-limited mass-energy field on space-time admits a sparse representation as a combination of basis elements. It is partitioned into non-equivariant and equivariant families, with mandatory representation of both enforced through the prior of Section 3.3.

### 3.1 Non-Equivariant Types

**Type 1. Cartesian hash lattice.** A multi-resolution grid of $L$ levels with per-level resolution $r_\ell$, mapped to fixed-capacity hash tables in the manner of [5]. Queries at continuous coordinates resolve by quadrilinear interpolation in 4D. This is the Earth4D primitive [7].

**Type 2. Collision-free tesseract grid.** A 4D embedding grid following [6] with a bijective minimal perfect hash

$$H_\ell(t, x, y, z) = t \cdot r_\ell^x r_\ell^y r_\ell^z + z \cdot r_\ell^x r_\ell^y + y \cdot r_\ell^x + x,$$

with per-level, per-axis resolutions adapted to the bounding box of observed entropy.

**Type 3. Anisotropic Cartesian lattice.** As Type 1 with independent per-axis resolutions, addressing scenes whose temporal entropy scale differs from the spatial, and scenes with a structurally distinguished axis.

**Type 4. Translation-equivariant convolutional patch.** A Cartesian convolutional kernel applied locally, equivariant under translation within the field's support. Suitable for repeated, locally stationary structure.

### 3.2 Equivariant Types

**Type 5. Spherical harmonic shell stack.** A radial stack of spherical harmonic decompositions $Y_\ell^m(\theta, \phi)$ centered at $c$, with internal coefficients $a_{r\ell m}$. Equivariant under arbitrary rotation about $c$. Suitable for orbital, droplet, and centered-distribution structure.

**Type 6. Circular harmonic stack.** Circular harmonics in a chosen plane with separate radial or Cartesian encoding along the orthogonal axis. Equivariant under rotation about the chosen axis. Suitable for vortices, plumes, and configurations with a privileged axis (gravity being the most universal example).

**Type 7. Radial basis stack.** A purely radial encoding under spherical, cylindrical, or planar geometry. Equivariant under all rotations preserving the chosen symmetry. Suitable for isotropic phenomena from a point source.

The seven types span the symmetries of the most physically prevalent transformations: arbitrary translation (Types 1, 2, 3, 4), rotation about a point (Type 5), rotation about an axis (Types 6, 7 in cylindrical instantiation), and identity. Reflection equivariance is recovered within Types 5 and 6 by restricting to even-parity harmonics. Scaling equivariance is recovered across the level structure of Types 1, 2, 3 and across the radial structure of Types 5, 6, 7.

### 3.3 The Equivariance Prior

Type assignments across the population of $P$ perceptive fields are drawn from a categorical prior

$$\pi(\mathcal{T}) = (\pi_1, \ldots, \pi_7), \qquad \sum_k \pi_k = 1,$$

subject to

$$\sum_{k \in \{1,2,3,4\}} \pi_k \ge \alpha_{\mathrm{ne}}, \qquad \sum_{k \in \{5,6,7\}} \pi_k \ge \alpha_{\mathrm{e}},$$

with default $\alpha_{\mathrm{ne}} = \alpha_{\mathrm{e}} = 0.3$. The constraint is enforced as a regularization on the empirical type distribution at instantiation; type assignments are then fixed.

The equivariance prior commits the model to symmetries that recur across physical phenomena and are well characterized by group theory. Rather than allocate parameters to gradient descent's rediscovery of these symmetries, we encode them in the basis. Following [8], the prior is asymmetric and partial: any individual perceptive field can become uninformative if the data does not support its symmetry, but the population always carries equivariant capacity.

---

## 4. Focusing

### 4.1 Initialization

A perceptive field is configured at instantiation by sampling $(c, R, s, \phi, \mathcal{T})$ from a prior $\rho$. The default is uniform over $\Omega$ for $c$, uniform over $SO(3)$ for $R$, log-uniform for $s$, type-dependent for $\phi$, and the categorical prior of Section 3.3 for $\mathcal{T}$. Optional data-driven initialization weights origins $c$ by the local Shannon entropy of $\Theta$, accelerating convergence when observations are highly non-uniform. More generally, $\rho$ itself is learnable: once Entropy4D has been trained across a corpus of space-time regions within a physical domain, the empirical distribution of converged canonical configurations across that corpus becomes the prior for deployment in new regions of the same domain.

### 4.2 The Focusing Operator

The focusing operator $\Phi$ is a graph neural network over the perceptive field population, with edges connecting spatially proximate or topologically related fields. At each step it produces an update $\Delta p_i^{(k)}$ composed onto the current configuration:

$$p_i^{(k+1)} = p_i^{(k)} \oplus \Phi_\theta\big(p_i^{(k)},\ \{p_j^{(k)}, O_{p_j}^{(k)}\}_{j \in \mathcal{N}(i)}\big),$$

where $\oplus$ denotes the appropriate composition on each parameter (vector addition for $c$ and $\phi$, multiplication on $SO(3)$ for $R$, multiplicative update for $s$) and $\mathcal{N}(i)$ is the neighborhood of field $i$ in the population graph. Weights $\theta$ are shared across all $K$ steps.

The operator monotonically reduces the focus energy

$$\mathcal{E}(\{p_i^{(k)}\}) = \mathcal{L}_{\mathrm{recon}}(O_P^{(k)}, \Theta) + \lambda \mathcal{R}(\{p_i^{(k)}\}),$$

where $\mathcal{L}_{\mathrm{recon}}$ measures how well the canonical embeddings reconstruct held-out observations, and $\mathcal{R}$ regularizes overlap among fields, enforces the equivariance prior, and constrains spectral parameters away from degenerate configurations. The schedule is annealed: early steps allow large global rearrangements, late steps converge to local minima. The structural analogy to denoising diffusion is direct, with the noise residing in the parameter space of the perceptive fields rather than in an output signal.

### 4.3 Causal Reconfiguration

When a new observation arrives at time $t$, the canonical state $\{p_i^{(K)}\}$ at $t - \delta t$ is used to initialize the focusing dynamics for $t$, rather than restarting from $\rho$. Only $K' \ll K$ further focusing steps are required to incorporate the incremental information. The model thus operates as a recurrent state machine: persistent structure is tracked slowly, sudden high-entropy events trigger rapid re-orientation, and the entire history of focus configurations conditions all future encoding.

---

## 5. Training Objective

Entropy4D is trained to minimize the surprise of its internal representations: the mismatch between the canonical state predicted from past observations and the canonical state produced from current observations. This is the mathematical content of the predictive-coding and free-energy framing [1, 2] introduced in Section 1. The engineering mechanism for asymmetric prediction in embedding space, with stop-gradient on the target to prevent representation collapse, is the joint-embedding predictive architecture of [3].

Given a sequence of canonical states $O_P(t_1), \ldots, O_P(t_n)$ at observation times $t_1 < \cdots < t_n$, the model is trained to predict each from its predecessors:

$$\mathcal{L}_{\mathrm{pred}} = \sum_{m=1}^{n-1} \big\| O_P(t_{m+1}) - f_\psi\big(O_P(t_m), O_P(t_{m-1}), \ldots\big) \big\|^2,$$

with $f_\psi$ a graph neural network over the perceptive field population sharing structural form with $\Phi$, and stop-gradient applied to the target $O_P(t_{m+1})$ in the pattern of [3].

The full objective combines prediction with reconstruction and regularization:

$$\mathcal{L} = \mathcal{L}_{\mathrm{pred}} + \beta \mathcal{L}_{\mathrm{recon}} + \lambda \mathcal{R}.$$

Training proceeds over a corpus of sequences, each drawn from a distinct region of space-time within the physical domain of interest. The predictor $f_\psi$, the focusing weights $\theta$, and the internal weights of each perceptive field type are shared across sequences. What varies across sequences is the observations $\Theta$ and the canonical configurations $\{p_i^{(K)}\}$ they induce. Parameters learned from any one sequence therefore apply to all others within the domain, and the learned initialization prior $\rho$ of Section 4.1 encodes the aggregated structure of the training corpus.

For dense observation regimes (a planetary atmosphere observed at every grid cell) $\mathcal{L}_{\mathrm{recon}}$ dominates and Entropy4D behaves as a reconstruction autoencoder with predictive consistency. For sparse regimes (a forest stand observed at a handful of photographs and soil cores) $\mathcal{L}_{\mathrm{pred}}$ dominates and Entropy4D behaves as a predictive world model whose representation is shaped by the requirement that future canonical states be inferable from past. The same architecture serves both regimes.

---

## 6. Worked Example: A Thrown Projectile Under Gravity

We illustrate the full mechanism on a deliberately minimal example: a rigid projectile thrown through air, observed by an unknown number of cameras at unknown viewpoints with unknown intrinsics. The example is minimal in subject and exhaustive in the operations it exercises.

The hypercube is sized to enclose the trajectory: $\Delta x = \Delta y = \Delta z \approx 5\,\mathrm{m}$, $\Delta t = 5\,\mathrm{s}$. Observations are image patches with pose-tagged camera centers. A population of $P = 256$ perceptive fields is initialized with type assignments drawn from the prior of Section 3.3 at the default constraint, yielding approximately 90 non-equivariant grids, 90 equivariant harmonic stacks, and 76 mixed types. Origins $c$ are sampled uniformly within the hypercube; orientations $R$ uniformly from $SO(3)$; extents $s$ log-uniform between $0.1\,\mathrm{m}$ and $5\,\mathrm{m}$ spatially and between $0.01\,\mathrm{s}$ and $5\,\mathrm{s}$ temporally.

Focusing proceeds for $K = 100$ steps. In the first phase, perceptive fields disperse from random origins toward regions of high observational entropy: the projectile's trajectory and the camera viewpoints. In the second phase, fields differentiate by type. Circular harmonic fields (Type 6) align their axes with the trajectory tangent, capturing rotational symmetry about the direction of motion. Spherical harmonic fields (Type 5) center on the projectile centroid at successive instants, capturing orientation-resolved geometry. Cartesian and tesseract grids (Types 1, 2) tile the swept volume at progressively finer levels. Anisotropic grids (Type 3) align their high-resolution axis along the trajectory, exploiting the one-dimensional character of the path through three-dimensional space. In the third phase, the population converges to a canonical configuration that locally minimizes the focus energy.

The canonical state $O_P(t)$ is a $256 \times D$ tensor. The predictor $f_\psi$ maps $\big(O_P(t - 2\delta), O_P(t - \delta)\big) \mapsto O_P(t)$ for small temporal increments $\delta$. Because the focusing dynamics have aligned the equivariant fields with the trajectory's intrinsic geometry, prediction reduces in those channels to low-dimensional regression on the projectile's center of mass and angular velocity. Because the non-equivariant fields tile the swept volume, prediction reduces in those channels to local interpolation on the canonical embeddings.

Two empirical claims follow, and constitute the criteria under which the architecture is judged. First, an Entropy4D model trained on a corpus of thrown projectiles, with no architectural specialization, generalizes to unseen throws under arbitrary launch angles, speeds, masses, and viewpoints, because the focusing dynamics rediscover the trajectory's intrinsic geometry for each new instance. Second, the same Entropy4D instantiation, trained without modification on a different physical regime, produces comparable generalization, because the perceptive field basis spans the symmetries of those regimes as well. The first claim is a baseline; the second is the core hypothesis of the architecture.

---

## 7. Target Domains

A single Entropy4D architecture is hypothesized to operate across scales spanning over twenty orders of magnitude. We enumerate twenty target domains against which the hypothesis is to be evaluated.

The first ten correspond to applied environmental and ecological systems where Entropy4D inherits the commitments of prior work at Ecological Intelligence: (1) flood risk modeling at parcel-to-watershed scale; (2) wildfire risk modeling at the wildland-urban interface; (3) wind risk modeling for structures and vegetation; (4) community ecology and biodiversity quantification at sub-meter resolution; (5) landscape architecture under multi-objective evaluation; (6) automatic municipal code enforcement for spatially defined ordinances; (7) agricultural yield optimization at per-plant resolution; (8) multi-objective ecologically optimized master planning of land development; (9) hydrological modeling of soil-water-vegetation systems with explicit root and macropore representation; (10) ecosystem-scale carbon, water, and nutrient flux estimation.

The second ten extend across the full scale range the entropy formulation is intended to cover: (11) reconstruction of the Standard Model of physics from particle interaction data, evaluating whether the equivariance prior is sufficient to recover $SU(3) \times SU(2) \times U(1)$ structure under training; (12) reconstruction of relativistic dynamics in strong-field regimes; (13) reconstruction of quantum mechanical orbital structure from electron density observation; (14) cellular biology simulation; (15) stellar interior simulation; (16) human physiology simulation; (17) human social and ecological systems at community to civilization scale; (18) ecosystem dynamics across regional scales; (19) planetary atmospheric, oceanic, and lithospheric coupling; (20) solar system and stellar neighborhood dynamics under gravity.

The list is not a deliverables roadmap but a statement of the scale range within which a single architecture is hypothesized to operate. Entropy4D earns its claim of generality if the same instantiation, trained on data from any of these domains, produces representations whose predictive accuracy matches or exceeds domain-specialized alternatives.

---

## 8. Implementation Specification

A reference implementation requires the following components, specified at the level of mathematical interface and computational complexity.

The perceptive field module stores $(c, R, s, \phi, \mathcal{T})$ as differentiable tensors, with $R$ in an over-parameterized representation projected onto $SO(3)$ to avoid the singularities of Euler angles. It exposes a forward method $E_p(\Theta) \rightarrow O_p \in \mathbb{R}^D$ implementing $\mathrm{Read} \circ \mathrm{Aggregate} \circ \mathrm{Sample}$, with $\mathrm{Aggregate}$ dispatching on $\mathcal{T}$ to one of the seven primitives. Per-pass cost is $O(|\Theta_p|)$ for sampling and $O(\dim(\mathrm{Aggregate}))$ for aggregation; for hash grid types this matches [5, 6, 7]; for harmonic types this is $O(L_{\max}^2)$ in maximum harmonic degree.

The focusing operator $\Phi_\theta$ and the predictor $f_\psi$ are graph neural networks over the perceptive field population with edges defined by spatial proximity (k-nearest in $c$) and shared type. Each invocation costs $O(P k D)$ for $P$ fields and edge degree $k$. We expect $K = 10\text{–}100$ for converged focusing in deployment and $K' = 1\text{–}5$ for incremental causal reconfiguration.

Memory is dominated by the hash table capacity of the non-equivariant types, inheriting the budget of [5]. The Earth4D instantiation [7] uses $7.24 \times 10^8$ trainable parameters at 24 levels per grid for four grids, fitting on a single high-memory GPU. Equivariant types have substantially smaller parameter counts because their internal representations are spectral coefficients rather than dense lattices.

Training is a standard sequence-prediction loop over windowed canonical states under the joint loss of Section 5, with the only non-standard element being the inner loop of $K$ focusing steps that produces each canonical state.

---

## 9. Discussion

The perceptive field is proposed as the representational primitive for learned physical models: a differentiable, geometrically situated, internally structured encoder, deployed in populations whose type distribution spans both equivariant and non-equivariant primitives, and whose configuration adapts to the entropy of observation through a focusing operator. The training objective is the minimization of surprise in the population's canonical state: future states of the internal representation are predicted from past states, in the brain-inspired tradition that frames perception, inference, and action as a single predictive process [1, 2]. The architecture is a generalization of the multi-resolution hash encoding lineage [5, 6, 7] under differentiable placement, frequency, and topology, and a structural commitment to the symmetries that recur across physical phenomena.

Polymathic AI [9] has demonstrated that a single transformer pre-trained on diverse physical systems produces representations that transfer across distinct physical domains, establishing the in-principle feasibility of cross-domain physical pre-training. Their result is restricted to domains already expressible as partial differential equations on regular grids. Our hypothesis is that the perceptive field's differentiable geometric and spectral configuration, combined with the equivariance prior and the focusing dynamics, extends this feasibility across the far broader range of scales, modalities, and symmetry regimes enumerated in Section 7.

The Hamiltonian formalism succeeded because it identified a representational primitive (the canonical pair of position and momentum, with its symplectic structure) sufficient to express the dynamics of every classical physical system. We propose that the perceptive field, with its differentiable position-orientation-extent-frequency-type configuration and its focusing dynamics, can play the analogous role for the data-driven, learned-physics regime that has emerged in the present decade.

---

## References

[1] Friston, K. The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138, 2010. https://doi.org/10.1038/nrn2787

[2] Rao, R. P. N., & Ballard, D. H. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79–87, 1999. https://doi.org/10.1038/4580

[3] Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 15619–15629, 2023. https://doi.org/10.1109/CVPR52729.2023.01499

[4] Legel, L. Inductive Neural Networks for Ecology. *Preprint*, 2025. https://doi.org/10.13140/RG.2.2.25523.90406

[5] Müller, T., Evans, A., Schied, C., & Keller, A. Instant Neural Graphics Primitives with a Multiresolution Hash Encoding. *ACM Transactions on Graphics*, 41(4), 102:1–102:15, 2022. https://doi.org/10.1145/3528223.3530127

[6] Sun, J., Lenz, D., Yu, H., & Peterka, T. F-Hash: Feature-Based Hash Design for Time-Varying Volume Visualization via Multi-Resolution Tesseract Encoding. *IEEE Transactions on Visualization and Computer Graphics*, 32(1), 396–406, 2026. https://doi.org/10.1109/TVCG.2025.3634812

[7] Legel, L., Huang, Q., Voelker, B., Neamati, D., Johnson, P. A., Bastani, F., Rose, J., Hennessy, J. R., Guralnick, R., Soltis, D., Soltis, P., & Wang, S. Self-Supervised Multi-Modal World Model with 4D Space-Time Embedding. *World Modeling Workshop, Mila Quebec AI Institute*, 2026. https://doi.org/10.48550/arXiv.2603.07039

[8] Ashman, M., Diaconu, C., Weller, A., Bruinsma, W., & Turner, R. E. Approximately Equivariant Neural Processes. *Advances in Neural Information Processing Systems*, 37, 97088–97123, 2024. https://doi.org/10.52202/079017-3078

[9] McCabe, M., Régaldo-Saint Blancard, B., Parker, L., Ohana, R., Cranmer, M., Bietti, A., Eickenberg, M., Golkar, S., Krawezik, G., Lanusse, F., Pettee, M., Tesileanu, T., Cho, K., & Ho, S. Multiple Physics Pretraining for Spatio-temporal Surrogate Models. *Advances in Neural Information Processing Systems*, 37, 119301–119335, 2024. https://doi.org/10.52202/079017-3791
