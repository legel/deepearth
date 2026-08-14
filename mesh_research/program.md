# PROGRAM — World Mesh Autoresearch

Authority is divided by subject:

- This file defines the scientific thesis and executable research loop.
- `model.py` is the only editable research surface.
- `data.py` defines the immutable data and split.
- `evaluate.py` defines the immutable canonical measurement.

If this document and the fixed implementation disagree, stop the loop and repair
the contract outside research. Never change data, evaluation, scoring, or a
baseline to make a candidate pass.

## Objective

Build a small world model whose shared state is a learned space-time mesh. Every
modality must write evidence into that mesh, fusion must read only the mesh, and
the resulting latent must improve DeepEarth's held-out human capabilities.

The current architecture thesis is:

```text
space + time -> multiresolution hash fields -> addressed mesh state
observations ------------------------------> gated residual writes
neighbor geometry + evidence --------------> relative mesh state
mesh state only -> latent fusion -> task decoders
```

- ECEF spatial hashes address a continuous planetary coordinate system.
- projected space-time hashes represent several spatial and temporal scales;
- relative fields situate neighbors in the query's local frame;
- modality adapters translate unlike measurements into one state language;
- gated writes compose evidence at the addressed mesh levels;
- fusion receives query and neighbor mesh state, never raw-modality bypasses;
- task heads read the fused latent and do not define the shared state.

The hash parameters persist across examples. Modality-conditioned writes are
currently transient within a forward pass rather than permanent observation
storage. Whether the useful world representation should remain a transient field
or become a more persistent, compositional memory is an open research question.

The falsifiable claim is: **as the mesh becomes a more faithful and usable model of
planetary state, canonical fusion capabilities improve with it.** Better hash
occupancy, training loss, ablation dependence, or private-head performance without
better human capabilities is not a breakthrough.

## Representation boundary

Mesh construction and mesh reading are separate scientific problems with one hard
interface between them:

```text
coordinates + time + observations
              -> addressed, composed mesh state
              -> one query-conditioned world latent
              -> scientific predictions
```

The mesh builder owns addressing, modality writes, spatial and temporal exchange,
and cross-scale composition. Reading and fusion are the same operation: given a
physical and semantic query, they reduce the structured mesh to one aggregate world
latent for that query. Fusion must not receive raw modalities through side paths.
Different questions may produce different reads, but every answer must come from the
same shared mesh and the task head may only decode the resulting latent.

The hash-addressed mesh is the model's shared residual stream. Its trainable tables
accumulate reusable structure about the ecological region across optimization, and
every modality contributes an addressed residual update in the same state language.
Space, time, resolution, and relative-position hashes provide the coordinates of
that stream; they are not auxiliary features appended after representation learning.
The long-term direction is a regional world model whose mesh can retain increasingly
rich ecological state. In the present implementation the learned hash tables persist
across examples, while observation-conditioned writes exist only for the current
forward pass. Durable observation memory must be introduced and evaluated as an
explicit future mechanism rather than assumed to exist already.

Each addressed cell is a structured fiber, not one undifferentiated feature blob.
Hash levels describe geometric resolution; lenses describe what kind of information
is stored at that address. Keep those axes distinct:

```text
cell[space, time, resolution]
    ├── abiotic lens       climate, soil, terrain, water
    ├── visual lens        ground and aerial observations
    ├── biological lens    identity, traits, phylogeny
    └── ecological lens    phenology, community and changing state
```

Phylogenetic structure is not itself a space-time coordinate. A biological
observation binds its graph-derived representation into the biological lens of the
addressed cell. Do not sum unlike modality adapters into one vector and ask the
reader to disentangle the collision later. Compose information within each lens,
then exchange information across lenses with an explicit learned operation. Fusion
reads that common fibered mesh with a query and returns one aggregate representation
for the requested inference. This lens boundary should remain visible to ablation
and information-retention measurements.

This boundary makes the central claim testable in both directions:

- hold the reader fixed while comparing mesh checkpoints to test mesh quality;
- hold a mesh checkpoint fixed while comparing readers to test fusion quality;
- require improvements in fixed-reader mesh quality to predict improvements in
  canonical fusion rather than merely improving an internal mesh diagnostic.

Mesh quality means held-out information recoverable from the mesh alone. Measure it
as a vector rather than inventing an unvalidated internal composite:

- masked reconstruction of observations through a fixed reader;
- held-out scientific capabilities through the same fixed, low-capacity probe;
- physical locality and consistency across nearby space, time, and scale;
- sensitivity to removing a relevant modality write or mesh component;
- information retained by query-conditioned world latents without modality bypasses.

Across matched seeds and training checkpoints, report the association between these
mesh measurements and canonical fusion harmonic and arithmetic. The architecture
claim is supported only when better mesh measurements predict better fusion output.
A utilization statistic that rises without that relationship remains a diagnostic,
not evidence that the mesh represents the world more accurately.

The evidence interface is singular, but aggregation must not destroy structure
prematurely. Spatial, temporal, neighborhood, resolution, and lens relationships
should compose inside the mesh before a query is reduced to its final world latent.

## Select the target

1. Read the latest complete scorecard produced by `evaluate.py` on the current
   control.
2. Rank active human-capability benchmarks from weakest to strongest. Select the
   weakest as the default target; do not substitute `val_bpb` or training loss.
3. Use the remaining capability rows and mesh diagnostics to locate the likely
   failure in addressing, writing, composition, reading, utilization, or scaling.
4. Check recorded experiments so a known dead end is not repeated.
5. State one causal hypothesis: what changes in the mesh, which named capability
   should rise, and which diagnostic movement would support the explanation.

If the selected benchmark, fixed data, or evaluator is structurally invalid, stop
and repair it outside this program. Skipping an inconvenient weak score is not
allowed.

## Run the experiment

Follow this loop exactly:

0. **THINK.** Choose a distinct scientific mechanism, not another incidental
   cleanup or tiny parameter nudge. Prefer meaningful architectural, objective,
   routing, or data-use swings.
1. **CLAIM.** Give the experiment one unique mechanism-level name and claim it in
   the shared research record before editing. If it is already claimed or already
   tested, choose another mechanism.
2. **HYPOTHESIZE.** Write the target, mechanism, expected score movement, and a
   clear failure condition before changing code.
3. **EDIT.** Change only `mesh_research/model.py`. Architecture, writes, fusion,
   objectives, optimization, and training may all change. `data.py`, `evaluate.py`,
   benchmark code, splits, prepared data, and scoring are read-only.
4. **COMMIT.** Commit the candidate before measurement. The commit is the
   reproducible experiment; do not score an unidentified dirty diff.
5. **RUN.** Execute the fixed 1,000-step screen on the fixed seed pair:

   ```bash
   MESH_SEED=1337 python mesh_research/evaluate.py --cache /path/to/deepcal-cache --device cuda
   MESH_SEED=1338 python mesh_research/evaluate.py --cache /path/to/deepcal-cache --device cuda
   ```

6. **SCORE.** Print and read the entire canonical human-capability scorecard first.
   Then inspect the fixed mesh receipt, signed Earth4D gains, `val_bpb`, runtime,
   parameter count, and training loss as mechanism diagnostics. The receipt's
   spatial locality, temporal locality, and write rank may support or falsify the
   stated mechanism; they never replace a human-capability result.
7. **DECIDE.** A screen advances only when all three statements hold:
   - the named target improves over its matched control;
   - capability harmonic improves;
   - capability arithmetic does not regress.
8. **RECORD.** Record the hypothesis, commit, cache, hardware, steps, seed, complete
   scorecard, diagnostics, and verdict after every completed run, including losses.
9. **REPEAT.** Keep an advancing candidate as the next control. Abandon a rejected
   experiment and begin the next materially different hypothesis from the current
   control. After three failures in one mechanism family, change the mechanism.

Candidate and control must share the same research base, full prepared cache,
spatial holdout, hardware class, step budget, evaluation protocol, and seed. The
committed hypothesis must be the only difference. Never compare a 1,000-step result
with a longer-trained model.

The two-seed 1,000-step result is a fast screen, not a record. A survivor must then
be compared with a matched longer-step control before it may be called a confirmed
breakthrough. Set that fixed budget explicitly, for example
`MESH_STEPS=8000 MESH_SEED=1337 ...`; candidate and control must use the same value.

## Research surface

`model.py` may be rewritten freely as long as the public model interface required
by `evaluate.py` remains valid and all scientific information still passes through
the mesh before fusion. Reuse base DeepEarth components when they express the
thesis; do not preserve them merely from convention.

The fixed full California data includes DINO, BioCLIP, identity and traits,
phylogeny, Daymet, soil, NAIP RGB/IR, Clay, topography, canopy height, hydrology,
phenology, pooled AlphaEarth, coordinates, time, and neighbor context. Missingness
is part of the task. A hand-picked data subset is not a comparable experiment.

Priority mechanism families are:

1. **Addressing:** collision behavior, coordinate systems, scale, and time.
2. **Writing:** where and how each modality edits shared state.
3. **Composition:** cross-modal and neighbor evidence without interference.
4. **Reading:** whether fusion can recover weak signal already present in the mesh.
5. **Utilization:** whether useful mesh dependence tracks capability gains.
6. **Scaling:** whether gains survive more steps, data, extent, and capacity.

## Judge the result

The fixed evaluator is authoritative:

- human-capability harmonic is the primary well-rounded score;
- human-capability arithmetic is the breadth guard;
- the named target is the causal check on the hypothesis;
- every individual capability must remain visible;
- `val_bpb` is a lower-is-better reconstruction diagnostic and never gates;
- spatial locality, temporal locality, write rank, state rank, collisions, and
  signed Earth4D gains are shadow diagnostics and never gate;
- loss, runtime, parameters, occupancy, attention, and ablation deltas diagnose a
  mechanism but are not capability claims.

The mesh census deliberately has no promotion composite. Across the fixed
two-seed trajectory, locality and write diversity tracked fusion breadth while
raw state effective rank moved in the opposite direction. Do not optimize any
one internal statistic or reinterpret lower state rank as either success or
failure. A diagnostic advance without improvement in the named human capability
and harmonic is evidence about the bottleneck, not a promotable result.

After every run, report:

```text
HUMAN CAPABILITIES (weakest first; every active row)
named target:  control -> candidate (delta)
harmonic:      control -> candidate (delta)
arithmetic:    control -> candidate (delta)
verdict:       advance | reject | invalid
diagnostics:   space/time locality, write rank, signed Earth4D gains,
               val_bpb, runtime, parameters
```

## Standing rules

- `model.py` is the only editable file.
- Commit before running; fixed steps, never fixed wall time.
- Show the complete scorecard after every result.
- Human capability gates; `val_bpb` diagnoses.
- Do not hide regressions in an aggregate.
- Do not tune the evaluator, data, split, floor, or baseline.
- Do not turn infrastructure cleanup into a research experiment.
- Do not stack repairs onto a rejected candidate.
- Prefer a simpler model when results are equal.
- Continue the loop until stopped or genuinely blocked.
