# DeepCal champion report

## Query-conditioned detail evidence record

The decoder now receives three detail tokens from every observed continuous modality and reads them with a
target-conditioned attention residual. The existing fused latent remains the primary state; a learned zero-initialized
gate admits detailed evidence only when it improves a target. This preserves the shared bottleneck while avoiding the
lossy one-token compression of high-dimensional imagery and environmental embeddings.

| Model | Seed | Steps | Harmonic | Arithmetic | Parameters | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Registered champion | 2-seed mean | 2,291 | 0.378407 | 0.587374 | 25.4M | 19,100 MB |
| Detail evidence | 1337 | 2,291 | 0.381355 | 0.588091 | 28.9M | 21,225.8 MB |
| Detail evidence | 1338 | 2,291 | 0.383536 | 0.594141 | 28.9M | 21,225.8 MB |
| **Detail-evidence mean** | **2 seeds** | **2,291** | **0.382446** | **0.591116** | **28.9M** | **21,225.8 MB** |
| **Delta** |  |  | **+0.004039 (+1.07%)** | **+0.003742** | **+3.5M** | **+2,125.8 MB** |

Both seeds independently exceed the registered champion on Lance's unchanged harmonic and arithmetic aggregates.
The comparison uses the public spatial holdout, all 58 active benchmarks, exactly 2,291 optimizer steps, and
checkpoint replay. Evaluator definitions, data preparation, and holdout membership are unchanged.

The largest capability gains are hydro reconstruction (+0.0410), topography reconstruction (+0.0389), growth form
(+0.0166), photo-only species top-1 (+0.0148), photo-conditioned species top-1 (+0.0142), phylogeny reconstruction
(+0.0140), LFMC (+0.0139), and NAIP-IR reconstruction (+0.0128). The result advances science rules 13-16 and 23:
fusion still consumes typed multimodal tokens through a shared latent field, while interface decoders can ask
target-specific questions of the observed evidence.

## Regressions

The aggregate record includes four capability regressions larger than 0.005. They are reported explicitly rather
than hidden: pollinator distribution fit 0.531038 -> 0.474724, species-to-pollinator recall 0.460037 -> 0.439075,
pollinator calibration 0.684013 -> 0.663654, and flowering fidelity 0.734281 -> 0.727470. The complete before/after
scorecard is in `BENCHMARKS.md`; exact unrounded values are in `champion_scores.json`.

## Reproduction

Run `autoresearch/champion.yaml` for seeds 1337 and 1338 at exactly 2,291 steps, then replay each checkpoint through
the public evaluator. The canonical `autoresearch/deepcal.yaml` carries the same model setting. No checkpoint or
generated cache is committed.
