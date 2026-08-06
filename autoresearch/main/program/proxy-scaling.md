# Proxy scaling — parameter/score curve

Measured to find the smallest fusion model whose results still steer the full one.
RTX PRO 6000, shared prepared cache, 621,558 obs / 29,668 held-out, ~1,000 steps
at the 120s screen budget, seed 1337. Architecture fixed at the screen config
(d_model 128, n_latents 16, n_layers 2, decoder_hidden 512, capacity 12, lr 5e-4);
only `absolute_log2_hashmap_size` varies.

| log2 hashmap | params | harmonic | arithmetic |
|---:|---:|---:|---:|
| 20 (previous default) | 172.6M | 0.322659 | 0.5112 |
| 18 | 59.4M | 0.323522 | 0.5176 |
| 16 | 31.1M | 0.329299 | 0.5184 |
| 14 | 24.0M | **0.332464** | **0.5229** |

**Smaller is better, monotonically.** 24.0M beats 172.6M by +0.0098 harmonic and
+0.0117 arithmetic — about 3x the noise floor below, so the effect is real. The
four absolute hash tables (4 x levels x 2^size x 2 features) were ~148M of the
172.6M and were not merely idle capacity; at this step budget they hurt. The
likely cause is that 2^20 entries cannot be trained in ~1,000 steps, while a
small table gets dense gradient sharing through hash collisions.

`spatial_log2_hashmap_size`/`temporal_log2_hashmap_size` were hardcoded at 20 in
the fusion constructor until `ad50262`, so this axis had never been searchable
from config. `d_model` and `n_layers` were never the parameter lever.

## Noise floor at this scale

Same config, two seeds, 172.6M: harmonic 0.322195 (1337) / 0.325487 (1338) —
spread **0.0033**. The full 796M model's two-seed spread is **0.027**, so the
proxy is ~8x more stable and ~3.5x faster (4 min vs 14 min).

For reference, the five experiments previously screened at this size were
rejected on deltas of 0.0002-0.0018, all below this 0.0033 floor.

## Open

- 2^12 and below: the trend has not turned yet.
- Rank correlation against known full-model outcomes is what certifies the proxy;
  the parameter curve alone does not.
- Low LR (5e-5, 5e-6) underfits at a fixed wall-clock budget: harmonic 0.260 and
  0.118 respectively vs 0.322 at 5e-4. The tiny-LR proxy regime in the literature
  assumes training to convergence.

## Refactor equivalence (2026-08-06)

Removing ~11,000 lines -- `probes/` entirely, `scoring/contract.py`, `definitions.py`,
`graduation.py`, `score.py`, `champion_report.py`, `run_experiment.py` -- and relocating both
encoders into `main/editable_files/encoders/` is score-neutral.

Same mirror codebase, same config (2^14, 24M, seed 1337, 120s, shared prepared cache):

| tree | params | harmonic | arithmetic |
|---|---:|---:|---:|
| `5a75482` + knob (pre-refactor) | 24.3M | 0.278777 | 0.4461 |
| `ecce317` (refactored) | 24.0M | 0.279769 | 0.4433 |
| delta | −0.3M | +0.000992 | −0.0028 |

Both deltas are inside the 0.0167 two-seed spread measured at this scale. The 0.3M parameter
difference is a dead import in `earth4d.py` that allocated the probe readout heads
(`LocalCrossEraHead`, `OrthogonalTemporalHead`, `build_probe_readout`) while referencing none of them.

**Mirror and public-main scores are not comparable.** The 2^14 sweep above scored 0.332464 / 0.5229
on public main `4d6cb44`; the same config on the mirror scores ~0.279 / ~0.445. The mirror's
`evaluate.py` is 158 lines diverged from public, so the ~0.05 gap is the evaluator, not the model.
Never compare a mirror run to a public-main run.
