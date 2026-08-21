# DeepCal champion report

## Fresh v2 held-species baseline

This is the first two-seed baseline trained with the deterministic species-level pollinator holdout. It activates B64
and replaces the provisional membership-only receipt; it is a protocol baseline, not a scientific improvement claim.

| Receipt | Seed | Harmonic | Arithmetic | B64 held-species NDCG@10 |
|---|---:|---:|---:|---:|
| Fresh v2 control | 1337 | 0.420846 | 0.581585 | 0.172044 |
| Fresh v2 control | 1338 | 0.419688 | 0.584098 | 0.174569 |
| **Fresh v2 baseline** | **2 seeds** | **0.420267** | **0.582842** | **0.173307** |

The previous `0.436640 / 0.598583` row reaggregated legacy checkpoints under new membership. Those checkpoints had
seen every species' interaction labels and could not report B64, so the number remains migration history and is not a
promotion target.

### Evaluation correction

The champion trains with diffusion noise, but evaluation previously sampled fresh masked-state noise on every pass.
Evaluation now reads the deterministic posterior mean; training behavior is unchanged. Replaying seed 1337 before the
fix produced harmonic values from `0.417799` to `0.420934`; the corrected model removes that explicit inference RNG.

Legacy B55 remains visible but quarantined because it scores a focal prediction against spatial neighbors' pollinator
union. B64 is the valid held-species transfer capability. Derived `*_gain` values remain visible mechanism diagnostics
and enter neither headline mean. This implements science rules 27, 30, and 32.
