# FluxFill report: outline

Draft structure of the scientific report for the dashboard's Report tab, modelled on deepearth.ai/native-habitats:
- a lede that states the result in one sentence;
- numbered figures, each comparing every method on the same data.

The report is published once FluxFill is evaluated on the test sites of FluxFill-Bench against every baseline, with
ablations. Until then this file only lists what each figure needs.

## Lede (written from measured results only)

One sentence naming the result, for example the CRPS reduction over the best published method at each gap length,
and the calibration of sums, on the 136 test towers.

## Figures

1. **Towers.** Map of the 678 towers with measured data since 2014, coloured by train, validation and test; inset of
   IGBP and Köppen counts. Data: `towers/sites.csv`, `split.csv`, `report.json`.
2. **What a fill looks like.** Three test windows (7 d, 30 d, 180 d gaps) at contrasting ecosystems. Each shows the
   masked truth, FluxFill's joint samples with 80% and 95% bands, and the best baseline for that gap length. Data:
   `bench/dashboard/windows/` exports.
3. **Every method, the same windows.** CRPS by gap-length class for NEE, LE, H and FCH4: one panel per flux, one line
   per method (climatology, MDS, RF, XGBoost, deep ensemble, SAITS, CSDI, Chronos-2, FluxFill). Regime R1, mask
   family M2.
4. **Calibration of values and of sums.** 95% coverage of half-hourly values and of daily, 30-day and window sums, by
   gap length. Every per-step method under-covers sums; joint samples are FluxFill's answer.
5. **Budgets.** Bias and CRPS of 30-day and window NEE sums in g C m⁻², and of LE and H in MJ m⁻², by gap length.
6. **No tower data (regime R3).** Skill from reanalysis and imagery alone at test towers, against climatology, the
   leave-site-out regime of the upscaling literature.
7. **What each input contributes.** Ablations, each a retrained model on the full benchmark:
   - no Sentinel-2;
   - no reanalysis;
   - no tower meteorology;
   - no site memory;
   - no cross-site retrieval;
   - marginal head instead of joint;
   - CLS tokens vs patch tokens.
8. **Where it works and where it does not.** CRPS skill over the best baseline by IGBP class, Köppen class and latitude
   band, with site counts.
9. **Imagery as a target.** Imputation of masked Clay embeddings: NLL and cosine against nearest-date persistence and
   seasonal-mean baselines.
10. **Methane.** FCH4 at the 21 test towers that measure it, against Irvin et al. 2021's protocol (mask family M1) and
    FluxFill-Bench's M2.

## Methods section

Data and quality control (`data/towers/README.md`), drivers (`data/drivers/README.md`), imagery and embeddings
(`data/sentinel2/README.md`, `embeddings/clay/README.md`), benchmark (`docs/BENCHMARK.md`), model (`docs/MODEL.md`,
`model/NOTES.md`), baselines (`baselines/README.md`), references (`docs/REFERENCES.md`).

## Data availability and attribution

Per-site citations and network acknowledgements as each data policy requires (`towers/sites.csv` columns
`shuttle_citation`, `shuttle_pid`, `acknowledgement`); Copernicus Sentinel data terms; ERA5 and ERA5-Land licences.
