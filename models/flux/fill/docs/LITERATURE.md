# Literature: long-gap filling of eddy-covariance fluxes, 2023 to 2026

Compiled 2026-10-26. Scope: CO2/NEE (and GPP/Reco), CH4, H2O/LE, H, N2O. Focus: contiguous gaps of weeks to many months. Earlier baseline papers (Irvin 2021, Liu 2025, Gao 2023, Li 2026 Sci China, Zhu 2022/2023, Vekuri 2023, Mahabbati 2021, Kim 2020, Lucas-Moffat 2022, Uyekawa 2024) are cited only where needed for comparison.

Verification legend used throughout:
- **FT**: read from full text (PDF, HTML, PMC XML, arXiv).
- **ABS**: abstract only (publisher, OpenAlex, Europe PMC, conference page).
- **SNIP**: only search-engine snippets of the abstract were available; numbers are unverified.
- **NV**: not verified (title or metadata only).

ScienceDirect and Wiley served a CAPTCHA or HTTP 403 to every automated request during this review. Elsevier papers without an open-access copy are therefore ABS or SNIP.

## Summary of findings

1. No 2023 to 2026 paper applies a time-series foundation model (TimesFM, Chronos, Chronos-2, Moirai, Lag-Llama, TimeGPT), TabPFN/TabImpute, SAITS/BRITS/PyPOTS, S4/Mamba, or masked-autoencoder pretraining to eddy-covariance flux data. GitHub code search for these model names combined with FLUXNET variable names returned nothing. The only diffusion model evaluated on tower fluxes is ZeroDiff (ICML 2026, FLUXNET-CH4, leave-location-out), and it reports point metrics only.
2. No paper in either family scores flux predictions with a proper scoring rule (CRPS, energy score, NLL on held-out data). The strongest uncertainty evidence for long gaps is Vekuri et al. 2025. They use known synthetic truth, gaps of 1 to 28 days placed at every position in the year, prediction-interval coverage (PICP), and a balance calibration ratio. Deep ensembles with a Laplace likelihood were calibrated for long gaps except during rapid ecosystem change. MDS intervals were too narrow (78 to 82% coverage at nominal 95%). Tree methods with Platt scaling were miscalibrated for annual sums and could not extrapolate.
3. The skill ceiling depends on the regime more than on the architecture:
   - **Gap inside a record**, with other years observed at the same site. NEE and GPP reach R2 about 0.8 to 0.9 on held-out years; CH4 reaches about 0.5 to 0.7.
   - **Leave-site-out**, the proxy for a site with no usable record. Daily NEE median R2 is only 0.22 to 0.28 (CarbonFluxBench). Hourly NEE NSE is about 0.66 to 0.71 (X-BASE, EC-STILT), but daily anomalies reach only NSE 0.25 to 0.5 and interannual variability is near unskilled. CH4 leave-site-out R2 is about 0 to 0.2 (X-MethaneWet) or 0.51 daily (WetCH4).
   - **Conditioning on the site's own history helps.** Support sets, historical-year encoders and site-year retrieval (TAM-RL, CHAM-net, RACI) exploit the first regime and are the most transferable deep-learning idea for long gaps.
4. The best-evidenced long-gap fillers are still per-site gradient-boosted trees or random forests with vegetation-state covariates. The covariates are PhenoCam GCC, Sentinel-2 or MODIS indices, and SIF. These were tested at 1 to 64 sites with contiguous gaps of 30 to 240 days. Deep sequence models win on interannual variability (Huang et al. 2024: LSTM IAV r = 0.81 vs RF r = -0.21, 82 sites, monthly) and on whole-year holdouts of energy fluxes (Hou et al. 2025). Neither has been tested with a realistic multi-site long-gap protocol.
5. Process-based and hybrid models have not been evaluated on contiguous long gaps at all. Every process-model gap test found uses random hours. A single-site LandscapeDNDC test was slightly worse than MDS on random hours while shifting annual NEE by about 25%. Hybrid work contributes three things:
   - identifiable structure (Q10, resistances, light-use efficiency);
   - knowledge-guided constraints (Zhu Q. et al. 2026 CH4: monotonic in GPP, bounded Q10);
   - pretraining on simulator output, which gives the largest data-efficiency gains reported (KGML-ag-Carbon; X-MethaneWet with TEM-MDM; AgroFlux).
6. Evaluation weaknesses are common:
   - random half-hour or random 80/20 splits inside a site;
   - hyperparameters tuned on data that later serves as test;
   - checkpoints selected on the test set (Montero et al. 2024, explicit);
   - a single split with 1 to 5 test sites per class (EcoPerceiver);
   - "zero-shot" claims that use the test site's own history (TAM-RL 2026);
   - in-sample fit reported as gap-filling skill (miniRECgap);
   - missing flux values treated as zero (exposed in the Kim et al. 2026 comment on an OpenET evaluation).

## (a) Ranked table: strongest methods for long gaps

Ranking criterion: demonstrated skill on contiguous gaps of at least 30 days, or on held-out years or sites. Within that, the order weighs rigor of the protocol, number of sites, annual-sum error reporting, and calibrated uncertainty. Point accuracy alone does not set the rank.

Abbreviations:
- Gap simulation and split: RHH = random half-hours or hours; CB = contiguous artificial blocks; LSO = leave-site-out; TH = temporal holdout of whole years at seen sites.
- Uncertainty: UQ = uncertainty quantification; PS = proper score.

| # | Method, citation | Year | Variables | Sites / ecosystems | Longest gap or holdout | Gap simulation and split | Headline metrics | UQ (calibrated? PS?) | Code / data | Weaknesses | Ver. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Deep ensemble (5 MLPs, Laplace NLL, aleatoric + epistemic) vs MDS, XGB+Platt, NGBoost. Vekuri et al., AFM 371:110558, doi:10.1016/j.agrformet.2025.110558 | 2025 | NEE (CO2) | 12 European forest site-years (ICOS Warm Winter 2020). Synthetic truth from process models (incl. APES) at FI-Sod, FI-Hyy, DE-Tha, DE-Hai | 28 d (1 to 28 d every 3 d, every start position in the year). Missing fraction 30 to 90% | CB on synthetic truth with an ICOS-derived gap pool. 20% and 50% random removal, plus a 14-d block, on measured data | Annual random uncertainty grows from about 10 to 25-75 g C m-2 y-1 as missingness rises from 30 to 90%. 4-week gap at DE-Tha: 21 (NN) vs 39 (MDS) g C m-2 y-1. Dry warm August gap at DE-Hai: up to 99. Annual error mostly below 10% of the balance | Yes. PICP@95 (NN 94-96%; MDS 78-82% for long gaps) and balance calibration ratio. Laplace NLL is the training loss, not a reported score. No CRPS | github.com/hvekuri/nn_gapfilling_unc (code). No separate data deposit | Long-gap hyperparameters chosen on one evaluation scenario (DE-Tha, 4-week June gap) against a target calibration score of 1.15. Only 3 drivers (Ta, SW, VPD) plus time encodings. Forests only | FT |
| 2 | Knowledge-guided NN for wetland CH4 (loss penalizes negative dFCH4/dGPP and Q10 outside 1.4-5.8). Zhu Q. et al., Global Change Biology, doi:10.1111/gcb.70899 | 2026 | CH4 (daily) | 11 longest AmeriFlux wetland records | Contiguous thirds of each record (multi-year) | Random 80/20, then contiguous-thirds TH, per site | R2 0.66 to 0.96 random vs 0.50 to 0.92 contiguous. 0.26 at US-LA2 (82% days missing). Budget spread up to 19% | Spread of the top-10 hyperparameter configurations. Not calibrated, no PS | Zenodo 10.5281/zenodo.19543891 | Models picked by test performance (test-set tuning risk). One model per site | FT (agent) |
| 3 | AutoML (H2O), site-specific, ERA5-Land + MODIS NDVI/LAI. Qian L. et al., ESSD 18:6707, doi:10.5194/essd-18-6707-2026 | 2026 | LE | 50 ChinaFlux sites (10 already PI-gap-filled, flagged) | 30-d blocks. Forward and backward prolongation over one third of each record | CB (30 min, 1, 7, 30 d, about 5% each). Random 80/20 x20 for gap tests. Contiguous thirds for prolongation | 30 d: CC 0.819, RMSE 37.26 W m-2 vs MDS 49.85. Prolongation CC 0.90, RMSE 35.3-35.9. Median PBias about 1.5% | Repeat spread only | Data Zenodo 10.5281/zenodo.18194590. Code not found | Random split for model fitting next to the block gaps. No UQ | FT (HTML) |
| 4 | Transformer-CNN (vs LSTM, GRU, Transformer, XGB, SVR, KNN, MDV, LUT). Hou et al., GMD 18:4625, doi:10.5194/gmd-18-4625-2025 | 2025 | H, LE (hourly) | 1 alpine steppe site (QOMS, Tibetan Plateau), 2007 to 2016 | Whole-year holdout (2012). Rolling year-out in the supplement | TH by year, 10% random validation | R2 0.95 (H), 0.90 (LE) | None | Zenodo 10.5281/zenodo.10005741 | Single site. Paper text also states an 80/10/10 split, which conflicts with the year holdout | FT |
| 5 | XGBoost with in-situ met + SWC + tower SIF, vs ERA5 + Sentinel-2, vs hybrid. De Cannière et al., AFM 378:110987, doi:10.1016/j.agrformet.2025.110987 | 2026 | NEE, GPP | 1 (BE-Bra, Scots pine, ICOS) | 56 d | CB (details NV) | Beats MDS in all scenarios. Minimal degradation to 56 d. SWC and SIF help at Ta > 25 C | NV | NV | Single site | SNIP |
| 6 | Full-factorial (decoupled Penman-Monteith, median-adjusted) + iterative filling, reanalysis-filled drivers. Wang X. et al., Sci Data, doi:10.1038/s41597-025-06144-9 (preprint essd-2024-291) | 2025 | LE (daily) | 265 sites, 1509 site-years published (339 screened; forest, grass, crop, wetland, shrub) | 5 to 50 d | Preprint: random 50% removal within a random 10% subset. Published: long-period test 5-50 d | RMSE 14.69-19.51 W m-2 (isolated), 24.41 (5-50 d). Beats ERA5-Land and GLEAM | None | Dataset public (Sci Data) | Validation mostly random. Daily only. Physics-based, so no learned site memory | ABS + preprint FT |
| 7 | Bias-corrected random forest (two-stage RF). Li W. et al., ESSD 17:3835, doi:10.5194/essd-17-3835-2025 | 2025 | LE (hourly, daily) | 64 FLUXNET2015 sites | 30 d | CB (30 min, 1, 7, 30 d, about 5% each). Random 80/20 x20 | Hourly RMSE 33.86 W m-2, CC 0.86. Long gaps 33.44 vs MDS 37.64 | Box-plot spread only | Data Zenodo 10.5281/zenodo.13853409. Code not released | Random split for fitting. No UQ | FT (HTML) |
| 8 | Meta-network: encoder over a site's historical years conditions an LSTM decoder. CHAM-net, Dong et al., arXiv:2606.00338 | 2026 | CH4 emission and uptake | 30 FLUXNET-CH4 wetlands + 28 upland sites | 1/7 of each record (temporal) | TH per site. Ordering not explicit in text | Emission nRMSE 0.88, R2 0.68 | None | zenodo.org/records/20450697 | Point metrics only | FT (agent) |
| 9 | Hierarchical daily/monthly/yearly encoder with neighbour-site and similar-site-year retrieval. RACI, Sun et al., arXiv:2603.03531 | 2026 | GPP, CO2, CH4 | AgroFlux FLUXNET crop sites, X-MethaneWet 30 CH4 sites, plus simulators | Last 5 years held out (2016-2020) | TH at seen sites | GPP R2 0.87 (LSTM 0.84). CO2 0.78 (0.71). CH4 0.49 (0.44). TimesNet, DLinear, PatchTST, TimeMixer had negative R2 on CH4 | None | Baselines via TSLib. No RACI code | Within-site R2. No sums | FT (agent) |
| 10 | Task-aware modulation (support set of site history) for upscaling. TAM-RL, Rozanov et al., arXiv:2603.09974; original Renganathan et al., arXiv:2310.04727 | 2026 / 2023 | GPP, Reco, NEE (daily) | 579 available; 164 held-out sites (2026). 145 train / 15 test (2023) | Site with 1 to 24 months of support data | LSO with support set from the held-out site | 2026: NEE RMSE/R2 1.62/0.23 vs FLUXCOM-X-BASE 1.76/0.16. GPP 1.97/0.43 vs 2.18/0.36 | None | NV | "Zero-shot" uses the test site's own past years | FT (agent) |
| 11 | LSTM with PFT conditioning, SIF + LAI + met. Huang et al., JGR-Atmos, doi:10.1029/2023JD040418 | 2024 | NEE (monthly) | 82 North American sites | Test set (split NV) | NV | Test R2 0.79, RMSE 0.89 g C m-2 d-1. IAV r = 0.81 vs RF -0.21 | None | NV | Monthly only. Split not verified | ABS |
| 12 | CNN-LSTM with ERA5, NOAA NDVI, and 4 ET products as inputs. Qian L. et al., J Hydrol 641:131787, doi:10.1016/j.jhydrol.2024.131787 | 2024 | ET (daily) | 241 FLUXNET sites | Random removal 5 to 50% (fill limit reached at 93%) | RHH-like random daily removal, x30 | R2 > 0.8 at 50% gaps. ET products add 10 to 30% | None | NV | Random daily removal is not a long-gap test. Gridded ET products as inputs partly encode the target | SNIP |
| 13 | MLP with Bayesian-optimized architecture and lags vs ImputeTS, mtsdi. Park et al., arXiv:2202.12441 | 2022 | NEE (hourly) | 1 (US-MMS) | Single contiguous 3, 12, 24, 36 months | CB, one start date | RMSE 0.96 / 2.68 / 2.88 / 2.94 vs ImputeTS 1.86 / 8.14 / 8.20 / 8.15 | None | NV | One site, one gap position, no sums. Pre-2023, included as the only multi-month deep-learning test | FT (agent) |
| 14 | MissForest (short) + Prophet (long). Abdaki et al., JGR-B, doi:10.1029/2024JG008375 | 2025 | CO2, H, LE + 3 others | 1 tropical dry forest (Costa Rica) + 3 validation sites | Gaps sized by percent of a 2013-2022 record. Both models fail at 10-13% (about 1 to 1.3 yr) | NV | Short R2 0.76. Prophet at 5-10% gaps R2 0.72. Validation R2 0.79 (CO2), 0.88 (H), 0.77 (LE) | None | NV | Prophet is driven mainly by seasonality. Long gaps unsolved | ABS |

Leave-site-out proxies for "entire record missing". These are not gap fillers, but they set the skill ceiling for very long gaps:

| Product / benchmark | Variables, sites | Split | Skill | UQ | Ver. |
|---|---|---|---|---|---|
| X-BASE / FLUXCOM-X (XGBoost), Nelson et al., Biogeosciences 21:5079, doi:10.5194/bg-21-5079-2024 | NEE, GPP, ET, T; 294 sites, hourly | 10-fold LSO, sites within 0.05 deg share a fold | Hourly NEE NSE about 0.71, diurnal-mean 0.81. Daily anomalies NSE 0.25 to 0.5. Poor IAV and between-site NEE | No calibrated per-site distribution | FT (agent) |
| EC-STILT, Upton et al., ACP 26:2561, doi:10.5194/acp-26-2561-2026 | NEE; same 294 + 3 tall towers | 10-fold LSO | Hourly NSE 0.66, diurnal-mean 0.76. Global IAV improved | Ensemble spread, unscored | FT (agent) |
| CarbonFluxBench (formerly CarbonBench), Rozanov et al., arXiv:2603.09868 | GPP, Reco, NEE daily; 474 sites (v2) | LSO, IGBP- or Koppen-stratified | Median NEE R2 0.22 to 0.28. GPP 0.57 to 0.65. NEE nMAE 2.0 to 2.7 | Seed ensembles only | FT (agent) |
| WetCH4 (RF), Ying et al., ESSD 17:2507 | CH4; 26 sites, 138 site-years | Leave-one-site-out | Daily R2 0.51, monthly 0.70 | Bootstrap | FT (agent) |
| X-MethaneWet, Sun et al., arXiv:2505.18355 | CH4 daily; 30 FLUXNET-CH4 sites + TEM-MDM | TH and 5-fold LSO | TH R2 about 0.55 (0.64 with simulator transfer). LSO R2 -0.01 to 0.19 | None | FT (agent) |
| EcoPerceiver (CarbonSense), Fortier et al., ICLR 2025, arXiv:2406.04940 | NEE (+GPP, Reco) hourly; 385 sites | One random site split per IGBP class, 1-5 test sites each | NSE CRO 0.848 vs XGB 0.807; WET 0.414 vs 0.453 | None | FT (agent) |
| FLUXtrapolation, Fries et al., arXiv:2605.19812 | NEE, GPP, ET hourly; 207 sites | Temporal, random spatial, and hottest-site spatial | Median and 90th-percentile RMSE across hourly to IAV scales | None (tabular baselines) | FT (agent) |

Earlier baselines, listed for ranking context:
- Zhu et al. 2022 / 2023: RFR, 94 FLUXNET sites, up to 30 d; RFR reliable for cumulative fluxes with gaps longer than 3 months.
- Liu et al. 2025: XGBoost + PhenoCam GCC, US-Bar, up to 240 d.
- Irvin et al. 2021: FLUXNET-CH4, realistic gap-length sampling, Platt-scaled calibration.

Vekuri 2025 shows that Irvin-style Platt scaling, calibrated on short-gap fluxes, does not transfer to annual balances.

## (b) Findings by family

### Family 1: deep learning, foundation models, multi-site and remote-sensing-driven models

**Direct deep-learning gap fillers.** Verified examples are few, and all are single-site or random-gap tests:
- Hou et al. 2025 (Transformer-CNN, H and LE, whole-year holdout, R2 0.95 / 0.90, FT).
- Qian et al. 2024 (CNN-LSTM, daily ET, 241 sites, random removal, SNIP).
- Vekuri et al. 2025 (MLP deep ensembles, FT).
- Park et al. 2022 (MLP, 3 to 36 month single gaps, FT).
- Tefera et al. 2026, AFM 111438, doi:10.1016/j.agrformet.2026.111438 (ABS via agent).

Tefera et al. benchmark LE, H and G reconstruction at 6 FLUXNET sites under contiguous 6 to 24 h outages. Gap structure controls skill more than missing fraction, and deep models produce larger energy-balance residuals than tree models. Hess et al. 2025 (ESSOAr, doi:10.22541/essoar.175977613.32796246/v1, ABS) train a multi-site LSTM for CH4 on 4 Alaskan sites and apply it to a fifth (leave-site-out); the abstract gives no skill numbers.

**Foundation models and generic imputers.** No application to eddy-covariance flux data was found for TimesFM (arXiv:2310.10688), Chronos (arXiv:2403.07815), Chronos-2 (arXiv:2510.15821), Moirai (arXiv:2402.02592), Lag-Llama (arXiv:2310.08278), TimeGPT (arXiv:2310.03589), MOMENT (arXiv:2402.03885), SAITS (arXiv:2202.08516), BRITS (arXiv:1805.10572), CSDI (arXiv:2107.03502), SSSD (S4-based diffusion, arXiv:2208.09399), PriSTI (arXiv:2302.09746), ImputeFormer (arXiv:2312.01728), GRIN (arXiv:2108.00298) or SPIN (arXiv:2205.13479).

Search coverage: arXiv, OpenAlex, Crossref, and the web with multiple phrasings, plus GitHub code search.

Three related results:
- **ZeroDiff** (Fan, Lu, Jia, ICML 2026, arXiv:2609.37078, FT via agent) builds an informed prior from exogenous drivers and calibrates its residual by diffusion. It is evaluated leave-one-location-out on FLUXNET-CH4 (16 test locations). Standard diffusion baselines failed on the CH4 data. Scoring is NSE, RMSE and MAE only.
- **RACI** found negative R2 for TimesNet, DLinear, PatchTST and TimeMixer on FLUXNET-CH4 temporal holdout. Generic forecasting backbones without driver conditioning and site memory do not transfer to fluxes.
- **TSI-Bench** (arXiv:2406.12747) and the imputation survey (arXiv:2402.04059) are the general references for block-missing protocols. Neither includes flux data.

**Multi-site and cross-tower models.** No graph neural network across towers was found. Zou et al. 2025 (RSE, doi:10.1016/j.rse.2025.114952) build their DeeperGCN graph within each tower footprint, and use a random 60/20/20 split that leaks in time. Three models use cross-site information:
- RACI retrieves neighbour-site and similar-site-year information.
- TAM-RL modulates a shared model with a site's support set.
- CHAM-net encodes a site's historical years.

The 2023 TAM-RL paper shows the expected cross-over. Task modulation wins with 1 to 3 months of site data; a fine-tuned CT-LSTM wins with 12 to 24 months.

**Remote-sensing and reanalysis covariates.**
- Vegetation-state covariates are the most consistent improvement for long gaps in tree models: PhenoCam GCC (Liu 2025), Sentinel-2 and tower SIF (De Cannière 2026), MODIS NDVI/LAI (Qian 2026).
- SIF, shortwave radiation and LAI rank top in the 82-site LSTM (Huang 2024).
- ERA5 or ERA5-Land replaces missing tower drivers in Zhu 2023, Wang X. 2025, Qian 2026 and CarbonFluxBench. Li T. et al. 2026 (AFM 111389, doi:10.1016/j.agrformet.2026.111389, NV) address multisource gap filling of the drivers themselves, which matters because driver and flux gaps coincide during instrument outages.
- EcoPerceiver uses tower meteorology plus MODIS MCD43A4 patches. Its current repository lists PhenoCam inputs and CH4 targets added after the ICLR paper.

**Probabilistic outputs.** Vekuri 2025 is the only flux study that verifies interval coverage for long gaps and for annual balances. The method that worked was a deep ensemble predicting Laplace location and scale, with epistemic spread across members and balance uncertainty computed from member-level balances. Zammit-Mangion et al. (arXiv:2610.12967) is not eddy covariance (atmospheric CH4 inversion), but it shows CRPS, energy score and 90% coverage scoring for guided diffusion against MCMC.

### Family 2: process-based, physics-informed and hybrid models

**Response-function and footprint-weighted Bayesian models.**
- miniRECgap (Premrov et al. 2025, Environ Model Softw, doi:10.1016/j.envsoft.2025.106611, ABS) fits Lloyd-Taylor plus light-response functions. At a cutaway peatland: R2 0.57 vs MDS 0.24 vs a shallow ANN 0.68. This is in-sample.
- Footprint-unmixing Bayesian models: Ludwig et al. 2024 (Biogeosciences 21:1301, FT; NEE and CH4, tundra) and Buzacott et al. (EGU23-12916; AFM 2024, doi:10.1016/j.agrformet.2024.110179, ABS; Dutch peatlands). They beat homogeneous gap filling on withheld random gaps, but the longest test gap was 5 h. A homogeneous-landscape assumption doubled the tundra CO2 sink and halved CH4 in upscaling.

**Land-surface models and data assimilation.**
- LandscapeDNDC at a mixed beech and Douglas-fir forest (Moutahir et al. 2026, Biogeosciences 23:1719, doi:10.5194/bg-23-1719-2026, FT via agent): hourly NEE R2 0.55 to 0.63; slightly worse than REddyProc MDS on 354 random gap hours. Annual NEE: -670 (model-filled) vs -898 kg C ha-1 yr-1 (MDS-filled), with no truth to adjudicate.
- CARDAMOM-FluxVal (DALEC with MCMC, 204 FLUXNET sites, Yang et al. 2022, GMD 15:1789): NEE and GPP modeling efficiency below 0 until 4 to 5 years are assimilated, at monthly resolution.
- No 2023 to 2026 Kalman, particle-filter or 4D-Var flux gap filler was found.

**Differentiable and hybrid models.**
- The Reichstein agenda (Nature 2019; 2022 hybrid ecosystem modeling chapter) and its descendants contribute identifiable structure, not gap-filling evaluations: double machine learning for Q10 (Cohrs et al. 2024, doi:10.1088/2632-2153/ad5a60), EasyHybrid.jl (JuliaCon 2026), ScIReN (arXiv:2506.14054) and DMDc respiration (arXiv:2402.17625).
- Hybrid Penman-Monteith with learned resistances (ElGhawi et al. 2023, ERL, doi:10.1088/1748-9326/acbbe0) is equifinal unless constrained by priors or multitask H and LE. Interannual forest anomalies reach only R2 about 0.47.
- Differentiable DALEC (DifferLand; Fang and Gentine 2024 JAMES; Fang et al. 2026 Nat Commun, doi:10.1038/s41467-026-73395-4) and differentiable FATES photosynthesis (Aboelyazeed et al. 2023, 2025) are monthly or leaf-scale.
- PERNN (Zia et al. 2025, Mach Learn 114:180, arXiv:2411.11497, FT via agent) is the only hybrid that explicitly gap-fills NEE: an Arrhenius ODE block, one UK site, 2018-2019 test years. Nighttime R2 0.73 vs random forest 0.721. Whether observed NEE leaks into gaps through the one-step recurrence is unclear.
- pyVPRNN (Glauch et al. 2026, EGUsphere-2026-4847, partial FT via agent): 32 ICOS sites, Sentinel-2, half-hourly footprints, Laplace NLL with a fixed scale from measured NEE uncertainty. The model does not predict its own scale.

**Simulator pretraining and knowledge-guided ML.**
- KGML-ag-Carbon (Liu L. et al. 2024, Nat Commun, doi:10.1038/s41467-023-43860-5): pretrained on ecosys simulations, then fine-tuned at 11 cropland EC sites. Daily NEE R2 0.96 on 2 held-out years, vs 0.59 to 0.88 for ecosys alone. Satellite GPP is an input.
- X-MethaneWet: TEM-MDM transfer raises CH4 temporal-holdout R2 from about 0.55 to 0.64.
- AgroFlux (arXiv:2602.01614): simulation pretraining raises LSTM crop CO2 leave-site-out R2 from 0.339 to 0.648.
- Zhu Q. et al. 2026 (rank 2 above): CH4 gradient and Q10 constraints in the loss.

**Upscaling products as site gap fillers.** X-BASE and EC-STILT train on ONEFlux MDS-gap-filled data with QC weighting. WetCH4 trains partly on gap-filled CH4, which makes its evaluation partly circular. No study uses these products to fill a site's long gaps with gap-length-resolved skill. Their leave-site-out scores (table above) are the reference skill when a tower has no usable data for a period and no site-specific model exists.

**N2O.** No eddy-covariance N2O deep model or benchmark exists:
- Random forest is the operating standard for EC N2O gap filling: Barczyk et al. 2024 AFM, doi:10.1016/j.agrformet.2024.110278 (3-year pasture, RF gap filling plus partitioning); Feigenwinter et al. 2023 STOTEN, doi:10.1016/j.scitotenv.2023.166389 (RF better for N2O than CH4).
- Tikkasalo et al. 2025 (Biogeosciences 22:1277, FT) run an RF/XGB/kNN ensemble on Irvin-style gaps covering 15% of the record: R2 0.92 for N2O, 0.75 for CO2, 0.66 for CH4. Hyperparameters were tuned on a random subset drawn before the test gaps were inserted.
- Other N2O models are chamber-based or very small: AgroFlux N2O (6 mesocosm chambers), Sharma et al. 2026 PNAS (chamber data, 17 sites, hybrid ensemble, held-out-site R2 0.84), Payrosangari et al. 2025 (egusphere-2025-5155, 5-site LSTM, relative RMSE 270%), and a 4-site PINN with negative leave-site-out R2 (arXiv:2607.23880).

**CH4 application papers 2023 to 2026.** These reuse the Irvin framework and add no methods:
- Buzacott et al. 2024 GCB, doi:10.1111/gcb.17590: 16 Dutch peatland sites; lasso, ANN, RF and XGB; gaps over 60 d filled; 95% interval from a 10-model ensemble (FT).
- Ma et al. 2025 J Clean Prod, doi:10.1016/j.jclepro.2025.145615: MDS plus ML for paddy CH4 (NV).
- Li X. et al. 2026 Ecol Indic, doi:10.1016/j.ecolind.2026.114695: MIC feature selection with RF, SVR and ANN at the Dajiuhu peatland (ABS).

## (c) Benchmarks, datasets and protocols

**Source data.**
- FLUXNET2015 / ONEFlux (MDS, with ANN for long gaps in some products).
- ICOS Warm Winter 2020 and ICOS 2023.
- AmeriFlux FLUXNET-format data.
- FLUXNET-CH4 V1.0 (81 sites). V2.0 is in preparation; an October 2024 workshop targeted CH4 integration into ONEFlux.
- The FLUXNET Data System (operational since early 2026, distributed through the FLUXNET Shuttle library and GUI). Its page states no change in gap-filling method.

**ML-ready benchmarks, 2024 to 2026.**
- CarbonSense V2 hourly (Zenodo 15586300).
- CarbonFluxBench (github.com/alexxxroz/CarbonFluxBench; daily; LSO).
- FLUXtrapolation (Dataverse doi:10.7910/DVN/Q1MPVG; temporal, spatial and temperature-extrapolation splits; multi-scale and tail RMSE).
- X-MethaneWet (HuggingFace ymsun99/X-MethaneWet; CH4 with simulator pairing).
- AgroFlux (crop CO2, GPP and chamber N2O with Ecosys and DayCent; code on acceptance).
- AMERI-FAR25 (FAR, arXiv:2512.01917; 30 m Landsat; includes a test_future split).

None defines contiguous gap masks of weeks to months, and none scores probabilistic forecasts.

**Gap-simulation protocols in use, ordered from least to most rigorous for long gaps:**
1. Random half-hours or a random 80/20 split within a site. Used widely; leaks temporal information and overstates long-gap skill.
2. Random gaps drawn from the site's empirical gap-length distribution (Irvin 2021; Tikkasalo 2025). Realistic for typical gaps, but dominated by short gaps.
3. Fixed contiguous blocks of 1 to 30 d, each covering a few percent of data (Zhu 2022; Li W. 2025; Qian 2026). Usually combined with random fitting splits.
4. Contiguous blocks at every start position and month, scored on annual balances against synthetic truth produced by process models with realistic noise (Vekuri 2025). The only protocol that measures true annual-sum error and calibration for long gaps.
5. Whole-year or contiguous-third temporal holdouts at seen sites (Hou 2025; Zhu Q. 2026; RACI; CHAM-net; FLUXtrapolation temporal). The proxy for multi-month gaps inside a record.
6. Leave-site-out with spatial blocking (X-BASE, 0.05 deg folds) and stratified site splits (CarbonFluxBench). The proxy for missing records.

**Most rigorous today.** Protocol 4 for annual sums and calibration. Protocols 5 and 6 for multi-month and multi-year skill. FLUXtrapolation's multi-scale and tail metrics for detecting smoothing. No benchmark combines realistic contiguous masks, multiple sites, annual-sum bias, interannual variability, and proper probabilistic scores.

## (d) Open problems and highest-value directions for a joint multi-site, multi-variable, space-time probabilistic model

**Open problems:**
1. **Non-stationarity inside long gaps.** Drought onset, start of season, mowing, harvest, grazing and disturbance cause the largest errors and the miscalibrated intervals (Vekuri 2025; Zhu 2023; Barczyk 2024). Standard drivers (Ta, SW, VPD) do not observe ecosystem state.
2. **Epistemic uncertainty for unobserved regimes.** MDS and trees cannot extrapolate. Deep ensembles are calibrated only until the gap covers a regime absent from training.
3. **Annual-sum error.** Most deep-learning papers report half-hourly R2 or NSE, never cumulative-sum bias or interannual variability. Correlated errors across a gap make independent per-step uncertainty invalid for budgets.
4. **CH4 and N2O pulses.** Heavy tails, ebullition, freeze-thaw and fertilization pulses. Leave-site-out CH4 skill is near zero, and no EC N2O benchmark exists.
5. **Coincident driver gaps.** Meteorology and soil sensors fail with the flux system, and reanalysis must substitute with site-specific bias.
6. **Footprint heterogeneity.** Homogeneous-landscape assumptions bias CO2 and CH4 by up to a factor of 2 at heterogeneous sites.
7. **Evaluation hygiene.** Leakage, test-set selection, and unproper scores (see Summary of findings, item 6).

**Highest-value directions, in order:**
1. **Build the long-gap benchmark before the model.**
   - Use the CarbonSense V2 / FLUXNET Data System sites plus FLUXNET-CH4.
   - Transplant real gap masks from other site-years.
   - Add synthetic contiguous blocks of 1 d, 1 wk, 1, 3, 6 and 12 months at every season.
   - Score three regimes separately: (i) gap inside a record; (ii) site with only other years observed; (iii) site never observed (spatially blocked leave-site-out).
   - Metrics: CRPS per half-hour and per daily mean; CRPS and bias of monthly and annual sums computed from joint samples; energy or variogram score across variables; PIT and interval coverage by gap length; interannual-variability correlation; FLUXtrapolation-style tail RMSE.
   - Add a synthetic-truth track (process-model simulations with Laplace, heteroscedastic noise, following Vekuri 2025) to measure annual-sum error exactly.
   - Tune hyperparameters on held-out sites or years that never enter the test set.
2. **Model the site x time x variable tensor generatively.**
   - Use a masked-modeling or conditional diffusion/flow backbone that outputs joint samples across time and variables (NEE, GPP and Reco decomposition, LE, H, CH4, N2O), not per-step marginals.
   - Joint samples are required for calibrated monthly and annual sums.
   - ZeroDiff's failure analysis suggests conditioning the diffusion on an informed prior from drivers; unconditioned diffusion baselines failed on CH4.
3. **Give the model site memory and cross-site context.**
   - A site encoder over all observed years (the CHAM-net and TAM-RL pattern).
   - Retrieval or attention over climatically similar site-years and spatial neighbours (RACI).
   - Spatial blocking during training, so regime (iii) is learned rather than assumed.
   - A graph across towers, with edges from distance, climate and plant functional type, is unexplored for flux imputation. ImputeFormer, GRIN and SPIN are direct methodological starting points.
4. **Observe ecosystem state with dense covariates:** daily Sentinel-2/HLS and MODIS/VIIRS indices, SIF (TROPOMI, OCO, tower), PhenoCam GCC, ERA5-Land soil moisture and snow, and management event flags. These are the covariates that improved long-gap skill in every study that tested them. Train with modality dropout so the model degrades gracefully when tower drivers are missing and ERA5-Land substitutes.
5. **Embed light physics as decoders and constraints, not as full process models:**
   - NEE = Reco - GPP, with Lloyd-Taylor or Q10 Reco and light-use-efficiency GPP heads; per-site latent parameters with priors.
   - Energy balance with an explicit, bounded closure residual (GCB 2026, doi:10.1111/gcb.70892, shows a persistent unexplained residual that should not be forced to zero).
   - Monotonic and bounded CH4 responses (Zhu Q. 2026).
   - A footprint-weighted land-cover mixture as the observation operator where land-cover maps exist.
6. **Pretrain on simulator ensembles and upscaling products, then fine-tune on towers.** Sources: ecosys, DayCent, TEM-MDM, CLM/ELM runs at tower locations, and X-BASE hourly fields. This is the only transfer result that is consistently positive for sparse targets (CH4, N2O, crops). It also supplies the synthetic-truth track for benchmark item 1.
7. **Train and select with proper scores; calibrate per gap length.**
   - Train with heteroscedastic, heavy-tailed likelihoods (Laplace or Student-t NLL) or CRPS.
   - Get epistemic spread from ensembles or diffusion sampling.
   - Apply post-hoc calibration on held-out contiguous blocks, stratified by gap length and season (block conformal or isotonic recalibration of the sum distribution). Vekuri 2025 shows that calibration fitted on short-gap fluxes does not transfer to balances.
8. **Run time-series foundation models as zero-shot baselines only.** Chronos-2 accepts covariates and multivariate inputs at low cost. No flux evaluation exists, and generic forecasting backbones had negative R2 on CH4 in RACI. Their value is as a reference row in the benchmark, not as the core model.

## Sources

Verified primary sources (DOIs and arXiv IDs as given above). Main links:
- Vekuri et al. 2025: https://doi.org/10.1016/j.agrformet.2025.110558 ; full text https://zenodo.org/records/16872479 ; code https://github.com/hvekuri/nn_gapfilling_unc
- Hou et al. 2025: https://gmd.copernicus.org/articles/18/4625/2025/
- Qian L. et al. 2026: https://essd.copernicus.org/articles/18/6707/2026/
- Li W. et al. 2025: https://essd.copernicus.org/articles/17/3835/2025/
- Wang X. et al. 2025: https://doi.org/10.1038/s41597-025-06144-9 ; preprint https://essd.copernicus.org/preprints/essd-2024-291/
- De Cannière et al. 2026: https://doi.org/10.1016/j.agrformet.2025.110987
- Qian L. et al. 2024: https://doi.org/10.1016/j.jhydrol.2024.131787
- Abdaki et al. 2025: https://doi.org/10.1029/2024JG008375
- Zhu Q. et al. 2026: https://doi.org/10.1111/gcb.70899
- Huang et al. 2024: https://doi.org/10.1029/2023JD040418
- Tefera et al. 2026: https://doi.org/10.1016/j.agrformet.2026.111438
- Tikkasalo et al. 2025: https://bg.copernicus.org/articles/22/1277/2025/
- Buzacott et al. 2024: https://doi.org/10.1111/gcb.17590
- Ludwig et al. 2024: https://bg.copernicus.org/articles/21/1301/2024/
- Barczyk et al. 2024: https://doi.org/10.1016/j.agrformet.2024.110278
- Feigenwinter et al. 2023: https://doi.org/10.1016/j.scitotenv.2023.166389
- Premrov et al. 2025: https://doi.org/10.1016/j.envsoft.2025.106611
- Moutahir et al. 2026: https://bg.copernicus.org/articles/23/1719/2026/
- Nelson et al. 2024 (X-BASE): https://bg.copernicus.org/articles/21/5079/2024/
- Upton et al. 2026 (EC-STILT): https://acp.copernicus.org/articles/26/2561/2026/
- Ying et al. 2025 (WetCH4): https://essd.copernicus.org/articles/17/2507/2025/
- CarbonSense / EcoPerceiver: https://arxiv.org/abs/2406.04940
- CarbonFluxBench: https://arxiv.org/abs/2603.09868
- FLUXtrapolation: https://arxiv.org/abs/2605.19812
- AgroFlux: https://arxiv.org/abs/2602.01614
- X-MethaneWet: https://arxiv.org/abs/2505.18355
- CHAM-net: https://arxiv.org/abs/2606.00338
- RACI: https://arxiv.org/abs/2603.03531
- TAM-RL: https://arxiv.org/abs/2603.09974 ; https://arxiv.org/abs/2310.04727
- FAR: https://arxiv.org/abs/2512.01917
- ZeroDiff: https://arxiv.org/abs/2609.37078
- PERNN: https://arxiv.org/abs/2411.11497
- Park et al.: https://arxiv.org/abs/2202.12441
- Guided diffusion (CRPS scoring example): https://arxiv.org/abs/2610.12967
- KGML-ag-Carbon: https://doi.org/10.1038/s41467-023-43860-5
- Cohrs et al. 2024: https://doi.org/10.1088/2632-2153/ad5a60
- ElGhawi et al. 2023: https://doi.org/10.1088/1748-9326/acbbe0
- Kim et al. 2026 comment (missing-as-zero): https://doi.org/10.31223/x5b51m
- FLUXNET Data System: https://fluxnet.org/fluxnet-data-system/
