# References

Every method, dataset and code base FluxFill uses or compares against. Each component's README gives the exact
version, commit, access date and checksum of what it used.

## Eddy-covariance data and processing

- Pastorello, G. et al. (2020). The FLUXNET2015 dataset and the ONEFlux processing pipeline for eddy covariance data.
  *Scientific Data* 7, 225. doi:10.1038/s41597-020-0534-3
- Delwiche, K. B. et al. (2021). FLUXNET-CH4: a global, multi-ecosystem dataset and analysis of methane seasonality
  from freshwater wetlands. *Earth System Science Data* 13, 3607-3689. doi:10.5194/essd-13-3607-2021
- Knox, S. H. et al. (2019). FLUXNET-CH4 synthesis activity: objectives, observations, and future directions.
  *Bulletin of the American Meteorological Society* 100, 2607-2632. doi:10.1175/BAMS-D-18-0268.1
- Papale, D. et al. (2006). Towards a standardized processing of net ecosystem exchange measured with eddy
  covariance technique: algorithms and uncertainty estimation. *Biogeosciences* 3, 571-583. doi:10.5194/bg-3-571-2006
- Reichstein, M. et al. (2005). On the separation of net ecosystem exchange into assimilation and ecosystem
  respiration: review and improved algorithm. *Global Change Biology* 11, 1424-1439.
  doi:10.1111/j.1365-2486.2005.001002.x
- Wutzler, T. et al. (2018). Basic and extensible post-processing of eddy covariance flux data with REddyProc.
  *Biogeosciences* 15, 5015-5030. doi:10.5194/bg-15-5015-2018
- Vuichard, N. and Papale, D. (2015). Filling the gaps in meteorological continuous data measured at FLUXNET sites
  with ERA-Interim reanalysis. *Earth System Science Data* 7, 157-171. doi:10.5194/essd-7-157-2015
- Foken, T., Gockede, M., Mauder, M., Mahrt, L., Amiro, B. and Munger, W. (2004). Post-field data quality control.
  In *Handbook of Micrometeorology*, Kluwer, 181-208. doi:10.1007/1-4020-2265-4_9
- Beck, H. E., McVicar, T. R., Vergopolan, N. et al. (2023). High-resolution (1 km) Koppen-Geiger maps for 1901-2099
  based on constrained CMIP6 projections. *Scientific Data* 10, 724. doi:10.1038/s41597-023-02549-6
- Network data policies and site citations: AmeriFlux, ICOS, TERN, JapanFlux and FLUXNET. These are listed per site in
  `data/towers/sites.csv` (columns `shuttle_citation`, `shuttle_pid`, `acknowledgement`).

## Gap-filling methods (baselines and literature)

- Irvin, J. et al. (2021). Gap-filling eddy covariance methane fluxes: comparison of machine learning model
  predictions and uncertainties at FLUXNET-CH4 wetlands. *Agricultural and Forest Meteorology* 308-309, 108528.
  doi:10.1016/j.agrformet.2021.108528. Code: github.com/stanfordmlgroup/methane-gapfill-ml (Apache-2.0).
- Liu, Y. et al. (2025). Robust filling of extra-long gaps in eddy covariance CO2 flux measurements from a temperate
  deciduous forest using eXtreme Gradient Boosting. *Agricultural and Forest Meteorology* 364, 110438.
  doi:10.1016/j.agrformet.2025.110438. Code: github.com/YujieLiu666/fluxgapfill_XGB_vs_MDS.
- Vekuri, H. et al. (2025). Improved uncertainty estimates for eddy covariance-based carbon dioxide balances using
  deep ensembles for gap-filling. *Agricultural and Forest Meteorology* 371, 110558.
  doi:10.1016/j.agrformet.2025.110558. Code: github.com/hvekuri/nn_gapfilling_unc.
- Vekuri, H. et al. (2023). A widely-used eddy covariance gap-filling method creates systematic bias in carbon balance
  estimates. *Scientific Reports* 13, 1720. doi:10.1038/s41598-023-28827-2
- Gao, D. et al. (2023). Eddy covariance CO2 flux gap filling for long data gaps: a novel framework based on machine
  learning and time series decomposition. *Remote Sensing* 15, 2695. doi:10.3390/rs15102695
- Li, T. et al. (2026). Filtering and gap-filling strategies in eddy covariance CO2 flux series of mountainous
  forests. *Science China Earth Sciences* 69, 125-138. doi:10.1007/s11430-024-1723-y
- Zhu, S. et al. (2022). Stable gap-filling for longer eddy covariance data gaps: a globally validated
  machine-learning approach for carbon dioxide, water, and energy fluxes. *Agricultural and Forest Meteorology* 314,
  108777. doi:10.1016/j.agrformet.2021.108777
- Moffat, A. M. et al. (2007). Comprehensive comparison of gap-filling techniques for eddy covariance net carbon
  fluxes. *Agricultural and Forest Meteorology* 147, 209-232. doi:10.1016/j.agrformet.2007.08.011
- Richardson, A. D. and Hollinger, D. Y. (2007). A method to estimate the additional uncertainty in gap-filled NEE
  resulting from long gaps in the CO2 flux record. *Agricultural and Forest Meteorology* 147, 199-208.
  doi:10.1016/j.agrformet.2007.06.004
- The 2023 to 2026 literature reviewed for this work, with its verification status: [LITERATURE.md](LITERATURE.md).

## Imputation and generative modelling

- Du, W., Cote, D. and Liu, Y. (2023). SAITS: self-attention-based imputation for time series. *Expert Systems with
  Applications* 219, 119619. doi:10.1016/j.eswa.2023.119619. PyPOTS: Du, W. (2023), arXiv:2305.18811.
- Tashiro, Y. et al. (2021). CSDI: conditional score-based diffusion models for probabilistic time series
  imputation. *NeurIPS 2021*. arXiv:2107.03502
- Ansari, A. F. et al. (2025). Chronos-2: from univariate to universal forecasting. arXiv:2510.15821
- Li, T., Tian, Y., Li, H., Deng, M. and He, K. (2024). Autoregressive image generation without vector quantization.
  arXiv:2406.11838
- Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS 2017*. arXiv:1706.03762

## Model components

- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M. and Le, M. (2023). Flow matching for generative modeling.
  *ICLR 2023*. arXiv:2210.02747
- Chang, H., Zhang, H., Jiang, L., Liu, C. and Freeman, W. T. (2022). MaskGIT: masked generative image transformer.
  *CVPR 2022*. arXiv:2202.04200
- Su, J. et al. (2021). RoFormer: enhanced transformer with rotary position embedding. arXiv:2104.09864
- Ho, J., Kalchbrenner, N., Weissenborn, D. and Salimans, T. (2019). Axial attention in multidimensional transformers.
  arXiv:1912.12180
- Peebles, W. and Xie, S. (2023). Scalable diffusion models with transformers. *ICCV 2023*. arXiv:2212.09748
- Kendall, A. and Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision?
  *NeurIPS 2017*. arXiv:1703.04977
- Bishop, C. M. (1994). Mixture density networks. Technical report NCRG/94/004, Aston University.

## Scoring

- Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the
  American Statistical Association* 102, 359-378. doi:10.1198/016214506000001437
- Ferro, C. A. T. (2014). Fair scores for ensemble forecasts. *Quarterly Journal of the Royal Meteorological Society*
  140, 1917-1923. doi:10.1002/qj.2270
- Jordan, A., Krueger, F. and Lerch, S. (2019). Evaluating probabilistic forecasts with scoringRules. *Journal of
  Statistical Software* 90(12). doi:10.18637/jss.v090.i12
- Gneiting, T., Balabdaoui, F. and Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness.
  *Journal of the Royal Statistical Society B* 69, 243-268. doi:10.1111/j.1467-9868.2007.00587.x

## Imagery and embeddings

- Drusch, M. et al. (2012). Sentinel-2: ESA's optical high-resolution mission for GMES operational services. *Remote
  Sensing of Environment* 120, 25-36. doi:10.1016/j.rse.2011.11.026. Data: Copernicus Sentinel data, Collection 1
  L2A via Element84 Earth Search, and L2A via Microsoft Planetary Computer.
- Main-Knorn, M., Pflug, B., Louis, J., Debaecker, V., Müller-Wilm, U. and Gascon, F. (2017). Sen2Cor for
  Sentinel-2. *Proc. SPIE* 10427, 1042704. doi:10.1117/12.2278218
- Xiong, Z., Wang, Y., Zhang, F. et al. (2024). Neural plasticity-inspired multimodal foundation model for Earth
  observation (DOFA). arXiv:2403.15356
- Clay Foundation (2024-2025). Clay: an open-source foundation model for Earth observation, v1.5.
  github.com/Clay-foundation/model; huggingface.co/made-with-clay/Clay. The checkpoint hash is in
  `embeddings/clay/README.md`.

## Drivers

- Muñoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications.
  *Earth System Science Data* 13, 4349-4383. doi:10.5194/essd-13-4349-2021
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*
  146, 1999-2049. doi:10.1002/qj.3803
- Dowell, D. C. et al. (2022). The High-Resolution Rapid Refresh (HRRR): an hourly updating convection-allowing
  forecast model. *Weather and Forecasting* 37, 1371-1395. doi:10.1175/WAF-D-21-0151.1
- European Space Agency (2021). Copernicus DEM GLO-30. doi:10.5270/ESA-c5d3d65
- Huffman, G. J. et al. (2023). GPM IMERG Late Precipitation L3 V07. NASA GES DISC. doi:10.5067/GPM/IMERG/3B-HH-L/07
- Natural Earth (public domain), 1:110m land, v5.1.2, for the dashboard map.

## Context of the report

- Baldocchi, D. D. (2003). Assessing the eddy covariance technique for evaluating carbon dioxide exchange rates of
  ecosystems: past, present and future. *Global Change Biology* 9, 479-492. doi:10.1046/j.1365-2486.2003.00629.x
- Falge, E. et al. (2001). Gap filling strategies for defensible annual sums of net ecosystem exchange. *Agricultural
  and Forest Meteorology* 107, 43-69. doi:10.1016/S0168-1923(00)00225-2
- Kim, Y. et al. (2020). Gap-filling approaches for eddy covariance methane fluxes: a comparison of three machine
  learning algorithms and a traditional method with principal component analysis. *Global Change Biology* 26,
  1499-1518. doi:10.1111/gcb.14845
- Jung, M. et al. (2020). Scaling carbon fluxes from eddy covariance sites to globe: synthesis and evaluation of the
  FLUXCOM approach. *Biogeosciences* 17, 1343-1365. doi:10.5194/bg-17-1343-2020
- Saunois, M. et al. (2020). The Global Methane Budget 2000-2017. *Earth System Science Data* 12, 1561-1623.
  doi:10.5194/essd-12-1561-2020
- He, K. et al. (2022). Masked autoencoders are scalable vision learners. *CVPR*. arXiv:2111.06377
- Schefzik, R., Thorarinsdottir, T. L. and Gneiting, T. (2013). Uncertainty quantification in complex simulation
  models using ensemble copula coupling. *Statistical Science* 28, 616-640. doi:10.1214/13-STS443
- Clark, M. et al. (2004). The Schaake shuffle: a method for reconstructing space-time variability in forecasted
  precipitation and temperature fields. *Journal of Hydrometeorology* 5, 243-262.
  doi:10.1175/1525-7541(2004)005<0243:TSSAMF>2.0.CO;2
- Kingma, D. P. and Ba, J. (2015). Adam: a method for stochastic optimization. *ICLR*. arXiv:1412.6980
- Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*. arXiv:1711.05101
- Cole, E. et al. (2023). Spatial implicit neural representations for global-scale species mapping. *ICML*.
  arXiv:2306.02564
