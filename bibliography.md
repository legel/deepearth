# DeepEarth / DeepCal — Bibliography

Formal scientific provenance for every phylogeny, dataset, foundation model, and method DeepCal depends on. Each
entry: authors (year), title, venue, DOI/identifier, and the role it plays here. Entries marked **[in use]** are
integrated in the current champion path; **[staged]** are downloaded/planned for integration; **[reference]** informs
the method. Cross-references to design rules live in `autoresearch/science.md`.

---

## 1. Phylogenies

### 1.1 Plants (the DeepCal species graph)
- **[P1] [in use]** California seed-plant dated phylogeny — `ca_subtree.dated.nwk`, restricted to the 2,141 modeled
  species; source of the `E1` evolutionary vectors and `patristic_ref.npy`. **Provenance to confirm/document:** the
  in-repo artifact carries no citation; it is consistent with a pruning of a published dated seed-plant megatree —
  most likely **Smith & Brown (2018)** "Constructing a broadly inclusive seed plant phylogeny," *Am. J. Bot.*
  105(3):302–314, DOI [10.1002/ajb2.1019](https://doi.org/10.1002/ajb2.1019) (GBOTB/ALLMB), or **Jin & Qian (2019)**
  "V.PhyloMaker," *Ecography* 42:1353–1359, DOI [10.1111/ecog.04434](https://doi.org/10.1111/ecog.04434). *Verify
  before formal publication.*

### 1.2 Pollinators — dense species-level trees (grafted for resolution)
- **[T1] [staged]** Bees — Henríquez-Piskulich, Hugall & Stuart-Fox (2024). "A supermatrix phylogeny of the world's
  bees (Hymenoptera: Anthophila)." *Mol. Phylogenet. Evol.* 190:107963. DOI
  [10.1016/j.ympev.2023.107963](https://doi.org/10.1016/j.ympev.2023.107963). 4,586 spp, dated (treePL). Dryad
  [10.5061/dryad.80gb5mkw1](https://doi.org/10.5061/dryad.80gb5mkw1), CC0. File `BEE_mat7_fulltree_tplo35_sf20lp.nwk`.
- **[T2] [staged]** Butterflies — Kawahara et al. (2023). "A global phylogeny of butterflies reveals their evolutionary
  history, ancestral hosts and biogeographic origins." *Nat. Ecol. Evol.* 7:903–913. DOI
  [10.1038/s41559-023-02041-9](https://doi.org/10.1038/s41559-023-02041-9). 2,244 spp, dated. Figshare
  [10.6084/m9.figshare.21774899](https://doi.org/10.6084/m9.figshare.21774899), CC BY 4.0.
- **[T3] [staged]** Butterflies (dated genus Newick) — Chazot et al. (2019). "Priors and Posteriors in Bayesian Timing
  of Divergence Analyses: The Age of Butterflies Revisited." *Syst. Biol.* 68(5):797–813. DOI
  [10.1093/sysbio/syz002](https://doi.org/10.1093/sysbio/syz002). Dryad [10.5061/dryad.fb88292](https://doi.org/10.5061/dryad.fb88292), CC0.
- **[T4] [staged]** Birds/hummingbirds — Jetz, Thomas, Joy, Hartmann & Mooers (2012). "The global diversity of birds in
  space and time." *Nature* 491:444–448. DOI [10.1038/nature11631](https://doi.org/10.1038/nature11631). All 9,993
  spp, dated; birdtree.org / vertlife.org.
- **[T5] [staged]** Ants (Formicidae) — Nelsen, Ree & Moreau (2018). "Ant–plant interactions evolved through increasing
  interdependence." *PNAS* 115(48):12253–12258. DOI [10.1073/pnas.1719794115](https://doi.org/10.1073/pnas.1719794115).
  1,731 spp, dated Newick. Dryad [10.5061/dryad.ft4sn88](https://doi.org/10.5061/dryad.ft4sn88).
- **[T6] [staged]** Hawkmoths (Sphingidae) — Couch et al. (2026). "The evolution of proboscis length and feeding
  behaviour in hawkmoths." *R. Soc. Open Sci.* DOI [10.1098/rsos.251697](https://doi.org/10.1098/rsos.251697). 310 spp,
  dated (MCMCtree). Dryad [10.5061/dryad.m37pvmdg3](https://doi.org/10.5061/dryad.m37pvmdg3), CC0.

### 1.3 Pollinators — dated order/clade backbones (topology + calibration)
- **[T7] [staged]** Coleoptera — Zhang et al. (2018). "Evolutionary history of Coleoptera revealed by extensive
  sampling of genes and species." *Nat. Commun.* 9:205. DOI
  [10.1038/s41467-017-02644-4](https://doi.org/10.1038/s41467-017-02644-4). 373 spp/124 fam, dated. Figshare
  [10.6084/m9.figshare.5306497](https://doi.org/10.6084/m9.figshare.5306497), CC BY 4.0.
- **[T8] [reference]** Coleoptera — McKenna et al. (2019). "The evolution and genomic basis of beetle diversity." *PNAS*
  116(49):24729–24737. DOI [10.1073/pnas.1909655116](https://doi.org/10.1073/pnas.1909655116). Zenodo
  [10.5281/zenodo.3522944](https://doi.org/10.5281/zenodo.3522944), CC BY 4.0.
- **[T9] [reference]** Coleoptera — Cai et al. (2022). "Integrated phylogenomics and fossil data illuminate the
  evolution of beetles." *R. Soc. Open Sci.* 9:211771. DOI [10.1098/rsos.211771](https://doi.org/10.1098/rsos.211771).
- **[T10] [staged]** Hymenoptera (whole order, dated) — Peters et al. (2017). "Evolutionary History of the Hymenoptera."
  *Curr. Biol.* 27(7):1013–1018. DOI [10.1016/j.cub.2017.01.027](https://doi.org/10.1016/j.cub.2017.01.027). Mendeley
  Data [10.17632/trbj94zm2n.3](https://doi.org/10.17632/trbj94zm2n.3), CC BY 4.0.
- **[T11] [staged]** Aculeata (bees+wasps+ants, dated) — Branstetter et al. (2017). "Phylogenomic Insights into the
  Evolution of Stinging Wasps and the Origins of Ants and Bees." *Curr. Biol.* 27(7):1019–1025. DOI
  [10.1016/j.cub.2017.03.027](https://doi.org/10.1016/j.cub.2017.03.027). 30/31 families. Dryad
  [10.5061/dryad.r8d4q](https://doi.org/10.5061/dryad.r8d4q), CC0.
- **[T12] [reference]** Hymenoptera — Blaimer et al. (2023). "Key innovations and the diversification of Hymenoptera."
  *Nat. Commun.* 14:1212. DOI [10.1038/s41467-023-36868-4](https://doi.org/10.1038/s41467-023-36868-4). Dryad
  [10.5061/dryad.08kprr54m](https://doi.org/10.5061/dryad.08kprr54m), CC0.
- **[T13] [reference]** Ants (genus backbone) — Borowiec et al. (2025). "Evaluating UCE Data Adequacy and Integrating
  Uncertainty in a Comprehensive Phylogeny of Ants." *Syst. Biol.* DOI
  [10.1093/sysbio/syaf001](https://doi.org/10.1093/sysbio/syaf001). 277/343 genera. Dryad
  [10.5061/dryad.547d7wmhb](https://doi.org/10.5061/dryad.547d7wmhb).
- **[T14] [reference]** Vespidae — Piekarski et al. (2018). "Phylogenomic Evidence Overturns Current Conceptions of
  Social Evolution in Wasps." *Mol. Biol. Evol.* 35(9):2097–2109. DOI
  [10.1093/molbev/msy124](https://doi.org/10.1093/molbev/msy124). Figshare 10.6084/m9.figshare.c.4135511.
- **[T15] [reference]** Apoid wasps (Crabronidae/Sphecidae) — Sann et al. (2018). "Phylogenomic analysis of Apoidea…"
  *BMC Evol. Biol.* 18:71. DOI [10.1186/s12862-018-1155-8](https://doi.org/10.1186/s12862-018-1155-8). Trees: Sann et
  al. (2021), Dryad [10.5061/dryad.pc866t1nj](https://doi.org/10.5061/dryad.pc866t1nj).
- **[T16] [reference]** Spider wasps (Pompilidae) — Waichert et al. (2015). "Molecular phylogeny and systematics of
  spider wasps." *Zool. J. Linn. Soc.* 175:271–287. DOI [10.1111/zoj.12212](https://doi.org/10.1111/zoj.12212).
- **[T17] [staged]** Lepidoptera (order backbone, dated) — Kawahara et al. (2019). "Phylogenomics reveals the
  evolutionary timing and pattern of butterflies and moths." *PNAS* 116(45):22657–22663. DOI
  [10.1073/pnas.1907847116](https://doi.org/10.1073/pnas.1907847116). Dryad
  [10.5061/dryad.j477b40](https://doi.org/10.5061/dryad.j477b40), CC0.
- **[T18] [reference]** Owlet moths (Noctuoidea) — Li et al. (2024). *Cladistics.* DOI
  [10.1111/cla.12559](https://doi.org/10.1111/cla.12559). Dryad [10.5061/dryad.1c59zw411](https://doi.org/10.5061/dryad.1c59zw411).
- **[T19] [reference]** Geometridae — Murillo-Ramos et al. (2019). "…phylogeny of geometrid moths." *PeerJ* 7:e7386.
  DOI [10.7717/peerj.7386](https://doi.org/10.7717/peerj.7386). 1,192 taxa (undated topology). CC BY.
- **[T20] [staged]** Diptera (fly tree of life, dated backbone) — Wiegmann et al. (2011). "Episodic radiations in the
  fly tree of life." *PNAS* 108(14):5690–5695. DOI [10.1073/pnas.1012675108](https://doi.org/10.1073/pnas.1012675108).
  Dryad [10.5061/dryad.21nf3](https://doi.org/10.5061/dryad.21nf3).
- **[T21] [reference]** Hoverflies (Syrphidae) — Wong et al. (2023), *Mol. Phylogenet. Evol.* 184:107759, DOI
  [10.1016/j.ympev.2023.107759](https://doi.org/10.1016/j.ympev.2023.107759); Young et al. (2016), *BMC Evol. Biol.*
  16:143, DOI [10.1186/s12862-016-0714-0](https://doi.org/10.1186/s12862-016-0714-0); Liao et al. (2026), *Biology*
  15(5):411 (open, dated genus backbone), DOI [10.3390/biology15050411](https://doi.org/10.3390/biology15050411).
- **[T22] [reference]** Bee flies (Bombyliidae) — Li et al. (2021). *Cladistics* 37:276–297. DOI
  [10.1111/cla.12436](https://doi.org/10.1111/cla.12436).
- **[T23] [reference]** Tachinidae — Stireman et al. (2019). "Molecular phylogeny and evolution of world Tachinidae."
  *Mol. Phylogenet. Evol.* 139:106358. DOI [10.1016/j.ympev.2019.106358](https://doi.org/10.1016/j.ympev.2019.106358).
  504 terminals / 359 genera.
- **[T24] [reference]** Muscidae — Li et al. (2023). *Insects* 14(3):286 (open, dated). DOI
  [10.3390/insects14030286](https://doi.org/10.3390/insects14030286).

### 1.4 Phylogenetic synthesis & dating tools
- **[T25] [staged]** Open Tree of Life — Hinchliff et al. (2015). "Synthesis of phylogeny and taxonomy into a
  comprehensive tree of life." *PNAS* 112(41):12764–12769. DOI
  [10.1073/pnas.1423041112](https://doi.org/10.1073/pnas.1423041112). `induced_subtree` / `tnrs` APIs.
- **[T26] [staged]** `rotl` (R interface to Open Tree) — Michonneau, Brown & Winter (2016). *Methods Ecol. Evol.*
  7:1476–1481. DOI [10.1111/2041-210X.12593](https://doi.org/10.1111/2041-210X.12593).
- **[T27] [staged]** `datelife` (chronogram synthesis / congruification) — Sánchez-Reyes & O'Meara (2024). *Syst. Biol.*
  73(2):470–479. DOI [10.1093/sysbio/syad044](https://doi.org/10.1093/sysbio/syad044). datelife.org.
- **[T28] [reference]** TimeTree — Kumar et al. (2022). "TimeTree 5." *Mol. Biol. Evol.* 39(8):msac174. DOI
  [10.1093/molbev/msac174](https://doi.org/10.1093/molbev/msac174).
- **[T29] [reference]** Ornstein–Uhlenbeck models on trees (basis of the OU-attention operator) — Butler & King (2004).
  "Phylogenetic Comparative Analysis: A Modeling Approach for Adaptive Evolution." *Am. Nat.* 164(6):683–695. DOI
  [10.1086/426002](https://doi.org/10.1086/426002).

---

## 2. Foundation models & encoders
- **[M1] [in use]** BioCLIP 2.5 (ViT-H/14, TreeOfLife-200M+) — `imageomics/bioclip-2.5-vith14`; text prior for the
  species-graph seed (rule 26). Part of the BioCLIP 2 line ([M3]).
- **[M2] [in use]** BioCLIP — Stevens et al. (2024). "BioCLIP: A Vision Foundation Model for the Tree of Life." *CVPR*.
  arXiv:[2311.18803](https://arxiv.org/abs/2311.18803). `imageomics/bioclip-2` supplies the 768-d image "bio" tokens.
- **[M3] [in use]** BioCLIP 2 — Gu et al. (2025). "BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive
  Learning." *NeurIPS* (Spotlight). arXiv:[2505.23883](https://arxiv.org/abs/2505.23883).
- **[M4] [in use]** DINOv3 — Siméoni et al. (2025). "DINOv3." arXiv:[2508.10104](https://arxiv.org/abs/2508.10104).
  `facebook/dinov3-vitl16-pretrain-lvd1689m` (ground vision, 1024-d) and `-sat493m` (NAIP aerial, RGB+IR).
- **[M5] [in use]** Clay v1.5 — Clay Foundation Model (open geospatial FM). <https://github.com/Clay-foundation/model>.
  Sentinel-2 embeddings.
- **[M6] [in use]** OpenCLIP (loader for BioCLIP checkpoints) — Ilharco et al. (2021). DOI
  [10.5281/zenodo.5143773](https://doi.org/10.5281/zenodo.5143773).

---

## 3. Datasets & data products
- **[D1] [in use]** GBIF occurrence download — GBIF.org. **iNaturalist Research-grade Observations** (datasetKey
  `50c9509d-22c7-4a22-a47d-8c48425ef4a7`). California vascular plants, 2025, ≤10 m coordinate uncertainty, with images.
  *Record the exact GBIF download DOI at acquisition.* iNaturalist contributors; GBIF Secretariat.
- **[D2] [in use]** iNaturalist — community observations & imagery. <https://www.inaturalist.org>.
- **[D3] [in use]** Global Biotic Interactions (GloBI) — Poelen, Simons & Mungall (2014). "Global Biotic Interactions:
  An open infrastructure to share and analyze species-interaction datasets." *Ecol. Inform.* 24:148–159. DOI
  [10.1016/j.ecoinf.2014.08.005](https://doi.org/10.1016/j.ecoinf.2014.08.005). Plant–pollinator interactions.
- **[D4] [in use]** Daymet V4 daily climate — Thornton et al. (2022). ORNL DAAC. DOI
  [10.3334/ORNLDAAC/2129](https://doi.org/10.3334/ORNLDAAC/2129). 180-day × 7-var climate window.
- **[D5] [in use]** SSURGO — USDA-NRCS Soil Survey Geographic Database. <https://sdmdataaccess.nrcs.usda.gov>.
- **[D6] [in use]** NAIP — USDA-FSA National Agriculture Imagery Program (aerial RGB+IR).
- **[D7] [in use]** USGS 3DEP — U.S. Geological Survey 3D Elevation Program (1 m DEM; topo + hydro).
  <https://www.usgs.gov/3d-elevation-program>.
- **[D8] [in use]** NAIP-CHM canopy height — Tolan et al. (2024). "Very high resolution canopy height maps from RGB
  imagery." *Remote Sens. Environ.* 300:113888. DOI
  [10.1016/j.rse.2023.113888](https://doi.org/10.1016/j.rse.2023.113888). Host: rangeland.ntsg.umt.edu / Meta-WRI.
- **[D9] [in use]** Copernicus Sentinel-2 — ESA (via Clay [M5]).
- **[D10] [staged]** gridMET — Abatzoglou (2013). "Development of gridded surface meteorological data…" *Int. J.
  Climatol.* 33:121–131. DOI [10.1002/joc.3413](https://doi.org/10.1002/joc.3413). Wind/humidity.
- **[D11] [staged]** HydroSHEDS — Lehner, Verdin & Jarvis (2008). *Eos* 89(10):93–94. DOI
  [10.1029/2008EO100001](https://doi.org/10.1029/2008EO100001). (Our hydro tokens use 3DEP-derived drainage.)
- **[D12] [staged]** PhenoVision / Phenobase — flowering/phenology probe. <https://www.phenobase.org>;
  model `phenobase/phenovision`.

---

## 4. Methods & architectures
- **[A1] [in use]** MADE — Germain, Gregor, Murray & Larochelle (2015). "MADE: Masked Autoencoder for Distribution
  Estimation." *ICML*, PMLR 37:881–889. arXiv:[1502.03509](https://arxiv.org/abs/1502.03509).
- **[A2] [in use]** Perceiver IO — Jaegle et al. (2022). *ICLR*. arXiv:[2107.14795](https://arxiv.org/abs/2107.14795).
- **[A3] [in use]** Instant-NGP multiresolution hash encoding (Earth4D basis) — Müller et al. (2022). *ACM Trans.
  Graph.* 41(4):102. DOI [10.1145/3528223.3530127](https://doi.org/10.1145/3528223.3530127).
- **[A4] [in use]** JEPA (I-JEPA) — Assran et al. (2023). *CVPR*. DOI
  [10.1109/CVPR52729.2023.01499](https://doi.org/10.1109/CVPR52729.2023.01499). arXiv:2301.08243.
- **[A5] [in use]** LP-FT (frozen-encoder + probe; why not to fine-tune BioCLIP) — Kumar et al. (2022). "Fine-Tuning can
  Distort Pretrained Features…" *ICLR*. arXiv:[2202.10054](https://arxiv.org/abs/2202.10054).
- **[A6] [reference]** GraphCast — Lam et al. (2023). *Science* 382:1416–1421. DOI
  [10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336).
- **[A7] [reference]** GenCast — Price et al. (2025). *Nature* 637:84–90. DOI
  [10.1038/s41586-024-08252-9](https://doi.org/10.1038/s41586-024-08252-9).
- **[A8] [reference]** Multimodal Deep Boltzmann Machines — Srivastava & Salakhutdinov (2014). *JMLR* 15:2949–2980.

---

*Maintenance: when a dataset/model/tree is integrated, flip its tag to **[in use]** and record the exact acquisition
identifier (GBIF download DOI, Dryad file, HF revision). Confirm **[P1]** (plant tree) provenance before publication.*
