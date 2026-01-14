![DeepEarth logo](https://github.com/legel/deepearth/blob/main/docs/deepearth_logo.png)
## DeepEarth: AI for Planetary Science & Sustainability

DeepEarth is a [self-supervised](https://en.wikipedia.org/wiki/Self-supervised_learning), [multi-modal](https://en.wikipedia.org/wiki/Multimodal_learning), [spatio-temporal](https://www.sciencedirect.com/topics/social-sciences/spatio-temporal-model) GeoAI model for global environmental intelligence and optimization.

![DeepEarth v.0.01 preview of architecture](https://github.com/legel/deepearth/blob/main/docs/deepearth_main_figure.png)

DeepEarth learns by jointly reconstructing masked multi-modal datasets (as seen above). It uses a novel space-time positional encoder, [Earth4D](https://github.com/legel/deepearth/tree/main/encoders/xyzt/README.md), especially for [earth observation](https://en.wikipedia.org/wiki/Earth_observation) data (as seen below).

![Earth4D space-time encoder](https://github.com/legel/deepearth/blob/main/docs/earth4d_spacetime_encoder.png) 

## Exciting News:

- _January 14, 2026_  
  **New geospatial coordinate system.** A refined (_x_, _y_, _z_, _t_) = (_latitude_, _longitude_, _elevation_, _time_) coordinate system in [Earth4D](https://github.com/legel/deepearth/tree/main/encoders/xyzt) improved a state-of-the-art forecasting benchmark by 4%. See [_commit_](https://github.com/legel/deepearth/commit/4d21a32).

- _December 22, 2025_  
  **10x faster.** Following state-of-the-art [Earth4D](https://github.com/legel/deepearth/blob/main/encoders/xyzt/earth4d.py) experiments by [Brandon Voelker](https://www.egr.uh.edu/news/202410/space-ground-%E2%80%93-phd-student-voelker-leads-team-transforming-remote-sensing-based) on small batches, [Lance Legel](https://www.linkedin.com/in/legel/) sped up small batch processing by 10x. See [_commit_](https://github.com/legel/deepearth/commit/69f5be4e35c29df43c302bd3580b47d3911997e3) and [_CUDA code_](https://github.com/legel/deepearth/blob/main/encoders/xyzt/hashencoder/src/precompute.cu). 

- _December 19, 2025_  
  **Supercomputing award.** US DOE [National Energy Research Scientific Computing Center](https://www.nersc.gov) has awarded a DeepEarth team with supercomputing access in 2026 through [EESSD](https://science.osti.gov/ber/Research/eessd).
  
- _December 2, 2025_  
  **Peer-reviewed presentation in top venue.** Accepted to the [2026 World Modeling Workshop](https://world-model-mila.github.io/) at the [Mila Quebec AI Institute](https://mila.quebec/en), alongside keynote talks by [Yoshua Bengio](https://yoshuabengio.org/) and [Yann LeCun](http://yann.lecun.com/). See [_paper_](https://github.com/legel/deepearth/blob/main/docs/DeepEarth.pdf). 
  
- _November 17, 2025_  
  **99% parameter reduction, 4× speedup.** [Earth4D](https://github.com/legel/deepearth/tree/main/encoders/xyzt) with [learned hash probing](https://arxiv.org/abs/2312.17241) tested on an [ecological benchmark](https://www.nature.com/articles/s41597-024-03159-6) demonstrates spectacular accuracy with 5M parameters. See [_code_](https://github.com/legel/deepearth/blob/main/encoders/xyzt/lfmc_grid_search.py).

- _November 16, 2025_  
  **23% error reduction in space-time encoder.** [Lance Legel](https://www.linkedin.com/in/legel/) and [Qin Huang](https://news.asu.edu/b/20250512-asu-phd-student-tackles-climate-change-and-extreme-weather) implemented [learned hash probing](https://arxiv.org/abs/2312.17241) in [Earth4D](https://github.com/legel/deepearth/tree/main/encoders/xyzt), achieving state-of-the-art R² on an ecological forecasting benchmark. See [_commit_](https://github.com/legel/deepearth/commit/aa2a4b7).

- _October 29, 2025_  
  **Predicting risk of fires.**  [Qin Huang](https://news.asu.edu/b/20250512-asu-phd-student-tackles-climate-change-and-extreme-weather), [Brandon Voelker](https://www.egr.uh.edu/news/202410/space-ground-%E2%80%93-phd-student-voelker-leads-team-transforming-remote-sensing-based), and [Lance Legel](https://www.linkedin.com/in/legel/) presented on simulating [live fuel moisture content](https://www.nature.com/articles/s41597-024-03159-6) through NSF's [Institute for Geospatial Understanding](http://i-guide.io/). See [_event_](https://i-guide.io/i-guide-vco/geospatial-simulation-of-fire-ecology-with-deepearth/).

- _October 27, 2025_  
  **Battle-hardened (_x_, _y_, _z_, _t_) AI.**  For our spatio-temporal [multi-resolution hash encoding](https://nvlabs.github.io/instant-ngp/), we've [fixed a numerical bug in NVIDIA's CUDA kernels](https://github.com/legel/deepearth/pull/7) based on [profiling of hash collisions](https://github.com/legel/deepearth/blob/main/encoders/xyzt/hash_collision_profiler.py).

- _September 30, 2025_  
  **Presentation at top AI lab.** 
  Thanks to the [Allen Institute for AI](https://allenai.org) for hosting a 1 hour talk with scientists pioneering [AI foundation models for the planet](https://allenai.org/earth-system). See [_video_](  https://www.youtube.com/watch?v=SHJwCInICiA) and [_slides_](https://github.com/legel/deepearth/blob/main/docs/DeepEarth_AI2_Presentation.pdf).

- _August 8, 2025_  
  **NSF summer school program.** NSF funded a week-long ["Spatial AI for Disaster Resilience"](https://i-guide.io/summer-school/summer-school-2025/) summer school program in Boulder, Colorado. 5 PhD students researched and developed DeepEarth.  See [_demos_](https://github.com/legel/deepearth/blob/main/docs/DeepEarth🔥_NSF_I-GUIDE_Final_Presentation.pdf).

- _June 23, 2025_  
  **Workshop in Chicago.** NSF funded a 3 hour workshop on DeepEarth in Chicago for a ["GeoAI for Sustainability"](https://i-guide.io/forum/forum-2025/workshops/) conference. 3 professors, 5 postdocs, and 2 PhD students contributed.  See [_slides_](https://github.com/legel/deepearth/blob/main/docs/NSF_DeepEarth_Workshop.pdf).


#### Planetary Intelligence for Everyone
DeepEarth is an open source project for solving intelligence across the planet 🌎. We aspire to help solve major sustainability challenges including [climate resilience and biodiversity](https://www.asla.org/climateandbiodiversityactionplan.aspx).

#### Invitation for Open Source Collaboration
Collaborators welcomed! Contact [Lance Legel](https://linkedin.com/in/legel) at lance@ecodash.ai or submit an issue/PR here.

For further details, see papers:
- [Self-Supervised Multi-Modal World Model with 4D Space-Time Embedding](https://github.com/legel/deepearth/blob/main/docs/DeepEarth.pdf) (2026)
- [Inductive Neural Networks for Ecology](https://doi.org/10.13140/RG.2.2.25523.90406) (2025)
- [AI Foundation Models for Biogeography and Ecophysiology](https://doi.org/10.13140/RG.2.2.12102.13123) (2024)
