**DeepEarth** is a [self-supervised](https://en.wikipedia.org/wiki/Self-supervised_learning), [multi-modal](https://en.wikipedia.org/wiki/Multimodal_learning), [spatio-temporal](https://www.sciencedirect.com/topics/social-sciences/spatio-temporal-model) AI world model for the planet.

#### News:
- _October 31, 2026_  
  **State-of-the-art climate simulation.** DeepEarth simulates sun, water, wind, temperature, and carbon across global flux towers more accurately than all existing models. See [_report_](https://deepearth.ai/fluxfill/#report), [_code_](https://github.com/legel/deepearth/tree/main/models/flux/fill).

- _October 5, 2026_  
  **#1 model of native plant habitats.** DeepEarth has exceeded all published models for predicting distributions of all native plant species across the United States. See [_report_](https://deepearth.ai/native-habitats/), [_code_](https://github.com/legel/deepearth/tree/main/models/habitat/plant)

- _July 16, 2026_  
  **NASA award.** [NASA's Commercial Satellite Data Acquisition](https://science.nasa.gov/earth-science/csda/) has awarded DeepEarth investigators at UC Berkeley 5 million km**2 of [Planet](https://www.planet.com/) data, approximately $200,000 in value.

- _March 7, 2026_  
  **Paper on arXiv.** [_"Self-Supervised Multi-Modal World Model with 4D Space-Time Embedding"_](https://arxiv.org/pdf/2603.07039), following peer-review through the [2026 World Modeling Workshop](https://world-model-mila.github.io/), is now on arXiv. See [_paper_](https://arxiv.org/abs/2603.07039).

- _January 28, 2026_  
  **Poster at World Modeling Workshop.** [Lance Legel](https://www.linkedin.com/in/legel/) and [Qin Huang](https://news.asu.edu/b/20250512-asu-phd-student-tackles-climate-change-and-extreme-weather) will present DeepEarth at the [2026 World Modeling Workshop](https://world-model-mila.github.io/). See [_poster_](docs/science/world_modeling_workshop_2026/poster/DeepEarth_2026_World_Modeling_Workshop_Poster.pdf).

- _January 14, 2026_  
  **New geospatial coordinate system.** A refined (_x_, _y_, _z_, _t_) = (_latitude_, _longitude_, _elevation_, _time_) coordinate system in [Earth4D](encoders/spacetime) improved a state-of-the-art forecasting benchmark by 4%. See [_commit_](https://github.com/legel/deepearth/commit/4d21a32).

- _December 22, 2025_  
  **10x faster.** Following state-of-the-art [Earth4D](encoders/spacetime/earth4d.py) experiments by [Brandon Voelker](https://www.egr.uh.edu/news/202410/space-ground-%E2%80%93-phd-student-voelker-leads-team-transforming-remote-sensing-based) on small batches, [Lance Legel](https://www.linkedin.com/in/legel/) sped up small batch processing by 10x. See [_commit_](https://github.com/legel/deepearth/commit/69f5be4e35c29df43c302bd3580b47d3911997e3). 

- _December 19, 2025_  
  **Supercomputing award.** US DOE [National Energy Research Scientific Computing Center](https://www.nersc.gov) has awarded a DeepEarth team with supercomputing access in 2026 through [BER](https://science.osti.gov/ber).
  
- _December 2, 2025_  
  **Peer-reviewed presentation in top venue.** Accepted to the [2026 World Modeling Workshop](https://world-model-mila.github.io/) at the [Mila Quebec AI Institute](https://mila.quebec/en), alongside keynote talks by [Yoshua Bengio](https://yoshuabengio.org/) and [Yann LeCun](http://yann.lecun.com/). See [_paper_](docs/deepearth.pdf). 
  
- _November 17, 2025_  
  **99% parameter reduction, 4× speedup.** [Earth4D](encoders/spacetime) with [learned hash probing](https://arxiv.org/abs/2312.17241) tested on an [ecological benchmark](https://www.nature.com/articles/s41597-024-03159-6) demonstrates spectacular accuracy with 5M parameters. 

- _November 16, 2025_  
  **23% error reduction in space-time encoder.** [Lance Legel](https://www.linkedin.com/in/legel/) and [Qin Huang](https://news.asu.edu/b/20250512-asu-phd-student-tackles-climate-change-and-extreme-weather) implemented [learned hash probing](https://arxiv.org/abs/2312.17241) in [Earth4D](encoders/spacetime), achieving state-of-the-art R² on an ecological forecasting benchmark. See [_commit_](https://github.com/legel/deepearth/commit/aa2a4b7).

- _October 29, 2025_  
  **Predicting risk of fires.**  [Qin Huang](https://news.asu.edu/b/20250512-asu-phd-student-tackles-climate-change-and-extreme-weather), [Brandon Voelker](https://www.egr.uh.edu/news/202410/space-ground-%E2%80%93-phd-student-voelker-leads-team-transforming-remote-sensing-based), and [Lance Legel](https://www.linkedin.com/in/legel/) presented on simulating [live fuel moisture content](https://www.nature.com/articles/s41597-024-03159-6) through NSF's [Institute for Geospatial Understanding](http://i-guide.io/). See [_event_](https://i-guide.io/i-guide-vco/geospatial-simulation-of-fire-ecology-with-deepearth/).

- _October 27, 2025_  
  **Battle-hardened (_x_, _y_, _z_, _t_) AI.**  For our spatio-temporal [multi-resolution hash encoding](https://nvlabs.github.io/instant-ngp/), we've [fixed a numerical bug in NVIDIA's CUDA kernels](https://github.com/legel/deepearth/pull/7) based on profiling of hash collisions.

- _September 30, 2025_  
  **Presentation at top AI lab.** 
  Thanks to the [Allen Institute for AI](https://allenai.org) for hosting a 1 hour talk with scientists pioneering [AI foundation models for the planet](https://allenai.org/earth-system). See [_video_](  https://www.youtube.com/watch?v=SHJwCInICiA).

- _August 8, 2025_  
  **NSF summer school program.** NSF funded a week-long ["Spatial AI for Disaster Resilience"](https://i-guide.io/summer-school/summer-school-2025/) summer school program in Boulder, Colorado. 5 PhD students researched and developed DeepEarth.

- _June 23, 2025_  
  **Workshop in Chicago.** NSF funded a 3 hour workshop on DeepEarth in Chicago for a ["GeoAI for Sustainability"](https://i-guide.io/forum/forum-2025/workshops/) conference. 3 professors, 5 postdocs, and 2 PhD students contributed.

#### Changing the World
[DeepEarth.ai](https://deepearth.ai) now simulates sun, wind, water, soil, & plants across the United States with state-of-the-art accuracy at centimeter scale. 

#### Sponsors
- **NASA** _Earth Science Division_ Commercial Satellite Data Acquisition Award 
- **U.S. National Science Foundation** _Institute for Geospatial Understanding_ Research Award (#2118329)
- **U.S. Department of Energy** _Office of Science_ Supercomputing Award (#BER‑ERCAP0036969) 
- **University of California Berkeley** _Quantitative Ecosystem Dynamics Lab_ (Visiting Scientist)
- **University of Florida** _Institute for Food & Agricultural Sciences_ Research Award (LIFT-AI)
- **Nebius** _2025 Nebius Robotics & Physical AI Awards_ Supercomputing Award
- **Google** _Startups Cloud Program_ Supercomputing Award
