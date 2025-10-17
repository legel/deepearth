![DeepEarth logo](https://github.com/legel/deepearth/blob/main/docs/deepearth_logo.png)
## DeepEarth: AI Foundation Model for Planetary Science & Sustainability

DeepEarth is an AI model for the planet that fuses [self-supervised](https://en.wikipedia.org/wiki/Self-supervised_learning), [multimodal](https://en.wikipedia.org/wiki/Multimodal_learning), and [spatiotemporal](https://www.sciencedirect.com/topics/social-sciences/spatio-temporal-model) deep learning.  The mission of DeepEarth is to solve global sustainability challenges (_e.g._ [climate and biodiversity](https://www.asla.org/climateandbiodiversityactionplan.aspx)) through AI for scientists, engineers, and designers.

![DeepEarth v.0.01 preview of architecture](https://github.com/legel/deepearth/blob/main/docs/deepearth_main_figure.png)

DeepEarth learns by jointly reconstructing masked multimodal datasets (as seen above). It uses a novel spacetime positional encoder, [Earth4D](https://github.com/legel/deepearth/tree/main/encoders/xyzt), especially for [earth observation](https://en.wikipedia.org/wiki/Earth_observation) data (as seen below).

![Earth4D spacetime encoder](https://github.com/legel/deepearth/blob/main/docs/earth4d_spacetime_encoder.png) 

## Exciting News:

- _September 30, 2025_  
  **Presentation at top AI lab.** 
  Thanks to the [Allen Institute for AI](https://allenai.org) for hosting a 1 hour talk with scientists pioneering [AI foundation models for the planet](https://allenai.org/earth-system). See [_video_](  https://www.youtube.com/watch?v=SHJwCInICiA) and [_slides_](https://github.com/legel/deepearth/blob/main/docs/DeepEarth_AI2_Presentation.pdf).

- _September 25, 2025_  
  **Breakthrough spacetime encoding!** [Earth4D](https://github.com/legel/deepearth/tree/main/encoders/xyzt) has been fine-tuned for simulating hourly data at sub-meter scale. See [_example code_](https://github.com/legel/deepearth/blob/main/encoders/xyzt/earth4d_to_lfmc.py).

- _August 8, 2025_  
  **Saving the world.** NSF funded a week-long ["Spatial AI for Disaster Resilience"](https://i-guide.io/summer-school/summer-school-2025/) summer school program in Boulder, Colorado. 5 PhD students developed DeepEarth (including Earth4D) for fire ecology.  See [_demos_](https://github.com/legel/deepearth/blob/main/docs/DeepEarth🔥_NSF_I-GUIDE_Final_Presentation.pdf).

- _June 23, 2025_  
  **Partnering with GeoAI leaders.** NSF funded a 3 hour workshop on DeepEarth in Chicago for a ["GeoAI for Sustainability"](https://i-guide.io/forum/forum-2025/workshops/) conference. 3 professors, 5 postdocs, and 2 PhD students contributed.  See [_slides_](https://github.com/legel/deepearth/blob/main/docs/NSF_DeepEarth_Workshop.pdf).

## Key Innovations:

#### Deep Bayesian Simulation 
DeepEarth is a deep neural network that learns to answer classical Bayesian questions, _e.g._ "As variable **α** changes across space and time, how is variable **β** most likely to change, given all available evidence?"

#### Maximizing Likelihood of the Planet
Following a [mathematical proof](https://proceedings.mlr.press/v37/germain15.html) from Google DeepMind, DeepEarth learns the _most probable_ statistical model for real world data across space and time.  It learns across (_x_, _y_, _z_, _t_, _energy_) metrics, where _energy_ can be any set of real-valued metrics ℝ<sup><em>d</em></sup>.  

#### Convergent Scientific Modeling 
A large number of DeepEarth models can be trained for diverse scientific domains: each model is trained by simply inputting domain-specific datasets, distributed across space and time. Deep inductive priors are automatically learned across all modalities.  

#### Physical Simulator _and_ Foundation Model 
DeepEarth models are trained as physical simulators of data observed across spacetime (_e.g._ predicting fire risk from historical data). Simulators can also be fine-tuned for specific applications, _i.e._ _ChatGPT_ from _GPT_.

#### Deep Spacetime Manifold
One of the great lessons from Einstein's _relativity_ is that _space_ and _time_ are not independent variables.  Following [Grid4D](https://jiaweixu8.github.io/Grid4D-web/), Earth4D extends NVIDIA's [3D multi-resolution hash encoding](https://nvlabs.github.io/instant-ngp/) to learn spatiotemporal distributions.

#### Top of the Class
Design and development of DeepEarth is led by award-winning scientists and engineers from Stanford University, University of Florida, and Ecodash.ai, along with one of the first engineers from Google DeepMind.  

#### Planetary Intelligence for Everyone
DeepEarth is an MIT-licensed open source project designed and built to solve planetary-scale problems 🌎, especially through AI-powered maximization of ecosystem services – _e.g._ for sustainable agriculture, environmental restoration, & ecological landscape design.

#### Invitation for Open Source Collaboration
Collaborators welcomed! Contact [Lance Legel](https://linkedin.com/in/legel) at lance@ecodash.ai or submit an issue/PR here.

For further details, see pre-print previews:
- [DeepEarth: Self-Supervised Multimodal Planetary Simulator with 4D Spacetime Embedding](https://github.com/legel/deepearth/blob/main/docs/deepearth.pdf) (2025)
- [Inductive Neural Networks for Ecology](https://doi.org/10.13140/RG.2.2.25523.90406) (2025)
- [AI Foundation Models for Biogeography and Ecophysiology](https://doi.org/10.13140/RG.2.2.12102.13123) (2024)
