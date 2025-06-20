# DeepEarth: Spatiotemporal Multimodal Self-Supervised Simulator of Earth Ecosystems

DeepEarth is an AI foundation model for deep multimodal geospatial simulation.  It can learn to inductively reconstruct masked spatiotemporal distributions of datasets from across Earth sciences, including from physics, chemistry, biology, and ecology.  

![DeepEarth Grid4D spacetime encoding](https://github.com/legel/deepearth/blob/main/docs/deepearth_spacetime_encoder.png)
![DeepEarth v.0.01 preview of architecture](https://github.com/legel/deepearth/blob/main/docs/deepearth_inductive_simulator.png)

#### Deep Physical Simulation  
DeepEarth is designed to deliver state-of-the-art modeling representations that can directly answer classical Bayesian questions, _e.g._ "As variable X changes across space S and time T, how is variable Y most likely to change, given all available evidence?"

#### Encoding the Planet
 DeepEarth is designed to discover the most predictive neural network parameters for multimodal observations across space and time.  It learns across (_x_, _y_, _z_, _t_, _energy_) metrics, where _energy_ can be any set of real-valued metrics ℝ<sup><em>d</em></sup>.  

#### Convergent Scientific Modeling 
Input any number of datasets distributed across space and time (_e.g._  satellite imagery, geological surveys, biodiversity records) and then automatically learn deep inductive priors across modalities.  

#### Deep Spacetime Manifold
One of the great lessons from Einstein's _relativity_ is that _space_ and _time_ are not independent variables.  DeepEarth learns unified (x,y,z,t) deep vector representations from _[Grid4D](https://github.com/JiaweiXu8/Grid4D/tree/main)_ as ["multi-resolution hash encodings"](https://graphics.stanford.edu/courses/cs348n-22-winter/LectureSlides/FinalSlides/ING.pdf).  This enables deep learning of complex spatiotemporal distributions.

#### Built for Lightspeed 
 DeepEarth is built on the [57x](https://www.youtube.com/watch?v=0VLAoVGf_74&ab_channel=WelchLabs) more memory-efficient Transformer architecture of _[DeepSeek](https://github.com/deepseek-ai/DeepSeek-V3)_ to unlock next-generation speed, accuracy, and scalability of multimodal cross-fusion attention.

#### Top of the Class
Design and development of DeepEarth is led by award-winning scientists and engineers from Stanford University, University of Florida, and Ecodash.ai, along with one of the first engineers from Google DeepMind.  

#### Planetary Intelligence for Everyone
DeepEarth is an MIT-licensed open source project designed and built to solve planetary-scale problems 🌎, especially through AI-powered maximization of ecosystem services – _e.g._ optimizing AI for sustainable agriculture, environmental restoration, & ecological landscape design.


## NSF Workshop + Summer School
#### Join us for an NSF-funded workshop on June 17th, 2025
We're proud to host an NSF I-GUIDE workshop on DeepEarth this summer in Chicago!  See ["DeepEarth Workshop: Self-Supervised AI for Spatiotemporal Modeling of Ecosystems"](https://i-guide.io/forum/forum-2025/workshops/) for more details. Looking forward to meeting you!

#### NSF summer program in AI for disaster resilience, August 4-8, 2025
5 PhD students will join us for a 5 day program in Boulder, Colorado on ["Spatial AI for Extreme Events and Disaster Resilience"](https://i-guide.io/summer-school/summer-school-2025/).  We will geospatially and temporally simulate fire and flood responses of plants at sub-meter scale.

## License

MIT License - see LICENSE file for details.
