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
One of the great lessons from Einstein's _relativity_ is that _space_ and _time_ are not independent variables.  Following [Grid4D](https://jiaweixu8.github.io/Grid4D-web/), Earth4D extends NVIDIA's [3D multi-resolution hash encoding](https://nvlabs.github.io/instant-ngp/) to learn spatio-temporal distributions.
