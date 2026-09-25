# Water simulation

The water model moves rain across a terrain grid in two dimensions and lets it soak into the soil where the soil
can take it. Each cell keeps its own soil water from one step to the next and from one storm to the next. The code is
PyTorch and runs on a CPU or a GPU; on a GPU the sub-step is compiled into fused kernels with `torch.compile`.

## Surface flow

Water depth h and the discharge per unit width q on each cell face follow the local-inertial form of the shallow
water equations (Bates, Horritt and Fewtrell 2010), which drops the convective acceleration and keeps gravity,
the water-surface slope and bed friction. Each face is updated semi-implicitly for Manning friction [solver-face]:

    q_new = (q - g h dt dη/dx) / (1 + g dt n² |q| / h^(7/3))

and capped at a Froude number of 0.9. Depth then changes by the net inflow over the cell and the rain, less what
infiltrates [solver-step]. The time step follows the wave speed of the deepest water, dt = 0.15 dx / sqrt(g h)
[solver-cfl]. Every term of the volume budget is carried separately (rain, inflow, infiltration, surface storage,
storage on the grid, outflow, and the small volume the positivity clamp creates), so each run reports how well
volume closed [solver-mass].

Roughness is a Manning n per cell: one value, 0.040, or by land cover, each NLCD class taking the normal value of Chow's
(1959) table for flood plains (woody wetlands 0.150, emergent wetlands, forest and shrub 0.100, grass and pasture
0.035, open water 0.030; developed land keeps 0.040, which the table has no class for) [roughness]. Nothing is
fitted to a gauge. Nodata cells are an open edge: water that reaches one leaves the domain and is counted as outflow.

## Infiltration and the soil state

Infiltration is Green-Ampt with redistribution (Ogden and Saghafian 1997), with the redistribution equation of
Smith, Corradini and Melone (1993) [infiltration]. While water stands on a cell, the soil takes the exact
Green-Ampt amount over the step, solving

    d - S ln(1 + d / (F + S)) = K_s dt,    S = G (θ_s - θ_b)

by Newton's method, rather than a rate times dt [ponded]. When no water stands, the wetted zone drains and spreads
downward:

    Z dθ/dt = r - (K(θ) - K(θ_b)) - p K_s G(θ_b, θ) / Z

with p = 1.7 while nothing enters and 1.0 while rain enters [rate]. This equation is stiff for a thin wetted zone, so
it is stepped linearly implicitly [redistribute]. Unsaturated conductivity and capillary drive are Brooks-Corey:
K(θ) = K_s S_e^(3 + 2/λ), with drive G scaled so that it equals the wetting-front suction ψ_f for a dry soil.

Each cell carries five numbers through time [bank]: the water and water content of a deep wetting front, the same
for a shallow front that a later storm starts on top of it, and whether the last step took in water. A run returns
this state, and the next run can start from it, so the soil's wetness before a storm is simulated, not assumed.
The total a cell can take is capped by the room above its water table.

The soil is coupled to the surface on a step of its own, dt_s = dx / (0.4 m/s), the time sheet flow at 0.4 m/s takes to
cross one cell: 0.5 s at 0.2 m, about ten flow sub-steps [soil-step], [soil-dt]. Each soil update takes infiltration and
then surface storage over the time since the last one. Green-Ampt's ponded increment is exact over any step and the
redistribution is implicit, so this is a first-order split in dt_s; setting dt_s to 0 updates the soil every flow
sub-step. It matters for cost: on every sub-step the soil, in float64, was 91 % of an NVIDIA L4's sub-step and 94 % of
an RTX PRO 6000's at 7.76 M cells, and on its own step the L4's sub-step is 7.2 times faster [soil-bench].

Soil parameters on the gauge site come from the USDA soil survey (SSURGO): saturated conductivity, saturated and
field-capacity water content from each map unit's dominant surface horizon. The pore-size index, residual water and
wetting-front suction come from the Rawls, Brakensiek and Miller (1983) texture table [texture], [gar-soil]. Roads
and buildings from OpenStreetMap seal their cells; elsewhere conductivity is reduced by the NLCD impervious fraction.
Without a soil the solver uses Horton's decay against a fixed store, the earlier parameterization.

## How the tower enters

The flux tower's precipitation P drives the rain on every cell, hour by hour. In the DeepEarth platform, a year-long
water balance on the same classes and soils supplies each storm's starting soil water and the room left in the root
zone, so the storm solver and the balance share one soil state.

For the gauge test the rain can instead come from the AORC record's 1 km grid (NOAA's Analysis of Record for
Calibration). Each cell then takes its own 1 km cell's storm total on the domain mean's hourly timing, or its own
cell's hours; either way each hour's shares average 1 over the grid, so the volume budget counts the grid's own rain
[aorc], [hourly]. The storm's starting soil can come from such a continuous balance run on the same grid: each
cell's water content and the room left above its water table at the storm's first hour [antecedent]. The domain can
be the gauge's whole drainage basin from the USGS Network-Linked Data Index [basin], with the depressions kept
(the terrain before its depressions are breached) so wetlands and ponds hold what they catch [depressions].

## Experiments

Checks against exact solutions [tests]:

    ponded infiltration against the Green-Ampt solution, 3 soils, dt 1 to 600 s    time error below 1e-9
    light rain (r = K_s/2) against an independent stiff solve of Smith et al.        water content within 0.002
    water infiltrated against water held in the soil state, 3,000 intermittent steps  equal to 1e-9
    same storm on a drained soil and on a wet one                                     drained soil takes over 5 % more
    Manning normal depth under a prescribed inflow                                    9.6e-5 relative
    volume budget with rain, inflow, infiltration and surface storage                 closes to 1e-6 (float64)
    soil on its own 0.5 s step against every sub-step, worst case: 7.76 M cells at    infiltration -0.86 %,
      0.2 m all ponded (1 cm standing, 40 % sealed at random, loam, 72 mm/h, 60 s)     outflow +0.29 %,
                                                                                      mass residual 3e-8
    the same on 0.2 m planes of sandy loam and clay loam, dt_s 0.5 to 16 s            converges; under 0.5 %
                                                                                      of the water at 1 s

Gauge test: Hurricane Ian (September 2022) at Gee Creek near Longwood, Florida, USGS gauge 02234400, 391.7 mm of rain
over 72 hours, 25 m grid, scored over the gauge's 175 samples. Terrain, rain and roughness are identical in both
runs; only the infiltration differs [validation-horton], [validation-gar].

                            Horton          GAR             observed
    peak discharge          342.0 m³/s      260.9 m³/s      32.4 m³/s
    runoff coefficient      0.80            0.63            0.29 to 0.31
    Nash-Sutcliffe          -39.9           -19.7
    Kling-Gupta             -5.23           -3.28
    volume residual         -0.0001 %       0.00003 %

Cost of that run on an 8-core CPU in float64: 530 s with Horton, 2,677 s with GAR.

The same storm over the gauge's whole drainage basin (USGS NLDI), with the depressions kept, the starting soil from a
90-day continuous balance on the same grid, and each cell's rain from AORC's 1 km grid (339.5 mm over the basin; the
gauge's runoff coefficient on that rain is 0.333 to 0.362). Nothing is fitted; each run changes one input from the
first [basin-scalar], [basin-nlcd], [basin-hourly].

                                              peak (gauge 32.4 m³/s    runoff        Nash-      Kling-
                                              at 37.5 h)               coefficient   Sutcliffe  Gupta
    Manning n 0.040, AORC totals on the       194.2 m³/s at 33.5 h     0.350         -8.92      -1.69
      domain mean's hours
    Manning n by NLCD class (Chow 1959)       151.1 m³/s at 34.5 h     0.344         -4.86      -1.03
    each cell its own AORC hours              204.1 m³/s at 33.5 h     0.348         -8.74      -1.66

The same basin with Manning n by land cover at 25 m and 5 m, on one GPU with the soil on its own step; the 25 m run
repeats the CPU run above (peak 151.09 against 151.06 m³/s) [basin-25], [basin-5]:

                                              peak                     runoff        Nash-      Kling-
                                                                       coefficient   Sutcliffe  Gupta
    25 m (556 x 560), 132 s on one L4         151.1 m³/s at 34.5 h     0.344         -4.87      -1.03
    5 m (1,392 x 1,401), 1,363 s              171.9 m³/s at 34.5 h     0.399         -7.71      -1.60

## Known errors

Gee Creek's discharge is not validated. Over the whole basin, with the starting soil from a continuous balance and
the depressions kept, the volume matches the gauge: a runoff coefficient of 0.344 to 0.350 against 0.333 to 0.362.
The peak does not. It is 4.7 to 6.3 times the gauge's and 3 to 4 hours early, so the water reaches the outlet too
fast. Land-cover roughness slows it by an hour and lowers it by 22 %; each cell's own rain timing does not help; a 5 m
grid, which resolves the channels, makes the peak 14 % higher and no later. Nothing was tuned to close the gap.

What the basin has that the solver lacks, in the order the basin suggests (lakes, wetlands and a water table near the
surface):
- **storage routing** through the lakes and wetland depressions, each with its outlet: kept depressions hold water, but
  nothing releases it through a control, so a full depression spills at once;
- **baseflow and a shallow water table**: one soil layer per cell and no groundwater, so recessions are too fast and
  water the soil takes never returns to the stream;
- **culverts**: flow under roads is not routed; a public inventory would supply them.

[solver-face]: https://github.com/legel/deepearth/blob/87fb913/models/hydro/solver.py#L255
[solver-step]: https://github.com/legel/deepearth/blob/87fb913/models/hydro/solver.py#L270
[solver-cfl]: https://github.com/legel/deepearth/blob/87fb913/models/hydro/solver.py#L248
[solver-mass]: https://github.com/legel/deepearth/blob/87fb913/models/hydro/solver.py#L125
[infiltration]: https://github.com/legel/deepearth/blob/87fb913/models/hydro/infiltration.py#L193
[ponded]: https://github.com/legel/deepearth/blob/87fb913/models/hydro/infiltration.py#L160
[rate]: https://github.com/legel/deepearth/blob/87fb913/models/hydro/infiltration.py#L176
[redistribute]: https://github.com/legel/deepearth/blob/87fb913/models/hydro/infiltration.py#L184
[bank]: https://github.com/legel/deepearth/blob/87fb913/models/hydro/infiltration.py#L47
[texture]: https://github.com/legel/deepearth/blob/87fb913/models/hydro/infiltration.py#L67
[gar-soil]: https://github.com/legel/deepearth/blob/87fb913/models/hydro/domain.py#L230
[tests]: https://github.com/legel/deepearth/blob/87fb913/models/hydro/tests/test_infiltration.py
[soil-step]: https://github.com/legel/deepearth/blob/018bd44/models/hydro/solver.py#L370
[soil-dt]: https://github.com/legel/deepearth/blob/018bd44/models/hydro/solver.py#L76
[soil-bench]: https://github.com/legel/deepearth/blob/018bd44/models/hydro/docs/soil_step_l4.json
[roughness]: https://github.com/legel/deepearth/blob/018bd44/models/hydro/domain.py#L127
[aorc]: https://github.com/legel/deepearth/blob/018bd44/models/hydro/forcing.py#L86
[hourly]: https://github.com/legel/deepearth/blob/018bd44/models/hydro/solver.py#L601
[antecedent]: https://github.com/legel/deepearth/blob/018bd44/models/hydro/domain.py#L347
[basin]: https://github.com/legel/deepearth/blob/018bd44/models/hydro/domain.py#L302
[depressions]: https://github.com/legel/deepearth/blob/018bd44/models/hydro/domain.py#L59
[basin-scalar]: https://github.com/legel/deepearth/blob/018bd44/models/hydro/docs/validation_ian_25m_gar_basin_depressions_antecedent_aorc.json
[basin-nlcd]: https://github.com/legel/deepearth/blob/018bd44/models/hydro/docs/validation_ian_25m_gar_basin_depressions_antecedent_aorc_nlcdn.json
[basin-hourly]: https://github.com/legel/deepearth/blob/018bd44/models/hydro/docs/validation_ian_25m_gar_basin_depressions_antecedent_aorch.json
[basin-25]: https://github.com/legel/deepearth/blob/simulator/models/hydro/docs/validation_ian_25m_gar_basin_depressions_antecedent_aorc_nlcdn_gpu_soildt.json
[basin-5]: https://github.com/legel/deepearth/blob/simulator/models/hydro/docs/validation_ian_5m_gar_basin_depressions_antecedent_aorc_nlcdn_gpu_soildt.json
[validation-horton]: https://github.com/legel/deepearth/blob/87fb913/models/hydro/docs/validation_site3_ian_25m.json
[validation-gar]: https://github.com/legel/deepearth/blob/87fb913/models/hydro/docs/validation_site3_ian_25m_gar.json
