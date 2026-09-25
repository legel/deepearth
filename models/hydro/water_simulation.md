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

Roughness is a Manning n per cell. Nodata cells are an open edge: water that reaches one leaves the domain and is
counted as outflow.

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

Soil parameters on the gauge site come from the USDA soil survey (SSURGO): saturated conductivity, saturated and
field-capacity water content from each map unit's dominant surface horizon. The pore-size index, residual water and
wetting-front suction come from the Rawls, Brakensiek and Miller (1983) texture table [texture], [gar-soil]. Roads
and buildings from OpenStreetMap seal their cells; elsewhere conductivity is reduced by the NLCD impervious fraction.
Without a soil the solver uses Horton's decay against a fixed store, the earlier parameterization.

## How the tower enters

The flux tower's precipitation P drives the rain on every cell, hour by hour. In the DeepEarth platform, a year-long
water balance on the same classes and soils supplies each storm's starting soil water and the room left in the root
zone, so the storm solver and the balance share one soil state.

## Experiments

Checks against exact solutions [tests]:

    ponded infiltration against the Green-Ampt solution, 3 soils, dt 1 to 600 s    time error below 1e-9
    light rain (r = K_s/2) against an independent stiff solve of Smith et al.        water content within 0.002
    water infiltrated against water held in the soil state, 3,000 intermittent steps  equal to 1e-9
    same storm on a drained soil and on a wet one                                     drained soil takes over 5 % more
    Manning normal depth under a prescribed inflow                                    9.6e-5 relative
    volume budget with rain, inflow, infiltration and surface storage                 closes to 1e-6 (float64)

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

## Known errors

The Gee Creek peak is 8 times the gauge's and the runoff coefficient twice the observed; this is not a validated
discharge. Three causes are measured. First, the grid box covers only the eastern half of the gauge's drainage basin
(the NLDI basin spans 11.3 km east to west, the box 6 km), and the score compares the whole box's outflow with the
gauge. Second, the survey puts the seasonal-high water table at the surface under a third of the catchment, so those
cells take nothing; that is the wettest condition of the year, not necessarily Ian's. Third, the depression
storage of Florida's wetlands and ponds is not represented. Refining the grid 5 times changes the runoff
coefficient by 0.4 %, so the grid is not the cause. A run over the whole basin is under way.

The solver has one soil layer per cell, no baseflow and no channel storage, so recessions are too fast. Culverts
are not routed.

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
[validation-horton]: https://github.com/legel/deepearth/blob/87fb913/models/hydro/docs/validation_site3_ian_25m.json
[validation-gar]: https://github.com/legel/deepearth/blob/87fb913/models/hydro/docs/validation_site3_ian_25m_gar.json
