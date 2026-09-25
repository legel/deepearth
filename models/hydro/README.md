# Hydro

Storm water over a site, in PyTorch: rain lands on the terrain, infiltrates into a soil whose state each cell
carries from one storm to the next, and the excess flows under gravity and bed friction in two dimensions. The
full account, with every experiment and known error, is [water_simulation.md](water_simulation.md).

## Equations

| process | equation | code |
|---|---|---|
| momentum, per face | $q^{n+1} = \dfrac{q^n - g h_f \Delta t\, \partial_x \eta}{1 + g \Delta t\, n^2 \lvert q^n \rvert / h_f^{7/3}}$, $\lvert q \rvert \le 0.9\, h_f \sqrt{g h_f}$ | [`solver.py` `_face_flux`](solver.py#L271) |
| continuity | $h^{n+1} = h^n + \Delta t\,(P + \nabla \cdot q) - i$ | [`solver.py` `_substep`](solver.py#L286) |
| time step | $\Delta t = \alpha\, \Delta x / \sqrt{g h_{\max}}$, $\alpha = 0.15$ | [`solver.py` `_cfl_dt`](solver.py#L264) |
| infiltration, ponded | $d - S \ln\!\left(1 + \dfrac{d}{F + S}\right) = K_s \Delta t$, $S = G(\theta_b, \theta_s)(\theta_s - \theta_b)$ | [`infiltration.py` `ponded_increment`](infiltration.py#L160) |
| infiltration, actual | $i = \min(d,\ h,\ F_{\max} - F_1 - F_2)$ | [`infiltration.py` `step`](infiltration.py#L193) |
| redistribution | $Z \dfrac{d\theta}{dt} = r - [K(\theta) - K(\theta_b)] - p\, K_s \dfrac{G(\theta_b, \theta)}{Z}$, $p = 1.7$ dry, $1.0$ wetting | [`infiltration.py` `_rate`](infiltration.py#L176) |
| conductivity | $K(\theta) = K_s S_e^{3 + 2/\lambda}$, $S_e = \dfrac{\theta - \theta_r}{\theta_s - \theta_r}$ | [`infiltration.py` `conductivity`](infiltration.py#L135) |
| capillary drive | $G(\theta_b, \theta) = \psi_f \dfrac{S_e^{c} - S_{e,b}^{c}}{1 - S_{e,b}^{c}}$, $c = 3 + 1/\lambda$ | [`infiltration.py` `capillary_drive`](infiltration.py#L140) |
| soil coupling | infiltration and surface storage over $\Delta t_s = \Delta x / 0.4\ \mathrm{m\,s^{-1}}$ (0.5 s at 0.2 m, about ten flow sub-steps), each over the time since the last: $i = \min(d(\Delta t_s),\ h,\ \ldots)$; $\Delta t_s = 0$ updates the soil every flow sub-step | [`solver.py` `_soil`](solver.py#L370), [`solver.py` `soil_dt_s`](solver.py#L76) |
| soil state | per cell: a deep front $(F_1, \theta_1)$, a surface front $(F_2, \theta_2)$ and a hiatus flag, carried between storms | [`infiltration.py` `BANK`](infiltration.py#L47) |
| mass balance | rain + initial + inflow + created = infiltrated + abstracted + stored + outflow | [`solver.py` `MassBalance`](solver.py#L138) |

The surface is the local-inertial scheme of Bates, Horritt and Fewtrell (2010) with Manning friction treated
semi-implicitly. Infiltration is Green-Ampt with redistribution (Ogden and Saghafian 1997; Smith, Corradini and
Melone 1993) with Brooks-Corey hydraulics and the Rawls, Brakensiek and Miller (1983) texture table; it is the
single-layer case of the LGAR scheme in NOAA's Next Generation Water Resources Modeling Framework. `created` is
the water the positivity clamp invents, reported rather than absorbed.

## Validation

Against exact solutions ([`tests/`](tests/)):

| check | measured | required |
|---|---|---|
| ponded infiltration against Green and Ampt's implicit solution, three soils, Δt 1 to 600 s | time error < 1e-9 of the run | < 1e-9 |
| light rain (r = K_s / 2) against an independent Radau solve of Smith et al. (1993), 6 h and 48 h | θ within bound | ≤ 0.002 |
| rain and drainage into the soil state, intermittent steps | F₁ + F₂ equals water infiltrated | 1e-9 relative |
| Manning normal depth under edge inflow ([`docs/analytic_inflow.json`](docs/analytic_inflow.json)) | 9.63e-5 relative, mass 8.5e-15 | ≤ 1e-4 |
| volume with rain, inflow, infiltration and storage | closes | 1e-6 (float64), 1e-4 (float32) |
| lake at rest over an uneven bed | depth and flux unchanged | to the bit |
| soil coupling at 0.5 s against every flow sub-step: 7.76 M cells at 0.2 m, 1 cm standing, 40 % sealed at random, loam elsewhere, 72 mm/h for 60 s (the worst case: every cell ponded, a sub-step near 0.05 s) ([`docs/soil_step_l4.json`](docs/soil_step_l4.json)) | infiltration −0.86 % (0.07 % of the water supplied), outflow +0.29 %, mass residual 3e-8 | reported |
| the same coupling on 0.2 m planes of sandy loam and clay loam, 120 then 20 mm/h, Δt_s 0.5 to 16 s | infiltrated, outflow (of the water supplied) and peak depth move less as Δt_s shrinks; at 1 s under 0.5 % | < 0.5 % |
| ponded Green-Ampt with the soil on a 60 s step | time error < 1e-6 of the run | < 1e-6 |

Against a stream gauge: Hurricane Ian at USGS 02234400 (Gee Creek near Longwood, Florida), 391.7 mm of rain,
25 m grid over a box east of the gauge, 175 gauge samples over 72 h. Same terrain, rain and roughness; only the
infiltration differs ([hydrograph](docs/hydrograph_ian.png); receipts [`docs/validation_site3_ian_25m.json`](docs/validation_site3_ian_25m.json),
[`docs/validation_site3_ian_25m_gar.json`](docs/validation_site3_ian_25m_gar.json)); refining that grid 5 times moved the runoff
coefficient 0.4 % ([`docs/resolution_site3_ian.json`](docs/resolution_site3_ian.json)).

| | Horton | Green-Ampt with redistribution | observed |
|---|---|---|---|
| peak discharge | 342.0 m³/s at 33.5 h | 260.9 m³/s at 34.5 h | 32.4 m³/s at 37.5 h |
| runoff coefficient | 0.800 | 0.628 | 0.289 to 0.314 |
| Nash-Sutcliffe | −39.9 | −19.7 | |
| Kling-Gupta (r) | −5.23 (0.54) | −3.28 (0.60) | |
| mass-balance residual | −0.00012 % | 0.00003 % | |

Over the gauge's whole drainage basin (USGS NLDI), depressions kept, the starting soil from a 90-day continuous balance
on the same grid, and each cell's rain from AORC's 1 km grid (339.5 mm over the basin). Nothing is fitted; each row
changes one input from the first.

| | peak (gauge 32.4 m³/s at 37.5 h) | runoff coefficient (gauge 0.333 to 0.362 on the same rain) | Nash-Sutcliffe | Kling-Gupta |
|---|---|---|---|---|
| Manning n 0.040 everywhere, AORC storm totals on the domain mean's hours | 194.2 m³/s at 33.5 h | 0.350 | −8.92 | −1.69 |
| Manning n by NLCD class (Chow 1959, flood plains) | 151.1 m³/s at 34.5 h | 0.344 | −4.86 | −1.03 |
| each cell its own AORC hours | 204.1 m³/s at 33.5 h | 0.348 | −8.74 | −1.66 |

The volume now matches the gauge; the peak arrives 3 to 4 h early and 4.7 to 6.3 times too high, so the water
reaches the outlet too fast. Receipts: [`docs/validation_ian_25m_gar_basin_depressions_antecedent_aorc.json`](docs/validation_ian_25m_gar_basin_depressions_antecedent_aorc.json),
[`docs/validation_ian_25m_gar_basin_depressions_antecedent_aorc_nlcdn.json`](docs/validation_ian_25m_gar_basin_depressions_antecedent_aorc_nlcdn.json),
[`docs/validation_ian_25m_gar_basin_depressions_antecedent_aorch.json`](docs/validation_ian_25m_gar_basin_depressions_antecedent_aorch.json).

The same basin with Manning n by land cover at 25 m and 5 m, on one GPU with the soil on its own step (the 25 m run
repeats the CPU run above: peak 151.09 against 151.06 m³/s, Nash-Sutcliffe −4.870 against −4.864):

| grid | peak | runoff coefficient | Nash-Sutcliffe | Kling-Gupta | wall, one L4 |
|---|---|---|---|---|---|
| 25 m, 556 x 560 | 151.1 m³/s at 34.5 h | 0.344 | −4.87 | −1.03 | 132 s |
| 5 m, 1,392 x 1,401 | 171.9 m³/s at 34.5 h | 0.399 | −7.71 | −1.60 | 1,363 s |

Resolving the channels makes the flood faster and larger, not slower, so the grid is not what holds the peak back.
Receipts: [`docs/validation_ian_25m_gar_basin_depressions_antecedent_aorc_nlcdn_gpu_soildt.json`](docs/validation_ian_25m_gar_basin_depressions_antecedent_aorc_nlcdn_gpu_soildt.json),
[`docs/validation_ian_5m_gar_basin_depressions_antecedent_aorc_nlcdn_gpu_soildt.json`](docs/validation_ian_5m_gar_basin_depressions_antecedent_aorc_nlcdn_gpu_soildt.json).

## Run it

```bash
cd models/hydro
python3 -m pip install --user -r requirements.txt

python3 cli.py fetch    --site site3 --storm ian
python3 cli.py terrain  --site site3
python3 cli.py simulate --site site3 --storm ian --cell-size 25
python3 cli.py validate --site site3 --storm ian --cell-size 25
```

```python
from infiltration import Soil
from solver import SolverConfig, Surface, simulate

soil = Soil.texture("silt loam", theta_i=0.20, shape=z.shape)          # or per-cell arrays
res = simulate(Surface(z=z, soil=soil, manning_n=n), rain_m_per_s, SolverConfig(dx=1.0, dt_s=60.0))
res = simulate(Surface(z=z, soil=soil, soil_state=res.soil_state), next_storm, cfg)   # the state carries
```

```bash
python3 -m pytest         # 184 tests, no network and no site data
```

## License

MIT, via the repository root [`LICENSE`](../../LICENSE).
