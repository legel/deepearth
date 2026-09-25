# Hydro

Storm water over a site, in PyTorch: rain lands on the terrain, infiltrates into a soil whose state each cell
carries from one storm to the next, and the excess flows under gravity and bed friction in two dimensions. The
full account, with every experiment and known error, is [water_simulation.md](water_simulation.md).

[![Flood depth and discharge in the viewer](docs/viewer_ian_peak.jpg)](docs/viewer_ian_peak.jpg)

## Equations

| process | equation | code |
|---|---|---|
| momentum, per face | $q^{n+1} = \dfrac{q^n - g h_f \Delta t\, \partial_x \eta}{1 + g \Delta t\, n^2 \lvert q^n \rvert / h_f^{7/3}}$, $\lvert q \rvert \le 0.9\, h_f \sqrt{g h_f}$ | [`solver.py` `_face_flux`](solver.py#L255) |
| continuity | $h^{n+1} = h^n + \Delta t\,(P + \nabla \cdot q) - i$ | [`solver.py` `_substep`](solver.py#L270) |
| time step | $\Delta t = \alpha\, \Delta x / \sqrt{g h_{\max}}$, $\alpha = 0.15$ | [`solver.py` `_cfl_dt`](solver.py#L248) |
| infiltration, ponded | $d - S \ln\!\left(1 + \dfrac{d}{F + S}\right) = K_s \Delta t$, $S = G(\theta_b, \theta_s)(\theta_s - \theta_b)$ | [`infiltration.py` `ponded_increment`](infiltration.py#L160) |
| infiltration, actual | $i = \min(d,\ h,\ F_{\max} - F_1 - F_2)$ | [`infiltration.py` `step`](infiltration.py#L193) |
| redistribution | $Z \dfrac{d\theta}{dt} = r - [K(\theta) - K(\theta_b)] - p\, K_s \dfrac{G(\theta_b, \theta)}{Z}$, $p = 1.7$ dry, $1.0$ wetting | [`infiltration.py` `_rate`](infiltration.py#L176) |
| conductivity | $K(\theta) = K_s S_e^{3 + 2/\lambda}$, $S_e = \dfrac{\theta - \theta_r}{\theta_s - \theta_r}$ | [`infiltration.py` `conductivity`](infiltration.py#L135) |
| capillary drive | $G(\theta_b, \theta) = \psi_f \dfrac{S_e^{c} - S_{e,b}^{c}}{1 - S_{e,b}^{c}}$, $c = 3 + 1/\lambda$ | [`infiltration.py` `capillary_drive`](infiltration.py#L140) |
| soil state | per cell: a deep front $(F_1, \theta_1)$, a surface front $(F_2, \theta_2)$ and a hiatus flag, carried between storms | [`infiltration.py` `BANK`](infiltration.py#L47) |
| mass balance | rain + initial + inflow + created = infiltrated + abstracted + stored + outflow | [`solver.py` `MassBalance`](solver.py#L125) |

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

Against a stream gauge: Hurricane Ian at USGS 02234400 (Gee Creek near Longwood, Florida), 391.7 mm of rain,
25 m grid, 175 gauge samples over 72 h. Same terrain, rain and roughness; only the infiltration differs
([hydrograph](docs/hydrograph_ian.png)).

| | Horton | Green-Ampt with redistribution | observed |
|---|---|---|---|
| peak discharge | 342.0 m³/s at 33.5 h | 260.9 m³/s at 34.5 h | 32.4 m³/s at 37.5 h |
| runoff coefficient | 0.800 | 0.628 | 0.289 to 0.314 |
| Nash-Sutcliffe | −39.9 | −19.7 | |
| Kling-Gupta (r) | −5.23 (0.54) | −3.28 (0.60) | |
| mass-balance residual | −0.00012 % | 0.00003 % | |

The peak is 8 times the gauge's. Refining the grid 5 times moves the runoff coefficient by 0.4 %, so the grid is
not the cause; the grid box covers about half the gauge's basin, and the soil survey puts the seasonal-high water
table at the surface under a third of it. Receipts: [`docs/validation_site3_ian_25m.json`](docs/validation_site3_ian_25m.json),
[`docs/validation_site3_ian_25m_gar.json`](docs/validation_site3_ian_25m_gar.json).

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
python3 -m pytest         # 176 tests, no network and no site data
```

## License

MIT, via the repository root [`LICENSE`](../../LICENSE).
