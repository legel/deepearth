# Wind

A steady 3D wind field over terrain, buildings and canopy, in PyTorch. The sides carry the steady wind of a column
over the site's mean canopy; the field is made mass-consistent around the obstacles by a Poisson projection, then
iterated to the steady state of the momentum equations with advection, turbulent mixing, canopy drag and wall stress. The converged field does not
depend on the pseudo-time step. The full account, with every experiment and known error, is
[wind_simulation.md](wind_simulation.md).

[![Speed and vorticity on the center section](docs/wind_tower_0.8m_section.png)](docs/wind_tower_0.8m_section.png)

## Equations

| process | equation | code |
|---|---|---|
| upwind profile | $u(z) = \dfrac{u_*}{\kappa} \ln\dfrac{z - d}{z_0}$ from the reference speed, held at the top | [`forcing.py` `LogProfile`](forcing.py#L24) |
| inflow at the sides | the steady column over the site's mean canopy, $\dfrac{d}{dh}\left(\nu_t \dfrac{dU}{dh}\right) = (\bar{c_d a} + c_w)\,U^2$, at each cell's height above its ground | [`solver.py` `equilibrium_column`](solver.py#L440) |
| mass consistency | $\nabla^2 \lambda = \nabla \cdot \mathbf{u}^*$, $\mathbf{u} = \mathbf{u}^* - \nabla \lambda$ | [`solver.py` `project`](solver.py#L714) |
| steady residual | $R(\mathbf{u}) = -\sum_f F_f \mathbf{u}_f + \nabla\cdot(\nu_t \nabla \mathbf{u}) - (c_w + c_d a)\lvert\mathbf{u}\rvert\mathbf{u}$ | [`solver.py` `fv_increment`](solver.py#L900) |
| convection | face value $\mathbf{u}_f$ by MUSCL, van Leer limited, on the divergence-free face fluxes $F_f$ | [`solver.py` `convection`](solver.py#L878), [`_muscl`](solver.py#L319) |
| turbulent mixing, as run (k-l) | $\nu_t = C_\mu^{1/4}\,\ell\sqrt{k}$; $\partial_t k + \nabla\cdot(\mathbf{u}k) = \nabla\cdot\left(\dfrac{\nu_t}{\sigma_k}\nabla k\right) + \nu_t\lvert S\rvert^2 - \dfrac{C_\mu^{3/4} k^{3/2}}{\ell} + c_d a\left(\beta_p\lvert\mathbf{u}\rvert^3 - \beta_d\lvert\mathbf{u}\rvert k\right)$; $C_\mu$ 0.09, $\sigma_k$ 1, $\beta_p$ 1, $\beta_d$ 5.1 (Katul et al. 2004); $\ell = \kappa(h_c - d)$ in the canopy, $\kappa(h - d)$ above, $d = 2h_c/3$ | [`solver.py` `k_step`](solver.py#L952), [`KL_C_MU`](solver.py#L39), [`_kl_length`](solver.py#L332) |
| drive, as run | a mean pressure gradient along the wind, $\Pi = u_*^2 / h_{top}$ (the column's stress over its depth), no stress through the top | [`solver.py` `_kl_column`](solver.py#L495) |
| turbulent mixing, mixing length (option) | $\nu_t = (\kappa\, \bar h)^2 \lvert S \rvert + \nu$, under-relaxed 0.5 between steps | [`solver.py` `viscosity`](solver.py#L814) |
| canopy drag | $c_d\, a\, \lvert \mathbf{u} \rvert \mathbf{u}$, $a = \mathrm{LAI}/h$ | [`physics.py` `drag_density`](physics.py#L97) |
| wall stress | $\left(\kappa / \ln(\delta / z_0)\right)^2 \lvert \mathbf{u} \rvert \mathbf{u}$ at half a cell | [`solver.py` `_wall`](solver.py#L326) |
| pseudo-time step | $M\,\delta\mathbf{u} = \Delta t\, R(\mathbf{u}) + V \nabla \Pi$, $M = V(1 + \Delta t\, c\lvert\mathbf{u}\rvert) + \Delta t\,(D + A_\mathrm{upwind})$; then project, $\Pi \mathrel{+}= \lambda$ | [`solver.py` `run`](solver.py#L1063), [`poisson.py` `Convective`](poisson.py#L154) |

Where $\delta\mathbf{u} = 0$ the steady equations hold whatever $\Delta t$, so the step only sets how fast the
iteration arrives. $M$ is solved by BiCGSTAB and the projection by conjugate gradients, both preconditioned by one
multigrid kernel; momentum runs in float32 and every projection in float64.

## At every return: the Year's wind run

The solver is linear in the reference speed. An hour's field is the two solved headings about its direction $\theta$
blended and scaled to the reference speed $u_{ref}$, set by the tower ([wind_simulation.md](wind_simulation.md#how-the-tower-sets-the-value)):

| quantity | equation | code |
|---|---|---|
| speed at a return | $U(p, h) = u_{ref}(h)\,\lvert (1 - a)\,\mathbf{S}_k(p) + a\,\mathbf{S}_{k+1}(p) \rvert$, $\theta(h)$ between headings $k$ and $k + 1$, $a$ its fraction in steps of 1/16 | [`unit_field`](year_run.py#L42), [`groups`](year_run.py#L31) |
| wind run, km | $R(p) = 3.6 \sum_g \lvert \mathbf{S}_g(p) \rvert \sum_{h \in g} u_{ref}(h)$ over the year's hours grouped by heading pair and fraction | [`run_km`](year_run.py#L49) |
| stored | one byte a return, $\lfloor 254\, R / R_{\max} \rceil$, $R_{\max}$ the record's p98 over every year and return rounded up to 1, 2 or 5 times a power of ten | [`quantize`](year_run.py#L72), [`nice_ceil`](year_run.py#L61) |
| flow drawn over it | the year's typical hour, its most frequent 22.5° sector (calm under 0.5 m/s aside) at the sector's median speed, on the lowest level | [`typical_hour`](year_run.py#L153), [`ribbon_field`](year_run.py#L166) |

Grouping moves the run under 0.2 % (median) and 1 % (max) from summing every hour alone, over 8,760 random hours ([`tests/test_year_run.py`](tests/test_year_run.py)).

### The unit field at a return

> **Being replaced.** The rule below is to give way to the solver's 3D field sampled at each return (trilinear over
> fluid cells, a fixed clearance off solid faces, one rule for every class). Neighboring returns under different
> rules here differ by 30 to 90 %, at crowns, walls and roofs.

| rule | equation | code |
|---|---|---|
| where it is read | a canopy return at its own height $z$; a ground or roof return 2 m above its surface | [`point_basis`](year_run.py#L103), [`PLANT_HEIGHT_M`](year_run.py#L24) |
| between levels | bilinear on each level; linear in $\ln z$ between the usable levels about $z$ | [`bilinear`](year_run.py#L84) |
| below the lowest, above the top | $\mathbf{S}(z) = \mathbf{S}(z_l)\,\dfrac{\ln(z / z_0)}{\ln(z_l / z_0)}$, $z_0$ by the surface's class | [`point_basis`](year_run.py#L103) |
| a return the solve does not reach | the nearest reached return in $(x, y, z)$, carried to $z$ by the same log law, clipped to a factor of 3 | [`fill_solids`](year_run.py#L132), [`FILL_MAX`](year_run.py#L25) |

## Validation

Physics checks, float64, 1 m cells ([`docs/verification_cpu.json`](docs/verification_cpu.json)):

| check | measured | required |
|---|---|---|
| largest cell divergence, cube, ridge, flat | 3.4e-7, 1.9e-7, 2.7e-13 s⁻¹ | ≤ 1e-6 s⁻¹ |
| mass flux in against out | 2.3e-9, 1.0e-9, 3.4e-15 | ≤ 1e-6 |
| inflow profile returned by an unobstructed domain | 6.3e-6 of its peak | the discretization's own |
| speed inside canopy against none, LAI 0.5, 2, 8 | 0.967, 0.886, 0.702× | falling |
| wake behind an 8 m cube, min u / u(H) | −0.366, reversed in 12.7 % of the near wake | reversed |
| westerly against southerly over the same cube, rotated | 1.1e-5 | ≤ 1 % |
| field at 2, 5 and 10 m/s against one field scaled | 1.4e-5 | ≤ 2 % |

Step independence, Harvard Forest, one heading, 10.6 M cells: the published levels' medians at CFL 8, 32 and
64 against 16.

| level | CFL 8 | CFL 32 | CFL 64 |
|---|---|---|---|
| 4 m | +0.14 % | −0.40 % | −0.80 % |
| 10 m | +0.04 % | −0.14 % | −0.31 % |
| 25 m | 0.00 % | −0.05 % | −0.15 % |

The earlier semi-Lagrangian scheme's settled field depended on its step, and its medians sat 15 to 18 % below the
converged field at Harvard's four levels. With the column inflow the field changes at most 2.6 % between a 512 m and
a 768 m square; with a log-law inflow the canopy slowed the sides for hundreds of meters and the 4 m median fell 44 %
between 384 and 768 m ([wind_simulation.md](wind_simulation.md#experiments)).

## Run it

```bash
cd models/wind
python3 -m pip install --user -r requirements.txt

python3 cli.py simulate  --site campanile --dx 0.8 --synthetic tower --speed 8 --direction 290 --cfl 16
python3 cli.py verify    --site campanile --cfl 16
python3 cli.py benchmark --site campanile --nx 128 --ny 128 --nz 64 --steps 5
```

`--bundle <dir>` solves a real site from a directory holding `semantics/class_top_<res>.tif` (a class per column)
with `parameters.json` (z0, cd, LAI and closure per class), and `surface/` DTM and DSM rasters; without one, the
46-class table in [`physics.py` `CLASSES`](physics.py#L46) applies. `--fast` runs momentum in float32. A site is
solved with `--scheme fv --fast --inflow canopy --closure k-l --drive pressure`.

```bash
python3 -m pytest         # 108 tests, no network and no site data
```

## License

MIT, via the repository root [`LICENSE`](../../LICENSE).
