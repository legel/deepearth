# Wind

A steady 3D wind field over terrain, buildings and canopy, in PyTorch. An upwind log profile is made
mass-consistent around the obstacles by one Poisson projection, then relaxed in pseudo-time with
advection, turbulent mixing, canopy drag and wall stress until the flow near the ground stops changing.
Velocity and the three vorticity components come out on one stretched grid.

[![Speed and vorticity on the centre section](docs/wind_tower_0.8m_section.png)](docs/wind_tower_0.8m_section.png)

A 94 m tower at 0.8 m cells, 8 m/s from 290°: the log profile arrives from the left, the flow speeds up
over the roof and separates, and the wake recovers downwind. Vorticity marks the shear layers.

## Equations

Velocity lives on a staggered grid; every implicit operator is a 7-point solve on one
multigrid-preconditioned conjugate-gradient kernel ([`poisson.py` `cg`](poisson.py#L116)).

| process | equation | code |
|---|---|---|
| inflow | $u(z) = \dfrac{u_*}{\kappa} \ln\dfrac{z - d}{z_0}$, $u_*$ from the reference speed, re-rooted onto the site's fetch at a 60 m blending height | [`forcing.py` `LogProfile`](forcing.py#L24), [`transfer`](forcing.py#L48) |
| mass consistency | $\nabla^2 \lambda = \nabla \cdot \mathbf{u}^*$, $\mathbf{u} = \mathbf{u}^* - \nabla \lambda$; solids are Neumann faces, the top is open | [`solver.py` `project`](solver.py#L322) |
| advection | semi-Lagrangian with a MacCormack correction, limited to the departure cell's neighbours | [`solver.py` `advect`](solver.py#L393) |
| turbulent mixing | $\nu_t = (\kappa\, \bar h)^2 \lvert S \rvert + \nu$, $\bar h$ the logarithmic mean of the heights above the local surface | [`solver.py` `viscosity`](solver.py#L414) |
| canopy drag | $-c_d\, a\, \lvert \mathbf{u} \rvert \mathbf{u}$, $a = \mathrm{LAI}/h$ | [`physics.py` `drag_density`](physics.py#L97) |
| wall stress | $\tau / \rho = \left(\dfrac{\kappa}{\ln(\delta / z_0)}\right)^2 \lvert \mathbf{u} \rvert \mathbf{u}$ at half a cell, per surface class | [`solver.py` `_wall`](solver.py#L161) |
| momentum step | $(I + \Delta t\,(c_w + c_d a)\lvert \mathbf{u} \rvert - \Delta t\, \nabla \cdot \nu_t \nabla)\, \mathbf{u}^{n+1} = \mathcal{A}(\mathbf{u}^n)$, then project | [`solver.py` `diffuse`](solver.py#L438) |
| settling | stop when the horizontal speed of fluid cells 2 to 30 m above their surface moves < 1 % in median and p95 and < 5 % RMS between 40-step window means, two windows running | [`solver.py` `settle_change`](solver.py#L480), [`run`](solver.py#L493) |

The mixing length and the wall law share the logarithmic mean, so the discrete log profile is an exact
steady state of the whole loop: an unobstructed domain returns its inflow to round-off.

**Linearity.** Every steady term is homogeneous of degree two in velocity, so for one heading the field
scales with the reference speed: at 2, 5 and 10 m/s the deviation from one field scaled is 1.1e-5 of the
peak. A year of hours is one unit-speed field per heading, blended between the two nearest headings and
scaled by the hour's measured speed (`cli.py basis`).

## Validation

Every number is produced by `python3 cli.py verify` and asserted in `tests/test_solver.py`, float64, 1 m
cells ([`docs/verification_cpu.json`](docs/verification_cpu.json)):

| check | measured | required |
|---|---|---|
| largest cell divergence after the solve, cube, ridge, flat | 7.7e-7, 8.9e-7, 0.0 s⁻¹ | ≤ 1e-6 s⁻¹ |
| inflow profile returned by an unobstructed domain | bit-exact, crossflow 0.0 | exact |
| mass flux in against out | 5.6e-10, 1.3e-8, 0 | ≤ 1e-6 |
| speed inside canopy against none, LAI 0.5, 2, 8 | 0.964, 0.881, 0.706× | falling |
| wake behind an 8 m cube, min u / u(H) | −0.339, reversed in 10.7 % of the near wake | reversed |
| speed-up over the cube's roof | 1.076 | > 1 |
| westerly against southerly over the same cube, rotated | 3.6e-7 | ≤ 1 % |
| field at 2, 5 and 10 m/s against one field scaled | 1.1e-5 | ≤ 2 % |
| curl of solid-body rotation over 2Ω | 1.0 | 1 |

**The steady state depends on the pseudo-time step.** Semi-Lagrangian interpolation diffuses in
proportion to the step, so the settled median speed 4 m above a 40-acre campus, per 1 m/s at the
reference, is 0.198, 0.189 and about 0.17 (still falling) at CFL 8, 4 and 2. Fields are delivered at
CFL 8 and the dependence is stated rather than hidden; a step-independent steady solver is the
open item.

## Throughput

`python3 cli.py benchmark`, projection to 1e-6 ([`docs/`](docs/)`benchmark_*.json`):

| device | dtype | cells | projection | one momentum step |
|---|---|---|---|---|
| 4 CPU cores | float64 | 1.05 M | 2.73 s, 24 iterations | 5.64 s |
| NVIDIA L4 | float64 | 1.05 M | 0.76 s, 24 iterations | 1.20 s |
| NVIDIA L4 | float32 | 9.44 M | 6.7 s, 62 iterations | 7.3 s |

float32 stalls near a true divergence of 5e-5, so the 1e-6 checks are float64.

## Run it

```bash
cd models/wind
python3 -m pip install --user -r requirements.txt

python3 cli.py simulate  --site campanile --dx 0.8 --synthetic tower --speed 8 --direction 290
python3 cli.py basis     --site campanile --year 2025 --dx 0.8 --synthetic tower
python3 cli.py verify    --site campanile
python3 cli.py benchmark --site campanile --nx 128 --ny 128 --nz 64 --steps 5
```

`--bundle <dir>` solves a real site from a directory holding `semantics/class_top_<res>.tif` (a uint8
class per column) with `parameters.json` (z0, cd, LAI and closure per class), and `surface/` DTM and DSM
rasters; without one, the 46-class table in [`physics.py`](physics.py#L46) applies.

```bash
python3 -m pytest         # 0 tests, no network and no site data
```

## License

MIT, via the repository root [`LICENSE`](../../LICENSE).
