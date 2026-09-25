# Wind simulation

The wind model computes a steady three-dimensional wind over a site: terrain, buildings and tree canopy on a grid
of 1 m cubes near the ground, stretched above. It is written in PyTorch and runs on a GPU; a typical site of 10
acres is 10 million cells.

## What is solved

The model finds the steady flow of air driven by a logarithmic wind profile at the domain's sides [inflow]. Two
conditions hold at every cell when it is done.

Mass is conserved. The face velocities are corrected by the gradient of one scalar field, found from a Poisson
equation, so that no cell gains or loses air [project]. Solid cells (buildings, trunks, the ground) have closed
faces.

Momentum balances. Four terms are in the balance [residual]: the transport of momentum by the flow itself, turbulent
mixing, the drag of leaves, and the stress of the wind on solid surfaces.

- Transport uses the volume flux through each face and a face value reconstructed from the two cells upwind of
  it, limited so that it never overshoots its neighbors (MUSCL with the van Leer limiter) [convection].
- Mixing is an eddy viscosity from Prandtl's mixing length: (0.4 h)² times the strain rate, where h is the height
  above the nearest surface below [viscosity].
- Leaf drag is c_d a |u| u, with a the leaf area per volume from each class's leaf area index spread over its
  height [drag].
- Wall stress uses the logarithmic law at half a cell from each solid face, with that surface class's roughness
  length [wall].

## How it is solved

The steady state is reached by stepping in pseudo-time. Each step solves a linear system for the change in velocity
whose right side is the steady residual above plus the pressure gradient accumulated from all earlier
projections [step]. When the change is zero, the residual is zero, whatever the step size. The step size only
decides how fast the iteration arrives; it does not change where it arrives.

The system matrix carries implicit diffusion, drag and first-order upwind transport. It is solved by BiCGSTAB with a
multigrid preconditioner [convective]; the projection by conjugate gradients with the same multigrid [cg]. The
momentum work runs in 32-bit floats; the projection keeps 64-bit residuals under a 32-bit multigrid, and the last
projection holds the delivered field's divergence below 1e-9 of the flow. The stencil and smoother are compiled
into single GPU kernels with `torch.compile`.

The eddy viscosity is under-relaxed, half the new value and half the previous, between steps. Without this the
mixing length and the strain feed back on each other and the iteration oscillates between two states on
alternate steps; the relaxation removes the oscillation and does not move the steady state.

A heading is done when the median, the 95th percentile and the RMS of the horizontal speed 2 to 30 m above the
surface each move less than 0.2 %, 0.2 % and 1 % between two successive 40-step means, twice running [settle].

## One field per heading, then any wind

Every steady term scales with the square of the velocity, so a field solved at 1 m/s scales to any speed: at 2, 5
and 10 m/s the scaled field matches a direct solve to 1.4e-5. Each site is solved once at 1 m/s for 16 headings,
22.5° apart. An hour's field is the two nearest headings blended and scaled to the hour's reference speed. Midway
between two headings the blend differs from a direct solve by 14 % of the peak in the worst cell, 3.4 % at 10°
spacing.

## How the tower sets the value

The reference speed is the flux tower's measured wind speed, WS. Where the solved domain reaches the tower's sonic
anemometer, the reference is chosen so that the simulated speed at the sonic's position and height equals WS; the
two agree by construction, up to the blend between headings. A tower outside the domain is carried to 10 m over
short grass through a 60 m blending height (the Wieringa transfer), with the roughness length fitted per 30°
sector from the tower's own near-neutral hours. Neither path is yet checked against an independent tower.

## Experiments

Physics checks, 64-bit, 1 m cells, 100 steps [verification]:

    largest cell divergence, cube, ridge, flat          3.4e-7, 1.9e-7, 2.7e-13 per second
    mass flux in against out                            2.3e-9, 1.0e-9, 3.4e-15
    open ground returns the inflow profile              to 6.3e-6 of its peak
    speed inside canopy, LAI 0.5, 2, 8, against none    0.967, 0.886, 0.702 times
    behind an 8 m cube, min u over u at its height      -0.366, reversed in 12.7 % of the near wake
    speed-up over the cube's roof, beside it            1.077, 1.87
    the same cube with the wind rotated 90°             1.1e-5
    wake of a porous tower, wind along the grid and 45° 1.0 % apart

Step independence. Harvard Forest, heading 270°, 10.6 million cells. Medians of the published levels at CFL 8, 32
and 64, against CFL 16 (CFL is the step times the top speed over the cell):

                 CFL 8     CFL 32    CFL 64
    4 m          +0.14 %   -0.40 %   -0.80 %
    5 m          +0.09 %   -0.28 %   -0.60 %
    10 m         +0.04 %   -0.14 %   -0.31 %
    25 m          0.00 %   -0.05 %   -0.15 %

The 95th percentiles agree within 0.2 %. The same run on an L4 and on an A100 agrees to 0.01 %. The earlier
semi-Lagrangian scheme settled to a field that did depend on its step: on the UC Berkeley campus its 4 m median
per 1 m/s was 0.189, 0.198 and 0.225 at CFL 4, 8 and 32.

The earlier scheme against the converged finite volumes, medians of the four published levels (4, 5, 10, 25 m):

    Harvard Forest     -15 %, -16 %, -18 %, -18 %
    UC Berkeley        +5 %, +12 %, +22 %, +15 %

Cost of one heading at 1 m (one GPU, list prices):

                                         steps   seconds   USD
    Harvard, 10.6 M cells, L4            160     241       0.057
    Harvard, A100 80 GB                  160     129       0.105
    UC Berkeley, 42.6 M cells, A100      320     418       0.34
    earlier scheme, UC, H100 (64-bit)    160     1,996     2.61

## Known errors

The 1 m grid is not converged near the ground. Against a 2 m grid, the 1 m medians change by 29 % at 4 m, 24 % at 5
m, 12 % at 10 m and 6 % at 25 m; a 0.5 m comparison has not run to completion.

The canopy's wake cells keep moving by about 1 % of the median between successive means after the level
statistics have settled; the steady residual falls to 3 to 10 % of its first value and holds there. The delivered
field is the mean over the last 40 steps.

Step independence has been measured at Harvard; at UC Berkeley it is running. The tower comparison of the
simulated wind at a second, independent sonic has not been made.

[inflow]: https://github.com/legel/deepearth/blob/b9b5b1c/models/wind/forcing.py#L24
[project]: https://github.com/legel/deepearth/blob/b9b5b1c/models/wind/solver.py#L380
[residual]: https://github.com/legel/deepearth/blob/b9b5b1c/models/wind/solver.py#L557
[convection]: https://github.com/legel/deepearth/blob/b9b5b1c/models/wind/solver.py#L536
[viscosity]: https://github.com/legel/deepearth/blob/b9b5b1c/models/wind/solver.py#L480
[drag]: https://github.com/legel/deepearth/blob/b9b5b1c/models/wind/physics.py#L97
[wall]: https://github.com/legel/deepearth/blob/b9b5b1c/models/wind/solver.py#L205
[step]: https://github.com/legel/deepearth/blob/b9b5b1c/models/wind/solver.py#L648
[convective]: https://github.com/legel/deepearth/blob/b9b5b1c/models/wind/poisson.py#L154
[cg]: https://github.com/legel/deepearth/blob/b9b5b1c/models/wind/poisson.py#L263
[settle]: https://github.com/legel/deepearth/blob/b9b5b1c/models/wind/solver.py#L635
[verification]: https://github.com/legel/deepearth/blob/b9b5b1c/models/wind/docs/verification_cpu.json
