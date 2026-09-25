# Wind simulation

The wind model computes a steady three-dimensional wind over a site: terrain, buildings and tree canopy on a grid
of 1 m cubes near the ground, stretched above. It is written in PyTorch and runs on a GPU; a typical site of 10
acres is 10 million cells.

## What is solved

The model finds the steady flow of air over a site. At the domain's sides it carries the steady wind of a
horizontally uniform column over the site's mean canopy: the same momentum balance with nothing varying across,
solved first, with the canopy drag averaged over the fluid cells at each height above their ground [column]. Each
side cell takes the column's speed at its own height above its ground [sides], and the top holds the speed of the
upwind logarithmic profile at the domain's top [inflow]. The column is a steady state of the model over a uniform
canopy, so the canopy does not keep slowing the inflow downwind. A logarithmic profile at the sides is not one, and
the canopy slowed it for hundreds of meters (see Experiments). The solved square reaches at least 128 m beyond the
site. Two conditions hold at every cell when the solve is done.

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
surface each move less than 0.2 %, 0.2 % and 1 % between two successive 40-step means, twice running [settle]. How
far that stop is from the fixed point is measured under Experiments. A stop on the estimated change still to come
is also available [tail].

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

Stopping error. Harvard Forest, heading 270°, run to a 40-step change of 0.02 %: CFL 8, 16 and 32 agree within 0.1 %
at every published level's median and p95, and the 0.2 % stop (160 steps) is within 0.2 % of them. UC Berkeley,
heading 270°, CFL 16: the 0.2 % stop (320 steps) against the same run at 1,000 steps, median and p95:

    4 m          +0.21 %   +0.27 %
    5 m          +0.55 %   +0.41 %
    10 m         +2.06 %   +0.82 %
    25 m         +1.75 %   +0.67 %

At 1,000 steps UC still moves 0.03 % a window at 10 m; the window changes decay by about 0.92 a window, which puts
0.1 to 0.2 % more beyond it. The levels fall monotonically in every window and the window-to-window change decays
smoothly, from 9 % to 0.04 %: a steady solution with a slow mode, not an unsteady wake. Tighter inner solves, a
warm start from a settled 4 m solve and Anderson acceleration (depth 5) each left that tail as it was.

Fetch. Harvard Forest, heading 270°, 1 m cells, medians per 1 m/s at 4, 5, 10 and 25 m against the side of the
solved square, with the earlier logarithmic inflow and with the column:

                      256 m                     384 m                     512 m                     768 m
    log law     .173 .188 .280 .725       .125 .140 .238 .674       .088 .102 .204 .622       .069 .080 .179 .588
    column      .066 .085 .188 .577       .074 .087 .186 .591       .069 .082 .184 .589       .069 .080 .179 .585

With the column, 512 m is within 2.6 % of 768 m at every level; with the log law the 4 m median fell 44 % between
384 and 768 m. On one 512 m square, 2 m cells against 1 m change the four medians by +0.5, -2.9, -3.2 and -0.4 %.

Cost of one heading at 1 m (one GPU, list prices):

                                         steps   seconds   USD
    Harvard, 512 m square, 18.9 M cells, L4   280     499       0.12
    Harvard, 384 m square, 10.6 M cells, L4   160     241       0.057
    Harvard, 384 m square, A100 80 GB         160     129       0.105
    UC Berkeley, 42.6 M cells, A100           320     418       0.34
    earlier scheme, UC, H100 (64-bit)         160     1,996     2.61

## Known errors

Before the column inflow, the wind below the canopy was overstated about 1.5 times at 4 m: on Harvard's 384 m square
the 4 m median over the 25 m median was 0.185, and with the column on a 512 m square it is 0.117. The fetch beyond the
survey is open ground in the model, and the column is the site's mean canopy, so the column is right for a site
inside a wide canopy and approximate at a canopy's edge. The residual fetch dependence from 512 to 768 m is at most
2.6 %. The modeled profile has not yet been scored against a tower's own profile.

At UC Berkeley the stop leaves the 10 and 25 m medians 0.7 to 2.1 % high and the 4 and 5 m medians at most 0.6 %
high (Stopping error).

The canopy's wake cells keep moving by about 1 % of the median between successive means after the level
statistics have settled; the steady residual falls to 3 to 10 % of its first value and holds there. The delivered
field is the mean over the last 40 steps.

The tower comparison of the simulated wind at a second, independent sonic has not been made.

[inflow]: https://github.com/legel/deepearth/blob/33490a5/models/wind/forcing.py#L24
[column]: https://github.com/legel/deepearth/blob/33490a5/models/wind/solver.py#L397
[sides]: https://github.com/legel/deepearth/blob/33490a5/models/wind/solver.py#L448
[project]: https://github.com/legel/deepearth/blob/33490a5/models/wind/solver.py#L574
[residual]: https://github.com/legel/deepearth/blob/33490a5/models/wind/solver.py#L751
[convection]: https://github.com/legel/deepearth/blob/33490a5/models/wind/solver.py#L730
[viscosity]: https://github.com/legel/deepearth/blob/33490a5/models/wind/solver.py#L674
[drag]: https://github.com/legel/deepearth/blob/33490a5/models/wind/physics.py#L97
[wall]: https://github.com/legel/deepearth/blob/33490a5/models/wind/solver.py#L311
[step]: https://github.com/legel/deepearth/blob/33490a5/models/wind/solver.py#L883
[convective]: https://github.com/legel/deepearth/blob/33490a5/models/wind/poisson.py#L154
[cg]: https://github.com/legel/deepearth/blob/33490a5/models/wind/poisson.py#L263
[settle]: https://github.com/legel/deepearth/blob/33490a5/models/wind/solver.py#L847
[tail]: https://github.com/legel/deepearth/blob/33490a5/models/wind/solver.py#L200
[verification]: https://github.com/legel/deepearth/blob/33490a5/models/wind/docs/verification_cpu.json
