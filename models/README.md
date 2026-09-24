# DeepEarth Simulator: methods

The equations and code the DeepEarth Simulator runs for sun, wind and water at every cell of a site, and
how flux-tower measurements drive them. Each model is a small, flat Python package on PyTorch that runs
on a CPU or a GPU, with its own tests and validation receipts.

| model | method | validated against | README |
|---|---|---|---|
| water | 2D local-inertial shallow water (Bates et al. 2010); Green-Ampt with redistribution and a per-cell soil state bank (Ogden and Saghafian 1997) | Green-Ampt's exact solution, an independent solve of Smith et al. (1993), USGS gauge 02234400 through Hurricane Ian | [hydro](hydro/README.md) |
| wind | mass-consistent projection, then pseudo-time relaxation with mixing-length turbulence, canopy drag and log-law walls, run until the near-ground flow settles | divergence to 1e-6 s⁻¹, bit-exact log profile, bluff-body wake, rotation invariance, linearity in speed | [wind](wind/README.md) |
| solar | horizon ray march, sky-view factor, beam, diffuse and ground reflection per point | *being published* | [solar](solar/README.md) |

## How a flux tower enters

One measured variable per simulation, from the nearest tower's hourly record (FLUXNET):

| simulation | tower variable | role | validated against the tower |
|---|---|---|---|
| solar | `SW_IN`, incoming shortwave | the sky: its clear-sky index sets beam and diffuse every hour | not yet: open level ground returns `SW_IN` by construction; the value at the sensor's own position is the open check |
| wind | `WS`, wind speed at the sonic | the reference: each heading's unit-speed field is scaled so the speed at the sonic equals `WS` | not yet: equal at the sonic by construction; no independent point |
| water | `P`, precipitation | the rain on every cell, hour by hour | the storm solver against a stream gauge (hydro); against the tower's evapotranspiration, not yet published |

## Not in this repository

The platform that turns an address into a site is not published: 3D reconstruction, the fusion of LiDAR,
aerial imagery and towers into one frame, ordering, the pipeline and its infrastructure. Each package
here documents the inputs it reads, so a site prepared by other means runs the same.

## Credit

The water model began as **Qin Huang**'s hydro package (September 2026): the shallow-water solver, terrain
conditioning, storm forcing, flood probability, gauge validation and the 3D viewer. That history is this
branch's first commits, and the solver here is its descendant, held to it numerically by
`hydro/tests/test_parity.py`.

## License

MIT, via the repository root [`LICENSE`](../LICENSE).
