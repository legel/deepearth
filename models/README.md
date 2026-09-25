# DeepEarth Simulator: methods

The equations and code the DeepEarth Simulator runs for sun, wind and water at every point of a site, and how
flux-tower measurements drive them. Each model is a small, flat Python package on PyTorch that runs on a CPU or a
GPU, with its own tests, validation receipts and a full account of its method, experiments and known errors.

| model | method | account |
|---|---|---|
| solar | the tower's SW_IN split against a REST2 clear sky (Engerer2 refit on 51 US towers), Hay-Davies transposition, horizons and sky-view factor by ray march, Beer-Lambert light through the canopy | [README](solar/README.md), [solar_simulation.md](solar/solar_simulation.md) |
| wind | mass-consistent projection and a steady finite-volume momentum balance (MUSCL convection, mixing-length turbulence, canopy drag, log-law walls), independent of the pseudo-time step | [README](wind/README.md), [wind_simulation.md](wind/wind_simulation.md) |
| water | 2D local-inertial shallow water (Bates et al. 2010), Green-Ampt with redistribution and a per-cell soil state carried between storms (Ogden and Saghafian 1997) | [README](hydro/README.md), [water_simulation.md](hydro/water_simulation.md) |
| flux | how the tower's measurements become the forcing: the measured-only rule, the P rule over several gauges, the clear-sky index, the sonic carried to the reference height, station calms, NEON months past a release | [README](flux/README.md) |

## How a flux tower enters

One measured variable per simulation, from the nearest tower's hourly record (FLUXNET):

| simulation | tower variable | role | against the tower |
|---|---|---|---|
| solar | `SW_IN` | the sky: its clear-sky index sets beam and diffuse every hour | open level ground returns `SW_IN` to 1e-6; carried 140 m to a second tower, hourly bias −1.0 % (US-xHA minus US-Ha1), RMSE 24.2 W m⁻², r 0.995 |
| wind | `WS` at the sonic | the reference: each heading's unit-speed field is scaled so the speed at the sonic equals `WS` | equal at the sonic by construction; against NEON's tower profile the crown is 1.2 to 1.7 times too fast and the air near the ground 0.5 to 0.7 times too slow (wind_simulation.md) |
| water | `P` | the rain on every cell, hour by hour, under the P rule ([flux](flux/README.md)) | under the P rule, the NEON ground gauge where it measured; in full years it matches AORC within 3 % ([flux](flux/README.md)); the storm solver against a stream gauge (hydro) |
