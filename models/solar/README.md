# Solar

Sunlight at any point of a site, any hour of any year, from the nearest flux tower's measured incoming shortwave
(SW_IN). The tower's hourly SW_IN is split into beam and diffuse against a clear sky from the measured atmosphere,
and each point receives the beam its horizon and the canopy let through, the sky it sees, and light reflected
from the ground. The full account, with every experiment and known error, is [solar_simulation.md](solar_simulation.md).

## Equations

| process | equation | code |
|---|---|---|
| clear sky | REST2 on MERRA-2 aerosol, water vapor and ozone (NSRDB v4), each hour's 5-minute Ineichen-Perez shape scaled to it | [`sky.py` `with_clear_sky`](sky.py#L52) |
| clear-sky index | $k_c = \mathrm{GHI} / \mathrm{GHI}_{cs}$; a daytime hour no source measured takes $k_c$ interpolated between measured hours | [`sky.py` `fill_ghi`](sky.py#L104) |
| diffuse fraction | $k_d = c + \dfrac{1 - c}{1 + e^{b_0 + b_1 k_t + b_2 \mathrm{AST} + b_3 z + b_4 \Delta k_{tc}}} + b_5 k_{de}$, refit on 51 US towers | [`sky.py` `engerer2`](sky.py#L88), [`ENGERER2_US`](sky.py#L20) |
| beam | $\mathrm{DNI} = (\mathrm{GHI} - \mathrm{DHI}) / \cos z$, capped at $1.1\,\mathrm{DNI}_{cs}$; the excess is diffuse, so $\mathrm{DNI}\cos z + \mathrm{DHI} = \mathrm{GHI}$ | [`sky.py` `split`](sky.py#L121) |
| transposition | $A = \mathrm{DNI}/E_0$, $B = \mathrm{DNI} + A\,\mathrm{DHI}/\cos z$, $D = (1 - A)\,\mathrm{DHI}$ (Hay and Davies 1980) | [`transposition.py` `hay_davies`](transposition.py#L23) |
| a point | $E = B\,L \max(\mathbf{n}\cdot\mathbf{s}, 0) + D\,V + \rho\,\mathrm{GHI}\,(1 - n_z)/2$, $\rho = 0.2$ | [`transposition.py` `irradiance`](transposition.py#L31) |
| horizon | ray march over the height field, 64 wedges of 3 rays, samples growing by 3.5 % to 280 m | [`horizon.py` `march`](horizon.py#L64) |
| sky-view factor | $V = \frac{1}{\pi}\iint_{\mathrm{visible}} \max(\mathbf{n}\cdot\boldsymbol{\omega}, 0)\,\sin\theta\,d\theta\,d\phi$ | [`horizon.py` `sky_view_bins`](horizon.py#L96) |
| canopy | $\tau_0 = -\ln\dfrac{n_{\mathrm{first,\,ground}} + 0.5}{n_{\mathrm{first}} + 1}$ per 2 m column; $\tau(t) = \tau_0 + G\Omega(\Delta L(t) - \Delta L_{\mathrm{survey}})\,\tau_0/\overline{\tau_0}$, $G = 0.5$, $\Omega = 0.8$ | [`canopy.py` `columns`](canopy.py#L32), [`seasonal_tau`](canopy.py#L63) |
| through the crown | beam $e^{-\tau/\cos z}$, diffuse $2E_3(\tau)$ | [`canopy.py` `beam_transmittance`](canopy.py#L81) |

## Validation

| check | result |
|---|---|
| beam and diffuse against the tower's GHI, every hour | $\mathrm{DNI}\cos z + \mathrm{DHI} = \mathrm{GHI}$ to 1e-6, including 35 % above clear sky |
| clear sky against Harvard's 2,531 clearest hours, median $k_c$ | 0.991 (Ineichen-Perez on the Linke climatology: 1.061) |
| separation, leave-one-site-out, DHI at 51 US towers | RMSE 53.0 W m⁻², MBE −1.6 (Erbs 57.6, NSRDB 72.4) |

The tower's SW_IN carried 140 m to a second tower, Harvard Forest, 2019 to 2024. Bias is US-xHA minus US-Ha1 (other
minus reference; US-Ha1's mean is 158.3 W m⁻²). US-Ha1's SW_IN is its PAR sensor converted at a fixed ratio, not a
pyranometer:

| SW_IN | hours | bias | RMSE | r |
|---|---|---|---|---|
| hour | 51,570 | −1.6 W m⁻² (−1.0 %) | 24.2 W m⁻² | 0.995 |
| day | 2,099 | −1.6 | 10.3 | 0.995 |
| month | 69 | −1.7 | 6.4 | 0.997 |

Below the canopy, PAR transmission at 12.7 m against HF004 (1991 to 2023), by season:

| | DJF | MAM | JJA | SON |
|---|---|---|---|---|
| measured | 0.49 | 0.54 | 0.16 | 0.19 |
| model | 0.56 | 0.46 | 0.163 | 0.244 |

## Run it

```bash
cd models/solar
python3 -m pip install --user -r requirements.txt
python3 -m pytest         # 12 tests, no network and no site data
```

`sky.sky(year, lat, lon, elevation, ghi, dhi, cs)` turns a tower-year of hourly SW_IN (and SW_DIF where measured,
and NSRDB's hourly clear sky where available) into k_c, beam and diffuse; `horizon.march` and `horizon.sky_view`
give each point's horizon and sky-view factor; `canopy` gives the crown's optical depth and its transmittance.

## License

MIT, via the repository root [`LICENSE`](../../LICENSE).
