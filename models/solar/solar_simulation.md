# Solar simulation

The solar model gives the sunlight at any point of a site, for any hour of any year the nearest flux tower has
measured. The tower supplies one number per hour: incoming shortwave on a level sensor above the canopy, SW_IN
(the global horizontal irradiance, GHI). The model divides it into direct beam and diffuse sky light and then
lets each point receive what its surroundings allow.

## The sky from one measurement

A clear-sky reference comes first. Where NSRDB v4 publishes it for the hour, the clear sky is REST2 (Gueymard 2008)
driven by the measured atmosphere of MERRA-2: aerosol optical depth, water vapor and ozone. Its hourly values are
spread over the hour's twelve 5-minute sun positions in the shape of the Ineichen-Perez clear sky, computed with
pvlib and the NREL solar position algorithm [clear-sky]. Before 1998, or outside NSRDB, Ineichen-Perez with SoDa's
monthly Linke turbidity is used and the year is marked.

The clear-sky index k_c = GHI / GHI_clear says how much of a clear day's light arrived. An hour that no source
measured in daylight takes k_c interpolated between the nearest measured hours, times its own clear sky, and is
flagged [fill]. Daytime light is never zero for want of data; night stays dark.

The diffuse share of GHI is measured where the tower measures diffuse (few do). Elsewhere it is the Engerer2
separation model (Engerer 2015), whose seven coefficients we refit on 518,865 hours of measured diffuse at 51 US
towers on the REST2 clear sky [engerer2], [coefficients]. The beam is what the diffuse leaves, divided by the cosine
of the sun's zenith, capped at 1.1 times the clear-sky beam; any light the cap withholds is counted as diffuse
[split]. The beam and diffuse therefore always add back to the tower's GHI.

## From the sky to a point

The diffuse sky is not uniform: the sky around the sun is brighter. Hay and Davies (1980) fold that circumsolar
part into the beam with the anisotropy index A = DNI / E0 [hay-davies]. A point with unit normal n then receives

    E = (DNI + A DHI / cos z) L max(n.s, 0) + (1 - A) DHI V + 0.2 GHI (1 - n_z) / 2

where L is the share of the sun's direction s that reaches the point, V its sky-view factor, and the last term
the light reflected from ground of albedo 0.2 [point]. On open level ground (L = 1, V = 1, n_z = 1) E is exactly
the tower's GHI.

L and V come from the terrain and buildings. From each point, rays leave in 64 azimuth wedges of three rays each;
each ray samples the height field at distances growing 3.5 % per step out to 280 m, and a wedge's horizon is the
highest elevation angle any sample reaches [march]. V is the cosine-weighted share of the sky above those horizons:
1 for open level ground, 1/2 at the foot of an infinite wall [sky-view].

Trees are not opaque. For each 2 m column, the LiDAR survey's first returns give the canopy's optical depth: the
share of first returns that reach within 1 m of the ground is the gap fraction, and tau0 = -ln(gap) [canopy]. Most
surveys are flown with the leaves off; the leaves of each day are added from MODIS leaf area index (LAI), with
spherical leaf angles (G = 0.5) and the clumping of broadleaf forest (Omega = 0.8, Chen et al. 2005), and a leaf that
has turned stays on the tree ten days (the Harvard Forest phenology record) [season]. The direct beam passes a
crown as exp(-tau / cos z) and the diffuse sky as 2 E3(tau) [transmittance].

## Experiments

The beam and diffuse add back to the tower's GHI to 1e-6 in every hour, including broken-cloud hours 35 % above clear
sky and the lowest sun [tests].

The clear sky, against Harvard's 2,531 clearest hours of 2019 to 2024 (sun above 20°, measured diffuse at most 20 %
of GHI, k_c steady across neighboring hours), median k_c and its 10th to 90th percentiles:

    REST2 on MERRA-2                           0.991   0.962 to 1.019   cap binds on 0.24 % of those hours
    Ineichen-Perez, Linke climatology          1.061   1.013 to 1.113   cap binds on 45 %

The separation, leave one tower out at a time over the 51 towers, error of the hourly diffuse:

    Engerer2, refit                            RMSE 53.0 W m-2, mean bias -1.6
    Erbs, Klein and Duffie (1982)              RMSE 57.6, bias +3.7
    NSRDB's own diffuse                        RMSE 72.4, bias -2.3

One tower's SW_IN standing for a point 150 m away: US-xHA against US-Ha1 at Harvard Forest, 2019 to 2024, over hours
both measured:

                 n        bias                 RMSE          r
    hour         51,570   +1.6 W m-2 (+1.0 %)  24.2 W m-2    0.995
    day          2,099    +1.6                 10.3          0.995
    month        69       +1.7                 6.4           0.997

The hourly RMSE is cloud edges passing one sensor before the other.

Light below the canopy, as a fraction of open-sky PAR, against Harvard Forest's HF004 sensor at 12.7 m, 1991 to 2023:

                     winter   spring   summer   fall
    measured         0.49     0.54     0.16     0.19
    model, 12.7 m    0.56     0.46     0.163    0.244
    model, 0.5 m     0.21     0.22     0.101    0.118

## Known errors

Below the canopy the model is right in summer and off in the other seasons: winter 14 % bright, spring 15 % dark and
fall 28 % bright. MODIS counts green leaves, and oaks keep brown ones into winter. The comparison is a site mean;
HF004 is not a mapped column.

The canopy's transmission is fitted to PAR and applied to all shortwave. Leaves pass far more near infrared than PAR,
so under a summer canopy the model's shortwave reads low by a factor of 1.7 at 12.7 m and 2.1 at 0.5 m; PAR itself is
as the table shows.

The value at the tower's own sensor equals the tower by construction; an independent point below the canopy has not
been measured.

[clear-sky]: https://github.com/legel/deepearth/blob/8670ff2/models/solar/sky.py#L52
[fill]: https://github.com/legel/deepearth/blob/8670ff2/models/solar/sky.py#L104
[engerer2]: https://github.com/legel/deepearth/blob/8670ff2/models/solar/sky.py#L88
[coefficients]: https://github.com/legel/deepearth/blob/8670ff2/models/solar/sky.py#L20
[split]: https://github.com/legel/deepearth/blob/8670ff2/models/solar/sky.py#L121
[hay-davies]: https://github.com/legel/deepearth/blob/8670ff2/models/solar/transposition.py#L23
[point]: https://github.com/legel/deepearth/blob/8670ff2/models/solar/transposition.py#L31
[march]: https://github.com/legel/deepearth/blob/8670ff2/models/solar/horizon.py#L64
[sky-view]: https://github.com/legel/deepearth/blob/8670ff2/models/solar/horizon.py#L96
[canopy]: https://github.com/legel/deepearth/blob/8670ff2/models/solar/canopy.py#L32
[season]: https://github.com/legel/deepearth/blob/8670ff2/models/solar/canopy.py#L63
[transmittance]: https://github.com/legel/deepearth/blob/8670ff2/models/solar/canopy.py#L81
[tests]: https://github.com/legel/deepearth/blob/8670ff2/models/solar/tests/test_solar.py
