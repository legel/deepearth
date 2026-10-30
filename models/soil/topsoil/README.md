# Topsoil

pH, texture and organic matter of the top 30 cm for a site, on its 1 m grid. The recipe is the one an independent
benchmark on NEON's soil plots supports (leave one site out): SOLUS100 for pH, sand and clay; a two-product ensemble
with a terrain correction for organic carbon. The soil survey (SSURGO) names the map units and gives each its
drainage class.

## Equations

| quantity | equation | code |
|---|---|---|
| a layer's value | $\bar x_{0,30} = \dfrac{\sum_l x_l\, t_l}{\sum_l t_l}$ over the layers 0 to 5, 5 to 15 and 15 to 30 cm, each the mean of SOLUS100's predictions at its bounding depths (USDA-NRCS, 100 m; Nauman et al. 2024), bilinear to the 3DEP 1/3 arc-second grid, then to 1 m | [`products.py` `interval_mean`](products.py#L94), [`site_layers.py` `layers`](site_layers.py#L55) |
| pH, sand, clay | SOLUS100, as above | [`site_layers.py` `layers`](site_layers.py#L55) |
| organic carbon | $\log_{10} C = \tfrac12\left(\log_{10} C_{SOLUS} + \log_{10} C_{SG}\right) + \sum_j \beta_j \dfrac{t_j - \bar t_j}{\sigma_j}$, $C_{SG}$ SoilGrids 2.0 (250 m), $t_j$ 10 m terrain covariates centred on the site, $\beta_j$ a ridge regression on site-centred NEON residuals; where SoilGrids is masked (urban), $\log_{10} C_{SG} = \log_{10} C_{SOLUS} + \Delta$, $\Delta$ the NEON-calibrated offset | [`site_layers.py` `layers`](site_layers.py#L88), [`fit`](site_layers.py#L28) |
| organic matter | $OM = 1.724\, C$ | [`scene.py`](scene.py#L206) |
| texture | silt $= 100 -$ sand $-$ clay; the USDA texture triangle's 12 classes (Soil Survey Manual 2017) | [`texture.py` `texture_class`](texture.py#L10), [`scene.py`](scene.py#L202) |
| drainage index | the map unit's drainage class on 0 (poor) to 3 (well drained), less $0.5\, z(\mathrm{TWI})$, clipped to [0, 3]: a position wetter than the site's mean drains worse | [`scene.py`](scene.py#L229), [`DRAIN`](scene.py#L27) |

## Against NEON's soil plots

From the docstring of [`site_layers.py`](site_layers.py), leave one site out: pH R² 0.81 nationally and 0.18 within a
site (SSURGO −0.35 within a site); organic carbon, log R² 0.59 nationally and 0.13 within a site. Within-site skill
for organic matter is low, so organic matter is shown in absolute agronomic classes (under 2 %, 2 to 4 %, over 4 %).

## Known errors

- SOLUS100 and SoilGrids predict the native soil from its surroundings. Under fill, compaction or imported topsoil
  they give the native soil the ground would hold undisturbed. A soil test measures what is there.
- Plant-available nitrogen, phosphorus and potassium are not served: no national product carries them.

## Layout

`site_layers.py`, `products.py`, `terrain.py` and `texture.py` read the public products remotely (SOLUS100 from its
public bucket, SoilGrids from ISRIC, 3DEP from USGS). `scene.py` places them on a Simulator scene's grid; it reads
the producer's configuration and site classes, which are not part of this export.
