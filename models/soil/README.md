# Soil

The hourly water balance of every ground cell, in PyTorch. It produces the soil water and the standing water a site
shows at any hour of its record. Each cell has:

- a canopy store;
- water standing on the surface;
- two soil layers: the evaporation layer, 0 to 0.1 m, and the root zone below it, down to the cover's root depth.

The layers gain water from rain, run-on and lateral flow. They lose it to evaporation, transpiration and drainage.
Storms at full resolution in two dimensions are the [hydro](../hydro/README.md) solver's job. This model is the
state between and through them, every hour of every year.

## Equations

| process | equation | code |
|---|---|---|
| interception | $c \leftarrow c + \min(P,\ S_{\max} - c)$, $S_{\max} = 0.935 + 0.498\,L - 0.00575\,L^2$, $L = -\ln P_{gap} / 0.5$ from the LiDAR column | [`balance.py` `rain_step`](balance.py#L189), [`soils.py` `s_max`](soils.py#L119), [`lai_from_gap`](soils.py#L124) |
| infiltration capacity, rain hours | $F_1 = F_0 + K_s + \psi_f \Delta\theta \ln\dfrac{F_1 + \psi_f \Delta\theta}{F_0 + \psi_f \Delta\theta}$, $\Delta\theta = \theta_s - \theta_1$; $F$ resets after 6 dry hours | [`green_ampt_capacity`](balance.py#L176), [`F_RESET_H`](balance.py#L28) |
| run-on | each cell in descending filled elevation takes $\min(a\,p,\ \text{cap})$ of what reaches it ($a$: rain plus run-on, $p$: pervious share), fills its depression, and passes the rest to its lower neighbors by $w_j \propto \tan\beta_j\, \ell_j$ | [`routing.py` `cascade`](routing.py#L181), [`network`](routing.py#L138) |
| standing water soaks in | $f_p = \min(w,\ K_s\,p,\ \text{room in both layers})$, layer 1 first | [`local_step`](balance.py#L117) |
| net radiation | $R_n = (1 - \alpha) R_s + \varepsilon\,(L_{in} - \sigma T^4)$ | [`local_step`](balance.py#L117) |
| reference ET | ASCE-EWRI (2005) hourly short reference, $C_d$ 0.24 by day and 0.96 at night, $G$ 0.1 and 0.5 of $R_n$ | [`et0_hourly`](balance.py#L90) |
| losses, in order | the canopy store, then standing water, then $T = K_w K_{cb} E$ and $E_s = K_e E$ on what remains ($E = ET_0 - E_c - E_w$) | [`local_step`](balance.py#L117) |
| water stress | $K_w = \dfrac{TAW - D_r}{(1 - p)\,TAW}$ clipped to [0, 1], $TAW = 1000 (\theta_{fc} - \theta_{wp}) z_r$, $p = 0.5$ (FAO-56 eq. 84) | [`P_STRESS`](balance.py#L27) |
| soil evaporation | $K_e = \min(K_r (K_{c,\max} - K_{cb}),\ f_{ew} K_{c,\max})$, $K_r = \dfrac{TEW - D_e}{TEW - REW}$ (FAO-56 eqs. 71 to 74), from layer 1 down to $\theta_{wp}/2$ | [`local_step`](balance.py#L117) |
| drainage, both layers | $\dfrac{d\theta}{dt} = -\dfrac{K_s S_e^{b}}{1000\,\Delta z}$, $b = 3 + 2/\lambda$, integrated exactly over the hour: $S_e(1\,\mathrm{h})^{1-b} = S_{e0}^{1-b} + \dfrac{(b-1) K_s}{1000\,\Delta z\,(\theta_s - \theta_r)}$, never below $\theta_{fc}$ | [`drain`](balance.py#L102) |
| lateral flow, layer 2 | $Q = \min\!\left(W,\ K(\theta_2) \tan\beta\, \dfrac{\Delta z_2}{\Delta x}\right)$, $K(\theta) = K_s S_e^{3 + 2/\lambda}$, $W = 1000 (\theta_2 - \theta_{fc})^+ \Delta z_2$; receivers fill layer 2, then layer 1, the rest returns to the surface | [`lateral_step`](balance.py#L235), [`routing.py` `lateral`](routing.py#L256) |
| soil water shown | $\theta_{root} = \dfrac{\theta_1 Z_e + \theta_2 (z_r - Z_e)}{z_r}$, $Z_e = 0.1$ m | [`root_zone`](balance.py#L78), [`ZE`](balance.py#L26) |
| standing water shown | $w$, mm: the surface store after the hour's run-on, soak-in, evaporation and return flow, drawn where $w > 0$ | [`local_step`](balance.py#L117), [`lateral_step`](balance.py#L235) |
| soil hydraulics | a survey's measured values where it has them (SSURGO 1/3 and 15 bar, $K_s$, $\theta_s$; POLARIS's Brooks-Corey fit), else the texture's Rawls, Brakensiek and Miller (1983) row; $\theta_{fc}$, $\theta_{wp}$ on Brooks-Corey at 33 and 1500 kPa | [`soils.py` `hydraulics`](soils.py#L65), [`RAWLS`](soils.py#L14) |
| cover | $K_{cb}$, $K_{c,\max}$, root depth, albedo and $f_{ew}$ per cover (FAO-56 Tables 17 and 22) | [`SURFACES`](balance.py#L31) |

The depressions come from Priority-Flood with epsilon (Barnes, Lehman and Mulla 2014), [`fill`](routing.py#L57). The
flow weights are Quinn et al.'s (1991) multiple flow directions, with contour lengths 0.5 and 0.354 cell widths.
Run-on uses the filled terrain. Lateral flow uses the unfilled terrain, so a buried hollow gathers the water above it.
A roof's rain goes to the storm sewer unless the site's downspouts are disconnected. Mass is conserved every hour,
from rain to AET, drainage and water leaving the grid ([`tests/test_balance.py`](tests/test_balance.py)).

Stored hourly frames, as the page draws them: $\theta_{root}$ in steps of 0.0025 m³/m³ (`.rw`), and standing water as
float32 mm per wet cell (`.pw`).

## Validation: Harvard Forest, NEON soil water

Every NEON soil water sensor (DP1.00094.001) at HARV is compared with the model layer that holds its depth at the
sensor's cell, hourly and UTC-aligned, over 2019 to 2025. The run is 0.5 m cells with SSURGO soils and lateral flow.
The rows use flag-0 half-hours only. The anomaly is each value less its own 31-day running mean. A storm's rise is its
peak less the value before it (rain of 5 mm or more, split by 6 dry hours). Receipt:
[`docs/validation_harv_swc.json`](docs/validation_harv_swc.json), which also holds the sensitivity set.

| sensor | depth | layer | hours | r, hourly | r, daily | mean, model / sensor (m³/m³) | r, anomaly | storm rise, model / sensor, median (m³/m³) | storms |
|---|---|---|---|---|---|---|---|---|---|
| 001.501 | 6 cm | 1 | 20,092 | 0.31 | 0.32 | 0.194 / 0.285 | 0.54 | 0.081 / 0.036 | 103 |
| 002.501 | 6 cm | 1 | 8,274 | 0.54 | 0.66 | 0.195 / 0.076 | 0.64 | 0.084 / 0.048 | 30 |
| 003.501 | 7 cm | 1 | 2,237 | 0.37 | 0.81 | 0.187 / 0.207 | 0.57 | 0.081 / 0.161 | 6 |
| 003.502 | 17 cm | 2 | 514 | 0.82 | 0.86 | 0.212 / 0.113 | n too small | 0.004 / 0.162 | 1 |

Three more sensors (004.501, 004.502, 005.502) fall outside the modeled area and are not scored.

Soil water against the terrain over the 168,380 soil cells, at the default hour (1 July, local standard noon) and over the year. The
topographic wetness index is $\ln(a / \tan\beta)$. The columns are Spearman correlations of the root zone's water.

| | TWI | slope | upslope area | cells within one 0.0025 step of the mode, 1 July local standard noon |
|---|---|---|---|---|
| without lateral flow, year mean | −0.08 | +0.03 | −0.08 | 0.84 |
| with lateral flow, year mean | **+0.55** | **−0.42** | **+0.42** | 0.73 |
| with lateral flow, July mean | +0.60 | −0.45 | +0.45 | |

## Known errors

- **The storm goes into the wrong layer.** At 6 cm the model's median rise per storm is 0.081 m³/m³, against the
  sensor's 0.036 over 103 storms (001.501) and 0.048 over 30 (002.501). Each hour's infiltration fills the 10 cm
  top layer before any reaches layer 2, and it passes on only by drainage over the following hours. The same
  hour's water should run deeper through a wetting front. At 17 cm the one clean storm rose 0.004 against 0.162,
  but the sensitivity set's three storms give 0.004 against 0.006, so the deep deficit rests on too few storms to
  size. **Next step:** carry a Green-Ampt wetting front through both layers in the rain hour, with the
  redistribution of Ogden and Saghafian (1997) that the [hydro](../hydro/infiltration.py) solver already runs, in
  place of filling layer 1 first.
- **One soil for the whole site.** Every cell at Harvard reads one SSURGO map unit, so the model's mean at every
  sensor is 0.19 to 0.21 while the sensors read 0.08 to 0.29. The spread between sensors is real and absent from
  the model. Most cells sit within one quantization step (0.0025) of a single value, the drainage attractor of one
  soil's Brooks-Corey curve (b = 8.3, K 0.20 mm/h at the July mode). Terrain now spreads it (above). Soil
  variation finer than the survey unit is not modeled.
- **No terrain signal without lateral flow.** A one-dimensional column cannot put more water in a hollow than on a
  ridge. Before lateral flow, the Spearman correlation with TWI was −0.08. Lateral flow moves only water above
  field capacity, at the layer's own K(θ₂), with isotropic conductivity (`LATERAL_ANISOTROPY` 1). Forest soils often
  conduct faster along the slope, which this does not assume.
- **Arrival within the hour.** Rain enters layer 1 in the hour it falls (model arrival 0 h). The shallow sensors
  respond a median 1 to 2 h later.
- **No water table.** Drainage leaves the root zone for good, and nothing rises from below.

## Run the tests

```bash
cd models/soil && python -m pytest -q
```

Requires `torch`, `numpy` and `numba`.
