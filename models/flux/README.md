# Flux

How the nearest flux tower's measurements become a simulation's forcing. The tower defines the three published
quantities: incoming shortwave (SW_IN), wind speed (WS) and precipitation (P). Each rule below decides which source
drives each hour. Every hour carries a code naming its source, so a filled hour is never shown as measured. The solar,
wind and hydro models read what these rules produce.

## Rules

| rule | condition or equation | code |
|---|---|---|
| measured only | an hour counts as measured only at QC 0 with a value; FLUXNET's gap-fills (QC 1 to 3) take the next source | [`qc.py` `measured`](qc.py#L23), [`layer`](qc.py#L31) |
| source codes | tower 0, fill 1, observed 2, reanalysis 3, tower SW_IN from PAR 5, reference gauge 6, station calm 7 | [`qc.py` `CODES`](qc.py#L14) |
| P, the order | the reference gauge (WMO double-fence or shielded weighing gauge, QC 0), else the vetted tower gauge, else AORC | [`rain.py` `rain`](rain.py#L70) |
| P, an hour | kept only if $p \le 2\,\max(\text{independent sources, 3 h window}) + 1$ mm | [`rain.py` `vet_rain`](rain.py#L36), [`RAIN_HOUR_FACTOR`](rain.py#L22) |
| P, a month | kept only if $\lvert \sum p - \mathrm{median} \rvert \le 0.35\,\mathrm{median}$ of the independent sources' sums over the same hours | [`RAIN_MONTH_BAND`](rain.py#L24) |
| SW_IN | $k_c = \mathrm{SW_{IN}} / \mathrm{GHI}_{cs}$ against REST2 on MERRA-2; a daytime gap takes $k_c$ interpolated between measured hours, marked | [`sky.py` `clear_sky_index`](sky.py#L17), [`fill_ghi`](sky.py#L24) |
| WS, carried | $u_b = u\,\dfrac{\ln((z_b - d_t)/z_{0,t})}{\ln((z_m - d_t)/z_{0,t})}$, $u_{ref} = u_b\,\dfrac{\ln((z_{ref} - d_s)/z_{0,s})}{\ln((z_b - d_s)/z_{0,s})}$, $z_b = 60$ m (Wieringa 1986) | [`wind.py` `transfer`](wind.py#L45) |
| WS, tower roughness | $z_0 = (z - d)\,e^{-\kappa U / u_*}$ per 30° sector from near-neutral hours ($\lvert z/L\rvert < 0.05$, $u_* > 0.2$) | [`wind.py` `z0_by_sector`](wind.py#L28) |
| WS, calm | a station hour reading 0 is a calm report (below 3 kt, 1.54 m/s): kept at 0, coded 7, never measured still air | [`wind.py` `calm_codes`](wind.py#L53) |
| NEON months | where a NEON tower's FLUXNET release has no measured step, NEON's own measurement at the same height takes it, QC 0; never over a measured step | [`neon_fill.py` `fill`](neon_fill.py#L17) |

## Validation: Harvard Forest's gauges

Annual precipitation in mm. The NEON ground gauge is a double-fence weighing gauge (DP1.00044). The model uses the P
rule's output.

| year | US-xHA bucket | US-Ha1 gauge | NEON ground | AORC | Barre Falls COOP | model |
|---|---|---|---|---|---|---|
| 2021 | 1,401 | 1,405 | 1,325 | 1,326 | 1,292 | 1,325 |
| 2022 | 1,024 | 1,142 | 1,196 | 1,163 | 989 | 1,196 |
| 2023 | 1,115 | 1,541 | 1,425 | 1,419 | 1,506 | 1,425 |
| 2024 | 1,533 | 1,439 | 1,280 (85 % of hours) | 1,272 | 1,039 | 1,337 |

- In years it covers fully, the ground gauge matches AORC within 3 %.
- US-xHA's tower-top bucket was flagged good throughout, yet read wrong in both directions:
  - it undercaught February to July 2023 (July: 88 mm, against 226 at the ground gauge);
  - it overcaught convective hours in 2024 (57.5 mm in the half hour of 9 July, against 39.9 mm in the whole hour at
    the ground gauge).
- The hour and month checks remove both errors without reading any flag.

## Sources

| data | provider | license |
|---|---|---|
| tower SW_IN, WS, P, QC | FLUXNET (AmeriFlux FLUXNET-1F), per tower DOI | CC-BY-4.0 |
| months past a release, ground and tower-top gauges | NEON DP1.00014, DP1.00044, DP1.00045 | CC-BY-4.0 (NEON data policy) |
| gridded rain | NOAA AORC v1.1, 1 km hourly | public domain |
| clear sky | NREL NSRDB v4 (REST2 on MERRA-2) | public |
| station wind and calms | NOAA ASOS via Iowa Environmental Mesonet | public |
| daily gauges | NOAA GHCN-Daily cooperative observers | public |

## Run the tests

```bash
cd models/flux && python -m pytest -q
```
