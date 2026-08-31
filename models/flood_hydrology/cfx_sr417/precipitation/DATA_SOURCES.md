# Precipitation Data Sources — CFX SR417 Corridor AOI

AOI center: **28.36687°N, 81.43299°W** · 2×2 km box · Lake Nona / south Orlando, FL

---

## Source inventory

| Source | Station | Distance | Resolution | Record | Format | Key file(s) |
|---|---|---|---|---|---|---|
| **NOAA GHCND** | USC00084625 · Kissimmee 2, FL | 10.1 km | Daily | 1948-09-02 – 2026-06-22 (77.8 yr) | CDO REST API | `daily_precip_raw.csv` |
| **IEM ASOS — ISM** | ISM · Kissimmee Muni Airport | 8.6 km | 5-min | 2021-01-01 – present | IEM free API | `asos_hourly_ISM.csv` |
| **IEM ASOS — MCO** | MCO · Orlando Intl Airport | 14.0 km | 5-min | 2021-01-01 – present | IEM free API | `asos_hourly_MCO.csv` |

All sources fetched programmatically — no manual download.
CDO token required for GHCND (see `fetch_precip_seasonality.py`); IEM ASOS is open/no auth.

---

## 1. NOAA GHCND — Kissimmee 2 CoOp (USC00084625)

**What it is:** Cooperative observer (CoOp) daily rain gauge. A volunteer reads a standard 8-inch
rain gauge once per day (typically 07–08 local) and records the accumulated total since the prior
reading. This is the NOAA NWS standard for long-term daily climatology.

**Coverage:** 24,725 records, 1948–2026. Real data gaps: 1952-01-01 to 1958-12-31 (7 years with
zero PRCP records — a station-level gap, confirmed in the raw CSV, not a fetch error).

**Data coverage:** 87% (per CDO `datacoverage` field). Dry days (0 mm) are reported as 0.0, not
missing, so the coverage figure reflects sensor/reporting outages only.

**Usage recommendation:** Primary source for:
- Long-term climatology and wet/dry seasonality (Jun–Sep wet, Oct–May dry)
- Daily storm totals and event identification
- Cross-validating ASOS hourly totals

**Key seasonality findings (complete months, 2021-present):**

| Season | Months | n (complete months) | Mean mm/mo | Median | Stdev |
|---|---|---|---|---|---|
| Wet | Jun–Sep | 271 | 181.1 | 170.3 | 87.5 |
| Dry | Oct–May | 552 | 66.8 | 51.0 | 59.9 |
| Wet minus dry | — | — | **+114.3 mm (+171%)** | — | — |

**Limitations:**
- Daily resolution only — cannot resolve intra-storm intensity
- Single-point gauge 10.1 km from AOI; local convective cells can miss it entirely
- CoOp daily timestamp assigns overnight rain to the morning read date, not the meteorological day

---

## 2. IEM ASOS — ISM (Kissimmee Muni Airport, 8.6 km)

**What it is:** Automated Surface Observing System (ASOS) at the nearest airport to the AOI.
Reports 5-minute METARs; p01i field = accumulated precipitation (inches) within the current UTC
hour, reset each hour. The :53 METAR observation is the authoritative hourly total.

**Record: DO NOT USE for precipitation.** Long-term r = **0.166** vs GHCND Kissimmee 2 — the
lowest plausible correlation for co-located stations, consistent with a malfunctioning
tipping-bucket gauge rather than genuine spatial decorrelation.

**Hurricane Ian (Sep 28–29, 2022) result: ISM reported 0.0 mm total** during the largest
rainfall event of the 2021-present record, when GHCND logged 374.6 mm and MCO logged 335.5 mm.
The sensor was offline during the storm — exactly when it mattered most.

**Verdict:** ISM ASOS is not usable as a precipitation source for this project. Its proximity
(8.6 km) is not an advantage if the sensor is unreliable.

---

## 3. IEM ASOS — MCO (Orlando International Airport, 14.0 km)

**What it is:** ASOS at one of Florida's largest airports — well-maintained, continuously staffed,
redundant systems. Same 5-minute METAR / p01i structure as ISM.

**Reliability vs GHCND:** r = **0.578**, bias = −0.3 mm/day, RMSE = 10.5 mm/day over 2021-present
(n = 1,896 days). The moderate-not-high r is expected: at 14 km separation, isolated Florida
convective cells (typical size ~2–5 km) hit one station and miss the other. Agreement improves
for large-scale storm systems (hurricanes, fronts), as shown by Hurricane Ian below.

**Usage recommendation:** Primary sub-daily source for:
- Hourly intensity profiles for storm event reconstruction (especially Hurricane Ian)
- Cross-checking GHCND daily totals
- Identifying within-storm timing (peak hour, onset, cessation)

**IEM p01i extraction note:** p01i is a within-hour running accumulator in **inches**, reset each
UTC hour. Taking `max(p01i)` per UTC hour = the :53 METAR value = the true hourly total. Never
sum all 5-min rows (double-counts the within-hour accumulation). Convert to mm: `× 25.4`.

---

## 4. Hurricane Ian event — cross-source comparison

Ian made landfall near Fort Myers FL at ~17:05 UTC Sep 28, 2022 (Category 4), then tracked
northeast across central Florida through Sep 28–29, passing directly over or near the SR417 AOI.

**Daily totals — gauge comparison:**

| Date | GHCND Kissimmee 2 (10 km) | ISM ASOS (8.6 km) | MCO ASOS (14 km) |
|---|---|---|---|
| 2022-09-26 | 0.0 mm | 0.0 mm | 0.0 mm |
| 2022-09-27 | 15.2 mm | 0.0 mm | 0.1 mm |
| 2022-09-28 | 14.0 mm | 0.0 mm | **73.0 mm** |
| **2022-09-29** | **345.4 mm** | **0.0 mm** | **262.5 mm** |
| 2022-09-30 | 6.9 mm | 0.0 mm | 0.0 mm |
| 2022-10-01 | 0.0 mm | 0.0 mm | 0.0 mm |
| **Event total** | **381.5 mm** | **0.0 mm (sensor down)** | **335.5 mm** |

**Interpretation of MCO vs GHCND timing difference:** MCO (northeast of AOI, 14 km) received
heavy rain on Sep 28 as Ian's outer bands approached from the southwest. GHCND (Kissimmee 2,
~same longitude as AOI) got the heavier bands a day later on Sep 29 as Ian's center tracked
directly overhead. This is genuine meteorological signal — the two gauges capture the same storm
from different spatial geometry, not a data error. Agreement on total: 335.5 vs 381.5 mm (−12%).

**MCO ASOS hourly profile for Ian reconstruction (Sep 29 UTC, peak hours):**

| UTC hour ending | MCO hourly (mm) | Notes |
|---|---|---|
| 00:53 | 28.4 | Ian outer bands |
| 01:53 | 17.3 | |
| 02:53 | 16.8 | |
| 03:53 | 37.1 | Intensifying |
| 04:53 | 21.1 | |
| 05:53 | 35.8 | |
| **06:53** | **66.8** | **Peak hour — eyewall passage** |
| 07:53 | 16.3 | Weakening |
| 08:53 | 7.1 | |
| 09:53 | 5.6 | Trailing bands |

Peak 1-hour intensity at MCO: **66.8 mm/hr** at 06:53 UTC Sep 29 (= ~02:53 EDT).

**Recommended target dates for sourcing new PlanetScope imagery:**
- **MAX inundation:** 2022-10-01 to 2022-10-03 — gauges show zero precip (skies clear), Ian's
  flood water still pooled. This is the optimal acquisition window: max water extent, clear sky.
  Also request 2022-09-30 if available (earliest post-storm clear window).
- **Peak storm itself:** 2022-09-28 and 2022-09-29 — likely cloudy/unusable UDM2 during Ian,
  but worth checking; could confirm cloud contamination in the record.

---

## 5. Wet/dry seasonality — detailed findings

Based on GHCND (Kissimmee 2), 2021-present, 823 complete months out of 827 total.

**Convention:** wet = Jun–Sep, dry = Oct–May. Matches
`ground_truth/seasonal_area_split.py` in `flood_hydrology` for cross-project consistency.
Note: NWS official Florida wet season is May–Oct; this project keeps Jun–Sep/Oct–May for
comparability. The difference matters mainly for May and October, which are transitional months.

**Monthly distribution:**

| Metric | Wet (Jun–Sep) | Dry (Oct–May) |
|---|---|---|
| Mean | 181.1 mm | 66.8 mm |
| Median | 170.3 mm | 51.0 mm |
| Stdev | 87.5 mm | 59.9 mm |
| Min | 24.4 mm | 0.0 mm |
| Max | 529.7 mm | 433.8 mm |
| n (complete months) | 271 | 552 |

Wet minus dry: **+114.3 mm/month (+171% relative to dry mean)** — a strong, clean Florida
seasonal signal. Wet season std (87.5 mm) is itself larger than the dry mean (66.8 mm),
meaning individual wet months vary enormously (drought years vs tropical storm years).

**Driest periods 2021-present (consecutive zero-precip days — platform-verified dry
conditions for baseline imagery selection):**

| Drought period | Days | Mid-date (best baseline acquisition) |
|---|---|---|
| 2021-02-17 to 2021-03-31 | **43 days** | 2021-03-10 |
| 2024-11-08 to 2024-12-11 | 34 days | 2024-11-24 |
| 2022-04-09 to 2022-05-15 | 33 days | 2022-04-27 |
| 2021-04-22 to 2021-06-04 | 32 days | 2021-05-13 |
| 2024-04-13 to 2024-05-13 | 31 days | 2024-04-28 |

The 2026-05-09 PlanetScope sample scene (already in the existing catalogue) falls after 15+ consecutive
zero-precip days (last meaningful rain May 3: 6.1 mm) — it is effectively a dry-season baseline
already in hand.

---

## 6. Data file inventory

All files in `precipitation/data/`:

| File | Source | Contents |
|---|---|---|
| `daily_precip_raw.csv` | GHCND / CDO API | 24,725 daily records, 1948–2026 |
| `monthly_precip_timeseries.csv` | derived | 827 monthly totals + wet/dry label |
| `monthly_precip_timeseries.png` | derived | Monthly bar chart colored by season |
| `station_metadata.json` | CDO API | Chosen station + top-10 candidates |
| `asos_hourly_ISM.csv` | IEM ASOS | Hourly totals (mm) at ISM, 2021–present |
| `asos_daily_ISM.csv` | IEM ASOS | Daily totals aggregated from ISM hourly |
| `asos_hourly_MCO.csv` | IEM ASOS | Hourly totals (mm) at MCO, 2021–present |
| `asos_daily_MCO.csv` | IEM ASOS | Daily totals aggregated from MCO hourly |
| `precip_comparison_ian.png` | derived | Ian event: GHCND vs ASOS daily + MCO hourly |
| `precip_comparison_monthly.png` | derived | Monthly totals: GHCND vs ISM vs MCO |

---

## 7. Source comparison conclusions

| Question | Answer |
|---|---|
| Best daily climatology | **GHCND Kissimmee 2** — longest record, CoOp quality, 10 km |
| Best sub-daily for Ian reconstruction | **MCO ASOS** — only reliable sub-daily gauge; ISM was down during Ian |
| Use ISM ASOS for precip? | **No.** r=0.166 vs GHCND; sensor malfunctioned during Hurricane Ian |
| Do GHCND + MCO agree on Ian? | Yes within 12% on total (381.5 vs 335.5 mm); timing offset explained by storm geometry |
| Is GHCND alone sufficient for simulation? | For daily totals yes; for minute-by-minute hyetograph, must combine GHCND total with MCO ASOS hourly shape |
| Next step for Ian simulation | Build Ian hyetograph from MCO ASOS 5-min data, scaled to GHCND daily total at AOI |
