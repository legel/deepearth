# gNATSGO Cross-Check — CFX SR417 Corridor AOI

Investigated 2026-06-29 as a cross-check against USDA gNATSGO ("double check for soil
dataset"). AOI: 28.36687N, -81.43299W, 1.0 km radius. Compared against the
9 SSURGO map units already ingested by `soil/ssurgo_download.py`
(mukeys 323133, 323139, 323143, 323148, 323151, 323156, 323157, 323159, 323176).

## Bottom line

**gNATSGO adds no new coverage for this AOI** (it already has complete live SSURGO coverage —
no STATSGO gap-filling needed). **gNATSGO's signature value-add table, Valu1, is NOT queryable
through the Soil Data Access (SDA) REST API** that `ssurgo_download.py` already uses — confirmed
empirically (see below), not just from documentation. Pulling it would require a full-state
gNATSGO File Geodatabase download from NRCS's `gdg.sc.egov.usda.gov` gateway (multi-GB,
whole-Florida extent) — out of scope for today per task instructions.

As a lighter-weight substitute, this check instead queried **`muaggatt`**, the standard
(non-gNATSGO-specific) SSURGO map-unit-aggregate table, which *is* SDA-queryable and provides
an independently-computed, dominant-condition-weighted view of several of the same properties
Valu1 packages (HSG, drainage class, available water storage, hydric %). This served as the
practical cross-check.

## Part 1 — Is Valu1 SDA-queryable? No (confirmed empirically)

Direct SQL POST to `https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest`:

```sql
SELECT TOP 5 * FROM Valu1
```
→ `Invalid query: Invalid object name 'Valu1'.` (HTTP 400)

Same result for `valu1` (case-insensitive SQL Server, so this isn't a casing issue — the table
genuinely does not exist in the SDA-exposed schema). SDA's own published "Advanced Queries"
documentation (`sdmdataaccess.sc.egov.usda.gov/documents/AdvancedQueries.html`) also does not
list Valu1, gNATSGO, or gSSURGO among its tables/tabular-functions/macros — only the standard
SSURGO tables (`legend`, `mapunit`, `component`, `chorizon`, spatial `mupolygon`/`mupoint`/etc.,
and AOI helper tables). This matches NRCS's stated distribution model: Valu1 ships bundled
*inside* the gSSURGO/gNATSGO state File Geodatabases (via `gdg.sc.egov.usda.gov` or Web Soil
Survey bulk download), not as a live queryable web-service table.

## Part 2 — muaggatt cross-check (SDA-queryable, live-queried for our 9 mukeys)

```sql
SELECT mukey, muname, hydgrpdcd, drclassdcd, drclasswettest, hydclprs,
       aws025wta, aws050wta, aws0100wta, aws0150wta, wtdepannmin, pondfreqprs
FROM muaggatt WHERE mukey IN (323133,323139,323143,323148,323151,323156,323157,323159,323176)
```

| mukey | muname | hydgrpdcd (muaggatt, map-unit dominant-condition) | our hsg_raw (ssurgo_download.py, dominant *component*) | Match? | drclassdcd | hydric % (hydclprs) | AWS 0-150cm (cm) |
|---|---|---|---|---|---|---|---|
| 323133 | Immokalee fine sand | B/D | B/D | ✓ | Poorly drained | 14% | 14.9 |
| 323139 | Ona fine sand | B/D | B/D | ✓ | Poorly drained | 5% | 14.73 |
| 323143 | Basinger fine sand, frequently ponded | A/D | A/D | ✓ | Poorly drained | 95% | 14.8 |
| 323148 | Pomello fine sand | A | A | ✓ | Moderately well drained | 0% | 14.31 |
| 323151 | St. Johns fine sand | B/D | B/D | ✓ | Poorly drained | 30% | 17.94 |
| 323156 | Samsula-Hontoon-Basinger, depressional | A/D | A/D | ✓ | Very poorly drained | 96% | 31.79 |
| 323157 | Sanibel muck | A/D | A/D | ✓ | Very poorly drained | 100% | 21.16 |
| 323159 | Smyrna, wet, fine sand | A/D | A/D | ✓ | Poorly drained | 23% | 9.65 |
| 323176 | Water | *None* (no HSG — water has no soil) | **B** (script default) | **✗ see finding below** | *None* | 0% | *None* |

**Result: HSG matches for all 8 real soil map units.** `ssurgo_download.py`'s per-dominant-
component approach agrees exactly with SDA's independently-computed, whole-map-unit
dominant-condition aggregate (`muaggatt.hydgrpdcd`) — this validates the HSG assignment
methodology already in place, and by extension the CN values derived from it (CN itself is not
a stored soil property in either muaggatt or Valu1 — it's a TR-55 lookup keyed on HSG × land
use, computed the same way by us as it would be from gNATSGO — so there is no independent
"gNATSGO CN" to diff against; the meaningful check is the HSG input to that lookup, which
checks out).

### Discrepancy found: mukey 323176 ("Water")

`ssurgo_download.py`'s `build_soil_parameters()` has no real `hydgrp` for the Water map unit
(SSURGO's `component` table returns a blank `hydgrp` for the single 100%-Water component — see
`ssurgo_components.csv` row `26973623,Water,100,,,,`). The code's fallback
(`if not hsg_raw: hsg_raw = "B"`) silently assigns **HSG B** to open water, which then drives a
residential CN of 68 and Horton infiltration parameters (`fc=10 mm/hr, f0=35 mm/hr`) — both
physically wrong for a water surface. `muaggatt` independently confirms the correct answer is
"no HSG" (all aggregate fields are null for this map unit, as expected — water isn't soil).
The project's own `cn_by_hsg.csv` already has a correct `water_body` CN=0 row that isn't
currently being used for mukey 323176. **Recommend**: special-case `mukey == water` (or
`hsg_raw` blank) in `build_soil_parameters()` to use `CN_WATER_BODY` and skip/zero Horton
infiltration, rather than falling through to the generic HSG-B default. Not fixed in this pass
(out of scope for a documentation/comparison task) — flagging for a follow-up edit.

### K-factor (erosion) — also SDA-queryable via chorizon, relevant to slope-stabilization scope

Not in muaggatt, but queried `chorizon.kwfact`/`kffact` (surface horizon, `hzdept_r=0`) for the
dominant components — directly relevant to this project's erosion/slope-stabilization goal:

- Sandy surface horizons (Immokalee, Ona, Basinger, Pomello, St. Johns, Smyrna): **K = 0.02–0.05**
  (low erodibility — expected for excessively-drained fine sands).
- Organic/muck surface horizons (Sanibel, Samsula, Hontoon components): **K = None** — RUSLE
  K-factor is undefined for organic soils in SSURGO, a real data gap to account for (not a bug)
  if/when erosion equations are extended across the full AOI; muck areas would need a different
  erodibility treatment than the K-factor lookup.

### Manual spot-check (citable, non-SDA source)

Per the "if not SDA-queryable, do lighter-weight validation" fallback: the NRCS Official Series
Description for Immokalee (`https://soilseries.sc.egov.usda.gov/OSD_Docs/I/IMMOKALEE.html`,
mukey 323133, our AOI's dominant map unit at 82% comppct) states: *"Drainage class: Poorly
drained; very poorly drained in depressional and ponded phases."* This matches `muaggatt`'s
`drclassdcd = "Poorly drained"` exactly, and is consistent with the B/D (drained/wet) HSG dual
classification already used in `soil_parameters.json`.

## Conclusion

1. **gNATSGO's Valu1 table is not usable today without a large bulk geodatabase download** —
   confirmed by direct SDA query failure, not just documentation. Not pursued further (per
   task scope).
2. **The `muaggatt` SDA table is a usable substitute for the authoritative independent check
   gNATSGO was expected to provide**, and it validates `ssurgo_download.py`'s current HSG/CN
   methodology for all 8 real soil map units — no discrepancy.
3. **One real, fixable bug found**: mukey 323176 ("Water") gets an incorrect HSG-B / CN-68
   default instead of using the project's own `CN_WATER_BODY=0` path. Recommend a small
   follow-up edit to `ssurgo_download.py`.
4. K-factor is available via SDA (`chorizon.kwfact/kffact`) for future erosion work, but is
   undefined for the AOI's organic/muck components — a genuine data gap to plan around, not
   a script bug.
