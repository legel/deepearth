# PlanetScope SuperDove — Derived Visualizations

_Scene `20251204_164906_45_24f6` · generated 2026-06-26 00:47 UTC by `visualize_psscene.py`_

## 1. Dataset summary

Planet **PSScene** delivery, bundle `analytic_8b_sr_udm2`: an 8-band SuperDove (PSB.SD) image as **bottom-of-atmosphere Surface Reflectance** with an accompanying UDM2 usable-data mask.

| Property | Value |
|---|---|
| Item ID | `20251204_164906_45_24f6` |
| Instrument / satellite | PSB.SD / 24f6 |
| Acquired (UTC) | 2025-12-04T16:49:06.455007Z |
| Ground sample distance | 3.3 m (pixel grid 3 m) |
| Raster size | 670 × 670 px, 8 bands, uint16 |
| CRS | EPSG:32617 |
| Bounds (CRS units) | (456567.0, 3136929.0, 458577.0, 3138939.0) |
| Sun elev / azim | 38.8° / 171.5° |
| View angle | 3.6° |
| Cloud / clear | 0% / 100% |
| Quality category | standard |
| UDM2 clear (in-footprint) | 99.5% |
| UDM2 mean confidence | 96.5% |
| Open water (sub-pixel) | 6.2% (24.4 ha; 233 bodies) |

## 2. Original data & paths

| Role | File |
|---|---|
| Surface-reflectance cube (8-band) | `PSScene/20251204_164906_45_24f6_3B_AnalyticMS_SR_8b_clip.tif` |
| UDM2 quality mask (8-band) | `PSScene/20251204_164906_45_24f6_3B_udm2_clip.tif` |
| Band radiometry XML | `PSScene/20251204_164906_45_24f6_3B_AnalyticMS_8b_metadata_clip.xml` |
| Item metadata (STAC props) | `PSScene/20251204_164906_45_24f6_metadata.json` |

**Band layout (SuperDove, 1-indexed).** Surface-reflectance DN are stored as `uint16`; physical reflectance = `DN × 1e-4` (range 0–1). `nodata = 0` marks the clipped scene corners.

| Band | Name | Center λ (nm) | Render colormap | Stretch lo–hi (reflectance) |
|---|---|---|---|---|
| 1 | Coastal Blue | 443 | `cividis` | 0.026 – 0.175 |
| 2 | Blue | 490 | `gray` | 0.022 – 0.185 |
| 3 | Green I | 531 | `gray` | 0.028 – 0.209 |
| 4 | Green | 565 | `gray` | 0.027 – 0.211 |
| 5 | Yellow | 610 | `gray` | 0.023 – 0.227 |
| 6 | Red | 665 | `gray` | 0.021 – 0.236 |
| 7 | Red Edge | 705 | `magma` | 0.022 – 0.244 |
| 8 | NIR | 865 | `inferno` | 0.034 – 0.335 |

<details><summary>Per-band TOA reflectance coefficients (from XML, for reference — the SR product is already atmospherically corrected)</summary>

| Band | reflectanceCoefficient | radiometricScaleFactor |
|---|---|---|
| 1 | 2.5606181241847283e-05 | 0.01 |
| 2 | 2.4729451133816905e-05 | 0.01 |
| 3 | 2.6430072141477853e-05 | 0.01 |
| 4 | 2.682101046959049e-05 | 0.01 |
| 5 | 2.8826289087113963e-05 | 0.01 |
| 6 | 3.228278904260674e-05 | 0.01 |
| 7 | 3.4512298024172836e-05 | 0.01 |
| 8 | 5.109553792292737e-05 | 0.01 |

</details>

## 3. Derived data & products

All rasters are **lossless PNG** (8-bit RGBA; `nodata` → transparent), written to the `derived/` directory (paths below are relative to this README).

| Product | Description |
|---|---|
| `derived/rgb_truecolor.png` | Natural-colour composite (R=Red, G=Green, B=Blue); independent per-channel 2–98% stretch (gray-world white balance) + gamma 1.8. |
| `derived/false_color_nir.png` | Colour-IR composite (R=NIR, G=Red, B=Green); healthy vegetation appears bright red. Same stretch as RGB. |
| `derived/nir_inferno.png` | Band 8 NIR (~865 nm) with the perceptually-uniform `inferno` colormap; reflectance stretched 2–98%. |
| `derived/bands/` | Per-band single-band renderings; colormap chosen per band (`inferno` for NIR, `magma` for red-edge, `cividis` for coastal blue, grayscale luminance for true-colour primaries). |
| `derived/ndvi.png / ndvi_annotated.png` | NDVI = (NIR-Red)/(NIR+Red) (vegetation vigour / greenness); range [-1,1], `RdYlGn`. Scene median +0.517. |
| `derived/ndre.png / ndre_annotated.png` | NDRE = (NIR-RedEdge)/(NIR+RedEdge) (canopy chlorophyll / N status); range [-1,1], `RdYlGn`. Scene median +0.372. |
| `derived/ndwi.png / ndwi_annotated.png` | NDWI = (Green-NIR)/(Green+NIR) (open water (McFeeters)); range [-1,1], `RdBu`. Scene median -0.529. |
| `derived/ndwire.png / ndwire_annotated.png` | NDWIre = (RedEdge-NIR)/(RedEdge+NIR) (red-edge water variant; separates dark water from shadow/wet vegetation); range [-1,1], `RdBu`. Scene median -0.372. |
| `derived/greennirratio.png / greennirratio_annotated.png` | Green/NIR band ratio (`YlGnBu`, scaled to its 98th pct); amplifies dark tannic-water contrast that normalized NDWI loses. |
| `derived/water_fraction.png / water_fraction_annotated.png` | Continuous water-area fraction from NIR linear unmixing (`Blues`); the sub-pixel shoreline is its f=0.4 isoline. |
| `derived/water_mask_hires.png` | Crisp 4x-supersampled sub-pixel water mask (for zoom inspection). |
| `derived/water_mask.png` | Open-water mask, antialiased by per-pixel water-area fraction (sub-pixel f=0.4 shoreline). |
| `derived/water_mask_overlay.png` | Sub-pixel shoreline (cyan) + translucent fractional fill on true colour. |
| `derived/water_boundaries.shp` | Fitted sub-pixel shoreline polygons (EPSG:32617). |
| `derived/water_boundaries.geojson` | Fitted sub-pixel shoreline polygons (EPSG:32617). |
| `derived/water_boundaries_wgs84.geojson` | Fitted sub-pixel shoreline polygons (WGS84). |
| `derived/udm2_class.png` | UDM2 categorical quality map (clear/shadow/snow/haze/cloud). |
| `derived/udm2_confidence.png` | UDM2 per-pixel usable-data confidence (0–100%), `viridis`. |
| `derived/overview.png` | Single-page contact sheet of the headline products. |

## 4. Processing & hyper-parameters

- **Radiometric scaling:** reflectance = `DN × 1e-4` (Planet SR convention).
- **Valid-pixel mask:** pixels non-`nodata` across all bands; nodata is rendered transparent in every output.
- **RGB / false-colour exposure:** independent per-channel percentile stretch at **2–98%** of valid pixels (a gray-world white balance), then **gamma 1.8**. Per-channel reflectance windows used: Red 0.021–0.236, Green 0.027–0.211, Blue 0.022–0.185.
- **Single-band & index renders:** linear 2–98% percentile stretch, gamma 1.0 (radiometrically honest).
- **Colormaps:** perceptually-uniform where possible — `inferno` (NIR, as requested), `magma` (red edge), `cividis` (coastal blue), grayscale for the true-colour primary bands, diverging `RdYlGn`/`RdBu` for indices fixed to [-1, 1].
- **Spectral indices:** NDVI=(B8−B6)/(B8+B6), NDRE=(B8−B7)/(B8+B7), NDWI=(B4−B8)/(B4+B8), NDWIre=(B7−B8)/(B7+B8).

- **Open-water seed (for dark tannic / turbid water that defeats McFeeters NDWI):** a pixel is *seed* water if it is **dark in NIR** (NIR reflectance < 0.16) **and** **green-dominant over NIR** (Green/NIR > 0.48). Low NIR rejects vegetation and bright impervious surfaces; the ratio rejects spectrally-flat shadow. Thresholds are on calibrated surface reflectance and are physically meaningful (`--water-nir-max` / `--water-gnr-min`).
- **Sub-pixel shoreline (the headline product).** A hard threshold pins the boundary to pixel centres and runs ~1–2 px conservative, because shoreline pixels are *mixed* land+water and get thrown wholesale to land. Instead each pixel's **water-area fraction** `f∈[0,1]` is recovered by linear spectral unmixing on NIR, `f = (L − NIR)/(L − W)`, with water endmember **W = 0.039** (robust median NIR of confident interior water) and a *local* land endmember **L** (mean NIR of nearby land in a 25-px box, since land brightness varies). The shoreline is the sub-pixel **marching-squares isoline at f = 0.4** (`--water-frac-level`; lower = less conservative / higher recall). This places the edge inside the mixed transition zone — the true sub-pixel shoreline — and grows gentle shores more than steep ones (physically correct), rather than a blunt uniform dilation.
- **Smooth lake-edge curves (PyTorch).** Each isoline is fit with a closed **truncated-Fourier curve** (auto ≤40 harmonics, `--water-curve-harmonics`) by Adam gradient descent minimising point distance + a `k²` curvature penalty (`--water-curve-smooth 0.04`). The curvature term is the 'don't go wild' guardrail: continuous, smooth shorelines that still hug the data, with interior holes filled.
- **Areas & exports.** Area is the exact polygon shoelace integral in EPSG:32617 (sub-pixel, not a pixel count): **24.4 ha across 233 bodies** (raw threshold mask was 42.5 ha). Rasters are oversampled 8× for the fractional mask; the fitted shorelines are exported as `water_boundaries.{shp,geojson}` (EPSG:32617) and `water_boundaries_wgs84.geojson`.
- **Why not classic NDWI>0 / Otsu:** these Central-Florida ponds are turbid/tannic, so NDWI stays negative everywhere (it never crosses 0); and a data-driven Otsu threshold splits vegetation/non-vegetation rather than water/land, flooding bright urban as false water.
- **Why no MNDWI:** SuperDove has no SWIR band, so the gold-standard MNDWI is not directly computable. For a decisive result, `--s2-fusion` fetches a same-window Sentinel-2 L2A scene and computes true MNDWI=(Green−SWIR)/(Green+SWIR).

## 5. Reproduce

```bash
python visualize_psscene.py Florida_Hydrology_Avg_Precipitation_3 \
    --low-pct 2 --high-pct 98 --gamma 1.8

# add a true SWIR-based MNDWI by fusing a same-window Sentinel-2 scene
# (free, no credentials; needs network):
python visualize_psscene.py Florida_Hydrology_Avg_Precipitation_3 --s2-fusion
```

_Bands, files and statistics above are read directly from the delivery; re-running regenerates this README from the data._