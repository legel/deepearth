# DeepEarth: Geospatial Deep Simulator of Earth's Ecosystems

This repository supports the development of **DeepEarth**, a deep neural network for modeling Earth's ecosystems at landscape scale.  DeepEarth will directly support prediction of plant-pollinator interactions, as well as plant-fire and plant-flood ecohydrology, across California and Florida.  Its development is led by Ecodash.ai, Stanford University, and University of Florida, with support from NSF.

DeepEarth learns from biological, geological, and ecological data – e.g., species presence records, satellite imagery, 3D topography, soil surveys, and climate models – across space and time. It learns by reconstructing partially or fully masked data, sampled around (x, y, z, t) coordinates of scientific records. Prior knowledge from validated scientific models guides optimization as Bayesian constraints.

Initial prototype results are expected by a DeepEarth workshop on June 17th, 2025, sponsored by NSF I-GUIDE in Chicago.  The prototype will focus on simulating flowering and pollination in native ecosystems of California and Florida between 2010 to 2025. Predictive accuracy will initially be validated through (i) _in silico_ accuracy of predicted plant-pollinator distributions vs. real observations
unseen during model training, and eventually by (ii) planting field trials seeking to maximize native plant & pollinator biodiversity, flowering abundance, and cross-pollination.

## Core Functionality Roadmap

The development is planned in phases:

### Phase 1: Foundational Geospatial, Reconstruction & Data Tools

*   [x] High-precision Geodetic Coordinate Conversions (Lat/Lon/Alt ↔ XYZ ↔ Normalized)
*   [ ] Develop Data Loaders for Diverse Ecological and Geospatial Datasets:
    *   [ ] **iNaturalist:** Load millions of timestamped/geotagged species photos for distributions, phenology, interactions.
        *   *In Situ* Nature Imagery (*3 × H × W × RGB*)
    *   [ ] **iDigBio:** Load millions of historical biodiversity records (pre-1900) for native ecosystem modeling.
        *   Species Distribution Records (*N × taxa*)
    *   [ ] **PhyloRegion:** Load geographic species distribution models with evolutionary constraints.
        *   Species Habitat Range Model (*12k × taxa*)
    *   [ ] **PhenoVision:** Load pre-trained model/data for flowering classification from images.
        *   Plant Flowering Phenology (*12k × taxa*)
    *   [ ] **GloBI:** Load indexed plant-pollinator interaction events (330k+ since 2014).
        *   Plant-Pollinator Flower Visits (*2 × taxa*)
    *   [ ] **InVEST:** Load pollination model outputs (LULC maps, pollinator presence, flight ranges).
        *   Floral Resources Index (*blooms/km²/season*)
        *   Habitat Nesting Suitability (*nests/km²*)
        *   Relative Species Abundance (*species/km²/season*)
        *   Pollinator Abundance (*visits/flower/season*)
    *   [ ] **USDA-Aerial:** Load HD aerial RGBI imagery (0.33m/px NAIP) for agricultural intelligence.
        *   Visible (aerial) (*H × W × RGBI*)
    *   [ ] **NASA-Thermal:** Load HD infrared satellite data (GOES-R) for hourly weather/cloud/temperature fields.
        *   Infrared (satellite) (*H × W × 16 × m*)
        *   Wind (speed, direction) (*m/s, °*)
        *   Downward Shortwave Radiation (*W/m²*)
    *   [ ] **ESA-Hyperspectral:** Load satellite hyperspectral data (DESIS, 235 frequencies) for spectrographic analysis.
        *   Hyperspectral (satellite) (*H × W × 235 × nm*)
    *   [ ] **WorldClim:** Load bioclimatic metrics (19 variables, 1970-2000, 1km res).
        *   Bioclimatic Metrics (*19 × kg/m², K*)
    *   [ ] **NOAA-Weather:** Load hourly land hydrology/energy simulations (evapotranspiration, heat flux).
        *   Precipitation (*kg/m²*)
        *   Air Temperature (2m) (*K*)
        *   Specific Humidity (2m) (*%*)
        *   Convective Available Potential Energy (*J/kg*)
        *   Surface Albedo (*%*)
        *   Sensible / Ground / Latent Heat Flux (*W/m²*)
        *   Surface / Subsurface Runoff (*kg/m²*)
        *   Streamflow (*m³/s*)
        *   Snow Water Equivalent (*kgm⁻²*)
        *   Evapotranspiration (mass/energy flux) (*kg, W/m²*)
    *   [ ] **SoilSurvey:** Load precision USDA soil surveys (pH, organic matter, texture).
        *   pH (*0 − 14*)
        *   Sand, Silt, Clay, Organic Matter (*4 × %*)
        *   Bulk Density (*g/cm³*)
        *   Depth to Water Table (*cm*)
        *   Saturated Hydraulic Conductivity (*𝜇m/s*)
        *   Available Water Capacity (*%*)
    *   [ ] **HydroSHEDS:** Load v2 global watersheds/river networks (12m scale).
        *   Mean Annual Discharge (*m³ s⁻¹*)
    *   [ ] **USGS-3DEP:** Load 3D topography/LiDAR data for microclimate/water flow modeling.
        *   Digital Elevation Model (*m*)
        *   3D Topographic LiDAR (`N × (x,y,z)`) *m*
        *   Slope / Aspect (*°*)
    *   [ ] **GeoFusion:** Fuse 3D data (LiDAR, ARKit, GCPs, GNSS RTK) into global geodetic coordinates for photorealistic 3D mapping & Gaussian Splatting.
        *   Geotagged 3D Gaussian Splats (*Neural Output: (latitude, longitude, altitude) → (color, opacity)*)
*   [ ] Implement Transformer-based encoders for unifying different data modalities.
*   [ ] Develop spatio-temporal embeddings (Multiresolution Hash, Time2Vec).
*   [ ] Implement self-supervised training via masked data reconstruction.

### Phase 2: Core DeepEarth Model & Simulation

*   [ ] Integrate Bayesian priors from external ecological models (e.g., Phyloregion).
*   [ ] Develop simulation capabilities for:
    *   [ ] Plant/Pollinator Habitat Distributions
    *   [ ] Plant Flowering Phenology
    *   [ ] Plant-Pollinator Interaction Networks
*   [ ] Implement observation bias modeling.
*   [ ] Develop APIs and tools for querying the simulator (e.g., for pollinator garden planning).

### Phase 3: Validation & Application

*   [ ] Validate predictions against environmental studies and field trials (California & Florida focus initially).
*   [ ] Scale model geographically and taxonomically.
*   [ ] Open-source the model and framework.

## Current Features

This package currently provides foundational tools for:

*   Converting between geodetic (latitude, longitude, altitude) and Earth-Centered, Earth-Fixed (ECEF) XYZ coordinates.
*   Handling orientation data (yaw, pitch, roll) and associated rotation matrices.
*   Supporting CSV data import/export with comprehensive metadata (e.g., for GeoFusion data).
*   Ensuring high-precision coordinate conversions suitable for landscape-scale modeling.
*   Enabling efficient batch processing via PyTorch tensors.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/legel/deepearth.git 
    cd deepearth
    ```
2.  Install dependencies (preferably in a virtual environment):
    ```bash
    pip install -e .[dev]
    ```
    *(The `-e` installs the package in editable mode. `[dev]` includes development tools like pytest, black, etc.)*

## Usage Example (Coordinate Conversion)

```python
import torch
from deepearth.geospatial.geo2xyz import GeospatialConverter

# Create a converter instance (uses float64 for precision)
converter = GeospatialConverter()
# Example coordinates from ecological landmarks around the world
coordinates = [
    # Hoover Tower, Stanford University
    [37.428889610708694, -122.16885901974715, 86.868],
    # Butterfly Rainforest, University of Florida
    [29.636335373496760, -82.37033779288247, 2.500],
    # Sky Garden, London
    [51.511218537276620, -0.083533446399636, 155.000],
    # Supertrees, Singapore
    [1.281931104253864, 103.86393021307455, 50.000],
    # High Line Garden, New York City
    [40.742766754019710, -74.00749599736363, 9.000],
    # Hoshun-in Bonsai Garden, Kyoto
    [35.044560764859480, 135.74464051040786, 0.000]
]

# Convert each set of coordinates
for lat, lon, alt in coordinates:
    # Convert from geodetic to XYZ
    xyz, _ = converter.geodetic_to_xyz(torch.tensor([[lat, lon, alt]]))
    
    # Convert XYZ to normalized coordinates
    norm = converter.xyz_to_norm(xyz)
    
    # Convert back to XYZ
    xyz2 = converter.norm_to_xyz(norm)
    
    # Convert back to geodetic
    geo2, _ = converter.xyz_to_geodetic(xyz2)
    
    print(f"\nOriginal geodetic: {lat:.8f}°, {lon:.8f}°, {alt:.3f}m")
    print(f"XYZ coordinates: {xyz[0,0].item():.3f}, {xyz[0,1].item():.3f}, {xyz[0,2].item():.3f}")
    print(f"Normalized coordinates: {norm[0,0].item():.6f}, {norm[0,1].item():.6f}, {norm[0,2].item():.6f}")
    print(f"Recovered geodetic: {geo2[0,0].item():.8f}°, {geo2[0,1].item():.8f}°, {geo2[0,2].item():.3f}m")
```
