# DeepEarth: Geospatial Deep Simulator of Earth's Ecosystems

**DeepEarth** is a neural network architecture for simulation of Earth's ecosystems at landscape scale.  It will help to more precisely predict _pollination_, _fire_, and _flood_ interactions with plant species in California and Florida.  

Preview our [paper](https://github.com/legel/deepearth/blob/main/docs/deepearth.pdf) to learn more.

### Join us for an NSF-funded workshop on June 17th, 2025

We're proud to host an NSF I-GUIDE workshop on DeepEarth this summer in Chicago!  See ["DeepEarth Workshop: Self-Supervised AI for Spatiotemporal Modeling of Ecosystems"](https://i-guide.io/forum/forum-2025/workshops/) for more details. Looking forward to meeting you!

### NSF summer program in AI for disaster resilience, August 4-8, 2025

We will work with 5 PhD students in geospatial, ecological, and computational scientists for a 5 day program themed ["Spatial AI for Extreme Events and Disaster Resilience"](https://i-guide.io/summer-school/summer-school-2025/).  We will geospatially and temporally simulate fire and flood responses of plants at sub-meter scale.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/legel/deepearth.git 
    cd deepearth
    ```
2.  Install dependencies (preferably in a virtual environment):
    ```bash
    pip install -e .[dev]
    ```

### Converting between (_lat_, _lon_, _alt_) and ECEF (_x_,_y_,_z_) coordinates

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
