**Note**: Temporal resolution depends on the application's time range. Values shown assume 200-year range (1900-2100).

## 🏗️ Architecture Details

### Decomposed 4D Encoding

Earth4D uses a decomposed architecture optimized for spacetime:

1. **Spatial Encoder (XYZ)**: 3D hash encoding of ECEF coordinates
   - Encodes full 3D position in Earth-Centered Earth-Fixed frame
   - Default: 24 levels × 2 features = 48D output
   - Hash table: Configurable (tested: 2^22 = 4M entries)

2. **Spatiotemporal Projections**: Three 3D encodings capturing orthogonal planes:
   - **XYT**: Equatorial plane + time (X-Y plane through Earth's center)
   - **YZT**: 90°E meridian plane + time (Y-Z plane through poles)
   - **XZT**: Prime meridian plane + time (X-Z plane through 0° longitude)
   - Default: 24 levels × 2 features = 48D output each
   - Hash table: Configurable (tested: 2^22 = 4M entries per projection)

Note: ECEF axes are NOT aligned with lat/lon/elevation:
- X: Points through 0° lat, 0° lon (equator/prime meridian intersection)
- Y: Points through 0° lat, 90°E lon (equator in Indian Ocean)
- Z: Points through North Pole

### Coordinate System

- **Input**: WGS84 geodetic coordinates (latitude, longitude, elevation, time)
- **Internal**: ECEF (Earth-Centered Earth-Fixed) for uniform spatial hashing
- **Normalization**: Automatic scaling to [-1, 1] for hash encoding

### Hash Encoding Algorithm

#### Multi-Resolution Decomposition
For each level L (configurable, default 0 to 23):
- Resolution at level L = `base_resolution * (growth_factor^L)`
- Creates progressively finer grids from coarse to fine resolution

#### Grid Mapping & Hashing
```cuda
// For each coordinate at each level:
1. Map to grid: pos_grid[d] = floor(input[d] * scale[d])
2. Calculate grid index:
   if (grid_size <= hashmap_size) {
      // Direct indexing for coarse levels (no collisions)
      index = x + y*stride_x + z*stride_xy
   } else {
      // Hash function for fine levels (with collisions)
      index = fast_hash(pos_grid) % hashmap_size
   }
```

#### XOR-Prime Hash Function
```cuda
uint32_t fast_hash(pos_grid[D]) {
    // Large primes for mixing (first is 1 for memory coherence)
    primes[] = {1, 2654435761, 805459861, 3674653429, ...}
    result = 0
    for d in D:
        result ^= pos_grid[d] * primes[d]
    return result
}
```

#### Smoothstep Interpolation
- Uses smoothstep function: `S(t) = 3t² - 2t³`
- Provides C¹ continuous gradients (derivative: `6t(1-t)`)
- Trilinear interpolation across 8 corners (2³ for 3D)
- Better than linear for continuous Earth phenomena

#### Feature Concatenation (Default Configuration)
- XYZ encoder → 48D features (24 levels × 2 features)
- XYT encoder → 48D features (24 levels × 2 features)
- YZT encoder → 48D features (24 levels × 2 features)
- XZT encoder → 48D features (24 levels × 2 features)
- **Total**: 192D feature vector

### Hash Table Properties

- **No Explicit Regularization**: No sparsity enforcement or uniform distribution constraints
- **Initialization**: Embeddings uniformly in [-0.1, 0.1] for strong gradient flow
- **Collision Handling**: Learned disambiguation through MLP decoder
- **Memory Efficiency**: Hash table size configurable based on application needs
