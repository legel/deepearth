# Energy4D: Relative Spatiotemporal Encoder

Energy4D encodes **relative** spatiotemporal offsets instead of absolute coordinates, enabling better generalization across different locations and times.

## Key Difference from Earth4D

| Encoder | Input | Learns | Generalizes? |
|---------|-------|--------|--------------|
| **Earth4D** | Absolute (lat=56.7°N, lon=12.3°E, t=April 3) | "What happens at 56.7°N?" | ❌ Memorizes locations/times |
| **Energy4D** | Relative (Δx=+10km, Δy=-5km, Δt=-3hr) | "What happens 10km north?" | ✅ Transfers to new locations/times |

## Why Relative Coordinates?

**Problem with Earth4D (Absolute):**
- Hash table learns: "At (56.7°N, 12.3°E) on April 3 at 15:00, temperature is X"
- Cannot generalize to unseen locations or dates
- Overfits severely on small datasets (memorization)

**Solution with Energy4D (Relative):**
- Hash table learns: "10km north and 3 hours ago, temperature typically changes by ΔT"
- Same dynamics apply everywhere (translational invariance)
- Learns spatiotemporal patterns, not specific coordinates

## Architecture

Energy4D has the **same architecture** as Earth4D:
- 24 spatial levels (multi-scale hash encoding)
- 24 temporal levels
- 4 encoders: XYZ (spatial), XYT, YZT, XZT (spatiotemporal)
- 192-dimensional output

**Only difference:** Operates on `(coords - reference_coords)` instead of `coords`

## Usage

```python
from encoders.energy4d.energy4d import Energy4D

# Initialize (same parameters as Earth4D)
encoder = Energy4D(
    spatial_levels=24,
    temporal_levels=24,
    features_per_level=2,
    coordinate_system="geographic",
    enable_learned_probing=False,  # Disabled for stability
).cuda()

# Encode relative to reference points
node_coords = torch.randn(100, 4).cuda()  # [lat, lon, elev, time]
reference_coords = node_coords.mean(dim=0, keepdim=True)  # Global mean

# Energy4D encodes relative offsets
features = encoder(node_coords, reference_coords)
# Returns: (100, 192) - encodes how far each point is from reference
```

## For GNN Weather Forecasting

In GNN applications, each node encodes its local spatiotemporal neighborhood:

```python
for node_i in range(num_nodes):
    # Reference: this node's position
    reference = coordinates[node_i:node_i+1]  # (1, 4)

    # Encode neighborhood relative to this node
    # "What's 10m away? 100m away? 1km away? ..." (multi-scale)
    local_features = encoder(coordinates, reference.expand(num_nodes, 4))
```

This enables:
- Same encoder moves to each node (translational invariance)
- Learns local spatiotemporal dynamics
- Generalizes to unseen locations and times

## Status

✅ **Fully Implemented** - Ready for training and testing

**Next Steps:**
1. Train on DANRA dataset
2. Compare with Earth4D (expect better generalization)
3. Test on larger datasets (MEPS, ERA5)

## Learned Probing

**Note:** `enable_learned_probing=False` by default due to PyTorch autograd in-place operation issues that cause gradient computation errors during backpropagation.

## Citation

Based on Earth4D architecture. Relative coordinate encoding inspired by:
- Convolutional neural networks (translational invariance)
- LSTMs (relative temporal encoding)
- Physics (spatiotemporal dynamics are local, not absolute)
