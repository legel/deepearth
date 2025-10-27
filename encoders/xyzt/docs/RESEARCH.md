# Research Results

**Note: Earth4D is in active research and development.**

Latest tested configuration (24 spatial, 24 temporal levels):
- **Hash Tables**: 4.2M entries spatial, 4.2M entries per temporal grid (2^22 each)
- **Model Size**: 724M parameters (2.76GB)
- **Training Memory**: ~11GB (including gradients and optimizer states, 4x multiplier)
- **Collision Rates**: <1% at spatial fine levels, 1-2% at temporal fine levels
- **Data**: Tested with 41,261 unique spatiotemporal coordinates (Globe LFMC 2.0)
- **Behavior**: Collisions at very fine spatial/temporal levels are expected when data doesn't vary at that resolution

### Latest Results: Live Fuel Moisture Content (LFMC) Prediction

Earth4D has been successfully applied to predict Live Fuel Moisture Content across CONUS using the Globe LFMC 2.0 dataset. This represents a critical application for wildfire risk assessment and vegetation monitoring.

#### Dataset Overview
- **89,961 samples** across Continental United States
- **182 unique plant species** with significant multi-species degeneracy (76% of samples)
- **Temporal coverage**: Multiple years of observations
- **Key challenge**: Multiple species at same spatiotemporal coordinates with different LFMC values

#### Model Configuration
- **Earth4D**: Spatiotemporal encoding of latitude, longitude, elevation, and time
- **Species embeddings**: 768-dimensional learnable vectors (140K parameters)
- **Note**: Parameter count varies by configuration (spatial/temporal levels and hash table sizes)

#### Performance Results (2500 epochs)

**Training Performance:**
- Mean Absolute Error: **2.3 percentage points**
- Median Absolute Error: **0.8 percentage points**

**Test Performance (15% holdout):**
- **Temporal test** (final 6 months): MAE = 19.7pp, Median = 14.1pp
- **Spatial test** (5 geographic clusters): MAE = 20.3pp, Median = 12.3pp
- **Random test** (random holdout): MAE = 20.2pp, Median = 12.3pp

#### Key Insights

1. **Strong Learning Capacity**: The sub-3pp training MAE demonstrates Earth4D's ability to capture complex spatiotemporal patterns in vegetation moisture content.

2. **Generalization Performance**: Test median errors around 12-14pp are promising given the complexity of predicting vegetation moisture across diverse ecosystems and species.

3. **Species Disambiguation**: The model successfully handles locations with multiple species through learned embeddings, with comparable performance on unique (single-species) vs degenerate (multi-species) locations.

4. **Temporal Dynamics**: The model captures seasonal patterns including wet summers and dry winters, as shown in the temporal visualization below.

#### Visualizations

![Temporal LFMC Predictions](../../docs/temporal_predictions_epoch_2500.png)
*Monthly temporal evolution showing ground truth (red) and predicted (blue) LFMC distributions across all test sets. The model captures seasonal moisture patterns while maintaining reasonable prediction distributions.*

![Geospatial Error Distribution](../../docs/geospatial_error_map_epoch_2500.png)
*Spatial distribution of prediction errors across CONUS on 100km × 100km grid. Colors indicate average LFMC error (capped at 75pp) with circle size representing sample density.*

#### Ablation Study: BioCLIP vs Random Embeddings

We compared pre-trained BioCLIP 2 biological embeddings against randomly initialized 768D embeddings:
- Random embeddings: Slightly better on temporal (-1.9%) and spatial (-0.3%) tests
- BioCLIP embeddings: 4.4% better on random test split
- **Conclusion**: Minimal difference suggests Earth4D's spatiotemporal encoding provides the primary signal

