# DeepEarth: Planetary Simulator for Ecological Intelligence 

DeepEarth is a new AI foundation model that fuses multimodal data and spatiotemporal simulation for scientific modeling of physical systems across the planet.  Similar to how ChatGPT was trained to reconstruct masked language datasets, DeepEarth learns to reconstruct masked earth science datasets, including from fields of physics, chemistry, biology, geology, and ecology.  DeepEarth welcomes contributors to a new open source project for AI that radically improves global sustainability and ecological intelligence.

![DeepEarth v.0.01 preview of architecture](https://github.com/legel/deepearth/blob/main/docs/deepearth_inductive_simulator.png)

## Latest Results: Earth4D → AlphaEarth Integration

**Breakthrough Achievement**: Earth4D successfully trained to predict Google DeepMind's AlphaEarth 64D embeddings with **3.61% MAPE** (Mean Absolute Percentage Error) on biodiversity data around iNaturalist flower visitations. This demonstrates Earth4D's capability as a universal spatiotemporal encoder for planetary-scale ecological intelligence.

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/legel/deepearth/blob/main/docs/earth4d_rgb_progression.gif" alt="Earth4D RGB Predictions" width="400"/>
        <br/>
        <sub><b>Earth4D learning to predict AlphaEarth RGB projections</b><br/>Bay Area biodiversity, epochs 0→200</sub>
      </td>
      <td align="center">
        <img src="https://github.com/legel/deepearth/blob/main/docs/earth4d_error_progression.gif" alt="Earth4D Error Reduction" width="400"/>
        <br/>
        <sub><b>Prediction error reduction during training</b><br/>MAPE: 15%→3.61% over 200 epochs</sub>
      </td>
    </tr>
  </table>
</div>

### Key Technical Specifications:
- **Spatial Encoding (XYZ)**: 24 levels × 2 features = 48D - encodes 3D position (lat/lon/elevation)
- **Spatiotemporal Encoding**: 19 levels × 2 features × 3 projections = 114D
  - XYT projection: 38D - captures longitude-time patterns (e.g., weather systems moving east-west)
  - YZT projection: 38D - captures latitude-elevation-time patterns (e.g., seasonal variations by latitude)
  - XZT projection: 38D - captures cross-section-time patterns (e.g., diurnal cycles)
- **Total Output**: 48D (spatial) + 114D (spatiotemporal) = 162D feature vector
- **Resolution**: 0.095m spatial, 0.84hr temporal (finest levels with 2^22 spatial, 2^18 temporal hashmaps)
- **Model Size**: 198M parameters (755MB) for Earth4D encoder
- **Training Performance**: 200 epochs in <2 hours on L4 GPU with 3.2M samples
- **Memory Usage**: ~3.8GB during training (includes model, gradients, optimizer states)

The first prototype version of DeepEarth is now validated with Earth4D encoding planetary coordinates (_x_, _y_, _z_, _t_) for predicting AlphaEarth embeddings. Earth4D uses multi-resolution hash encoding for efficient spatiotemporal representation learning. Through this approach, DeepEarth enables breakthrough AI representations for global scientific simulation and discovery.

![DeepEarth Grid4D spacetime encoding](https://github.com/legel/deepearth/blob/main/docs/deepearth_spacetime_encoder_grid4d.png) 

Below you can discover more about the core model architecture and features.  Currently, active contributors to DeepEarth include scientists from Stanford University and the University of Florida, as well as AI engineers from Meta and Google DeepMind.  For collaboration and partnership, please reach out to the DeepEarth principal investigator, Lance Legel (lance@ecodash.ai).

### NSF DeepEarth Workshop
We hosted an [NSF I-GUIDE workshop on DeepEarth](https://i-guide.io/forum/forum-2025/workshops/) this summer.  See a detailed summary [here](https://github.com/legel/deepearth/blob/main/docs/NSF_DeepEarth_Workshop.pdf).  

#### DeepEarth🔥
5 PhD students met for a 5 day NSF program in Boulder, Colorado on ["Spatial AI for Extreme Events and Disaster Resilience"](https://i-guide.io/summer-school/summer-school-2025/).  The team simulated _live fuel moisture content_, and saw spectacular results from Grid4D.  See a detailed summary [here](https://github.com/legel/deepearth/blob/main/docs/DeepEarth🔥_NSF_I-GUIDE_Final_Presentation.pdf).

#### Deep Physical Simulation  
DeepEarth is designed to deliver state-of-the-art modeling representations that can directly answer classical Bayesian questions, _e.g._ "As variable X changes across space S and time T, how is variable Y most likely to change, given all available evidence?"

#### Encoding the Planet
 DeepEarth is designed to discover the most predictive neural network parameters for multimodal observations across space and time.  It learns across (_x_, _y_, _z_, _t_, _energy_) metrics, where _energy_ can be any set of real-valued metrics ℝ<sup><em>d</em></sup>.  

#### Convergent Scientific Modeling 
Input any number of datasets distributed across space and time (_e.g._  satellite imagery, geological surveys, biodiversity records) and then automatically learn deep inductive priors across modalities.  

#### Deep Spacetime Manifold
One of the great lessons from Einstein's _relativity_ is that _space_ and _time_ are not independent variables.  DeepEarth learns unified (x,y,z,t) deep vector representations from _[Grid4D](https://github.com/JiaweiXu8/Grid4D/tree/main)_ as ["multi-resolution hash encodings"](https://graphics.stanford.edu/courses/cs348n-22-winter/LectureSlides/FinalSlides/ING.pdf).  This enables deep learning of complex spatiotemporal distributions.

#### Built for Lightspeed 
 DeepEarth is built on the [57x](https://www.youtube.com/watch?v=0VLAoVGf_74&ab_channel=WelchLabs) more memory-efficient Transformer architecture of _[DeepSeek](https://github.com/deepseek-ai/DeepSeek-V3)_ to unlock next-generation speed, accuracy, and scalability of multimodal cross-attention fusion.

#### Top of the Class
Design and development of DeepEarth is led by award-winning scientists and engineers from Stanford University, University of Florida, and Ecodash.ai, along with one of the first engineers from Google DeepMind.  

#### Planetary Intelligence for Everyone
DeepEarth is an MIT-licensed open source project designed and built to solve planetary-scale problems 🌎, especially through AI-powered maximization of ecosystem services – _e.g._ optimizing AI for sustainable agriculture, environmental restoration, & ecological landscape design.

# Code Implementation

## Earth4D: Production-Ready Spatiotemporal Encoder

Earth4D provides efficient multi-resolution hash encoding for planetary-scale (x,y,z,t) coordinates. The encoder has been tested in production, achieving state-of-the-art results predicting AlphaEarth embeddings.

### Quick Start

```python
from encoders.xyzt import Earth4D
import torch

# Initialize Earth4D with production-tested parameters
encoder = Earth4D(
    spatial_levels=24,              # 24 levels: 0.095m finest resolution globally
    temporal_levels=19,              # 19 levels: 200-year coverage at 0.84hr precision  
    spatial_log2_hashmap_size=22,   # 4M entries (755MB model memory)
    temporal_log2_hashmap_size=18,  # 256K entries
    verbose=True                     # Print resolution table
)

# Input: [latitude, longitude, elevation_m, time_normalized]
coordinates = torch.tensor([
    [37.7749, -122.4194, 50.0, 0.5],   # San Francisco, mid-timerange
    [40.7128, -74.0060, 100.0, 0.7],   # New York, later time
])

# Get spatiotemporal features
features = encoder(coordinates)  # Shape: [2, 162] 
print(f"Output dimension: {encoder.get_output_dim()}")  # 162 features
```

### Training Example: Earth4D → AlphaEarth

```python
import torch
import torch.nn as nn
from encoders.xyzt import Earth4D

class DeepEarth(nn.Module):
    """DeepEarth model using Earth4D to predict AlphaEarth embeddings."""
    
    def __init__(self, output_dim=64):
        super().__init__()
        
        # Earth4D encoder with optimized parameters
        self.earth4d = Earth4D(
            spatial_levels=24,
            temporal_levels=19,
            spatial_log2_hashmap_size=22,
            temporal_log2_hashmap_size=18,
            verbose=False
        )
        
        # Get encoder dimension and build MLP decoder
        encoder_dim = self.earth4d.get_output_dim()  # 162
        
        # MLP with normalization for stability
        self.decoder = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
            nn.Tanh()  # Output in [-1, 1] for normalized embeddings
        )
        
    def forward(self, coords):
        # coords: [batch, 4] with [lat, lon, elev, time]
        spacetime_features = self.earth4d(coords)
        embeddings = self.decoder(spacetime_features)
        return embeddings

# Training setup
model = DeepEarth(output_dim=64)
optimizer = torch.optim.Adam([
    {'params': model.earth4d.parameters(), 'lr': 1e-4},  # Lower LR for encoder
    {'params': model.decoder.parameters(), 'lr': 1e-3}
])
criterion = nn.MSELoss()

# Example training step
coords = torch.randn(32, 4)  # Batch of 32 samples
target_embeddings = torch.randn(32, 64)  # AlphaEarth targets

predictions = model(coords)
loss = criterion(predictions, target_embeddings)
loss.backward()
optimizer.step()
```

### Memory and Performance Guidelines

| Configuration | Hashmap Size | Model Memory | Training Memory | Finest Resolution | Use Case |
|--------------|-------------|--------------|-----------------|-------------------|----------|
| Light (L=16, log2=19) | 512K | ~200MB | ~800MB | 1.2km | Regional models |
| **Planetary (L=24, log2=22)** | **4M** | **755MB** | **3.8GB** | **0.095m** | **Global (default)** |
| Research (L=28, log2=24) | 16M | ~1.5GB | ~6GB | 0.006m | High-precision |
| Extreme (L=32, log2=26) | 64M | ~4GB | ~14GB | 0.37mm | Ultra-fine local |

### Coordinate Format

Earth4D expects input coordinates in the following format:
- **Latitude**: -90 to 90 degrees
- **Longitude**: -180 to 180 degrees  
- **Elevation**: Meters above sea level
- **Time**: Normalized to [0, 1] range (e.g., 0=year 1900, 1=year 2100)

The encoder automatically:
1. Converts lat/lon/elevation to ECEF (Earth-Centered, Earth-Fixed) coordinates using WGS84
2. Normalizes spatial coordinates for optimal hash encoding
3. Applies multi-resolution encoding from coarse (km-scale) to fine (0.095m at level 24)
4. Returns concatenated features: 48D spatial (XYZ) + 114D spatiotemporal (XYT+YZT+XZT) = 162D total

Note: While the finest resolution is 0.095m, hash collisions are managed through learned disambiguation. The sparsity of Earth observation data (e.g., biodiversity observations) allows the model to effectively utilize the fine resolutions despite the 4M hash table limit.

See [SPECIFICATIONS.md](https://github.com/legel/deepearth/blob/main/SPECIFICATIONS.MD) for full architectural details.

```python
class DeepEarthModel(nn.Module):
    """DeepEarth: Hierarchical Multimodal Transformer for Scientific Simulation
    
    This model implements a hierarchical architecture that processes arbitrary integrations of
    multimodal observational datasets through three stages:
    1. Spatiotemporal encoding: Multi-resolution hash encoding of coordinates
    2. Modality-specific encoding: Small DeepSeek Transformers process each data type
    3. Cross-modal fusion: Large DeepSeek Transformer integrates all information
    
    The model learns through self-supervised reconstruction of masked inputs,
    enabling it to understand complex physical system dynamics without labels.
    
    Args:
        config: DeepEarthConfig with all model specifications
    
    Example:
        >>> config = DeepEarthConfig()
        >>> config.modality_configs = {
        ...     'species': ModalityConfig(
        ...         name='species',
        ...         encoding_type='learned_embedding',
        ...         input_type='categorical',
        ...         column_name='species_name',
        ...         embed_dim=64
        ...     ),
        ...     'climate': ModalityConfig(
        ...         name='climate',
        ...         encoding_type='continuous_values',
        ...         input_type='numerical',
        ...         column_names=['temperature', 'precipitation']
        ...     )
        ... }
        >>> model = DeepEarthModel(config)
        >>> outputs, embeddings = model(batch)
    """
    
    def __init__(self, config: DeepEarthConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger('DeepEarth.Model')
        
        # Initialize components
        self._build_spatiotemporal_encoder()
        self._build_modality_processors()
        self._build_cross_modal_fusion()
        
        # Log model statistics
        self._log_model_info()
    
    def _build_spatiotemporal_encoder(self):
        """Build the spatiotemporal coordinate encoding system"""
        self.coord_encoder = Grid4DEncoder(self.config)
        
        # Dedicated decoders for coordinate reconstruction
        self.spatial_decoder = SpatiotemporalDecoder(
            'spatial', output_dim=3, config=self.config
        )
        self.temporal_decoder = SpatiotemporalDecoder(
            'temporal', output_dim=1, config=self.config
        )
    
    def _build_modality_processors(self):
        """Build encoders and decoders for each configured modality"""
        self.modality_encoders = nn.ModuleDict()
        self.modality_decoders = nn.ModuleDict()
        
        # Validate modality names
        reserved_names = {'spatial', 'temporal', 'xyz', 't'}
        for name in self.config.modality_configs:
            if name in reserved_names:
                raise ValueError(f"Modality name '{name}' is reserved. Please use a different name.")
        
        for modality_name, modality_config in self.config.modality_configs.items():
            # Determine input dimension
            if modality_config.encoding_type == 'learned_embedding':
                input_dim = modality_config.embed_dim
            elif modality_config.encoding_type == 'continuous_values':
                input_dim = len(modality_config.column_names)
            else:
                self.logger.warning(f"Unknown encoding type for {modality_name}, skipping")
                continue
            
            # Create encoder with optional custom config
            encoder_config = self.config.modality_encoder_config
            if modality_config.custom_encoder_config:
                # Override default settings with custom ones
                encoder_config = dataclasses_replace(
                    encoder_config,
                    **modality_config.custom_encoder_config
                )
            
            self.modality_encoders[modality_name] = ModalityEncoder(
                modality_name=modality_name,
                input_dim=input_dim,
                config=self.config,
                encoder_config=encoder_config
            )
            
            self.modality_decoders[modality_name] = ModalityDecoder(
                modality_name=modality_name,
                output_dim=input_dim,
                config=self.config
            )
    
    def _build_cross_modal_fusion(self):
        """Build the main cross-modal fusion network"""
        self.cross_modal_fusion = Transformer(self.config.cross_modal_fusion_config)
    
    def _log_model_info(self):
        """Log model configuration and statistics"""
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.logger.info(f"DeepEarth Model initialized:")
        self.logger.info(f"  Total parameters: {n_params:,}")
        self.logger.info(f"  Trainable parameters: {n_trainable:,}")
        self.logger.info(f"  Coordinate system: {self.config.spatial_coordinate_system}")
        self.logger.info(f"  Modalities: {list(self.config.modality_configs.keys())}")
        self.logger.info(f"  Grid4D resolutions: {len(self.config.spatial_resolutions)} spatial, "
                        f"{len(self.config.temporal_resolutions)} temporal")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass through the hierarchical architecture.
        
        Args:
            batch: Dictionary containing:
                - xyzt: (B, 4) normalized spatiotemporal coordinates
                - modalities: Dict of modality_name -> (B, D) tensors
                - modality_masks: Dict of modality_name -> (B,) bool masks
                - internal_masks: Dict of modality_name -> (B, D) bool masks
                - ground_truth: Dict of ground truth values for reconstruction loss
        
        Returns:
            reconstructions: Dict of modality_name -> reconstructed values
            embeddings: (B, N, D) full sequence of embeddings from fusion network
        """
        if 'xyzt' not in batch:
            raise ValueError("Batch must contain 'xyzt' key with spatiotemporal coordinates")
        
        device = batch['xyzt'].device
        B = batch['xyzt'].shape[0]
        
        # Stage 1: Encode spatiotemporal coordinates
        coord_embeddings = self._encode_coordinates(batch)
        
        # Stage 2: Encode modalities with masking
        modality_embeddings = self._encode_modalities(batch)
        
        # Stage 3: Cross-modal fusion
        fused_embeddings = self._fuse_modalities(coord_embeddings, modality_embeddings)
        
        # Stage 4: Decode all modalities
        reconstructions = self._decode_all(fused_embeddings, batch)
        
        return reconstructions, fused_embeddings
    
    def _encode_coordinates(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode spatiotemporal coordinates with Grid4D.
        
        Args:
            batch: Input batch with 'xyzt' and optional masks
        
        Returns:
            coord_embeddings: (B, D) coordinate embeddings
        """
        coord_embeddings = self.coord_encoder(
            batch['xyzt'],
            spatial_mask=batch['modality_masks'].get('spatial'),
            temporal_mask=batch['modality_masks'].get('temporal')
        )
        return coord_embeddings
    
    def _encode_modalities(self, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Encode each modality through its specific encoder.
        
        Args:
            batch: Input batch with modality data and masks
        
        Returns:
            modality_embeddings: List of (B, D) embeddings for each modality
        """
        modality_embeddings = []
        
        for modality_name, modality_data in batch['modalities'].items():
            if modality_name not in self.modality_encoders:
                self.logger.warning(f"No encoder for modality {modality_name}")
                continue
            
            # Apply internal feature-level masking if provided
            if modality_name in batch.get('internal_masks', {}):
                modality_data = modality_data * batch['internal_masks'][modality_name].float()
            
            # Encode with modality-level masking
            encoded = self.modality_encoders[modality_name](
                modality_data,
                mask=batch['modality_masks'].get(modality_name)
            )
            
            modality_embeddings.append(encoded)
        
        return modality_embeddings
    
    def _fuse_modalities(self, coord_embeddings: torch.Tensor, 
                        modality_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Fuse coordinate and modality embeddings through cross-modal transformer.
        
        Args:
            coord_embeddings: (B, D) spatiotemporal embeddings
            modality_embeddings: List of (B, D) modality embeddings
        
        Returns:
            fused_embeddings: (B, N, D) fused representation
        """
        # Build token sequence: [coordinates, modality_1, modality_2, ...]
        tokens = [coord_embeddings.unsqueeze(1)]
        
        for modality_embedding in modality_embeddings:
            if modality_embedding.dim() == 2:
                modality_embedding = modality_embedding.unsqueeze(1)
            tokens.append(modality_embedding)
        
        # Concatenate all tokens
        token_sequence = torch.cat(tokens, dim=1)  # (B, N, D)
        
        # Apply cross-modal fusion
        fused_embeddings = self.cross_modal_fusion(token_sequence)
        
        return fused_embeddings
    
    def _decode_all(self, embeddings: torch.Tensor, 
                    batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decode all modalities from fused embeddings.
        
        Args:
            embeddings: (B, N, D) fused embeddings
            batch: Original batch for modality ordering
        
        Returns:
            reconstructions: Dict mapping modality names to reconstructed values
        """
        reconstructions = {}
        
        # Decode spatiotemporal coordinates from first token
        coord_token = embeddings[:, 0]
        reconstructions['xyz'] = self.spatial_decoder(coord_token)
        reconstructions['t'] = self.temporal_decoder(coord_token)
        
        # Decode each modality from its corresponding token
        token_idx = 1
        for modality_name in batch['modalities'].keys():
            if modality_name in self.modality_decoders and token_idx < embeddings.shape[1]:
                modality_token = embeddings[:, token_idx]
                reconstructions[modality_name] = self.modality_decoders[modality_name](
                    modality_token
                )
                token_idx += 1
        
        return reconstructions
    
    def get_modality_embedding(self, batch: Dict[str, torch.Tensor], 
                              modality_name: str) -> torch.Tensor:
        """Extract embedding for a specific modality.
        
        Useful for downstream tasks or analysis.
        
        Args:
            batch: Input batch
            modality_name: Name of modality to extract
        
        Returns:
            embedding: (B, D) embedding for the specified modality
        """
        with torch.no_grad():
            _, embeddings = self.forward(batch)
            
            # Find the token index for this modality
            token_idx = 1  # 0 is coordinates
            for name in batch['modalities'].keys():
                if name == modality_name:
                    return embeddings[:, token_idx]
                token_idx += 1
        
        raise ValueError(f"Modality {modality_name} not found in batch")
```

## Development

### Recent Achievements
- ✅ Earth4D encoder validated with 3.61% MAPE on AlphaEarth prediction task
- ✅ Optimized for single GPU training (200 epochs in <2 hours on L4)
- ✅ Memory-efficient design fits 3.2M samples + model in 24GB GPU
- ✅ Production-ready coordinate conversion with WGS84 ellipsoid

Collaborators are invited to become core DeepEarth model contributors. The Earth4D encoder is now production-ready and can be used as a foundation for diverse spatiotemporal prediction tasks beyond AlphaEarth embeddings.

### Applications

While Earth4D has been validated on AlphaEarth embeddings, it can predict any spatiotemporal target:
- Climate variables (temperature, precipitation, wind)
- Biodiversity metrics (species occurrence, abundance)
- Environmental indicators (vegetation indices, soil moisture)
- Human activity patterns (land use, population density)
- Any georeferenced time-series data

For more details see the [technical specifications](https://github.com/legel/deepearth/blob/main/SPECIFICATIONS.MD).

## License

MIT License - see LICENSE file for details.
