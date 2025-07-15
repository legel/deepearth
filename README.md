# DeepEarth: Planetary Simulator for Ecological Intelligence 

DeepEarth is a new AI foundation model that fuses deep neural networks, multimodal datasets, and spatiotemporal simulation techniques, for scientific modeling of physical systems across the planet.  Similar to how ChatGPT was originally trained to reconstruct masked language datasets, DeepEarth will initially learn by reconstructing masked spatiotemporal distributions of vast and varied earth science datasets, including from fields of physics, chemistry, biology, geology, and ecology.  The vision for the DeepEarth project is to ultimately lead a new open source consortium for AI that radically improves global sustainability and ecological intelligence.

![DeepEarth v.0.01 preview of architecture](https://github.com/legel/deepearth/blob/main/docs/deepearth_inductive_simulator.png)

The first prototype version of DeepEarth is now in development, focused on native plant species distributed across Central Florida between 2010-2025 (see a live demo of initial AI feature extractions [here](https://deepearth.ecodash.ai)).  This first model integrates [Grid4D](https://github.com/JiaweiXu8/Grid4D/tree/main) (_x_, _y_, _z_, _t_) deep spacetime encoding with [V-JEPA 2](https://ai.meta.com/vjepa/) (vision) and [DeepSeek V3](https://github.com/deepseek-ai/DeepSeek-V3) (language + latent attention) world models: all of these "encoders" represent state-of-the-art ways of revealing the structure of space, time, vision, and language to deep neural networks.  Through this approach, DeepEarth will machine learn breakthrough new AI representations for global scientific simulation and discovery.  

![DeepEarth Grid4D spacetime encoding](https://github.com/legel/deepearth/blob/main/docs/deepearth_spacetime_encoder_grid4d.png) 

Below you can discover more about the core model architecture and features.  Currently, active contributors to DeepEarth include scientists from Stanford University and the University of Florida, as well as AI engineers from Meta and Google DeepMind.  For collaboration and partnership, please reach out to the DeepEarth principal investigator, Lance Legel (lance@ecodash.ai).

### NSF DeepEarth Workshop
We hosted an [NSF I-GUIDE workshop on DeepEarth](https://i-guide.io/forum/forum-2025/workshops/) this summer.  See a detailed summary [here](https://github.com/legel/deepearth/blob/main/docs/NSF_DeepEarth_Workshop.pdf).  

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

#### NSF summer program in AI for disaster resilience, August 4-8, 2025
5 PhD students will join us for a 5 day program in Boulder, Colorado on ["Spatial AI for Extreme Events and Disaster Resilience"](https://i-guide.io/summer-school/summer-school-2025/).  We will geospatially and temporally simulate fire responses of plants (live fuel moisture content) at sub-meter scale.

# Code Implementation Preview
See [SPECIFICATIONS.md](https://github.com/legel/deepearth/blob/main/SPECIFICATIONS.MD) for a full preview of the entire DeepEarth architecture.  Below is a sample focused on the core model architecture.

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
Collaborators are invited to become core DeepEarth model contributors to bring this to life.  DeepEarth v0.01 prototyping is now underway, for simulation of native plants and pollinators in Florida and California between 2010-2025.  For more insight see this [pre-print preview](https://github.com/legel/deepearth/blob/main/docs/deepearth.pdf).

## License

MIT License - see LICENSE file for details.
