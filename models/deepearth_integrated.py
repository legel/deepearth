"""
Integrated DeepEarth Model with Universal Token Architecture
Combines native encoders, universal projection, and cross-modal fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from encoders.universal_encoder import (
    UniversalEncoderModule, 
    EncoderConfig,
    UniversalDecoder
)
from models.cross_modal_fusion import (
    CrossModalFusion,
    HierarchicalFusion,
    FusionConfig
)
from models.encoders import Grid4DEncoder
from models.configs import DeepEarthConfig


@dataclass
class IntegratedDeepEarthConfig:
    """Configuration for the integrated DeepEarth model"""
    # Universal token configuration
    universal_dim: int = 2048
    
    # Encoder configurations
    vision_config: EncoderConfig = None
    language_config: EncoderConfig = None
    grid4d_config: DeepEarthConfig = None
    
    # Fusion configuration
    fusion_config: FusionConfig = None
    use_hierarchical_fusion: bool = True
    num_fusion_levels: int = 3
    
    # Task-specific configurations
    enable_reconstruction: bool = True
    enable_prediction: bool = True
    enable_generation: bool = False
    
    def __post_init__(self):
        # Set defaults if not provided
        if self.vision_config is None:
            self.vision_config = EncoderConfig(
                name="vision",
                native_dim=768,  # V-JEPA base
                universal_dim=self.universal_dim,
                num_tokens_per_sample=4,
                projection_type="attention",
                freeze_backbone=True
            )
        
        if self.language_config is None:
            self.language_config = EncoderConfig(
                name="language",
                native_dim=4096,  # DeepSeek large
                universal_dim=self.universal_dim,
                num_tokens_per_sample=1,
                projection_type="mlp",
                freeze_backbone=True
            )
        
        if self.grid4d_config is None:
            self.grid4d_config = DeepEarthConfig(
                hidden_dim=self.universal_dim,
                n_spatial_levels=16,
                n_temporal_levels=8
            )
        
        if self.fusion_config is None:
            self.fusion_config = FusionConfig(
                universal_dim=self.universal_dim,
                num_fusion_layers=24,
                num_heads=16,
                cross_attention_freq=3,
                spatial_aware=True,
                temporal_aware=True
            )


class DeepEarthIntegrated(nn.Module):
    """
    Integrated DeepEarth model with universal token architecture
    
    This model:
    1. Extracts native embeddings from pretrained encoders (V-JEPA, DeepSeek)
    2. Projects them to universal token space (2048-dim)
    3. Adds spatiotemporal embeddings via Grid4D
    4. Performs cross-modal fusion
    5. Supports various downstream tasks
    """
    
    def __init__(self, config: IntegratedDeepEarthConfig):
        super().__init__()
        self.config = config
        
        # Initialize universal encoder module
        encoder_configs = {
            "vision": config.vision_config,
            "language": config.language_config
        }
        self.universal_encoder = UniversalEncoderModule(encoder_configs)
        
        # Initialize Grid4D encoder for spatiotemporal information
        self.grid4d_encoder = Grid4DEncoder(config.grid4d_config)
        
        # Project Grid4D output to universal dimension
        self.grid4d_projector = nn.Linear(
            config.grid4d_config.hidden_dim,
            config.universal_dim
        )
        
        # Initialize fusion module
        if config.use_hierarchical_fusion:
            self.fusion = HierarchicalFusion(
                config.fusion_config,
                num_levels=config.num_fusion_levels
            )
        else:
            self.fusion = CrossModalFusion(config.fusion_config)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        
        if config.enable_reconstruction:
            self._init_reconstruction_heads()
        
        if config.enable_prediction:
            self._init_prediction_heads()
        
        if config.enable_generation:
            self._init_generation_heads()
        
        # Additional modality encoders (can be added dynamically)
        self.additional_encoders = nn.ModuleDict()
        self.additional_projectors = nn.ModuleDict()
    
    def _init_reconstruction_heads(self):
        """Initialize reconstruction heads for self-supervised learning"""
        # Universal decoder for reconstructing native embeddings
        self.universal_decoder = UniversalDecoder(
            target_configs={
                'vision': {'dim': self.config.vision_config.native_dim, 'type': 'mlp'},
                'language': {'dim': self.config.language_config.native_dim, 'type': 'mlp'},
                'spatial': {'dim': 3, 'type': 'mlp'},  # xyz coordinates
                'temporal': {'dim': 1, 'type': 'mlp'}  # t coordinate
            },
            universal_dim=self.config.universal_dim
        )
    
    def _init_prediction_heads(self):
        """Initialize prediction heads for downstream tasks"""
        self.task_heads['temperature_prediction'] = nn.Sequential(
            nn.Linear(self.config.universal_dim, self.config.universal_dim // 2),
            nn.LayerNorm(self.config.universal_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.universal_dim // 2, 1)
        )
        
        self.task_heads['land_cover_classification'] = nn.Sequential(
            nn.Linear(self.config.universal_dim, self.config.universal_dim // 2),
            nn.LayerNorm(self.config.universal_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.universal_dim // 2, 10)  # 10 land cover classes
        )
    
    def _init_generation_heads(self):
        """Initialize generation heads for synthesis tasks"""
        # Placeholder for future generation capabilities
        pass
    
    def add_modality(
        self,
        name: str,
        encoder: nn.Module,
        native_dim: int,
        num_tokens: int = 1,
        projection_type: str = "mlp"
    ):
        """Add a new modality encoder dynamically"""
        # Add the encoder
        self.additional_encoders[name] = encoder
        
        # Create projector configuration
        projector_config = EncoderConfig(
            name=name,
            native_dim=native_dim,
            universal_dim=self.config.universal_dim,
            num_tokens_per_sample=num_tokens,
            projection_type=projection_type
        )
        
        # Add projector
        from encoders.universal_encoder import UniversalProjector
        self.additional_projectors[name] = UniversalProjector(projector_config)
        
        # Add to fusion module's modality embeddings
        self.fusion.st_embedding.add_modality(name)
    
    def forward(
        self,
        xyzt: torch.Tensor,
        vision_input: Optional[torch.Tensor] = None,
        language_input: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        additional_modalities: Optional[Dict[str, torch.Tensor]] = None,
        task: Optional[str] = None,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through integrated DeepEarth model
        
        Args:
            xyzt: (B, 4) spatiotemporal coordinates
            vision_input: (B, C, H, W) images
            language_input: Text tokens or dict with input_ids and attention_mask
            additional_modalities: Dict of additional modality inputs
            task: Specific task to perform (if None, returns embeddings)
            return_intermediates: Return intermediate representations
            
        Returns:
            Dict containing results based on task and settings
        """
        B = xyzt.shape[0]
        outputs = {}
        
        # Step 1: Extract spatiotemporal embeddings
        spatial_embedding = self.grid4d_encoder(xyzt)  # (B, D_grid4d)
        spatial_universal = self.grid4d_projector(spatial_embedding)  # (B, D_universal)
        spatial_universal = spatial_universal.unsqueeze(1)  # (B, 1, D_universal)
        
        # Step 2: Extract and project modality embeddings to universal space
        modality_tokens = {'spatial': spatial_universal}
        
        # Prepare inputs for universal encoder
        encoder_inputs = {}
        if vision_input is not None:
            encoder_inputs['vision'] = vision_input
        if language_input is not None:
            encoder_inputs['language'] = language_input
        
        # Get universal tokens
        if encoder_inputs:
            universal_tokens = self.universal_encoder(encoder_inputs)
            modality_tokens.update(universal_tokens)
        
        # Step 3: Process additional modalities
        if additional_modalities is not None:
            for name, input_data in additional_modalities.items():
                if name in self.additional_encoders:
                    # Extract native embeddings
                    native_embeds = self.additional_encoders[name](input_data)
                    
                    # Project to universal space
                    universal = self.additional_projectors[name](native_embeds)
                    modality_tokens[name] = universal
        
        # Step 4: Prepare position information
        spatial_positions = {
            'spatial': xyzt[:, :2].unsqueeze(1),  # (B, 1, 2) - xy coordinates
        }
        
        temporal_positions = {
            'spatial': xyzt[:, 3:4].unsqueeze(1),  # (B, 1, 1) - t coordinate
        }
        
        # Add positions for vision tokens if present
        if 'vision' in modality_tokens:
            num_vision_tokens = modality_tokens['vision'].shape[1]
            # Create grid positions for vision tokens
            grid_size = int(num_vision_tokens ** 0.5)
            x_pos = torch.linspace(0, 1, grid_size, device=xyzt.device)
            y_pos = torch.linspace(0, 1, grid_size, device=xyzt.device)
            grid_x, grid_y = torch.meshgrid(x_pos, y_pos, indexing='xy')
            vision_positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            spatial_positions['vision'] = vision_positions.unsqueeze(0).expand(B, -1, -1)
            
            # Vision tokens share same temporal position as spatial
            temporal_positions['vision'] = temporal_positions['spatial'].expand(B, num_vision_tokens, -1)
        
        # Step 5: Cross-modal fusion
        fusion_outputs = self.fusion(
            modality_tokens,
            spatial_positions,
            temporal_positions,
            return_all_layers=return_intermediates
        )
        
        outputs.update(fusion_outputs)
        
        # Step 6: Task-specific processing
        if task is not None and task in self.task_heads:
            task_output = self.task_heads[task](fusion_outputs['fused_representation'])
            outputs['task_output'] = task_output
        
        # Step 7: Reconstruction outputs (if enabled)
        if self.config.enable_reconstruction and hasattr(self, 'universal_decoder'):
            reconstructions = {}
            
            # Reconstruct spatial coordinates
            reconstructions['spatial'] = self.universal_decoder(
                fusion_outputs['fused_representation'],
                'spatial'
            )
            reconstructions['temporal'] = self.universal_decoder(
                fusion_outputs['fused_representation'],
                'temporal'
            )
            
            # Reconstruct modality-specific embeddings
            for name in ['vision', 'language']:
                if name in modality_tokens:
                    # Use pooled representation for reconstruction
                    pooled = modality_tokens[name].mean(dim=1)
                    reconstructions[name] = self.universal_decoder(pooled, name)
            
            outputs['reconstructions'] = reconstructions
        
        # Step 8: Add intermediate representations if requested
        if return_intermediates:
            outputs['intermediates'] = {
                'spatial_embedding': spatial_embedding,
                'modality_tokens': modality_tokens,
                'spatial_positions': spatial_positions,
                'temporal_positions': temporal_positions
            }
        
        return outputs
    
    def extract_features(
        self,
        xyzt: torch.Tensor,
        vision_input: Optional[torch.Tensor] = None,
        language_input: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        level: str = 'fused'
    ) -> torch.Tensor:
        """
        Extract features at different levels
        
        Args:
            level: 'native', 'universal', or 'fused'
        """
        with torch.no_grad():
            outputs = self.forward(
                xyzt, vision_input, language_input,
                return_intermediates=True
            )
            
            if level == 'fused':
                return outputs['fused_representation']
            elif level == 'universal':
                # Return concatenated universal tokens
                tokens = outputs['intermediates']['modality_tokens']
                return torch.cat([tokens[k].mean(dim=1) for k in tokens], dim=-1)
            elif level == 'native':
                # Return spatial embedding (native Grid4D)
                return outputs['intermediates']['spatial_embedding']
            else:
                raise ValueError(f"Unknown level: {level}")


def create_integrated_deepearth(
    universal_dim: int = 2048,
    vision_model: str = "vjepa2-base",
    language_model: str = "deepseek-v3",
    num_fusion_layers: int = 24,
    freeze_backbones: bool = True,
    enable_all_tasks: bool = True
) -> DeepEarthIntegrated:
    """Factory function to create integrated DeepEarth model"""
    
    # Create configuration
    config = IntegratedDeepEarthConfig(
        universal_dim=universal_dim,
        enable_reconstruction=enable_all_tasks,
        enable_prediction=enable_all_tasks,
        enable_generation=False  # Coming soon
    )
    
    # Override backbone settings
    config.vision_config.freeze_backbone = freeze_backbones
    config.language_config.freeze_backbone = freeze_backbones
    config.fusion_config.num_fusion_layers = num_fusion_layers
    
    # Create model
    model = DeepEarthIntegrated(config)
    
    # Add common Earth observation modalities
    # Temperature/Weather
    from models.encoders import ModalityEncoder as SimpleModalityEncoder
    weather_encoder = SimpleModalityEncoder(
        modality_name="weather",
        input_dim=5,  # temp, humidity, pressure, wind_u, wind_v
        config=config.grid4d_config,
        encoder_config=config.grid4d_config.modality_encoder_config
    )
    model.add_modality("weather", weather_encoder, native_dim=config.universal_dim // 2)
    
    # Species observations
    species_encoder = SimpleModalityEncoder(
        modality_name="species",
        input_dim=64,  # Species embedding dimension
        config=config.grid4d_config,
        encoder_config=config.grid4d_config.modality_encoder_config
    )
    model.add_modality("species", species_encoder, native_dim=config.universal_dim // 2)
    
    # Soil properties
    soil_encoder = SimpleModalityEncoder(
        modality_name="soil",
        input_dim=10,  # Various soil properties
        config=config.grid4d_config,
        encoder_config=config.grid4d_config.modality_encoder_config
    )
    model.add_modality("soil", soil_encoder, native_dim=config.universal_dim // 2)
    
    return model


class DeepEarthLightning(nn.Module):
    """PyTorch Lightning wrapper for easier training"""
    
    def __init__(
        self,
        model_config: IntegratedDeepEarthConfig,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        weight_decay: float = 0.05
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        self.model = DeepEarthIntegrated(model_config)
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.contrastive_temperature = 0.07
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        # Extract inputs
        xyzt = batch['xyzt']
        vision = batch.get('images')
        language = batch.get('language')
        modalities = batch.get('modalities', {})
        
        # Forward pass
        outputs = self.model(
            xyzt=xyzt,
            vision_input=vision,
            language_input=language,
            additional_modalities=modalities,
            return_intermediates=True
        )
        
        # Compute losses
        total_loss = 0.0
        loss_dict = {}
        
        # Reconstruction losses
        if 'reconstructions' in outputs:
            # Spatial reconstruction
            if 'spatial' in outputs['reconstructions']:
                spatial_loss = self.reconstruction_loss(
                    outputs['reconstructions']['spatial'],
                    xyzt[:, :3]
                )
                total_loss += spatial_loss
                loss_dict['spatial_recon'] = spatial_loss
            
            # Temporal reconstruction
            if 'temporal' in outputs['reconstructions']:
                temporal_loss = self.reconstruction_loss(
                    outputs['reconstructions']['temporal'],
                    xyzt[:, 3:4]
                )
                total_loss += temporal_loss
                loss_dict['temporal_recon'] = temporal_loss
        
        # Contrastive losses between modalities
        if len(outputs.get('modality_tokens', {})) > 1:
            contrastive_loss = self.compute_contrastive_loss(
                outputs['modality_tokens']
            )
            total_loss += contrastive_loss
            loss_dict['contrastive'] = contrastive_loss
        
        # Task-specific losses
        if 'task_output' in outputs and 'task_target' in batch:
            task_name = batch.get('task_name', 'unknown')
            if 'classification' in task_name:
                task_loss = self.classification_loss(
                    outputs['task_output'],
                    batch['task_target']
                )
            else:
                task_loss = self.reconstruction_loss(
                    outputs['task_output'],
                    batch['task_target']
                )
            total_loss += task_loss
            loss_dict[f'{task_name}_loss'] = task_loss
        
        # Log losses
        for name, value in loss_dict.items():
            self.log(f'train/{name}', value, prog_bar=True)
        self.log('train/total_loss', total_loss)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Similar to training step but without gradients
        xyzt = batch['xyzt']
        vision = batch.get('images')
        language = batch.get('language')
        modalities = batch.get('modalities', {})
        
        outputs = self.model(
            xyzt=xyzt,
            vision_input=vision,
            language_input=language,
            additional_modalities=modalities
        )
        
        # Compute validation metrics
        metrics = {}
        
        # Feature quality metrics
        if 'fused_representation' in outputs:
            # Measure representation diversity
            rep_std = outputs['fused_representation'].std(dim=0).mean()
            metrics['representation_std'] = rep_std
            
            # Measure representation sparsity
            sparsity = (outputs['fused_representation'].abs() < 0.01).float().mean()
            metrics['representation_sparsity'] = sparsity
        
        # Log metrics
        for name, value in metrics.items():
            self.log(f'val/{name}', value)
        
        return metrics
    
    def compute_contrastive_loss(self, modality_tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute contrastive loss between different modalities"""
        # Get pooled representations for each modality
        pooled_reps = {}
        for name, tokens in modality_tokens.items():
            if tokens.dim() == 3:
                pooled_reps[name] = tokens.mean(dim=1)  # Average pooling
            else:
                pooled_reps[name] = tokens
        
        # Compute pairwise contrastive losses
        total_loss = 0.0
        num_pairs = 0
        
        modality_names = list(pooled_reps.keys())
        for i in range(len(modality_names)):
            for j in range(i + 1, len(modality_names)):
                name_i, name_j = modality_names[i], modality_names[j]
                rep_i = F.normalize(pooled_reps[name_i], dim=-1)
                rep_j = F.normalize(pooled_reps[name_j], dim=-1)
                
                # Compute similarity matrix
                sim_matrix = torch.matmul(rep_i, rep_j.T) / self.contrastive_temperature
                
                # Contrastive loss (assuming batch items are positive pairs)
                labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
                loss_ij = F.cross_entropy(sim_matrix, labels)
                loss_ji = F.cross_entropy(sim_matrix.T, labels)
                
                total_loss += (loss_ij + loss_ji) / 2
                num_pairs += 1
        
        return total_loss / max(num_pairs, 1)
    
    def configure_optimizers(self):
        # Separate parameters by learning rate
        backbone_params = []
        projection_params = []
        fusion_params = []
        
        for name, param in self.model.named_parameters():
            if 'encoder' in name and ('vjepa' in name or 'deepseek' in name):
                backbone_params.append(param)
            elif 'projector' in name or 'projection' in name:
                projection_params.append(param)
            else:
                fusion_params.append(param)
        
        # Different learning rates for different components
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.learning_rate * 0.1},  # Lower LR for pretrained
            {'params': projection_params, 'lr': self.learning_rate},
            {'params': fusion_params, 'lr': self.learning_rate}
        ], weight_decay=self.weight_decay)
        
        # Learning rate scheduler with warmup
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[self.learning_rate * 0.1, self.learning_rate, self.learning_rate],
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.warmup_steps / self.trainer.estimated_stepping_batches
            ),
            'interval': 'step'
        }
        
        return [optimizer], [scheduler]


# Example usage and testing
if __name__ == "__main__":
    print("Creating Integrated DeepEarth Model...")
    
    # Create model
    model = create_integrated_deepearth(
        universal_dim=2048,
        num_fusion_layers=24,
        freeze_backbones=True
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    test_inputs = {
        'xyzt': torch.randn(batch_size, 4),
        'vision_input': torch.randn(batch_size, 3, 224, 224),
        'language_input': {
            'input_ids': torch.randint(0, 1000, (batch_size, 32)),
            'attention_mask': torch.ones(batch_size, 32)
        },
        'additional_modalities': {
            'weather': torch.randn(batch_size, 5),
            'species': torch.randn(batch_size, 64)
        }
    }
    
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(**test_inputs, return_intermediates=True)
    
    print("\nOutput shapes:")
    print(f"Fused representation: {outputs['fused_representation'].shape}")
    print(f"All tokens: {outputs['all_tokens'].shape}")
    
    if 'modality_tokens' in outputs:
        print("\nModality-specific tokens:")
        for name, tokens in outputs['modality_tokens'].items():
            print(f"  {name}: {tokens.shape}")
    
    if 'reconstructions' in outputs:
        print("\nReconstructions:")
        for name, recon in outputs['reconstructions'].items():
            print(f"  {name}: {recon.shape}")
    
    # Test feature extraction
    print("\nTesting feature extraction...")
    features_fused = model.extract_features(**test_inputs, level='fused')
    features_universal = model.extract_features(**test_inputs, level='universal')
    features_native = model.extract_features(**test_inputs, level='native')
    
    print(f"Fused features: {features_fused.shape}")
    print(f"Universal features: {features_universal.shape}")
    print(f"Native features: {features_native.shape}")
    
    # Test task-specific outputs
    print("\nTesting task outputs...")
    task_outputs = model(
        test_inputs['xyzt'],
        test_inputs['vision_input'],
        test_inputs['language_input'],
        task='temperature_prediction'
    )
    
    if 'task_output' in task_outputs:
        print(f"Temperature prediction shape: {task_outputs['task_output'].shape}")
    
    print("\nModel successfully created and tested!")
