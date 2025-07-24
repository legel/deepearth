"""
DeepEarth Multimodal Model Integration
Combines Grid4D spatiotemporal encoding with V-JEPA vision and DeepSeek language encoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

# Import existing multimodal components
from encoders.vision.vjepa2_extractor import VJEPA2Extractor as VJEPAEncoder
from encoders.language.deepseek_v3_encoder import DeepSeekV3Encoder as DeepSeekEncoder

# Import Grid4D components
from models.encoders import Grid4DEncoder, ModalityEncoder
from models.decoders import ModalityDecoder, SpatiotemporalDecoder
from models.transformers import Transformer
from models.configs import DeepEarthConfig, TransformerConfig

# Import geospatial utilities
from geospatial.geo2xyz import geo2xyz  # Assuming this converts lat/lon to xyz


class DeepEarthMultimodal(nn.Module):
    """
    Unified DeepEarth model combining:
    - Grid4D spatiotemporal encoding
    - V-JEPA vision encoding
    - DeepSeek language encoding
    - Cross-modal fusion with spatial awareness
    """
    
    def __init__(
        self,
        config: DeepEarthConfig,
        vision_hidden_size: int = 768,
        text_hidden_size: int = 1024,
        freeze_vision: bool = True,
        freeze_language: bool = True,
    ):
        super().__init__()
        self.config = config
        
        # Initialize encoders
        self.grid4d_encoder = Grid4DEncoder(config)
        self.vision_encoder = VJEPAEncoder()
        self.language_encoder = DeepSeekEncoder(out_dim=config.hidden_dim, freeze=freeze_language)
        
        # Freeze pre-trained encoders if requested
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        
        # Projection heads to align dimensions
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_hidden_size, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.language_proj = nn.Sequential(
            nn.Linear(text_hidden_size, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Modality-specific encoders for additional data types
        self.modality_encoders = nn.ModuleDict()
        self.modality_decoders = nn.ModuleDict()
        
        # Cross-modal fusion transformer with spatial awareness
        self.cross_modal_fusion = Transformer(config.cross_modal_fusion_config)
        
        # Spatiotemporal decoders
        self.spatial_decoder = SpatiotemporalDecoder('spatial', output_dim=3, config=config)
        self.temporal_decoder = SpatiotemporalDecoder('temporal', output_dim=1, config=config)
        
        # Task-specific heads
        self.reconstruction_heads = nn.ModuleDict({
            'vision': nn.Linear(config.hidden_dim, vision_hidden_size),
            'language': nn.Linear(config.hidden_dim, text_hidden_size),
        })
        
        # Learnable modality tokens
        self.modality_tokens = nn.ParameterDict({
            'spatial': nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02),
            'vision': nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02),
            'language': nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02),
        })
        
    def add_modality(self, name: str, input_dim: int, output_dim: int):
        """Add a new modality encoder/decoder pair"""
        encoder_config = self.config.modality_encoder_config
        self.modality_encoders[name] = ModalityEncoder(
            name, input_dim, self.config, encoder_config
        )
        self.modality_decoders[name] = ModalityDecoder(
            name, output_dim, self.config
        )
        self.modality_tokens[name] = nn.Parameter(
            torch.randn(1, 1, self.config.hidden_dim) * 0.02
        )
        
    def encode_spatiotemporal(self, xyzt: torch.Tensor) -> torch.Tensor:
        """Encode spatiotemporal coordinates"""
        return self.grid4d_encoder(xyzt)
    
    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode visual input"""
        vision_features = self.vision_encoder(images)  # (B, N_patches, D_v)
        return self.vision_proj(vision_features)  # (B, N_patches, D)
    
    def encode_language(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode language input"""
        outputs = self.language_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        text_features = outputs.last_hidden_state  # (B, N_tokens, D_t)
        return self.language_proj(text_features)  # (B, N_tokens, D)
    
    def forward(
        self,
        xyzt: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        modalities: Optional[Dict[str, torch.Tensor]] = None,
        return_reconstructions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multimodal model
        
        Args:
            xyzt: (B, 4) spatiotemporal coordinates
            images: (B, C, H, W) visual input
            input_ids: (B, L) language tokens
            attention_mask: (B, L) attention mask for language
            modalities: Dict of additional modality data
            return_reconstructions: Whether to compute reconstructions
            
        Returns:
            Dictionary containing embeddings and optional reconstructions
        """
        B = xyzt.shape[0]
        tokens = []
        token_types = []
        
        # Always encode spatiotemporal coordinates
        spatial_emb = self.encode_spatiotemporal(xyzt)
        tokens.append(self.modality_tokens['spatial'].expand(B, -1, -1))
        tokens.append(spatial_emb.unsqueeze(1))
        token_types.extend(['spatial_token', 'spatial_data'])
        
        # Encode vision if provided
        if images is not None:
            vision_emb = self.encode_vision(images)
            tokens.append(self.modality_tokens['vision'].expand(B, -1, -1))
            tokens.append(vision_emb)
            token_types.extend(['vision_token'] + ['vision_data'] * vision_emb.shape[1])
        
        # Encode language if provided
        if input_ids is not None:
            language_emb = self.encode_language(input_ids, attention_mask)
            tokens.append(self.modality_tokens['language'].expand(B, -1, -1))
            tokens.append(language_emb)
            token_types.extend(['language_token'] + ['language_data'] * language_emb.shape[1])
        
        # Encode additional modalities
        if modalities is not None:
            for name, data in modalities.items():
                if name in self.modality_encoders:
                    mod_emb = self.modality_encoders[name](data)
                    tokens.append(self.modality_tokens[name].expand(B, -1, -1))
                    tokens.append(mod_emb.unsqueeze(1))
                    token_types.extend([f'{name}_token', f'{name}_data'])
        
        # Concatenate all tokens
        token_sequence = torch.cat(tokens, dim=1)  # (B, N_total, D)
        
        # Cross-modal fusion
        fused_embeddings = self.cross_modal_fusion(token_sequence)
        
        # Extract pooled representations
        outputs = {
            'fused_embeddings': fused_embeddings,
            'spatial_pool': fused_embeddings[:, 0],  # First spatial token
        }
        
        # Extract other modality pools
        idx = 2  # Skip spatial token and data
        if images is not None:
            outputs['vision_pool'] = fused_embeddings[:, idx]
            idx += 1 + vision_emb.shape[1]
        
        if input_ids is not None:
            outputs['language_pool'] = fused_embeddings[:, idx]
            idx += 1 + language_emb.shape[1]
            
        # Reconstructions if requested
        if return_reconstructions:
            outputs['reconstructions'] = {}
            
            # Spatial reconstruction
            outputs['reconstructions']['xyz'] = self.spatial_decoder(outputs['spatial_pool'])
            outputs['reconstructions']['t'] = self.temporal_decoder(outputs['spatial_pool'])
            
            # Vision reconstruction
            if images is not None:
                vision_start = 2  # After spatial token and data
                vision_end = vision_start + vision_emb.shape[1]
                vision_recon = self.reconstruction_heads['vision'](
                    fused_embeddings[:, vision_start:vision_end]
                )
                outputs['reconstructions']['vision'] = vision_recon
            
            # Language reconstruction  
            if input_ids is not None:
                lang_start = idx - language_emb.shape[1]
                lang_end = idx
                lang_recon = self.reconstruction_heads['language'](
                    fused_embeddings[:, lang_start:lang_end]
                )
                outputs['reconstructions']['language'] = lang_recon
                
            # Additional modality reconstructions
            if modalities is not None:
                for name in modalities:
                    if name in self.modality_decoders:
                        # Find the token position for this modality
                        for i, t_type in enumerate(token_types):
                            if t_type == f'{name}_token':
                                mod_pool = fused_embeddings[:, i]
                                outputs['reconstructions'][name] = self.modality_decoders[name](mod_pool)
                                break
        
        return outputs


def create_deepearth_model(
    spatial_resolutions: List[int] = None,
    temporal_resolutions: List[int] = None,
    hidden_dim: int = 1024,
    freeze_backbones: bool = True,
) -> DeepEarthMultimodal:
    """
    Factory function to create a DeepEarth multimodal model
    """
    config = DeepEarthConfig(
        hidden_dim=hidden_dim,
        n_heads=16,
        n_layers=12,
        spatial_resolutions=spatial_resolutions,
        temporal_resolutions=temporal_resolutions,
    )
    
    # Assuming V-JEPA outputs 768-dim features
    vision_hidden = 768
    # DeepSeek language model hidden size
    text_hidden = 1024
    
    model = DeepEarthMultimodal(
        config=config,
        vision_hidden_size=vision_hidden,
        text_hidden_size=text_hidden,
        freeze_vision=freeze_backbones,
        freeze_language=freeze_backbones,
    )
    
    # Add common environmental modalities
    model.add_modality('temperature', input_dim=5, output_dim=5)  # temp, humidity, pressure, etc
    model.add_modality('species', input_dim=64, output_dim=64)    # species embeddings
    
    return model


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_deepearth_model()
    
    # Sample inputs
    batch_size = 4
    xyzt = torch.rand(batch_size, 4)  # Spatiotemporal coordinates
    images = torch.randn(batch_size, 3, 224, 224)  # Images
    input_ids = torch.randint(0, 1000, (batch_size, 32))  # Text tokens
    attention_mask = torch.ones_like(input_ids)
    
    # Additional modalities
    modalities = {
        'temperature': torch.randn(batch_size, 5),
        'species': torch.randn(batch_size, 64),
    }
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            xyzt=xyzt,
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            modalities=modalities,
            return_reconstructions=True,
        )
        
    print("Output shapes:")
    print(f"Spatial pool: {outputs['spatial_pool'].shape}")
    print(f"Vision pool: {outputs['vision_pool'].shape}")
    print(f"Language pool: {outputs['language_pool'].shape}")
    print(f"Fused embeddings: {outputs['fused_embeddings'].shape}")
    
    print("\nReconstructions:")
    for key, value in outputs['reconstructions'].items():
        print(f"{key}: {value.shape}")
