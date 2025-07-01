"""
Vision Encoder with DeepSeek Decoders for DeepEarth
Uses V-JEPA or other vision backbones with DeepSeek Transformer decoders
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
import numpy as np

# Import vision backbones
from encoders.vision.vjepa2_extractor import VJEPA2Extractor

from deepseek.modeling import DeepSeekTransformer, DeepSeekConfig


@dataclass  
class VisionEncoderConfig:
    """Configuration for vision encoders"""
    backbone: str = "vjepa2"  # "vjepa2", "dinov2", "clip", "sam"
    model_size: str = "base"  # "base", "large", "huge"
    freeze_backbone: bool = True
    patch_size: int = 16
    image_size: int = 224
    
    def get_native_dim(self) -> int:
        """Get native embedding dimension for backbone"""
        dims = {
            ("vjepa2", "base"): 768,
            ("vjepa2", "large"): 1024,
            ("vjepa2", "huge"): 1280,
            ("dinov2", "base"): 768,
            ("dinov2", "large"): 1024,
            ("dinov2", "giant"): 1536,
            ("clip", "base"): 768,
            ("clip", "large"): 1024,
            ("sam", "base"): 768,
        }
        return dims.get((self.backbone, self.model_size), 768)
    
    def get_num_patches(self) -> int:
        """Get number of patches for image"""
        return (self.image_size // self.patch_size) ** 2


class VisionBackbone(nn.Module):
    """Wrapper for various vision backbones"""
    
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config
        
        # Load appropriate backbone
        if config.backbone == "vjepa2":
            self.model = VJEPA2Extractor()
        elif config.backbone == "dinov2":
            # Would load DINOv2
            self.model = self._load_dinov2()
        elif config.backbone == "clip":
            # Would load CLIP vision encoder
            self.model = self._load_clip_vision()
        else:
            raise ValueError(f"Unknown backbone: {config.backbone}")
        
        # Freeze if requested
        if config.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.native_dim = config.get_native_dim()
        self.num_patches = config.get_num_patches()
    
    def _load_dinov2(self):
        """Load DINOv2 model"""
        # Placeholder - would actually load DINOv2
        import torchvision.models as models
        return models.vit_b_16(pretrained=True)
    
    def _load_clip_vision(self):
        """Load CLIP vision encoder"""
        # Placeholder - would actually load CLIP
        import torchvision.models as models
        return models.vit_b_16(pretrained=True)
    
    def extract_features(
        self,
        images: torch.Tensor,
        return_all_layers: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Extract features from images"""
        if self.config.backbone == "vjepa2":
            # V-JEPA specific extraction
            if hasattr(self.model, 'extract_native_embeddings'):
                return self.model.extract_native_embeddings(images)
            else:
                # Fallback
                features = self.model(images)
                return {
                    'patch_embeddings': features,
                    'global_embedding': features.mean(dim=1) if features.dim() > 2 else features,
                    'cls_embedding': features[:, 0] if features.dim() > 2 else features
                }
        else:
            # Generic extraction
            features = self.model(images)
            if hasattr(features, 'last_hidden_state'):
                patch_features = features.last_hidden_state
            else:
                patch_features = features
            
            return {
                'patch_embeddings': patch_features,
                'global_embedding': patch_features.mean(dim=1),
                'cls_embedding': patch_features[:, 0]
            }


class DeepSeekVisionDecoder(nn.Module):
    """
    Vision decoder using DeepSeek Transformer
    Decodes patch embeddings into universal tokens with spatial attention
    """
    
    def __init__(self, config: ModalityDecoderConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = DeepSeekMLP(
            config.input_dim,
            config.output_dim,
            config.intermediate_size
        )
        
        # Positional embeddings for patches
        max_patches = 1024  # Support up to 32x32 patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, max_patches, config.output_dim) * 0.02
        )
        
        # DeepSeek Transformer
        deepseek_config = config.to_deepseek_config()
        self.transformer = DeepSeekTransformer(deepseek_config)
        
        # Token generation strategy
        if config.num_tokens > 1:
            # Learnable query tokens for multi-token output
            self.query_tokens = nn.Parameter(
                torch.randn(1, config.num_tokens, config.output_dim) * 0.02
            )
            # Cross-attention from queries to patches
            self.cross_attention = nn.MultiheadAttention(
                config.output_dim,
                config.num_heads,
                dropout=config.attention_dropout_prob,
                batch_first=True
            )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(config.output_dim)
        
        # Spatial pooling options
        self.spatial_pool = nn.AdaptiveAvgPool1d(1) if config.num_tokens == 1 else None
    
    def forward(
        self,
        patch_embeddings: torch.Tensor,
        spatial_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode patch embeddings to universal tokens
        
        Args:
            patch_embeddings: (B, N_patches, D_native) patch features from vision backbone
            spatial_positions: Optional (B, N_patches, 2) normalized x,y positions
            
        Returns:
            universal_tokens: (B, K, D_universal) where K = num_tokens
        """
        B, N, _ = patch_embeddings.shape
        
        # Project patches to universal dimension
        hidden_states = self.input_projection(patch_embeddings)  # (B, N, D_universal)
        
        # Add positional embeddings
        if N <= self.position_embeddings.shape[1]:
            hidden_states = hidden_states + self.position_embeddings[:, :N]
        else:
            # Interpolate position embeddings for larger images
            pos_embed = F.interpolate(
                self.position_embeddings.transpose(1, 2),
                size=N,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            hidden_states = hidden_states + pos_embed
        
        # Apply transformer to encode spatial relationships
        transformed = self.transformer(hidden_states)
        
        # Generate output tokens
        if self.config.num_tokens > 1:
            # Use cross-attention from learnable queries to patches
            queries = self.query_tokens.expand(B, -1, -1)
            output_tokens, _ = self.cross_attention(
                query=queries,
                key=transformed,
                value=transformed
            )
        else:
            # Single token output - pool all patches
            if self.spatial_pool is not None:
                # Adaptive pooling
                output_tokens = self.spatial_pool(transformed.transpose(1, 2)).transpose(1, 2)
            else:
                # Weighted pooling based on attention
                attn_weights = torch.softmax(transformed.sum(dim=-1), dim=-1)  # (B, N)
                output_tokens = torch.einsum('bn,bnd->bd', attn_weights, transformed).unsqueeze(1)
        
        # Final normalization
        output_tokens = self.output_norm(output_tokens)
        
        return output_tokens


class VisionModalityProcessor(nn.Module):
    """
    Complete vision modality processor
    Combines vision backbone with DeepSeek decoder
    """
    
    def __init__(
        self,
        vision_config: VisionEncoderConfig,
        decoder_config: ModalityDecoderConfig
    ):
        super().__init__()
        
        # Vision backbone (frozen)
        self.backbone = VisionBackbone(vision_config)
        
        # Update decoder config with correct input dimension
        decoder_config.input_dim = self.backbone.native_dim
        
        # DeepSeek decoder to universal space
        self.decoder = DeepSeekVisionDecoder(decoder_config)
        
        # Store configs
        self.vision_config = vision_config
        self.decoder_config = decoder_config
    
    def forward(
        self,
        images: torch.Tensor,
        return_native: bool = False,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Process images to universal tokens
        
        Args:
            images: (B, C, H, W) input images
            return_native: Whether to return native embeddings
            return_attention: Whether to return attention maps
            
        Returns:
            universal_tokens: (B, K, D_universal) universal tokens
            native_features: Optional dict of native features
        """
        # Extract native features
        with torch.no_grad() if self.vision_config.freeze_backbone else torch.enable_grad():
            native_features = self.backbone.extract_features(images)
        
        # Decode to universal space
        patch_embeddings = native_features['patch_embeddings']
        universal_tokens = self.decoder(patch_embeddings)
        
        if return_native:
            return universal_tokens, native_features
        else:
            return universal_tokens
    
    def get_patch_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Get patch-level embeddings before pooling"""
        with torch.no_grad():
            native_features = self.backbone.extract_features(images)
            patch_embeddings = native_features['patch_embeddings']
            # Project through decoder but before final pooling
            hidden_states = self.decoder.input_projection(patch_embeddings)
            return hidden_states


# Factory functions for common configurations
def create_vision_processor(
    backbone: str = "vjepa2",
    model_size: str = "base",
    num_universal_tokens: int = 4,
    universal_dim: int = 2048,
    use_moe: bool = False
) -> VisionModalityProcessor:
    """
    Create a vision processor with sensible defaults
    
    Args:
        backbone: Vision backbone to use ("vjepa2", "dinov2", "clip", "sam")
        model_size: Size of vision model ("base", "large", "huge")
        num_universal_tokens: Number of universal tokens to generate
        universal_dim: Dimension of universal token space
        use_moe: Whether to use Mixture of Experts in decoder
        
    Returns:
        VisionModalityProcessor ready for use
    """
    # Vision encoder config
    vision_config = VisionEncoderConfig(
        backbone=backbone,
        model_size=model_size,
        freeze_backbone=True
    )
    
    # Decoder config with DeepSeek Transformer
    decoder_config = ModalityDecoderConfig(
        name="vision",
        input_dim=vision_config.get_native_dim(),
        output_dim=universal_dim,
        num_tokens=num_universal_tokens,
        num_layers=6,  # Deeper for vision due to spatial complexity
        num_heads=16 if universal_dim >= 1024 else 8,
        intermediate_size=universal_dim * 4,
        use_moe=use_moe,
        num_experts=8 if use_moe else 0,
        num_experts_per_tok=2
    )
    
    return VisionModalityProcessor(vision_config, decoder_config)


# Specialized processors for different use cases
def create_satellite_vision_processor(**kwargs) -> VisionModalityProcessor:
    """Create processor optimized for satellite imagery"""
    return create_vision_processor(
        backbone="vjepa2",
        model_size="large",  # Better for fine details
        num_universal_tokens=16,  # More tokens for spatial coverage
        **kwargs
    )


def create_agricultural_vision_processor(**kwargs) -> VisionModalityProcessor:
    """Create processor for agricultural monitoring"""
    return create_vision_processor(
        backbone="vjepa2",
        model_size="base",
        num_universal_tokens=9,  # 3x3 grid for field analysis
        use_moe=True,  # MoE for diverse crop types
        **kwargs
    )


def create_realtime_vision_processor(**kwargs) -> VisionModalityProcessor:
    """Create fast processor for real-time applications"""
    return create_vision_processor(
        backbone="vjepa2",
        model_size="base",
        num_universal_tokens=1,  # Single token for speed
        **kwargs
    )