"""
Universal Encoder V2 for DeepEarth
Uses DeepSeek Transformers for all modality decoders
"""

import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass

# Import the new modality processors
from encoders.vision.vision_encoder import (
    create_vision_processor,
    VisionModalityProcessor,
    VisionEncoderConfig
)
from encoders.language.language_encoder import (
    create_language_processor,
    LanguageModalityProcessor,
    LanguageModelConfig
)

# Import DeepSeek components for additional modalities
try:
    from deepseek.modeling import DeepSeekTransformer, DeepSeekConfig
    from deepseek.modules import DeepSeekMLP, DeepSeekMoE
except ImportError:
    print("Warning: DeepSeek modules not found.")


@dataclass
class UniversalEncoderConfig:
    """Configuration for universal encoder"""
    universal_dim: int = 2048
    
    # Vision settings
    vision_backbone: str = "vjepa2"
    vision_size: str = "base"
    vision_tokens: int = 4
    
    # Language settings
    language_model: str = "deepseek"
    language_size: str = "7b"
    language_precision: str = "int8"
    language_tokens: int = 1
    
    # Additional modality defaults
    default_decoder_layers: int = 4
    default_decoder_heads: int = 8
    use_moe_for_complex_modalities: bool = False


class AdditionalModalityProcessor(nn.Module):
    """
    Generic processor for additional modalities using DeepSeek decoders
    Used for weather, soil, species, etc.
    """
    
    def __init__(
        self,
        name: str,
        input_dim: int,
        output_dim: int = 2048,
        num_tokens: int = 1,
        num_layers: int = 4,
        use_moe: bool = False
    ):
        super().__init__()
        self.name = name
        
        # Create DeepSeek config
        config = DeepSeekConfig(
            hidden_size=output_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=max(8, output_dim // 128),
            intermediate_size=output_dim * 4,
            use_moe=use_moe,
            num_experts=8 if use_moe else 0,
            num_experts_per_tok=2
        )
        
        # Input projection
        self.input_projection = DeepSeekMLP(
            input_dim,
            output_dim,
            output_dim * 2
        )
        
        # DeepSeek Transformer
        self.transformer = DeepSeekTransformer(config)
        
        # Output tokens
        if num_tokens > 1:
            self.output_queries = nn.Parameter(
                torch.randn(1, num_tokens, output_dim) * 0.02
            )
        
        self.num_tokens = num_tokens
        self.output_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process modality data to universal tokens"""
        B = x.shape[0]
        
        # Handle different input shapes
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
        
        # Project to universal dimension
        hidden = self.input_projection(x)
        
        # Add output queries if multiple tokens
        if self.num_tokens > 1:
            queries = self.output_queries.expand(B, -1, -1)
            hidden = torch.cat([queries, hidden], dim=1)
        
        # Transform
        output = self.transformer(hidden)
        
        # Extract output tokens
        if self.num_tokens > 1:
            output = output[:, :self.num_tokens]
        else:
            output = output.mean(dim=1, keepdim=True)
        
        return self.output_norm(output)


class UniversalEncoderV2(nn.Module):
    """
    Universal Encoder using DeepSeek decoders for all modalities
    
    Key improvements:
    - All modality decoders use DeepSeek Transformers
    - Support for flexible language models (7B to 70B)
    - Automatic handling of various data types
    - Easy registration of new modalities
    """
    
    def __init__(self, config: UniversalEncoderConfig):
        super().__init__()
        self.config = config
        
        # Core modality processors
        self.processors = nn.ModuleDict()
        
        # Vision processor
        self.processors['vision'] = create_vision_processor(
            backbone=config.vision_backbone,
            model_size=config.vision_size,
            num_universal_tokens=config.vision_tokens,
            universal_dim=config.universal_dim
        )
        
        # Language processor
        self.processors['language'] = create_language_processor(
            model_family=config.language_model,
            model_size=config.language_size,
            precision=config.language_precision,
            num_universal_tokens=config.language_tokens,
            universal_dim=config.universal_dim
        )
        
        # Registry for additional modalities
        self.additional_processors = nn.ModuleDict()
    
    def register_modality(
        self,
        name: str,
        input_dim: int,
        num_tokens: int = 1,
        use_moe: bool = None,
        **kwargs
    ):
        """
        Register a new modality with automatic DeepSeek decoder creation
        
        Args:
            name: Modality name
            input_dim: Input dimension
            num_tokens: Number of universal tokens to generate
            use_moe: Whether to use MoE (auto-determined if None)
            **kwargs: Additional config options
        """
        # Auto-determine MoE usage based on complexity
        if use_moe is None:
            use_moe = (
                self.config.use_moe_for_complex_modalities and 
                (input_dim > 100 or num_tokens > 4)
            )
        
        processor = AdditionalModalityProcessor(
            name=name,
            input_dim=input_dim,
            output_dim=self.config.universal_dim,
            num_tokens=num_tokens,
            num_layers=kwargs.get('num_layers', self.config.default_decoder_layers),
            use_moe=use_moe
        )
        
        self.additional_processors[name] = processor
        print(f"✓ Registered modality '{name}' with DeepSeek decoder "
              f"(tokens={num_tokens}, moe={use_moe})")
    
    def forward(
        self,
        inputs: Dict[str, Any],
        return_native: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process multiple modalities to universal token space
        
        Args:
            inputs: Dict mapping modality names to inputs
            return_native: Whether to return native embeddings
            
        Returns:
            Dict mapping modality names to universal tokens
        """
        outputs = {}
        native_embeddings = {} if return_native else None
        
        # Process each modality
        for name, data in inputs.items():
            if name == 'vision' and 'vision' in self.processors:
                result = self.processors['vision'](data, return_native=return_native)
                if return_native:
                    outputs[name], native_embeddings[name] = result
                else:
                    outputs[name] = result
                    
            elif name == 'language' and 'language' in self.processors:
                result = self.processors['language'](data, return_native=return_native)
                if return_native:
                    outputs[name], native_embeddings[name] = result
                else:
                    outputs[name] = result
                    
            elif name in self.additional_processors:
                outputs[name] = self.additional_processors[name](data)
                
            else:
                print(f"Warning: No processor for modality '{name}'")
        
        if return_native:
            return outputs, native_embeddings
        else:
            return outputs
    
    def list_modalities(self) -> List[str]:
        """List all registered modalities"""
        core = list(self.processors.keys())
        additional = list(self.additional_processors.keys())
        return core + additional


def create_universal_encoder(
    vision_backbone: str = "vjepa2",
    language_model: str = "deepseek",
    language_size: str = "7b",
    universal_dim: int = 2048,
    **kwargs
) -> UniversalEncoderV2:
    """
    Create a universal encoder with specified settings
    
    Args:
        vision_backbone: Vision model to use
        language_model: Language model family
        language_size: Language model size
        universal_dim: Universal token dimension
        **kwargs: Additional config options
        
    Returns:
        Configured UniversalEncoderV2
    """
    config = UniversalEncoderConfig(
        universal_dim=universal_dim,
        vision_backbone=vision_backbone,
        language_model=language_model,
        language_size=language_size,
        **kwargs
    )
    
    encoder = UniversalEncoderV2(config)
    
    # Register common Earth observation modalities
    common_modalities = {
        'weather': {'input_dim': 5, 'num_tokens': 1},
        'weather_forecast': {'input_dim': 120, 'num_tokens': 4, 'use_moe': True},
        'soil': {'input_dim': 10, 'num_tokens': 1},
        'elevation': {'input_dim': 1, 'num_tokens': 1},
        'species': {'input_dim': 64, 'num_tokens': 2},
        'ndvi': {'input_dim': 1, 'num_tokens': 1},
        'temperature': {'input_dim': 1, 'num_tokens': 1},
        'precipitation': {'input_dim': 1, 'num_tokens': 1},
    }
    
    for name, config in common_modalities.items():
        encoder.register_modality(name, **config)
    
    return encoder


# Example usage
if __name__ == "__main__":
    # Create encoder with local-friendly settings
    encoder = create_universal_encoder(
        vision_backbone="vjepa2",
        language_model="deepseek",
        language_size="7b",
        language_precision="int8",
        universal_dim=2048
    )
    
    # Register custom modalities
    encoder.register_modality("hyperspectral", input_dim=224, num_tokens=8, use_moe=True)
    encoder.register_modality("lidar", input_dim=128, num_tokens=4)
    
    # Process multimodal inputs
    batch_size = 2
    inputs = {
        'vision': torch.randn(batch_size, 3, 224, 224),
        'language': ["Drought conditions observed", "Healthy crop growth"],
        'weather': torch.randn(batch_size, 5),
        'soil': torch.randn(batch_size, 10),
        'hyperspectral': torch.randn(batch_size, 224)
    }
    
    # Get universal tokens
    universal_tokens = encoder(inputs)
    
    print("\nUniversal token shapes:")
    for name, tokens in universal_tokens.items():
        print(f"  {name}: {tokens.shape}")
    
    # List all modalities
    print(f"\nRegistered modalities: {encoder.list_modalities()}")