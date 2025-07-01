"""
Shared configuration for modality decoders using DeepSeek Transformers
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from encoders.deepseek_components import DeepSeekConfig


@dataclass
class ModalityDecoderConfig:
    """
    Configuration for modality-specific decoders using DeepSeek Transformers
    
    As per SPECIFICATIONS.md:
    - Each modality decoder is a "small DeepSeek Transformer"
    - Capable of discovering attention patterns
    - Projects from native embeddings to universal token space
    """
    name: str
    input_dim: int  # Native dimension from pretrained encoder
    output_dim: int  # Universal dimension (e.g., 2048)
    num_tokens: int = 1  # Number of output tokens
    
    # DeepSeek Transformer config
    num_layers: int = 4
    num_heads: int = 8
    num_key_value_heads: Optional[int] = None  # For GQA
    intermediate_size: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    
    # MoE settings (for complex modalities)
    use_moe: bool = False  # Use Mixture of Experts
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: Optional[int] = None
    n_shared_experts: Optional[int] = None
    norm_topk_prob: bool = True
    moe_layer_freq: int = 1  # MoE every N layers
    
    # Additional settings
    use_gate: bool = True  # Use gated projections
    use_normalization: bool = True
    activation: str = "silu"
    
    # Token generation settings
    use_cross_attention: bool = True  # For multi-token generation
    use_position_encoding: bool = True
    input_use_moe: bool = False  # Use MoE for input projection
    output_use_moe: bool = False  # Use MoE for output generation
    
    def to_deepseek_config(self) -> DeepSeekConfig:
        """Convert to DeepSeek configuration object"""
        config = DeepSeekConfig(
            hidden_size=self.output_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads or self.num_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.activation,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            attention_dropout=self.attention_dropout_prob,
            # MoE settings
            n_routed_experts=self.num_experts if self.use_moe else None,
            num_experts_per_tok=self.num_experts_per_tok,
            moe_intermediate_size=self.moe_intermediate_size or (self.intermediate_size // 4),
            n_shared_experts=self.n_shared_experts,
            norm_topk_prob=self.norm_topk_prob,
        )
        return config
    
    @classmethod
    def create_default(cls, name: str, input_dim: int, output_dim: int = 2048) -> "ModalityDecoderConfig":
        """Create default config for a modality"""
        return cls(
            name=name,
            input_dim=input_dim,
            output_dim=output_dim,
            num_tokens=1,
            num_layers=4,
            num_heads=max(8, output_dim // 256),
            intermediate_size=output_dim * 4,
            moe_intermediate_size=output_dim // 2
        )
    
    @classmethod
    def create_vision_config(cls, input_dim: int = 768, output_dim: int = 2048, num_tokens: int = 4) -> "ModalityDecoderConfig":
        """Create config optimized for vision modality"""
        return cls(
            name="vision",
            input_dim=input_dim,
            output_dim=output_dim,
            num_tokens=num_tokens,
            num_layers=6,  # Deeper for spatial relationships
            num_heads=16,
            num_key_value_heads=4,  # GQA for efficiency
            intermediate_size=output_dim * 4,
            use_moe=num_tokens > 4,  # Use MoE for many tokens
            n_shared_experts=1,
            use_cross_attention=True,
            use_position_encoding=True
        )
    
    @classmethod
    def create_language_config(cls, input_dim: int = 4096, output_dim: int = 2048, num_tokens: int = 1) -> "ModalityDecoderConfig":
        """Create config optimized for language modality"""
        return cls(
            name="language",
            input_dim=input_dim,
            output_dim=output_dim,
            num_tokens=num_tokens,
            num_layers=4,
            num_heads=16,
            intermediate_size=output_dim * 4,
            attention_dropout_prob=0.1,  # Higher dropout for language
            use_moe=num_tokens > 2,
            input_use_moe=True,  # MoE for diverse text inputs
            moe_layer_freq=2  # MoE every other layer
        )
    
    @classmethod
    def create_timeseries_config(cls, input_dim: int, output_dim: int = 2048, sequence_length: int = 100) -> "ModalityDecoderConfig":
        """Create config optimized for time series data"""
        # More tokens for longer sequences
        num_tokens = min(4, max(1, sequence_length // 50))
        
        return cls(
            name="timeseries",
            input_dim=input_dim,
            output_dim=output_dim,
            num_tokens=num_tokens,
            num_layers=4,
            num_heads=8,
            intermediate_size=output_dim * 3,
            use_moe=sequence_length > 200,
            use_position_encoding=True
        )
    
    @classmethod
    def create_tabular_config(cls, input_dim: int, output_dim: int = 2048) -> "ModalityDecoderConfig":
        """Create config for tabular/structured data"""
        return cls(
            name="tabular",
            input_dim=input_dim,
            output_dim=output_dim,
            num_tokens=1,
            num_layers=3,  # Shallower for tabular
            num_heads=8,
            intermediate_size=output_dim * 3,
            use_moe=False  # Usually not needed for tabular
        )
    
    @classmethod
    def create_satellite_config(cls, input_dim: int = 1024, output_dim: int = 2048) -> "ModalityDecoderConfig":
        """Create config for satellite imagery"""
        return cls(
            name="satellite",
            input_dim=input_dim,
            output_dim=output_dim,
            num_tokens=64,  # 8x8 grid for high resolution
            num_layers=8,  # Deeper for complex spatial patterns
            num_heads=16,
            num_key_value_heads=4,  # GQA for efficiency with many tokens
            intermediate_size=output_dim * 4,
            use_moe=True,
            num_experts=16,  # More experts for diverse terrain
            num_experts_per_tok=4,
            n_shared_experts=2,
            use_cross_attention=True,
            use_position_encoding=True,
            output_use_moe=True  # MoE for token specialization
        )
    
    def scale_for_precision(self, precision_level: str = "standard"):
        """Scale configuration based on required precision"""
        scaling = {
            "low": 0.5,      # Fast inference
            "standard": 1.0, # Default
            "high": 1.5,     # High precision
            "ultra": 2.0     # Maximum precision
        }
        
        scale = scaling.get(precision_level, 1.0)
        
        if scale != 1.0:
            self.num_layers = max(2, int(self.num_layers * scale))
            self.intermediate_size = int(self.intermediate_size * scale)
            
            # Enable MoE for ultra precision
            if precision_level == "ultra" and not self.use_moe:
                self.use_moe = True
                self.num_experts = 4
        
        return self


# Preset configurations for common use cases
PRESET_CONFIGS = {
    # Vision presets
    "vision_standard": ModalityDecoderConfig.create_vision_config(),
    "vision_satellite": ModalityDecoderConfig.create_satellite_config(),
    "vision_realtime": ModalityDecoderConfig.create_vision_config(num_tokens=1).scale_for_precision("low"),
    
    # Language presets
    "language_standard": ModalityDecoderConfig.create_language_config(),
    "language_agricultural": ModalityDecoderConfig.create_language_config(num_tokens=4).scale_for_precision("high"),
    "language_chat": ModalityDecoderConfig.create_language_config(num_tokens=2),
    
    # Earth observation presets
    "weather": ModalityDecoderConfig.create_default("weather", input_dim=5),
    "weather_forecast": ModalityDecoderConfig.create_timeseries_config(input_dim=5, sequence_length=168),
    "soil": ModalityDecoderConfig.create_tabular_config(input_dim=10),
    "species": ModalityDecoderConfig.create_tabular_config(input_dim=64).scale_for_precision("high"),
    "ndvi_timeseries": ModalityDecoderConfig.create_timeseries_config(input_dim=1, sequence_length=365),
    "hyperspectral": ModalityDecoderConfig(
        name="hyperspectral",
        input_dim=224,  # 224 spectral bands
        output_dim=2048,
        num_tokens=8,
        num_layers=6,
        use_moe=True,
        num_experts=8,
        input_use_moe=True  # MoE for complex spectral patterns
    ),
}


def get_preset_config(name: str) -> ModalityDecoderConfig:
    """Get a preset configuration by name"""
    if name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESET_CONFIGS.keys())}")
    
    # Return a copy to avoid modifying the preset
    import copy
    return copy.deepcopy(PRESET_CONFIGS[name])


# Example usage
if __name__ == "__main__":
    # Create custom config
    custom_config = ModalityDecoderConfig(
        name="hyperspectral",
        input_dim=224,  # 224 spectral bands
        output_dim=2048,
        num_tokens=8,   # Multiple tokens for spectral diversity
        num_layers=6,
        use_moe=True    # MoE for complex spectral patterns
    )
    
    print(f"Custom config: {custom_config.name}")
    print(f"  Input: {custom_config.input_dim} -> Output: {custom_config.output_dim}")
    print(f"  Tokens: {custom_config.num_tokens}, Layers: {custom_config.num_layers}")
    print(f"  MoE: {custom_config.use_moe}")
    
    # Use preset
    vision_config = get_preset_config("vision_satellite")
    print(f"\nPreset config: {vision_config.name}")
    print(f"  Optimized for satellite imagery with {vision_config.num_tokens} tokens")
    
    # Convert to DeepSeek config
    deepseek_cfg = custom_config.to_deepseek_config()
    print(f"\nDeepSeek config created with {deepseek_cfg.num_hidden_layers} layers")