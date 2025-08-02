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
    """
    name: str
    input_dim: int
    output_dim: int
    num_tokens: int = 1
    
    # DeepSeek Transformer config
    num_layers: int = 4
    num_heads: int = 8
    num_key_value_heads: Optional[int] = None
    intermediate_size: int = 4096
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    dropout_prob: float = 0.1
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    
    # MoE settings
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: Optional[int] = None
    n_shared_experts: Optional[int] = None
    norm_topk_prob: bool = True
    moe_layer_freq: int = 1
    
    # Additional settings
    use_gate: bool = True
    use_normalization: bool = True
    activation: str = "silu"
    
    # Token generation settings
    use_cross_attention: bool = True
    use_position_encoding: bool = True
    input_use_moe: bool = False
    output_use_moe: bool = False
    embedding_selection: Optional[str] = None

    def to_deepseek_config(self) -> DeepSeekConfig:
        return DeepSeekConfig(
            hidden_size=self.output_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads or self.num_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.activation,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            attention_dropout=self.attention_dropout_prob,
            n_routed_experts=self.num_experts if self.use_moe else None,
            num_experts_per_tok=self.num_experts_per_tok,
            moe_intermediate_size=self.moe_intermediate_size or (self.intermediate_size // 4),
            n_shared_experts=self.n_shared_experts,
            norm_topk_prob=self.norm_topk_prob,
        )

    @classmethod
    def create_default(cls, name: str, input_dim: int, output_dim: int, **kwargs) -> "ModalityDecoderConfig":
        return cls(name=name, input_dim=input_dim, output_dim=output_dim, **kwargs)

    @classmethod
    def create_vision_config(cls, name: str, input_dim: int, output_dim: int, **kwargs) -> "ModalityDecoderConfig":
        return cls(name=name, input_dim=input_dim, output_dim=output_dim, num_layers=6, num_heads=16, use_cross_attention=True, use_position_encoding=True, **kwargs)

    @classmethod
    def create_language_config(cls, name: str, input_dim: int, output_dim: int, **kwargs) -> "ModalityDecoderConfig":
        return cls(name=name, input_dim=input_dim, output_dim=output_dim, num_layers=4, num_heads=16, attention_dropout_prob=0.1, embedding_selection='token_embeddings', **kwargs)

    @classmethod
    def create_timeseries_config(cls, name: str, input_dim: int, output_dim: int, **kwargs) -> "ModalityDecoderConfig":
        return cls(name=name, input_dim=input_dim, output_dim=output_dim, num_layers=4, num_heads=8, use_position_encoding=True, **kwargs)

    @classmethod
    def create_tabular_config(cls, name: str, input_dim: int, output_dim: int, **kwargs) -> "ModalityDecoderConfig":
        return cls(name=name, input_dim=input_dim, output_dim=output_dim, num_layers=3, num_heads=8, **kwargs)

    @classmethod
    def create_satellite_config(cls, name: str, input_dim: int, output_dim: int, **kwargs) -> "ModalityDecoderConfig":
        return cls(name=name, input_dim=input_dim, output_dim=output_dim, num_layers=8, num_heads=16, num_key_value_heads=4, use_moe=True, num_experts=16, n_shared_experts=2, use_cross_attention=True, use_position_encoding=True, output_use_moe=True, **kwargs)

# Preset configurations for common use cases
# FIX: Provide the required input_dim and output_dim for each preset
# We know the universal dimension is 2048 from other files.
UNIVERSAL_DIM = 2048
PRESET_CONFIGS = {
    "vision_standard": ModalityDecoderConfig.create_vision_config(name="vision", input_dim=1408, output_dim=UNIVERSAL_DIM),
    "vision_satellite": ModalityDecoderConfig.create_satellite_config(name="satellite", input_dim=1408, output_dim=UNIVERSAL_DIM, num_tokens=64),
    "language_standard": ModalityDecoderConfig.create_language_config(name="language", input_dim=4096, output_dim=UNIVERSAL_DIM),
    
    "weather": ModalityDecoderConfig.create_default("weather", input_dim=5, output_dim=UNIVERSAL_DIM),
    "weather_forecast": ModalityDecoderConfig.create_timeseries_config(name="weather_forecast", input_dim=5, output_dim=UNIVERSAL_DIM, num_tokens=4),
    "soil": ModalityDecoderConfig.create_tabular_config(name="soil", input_dim=10, output_dim=UNIVERSAL_DIM),
    "species": ModalityDecoderConfig.create_tabular_config(name="species", input_dim=64, output_dim=UNIVERSAL_DIM),
    "ndvi_timeseries": ModalityDecoderConfig.create_timeseries_config(name="ndvi_timeseries", input_dim=1, output_dim=UNIVERSAL_DIM, num_tokens=8),
    "hyperspectral": ModalityDecoderConfig(name="hyperspectral", input_dim=224, output_dim=UNIVERSAL_DIM, num_tokens=8, num_layers=6, use_moe=True, num_experts=8, input_use_moe=True)
}

def get_preset_config(name: str) -> ModalityDecoderConfig:
    import copy
    if name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESET_CONFIGS.keys())}")
    return copy.deepcopy(PRESET_CONFIGS[name])