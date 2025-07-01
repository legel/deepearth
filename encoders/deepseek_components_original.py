# deepseek_components.py
from transformers import PretrainedConfig
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class DeepSeekConfig(PretrainedConfig):
    """Configuration for DeepSeek components"""
    hidden_size: int = 2048
    intermediate_size: int = 5504
    num_hidden_layers: int = 6
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    # MoE specific
    n_routed_experts: Optional[int] = None
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 1408
    n_shared_experts: Optional[int] = None
    norm_topk_prob: bool = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

from encoders.modeling_deepseek import (
    DeepseekV3MLP as _DeepSeekMLP,
    DeepseekV3MoE as _DeepSeekMoE,
    DeepseekV3Model,
    DeepseekV3DecoderLayer,
    DeepseekV3RMSNorm
)

class DeepSeekMLP(_DeepSeekMLP):
    """DeepSeek MLP module"""
    pass

class DeepSeekMoE(_DeepSeekMoE):
    """DeepSeek Mixture of Experts module"""
    pass

class DeepSeekTransformer(nn.Module):
    """Lightweight DeepSeek Transformer for modality decoding"""
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        
        # Build transformer layers
        self.layers = nn.ModuleList([
            DeepseekV3DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]
        
        return self.norm(hidden_states)
