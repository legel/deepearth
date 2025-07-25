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
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    
    # V3 specific attributes
    q_lora_rank: Optional[int] = None
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    topk_method: str = "greedy"
    n_group: int = 8
    topk_group: int = 3
    
    # Additional V3 attributes
    attention_bias: bool = False
    rope_scaling: Optional[dict] = None
    scoring_func: str = "softmax"
    routed_scaling_factor: float = 1.0
    
    # Default attention implementation
    _attn_implementation: str = "eager"  # or "flash_attention_2"
    
    def __init__(self, **kwargs):
        # Set defaults for V3
        if "kv_lora_rank" not in kwargs:
            kwargs["kv_lora_rank"] = 512
        if "qk_rope_head_dim" not in kwargs:
            kwargs["qk_rope_head_dim"] = 64
        if "v_head_dim" not in kwargs:
            kwargs["v_head_dim"] = 128
        if "qk_nope_head_dim" not in kwargs:
            kwargs["qk_nope_head_dim"] = 128
        if "_attn_implementation" not in kwargs:
            kwargs["_attn_implementation"] = "eager"  # Use standard attention
            
        super().__init__(**kwargs)

# Import the actual components from modeling_deepseek.py
try:
    from encoders.modeling_deepseek import (
        DeepseekV3MLP,
        DeepseekV3MoE,
        DeepseekV3Model,
        DeepseekV3DecoderLayer,
        DeepseekV3RMSNorm
    )
    
    # Create aliases for compatibility
    DeepSeekMLP = DeepseekV3MLP
    DeepSeekMoE = DeepseekV3MoE
    DeepseekV3RMSNorm = DeepseekV3RMSNorm
    
except ImportError as e:
    print(f"Warning: Could not import DeepSeek V3 components: {e}")
    # Fallback implementations
    
    class DeepSeekMLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
            self.act_fn = nn.SiLU()
            
        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    
    DeepSeekMoE = DeepSeekMLP  # Simplified fallback
    DeepseekV3RMSNorm = nn.LayerNorm  # Simplified fallback

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
