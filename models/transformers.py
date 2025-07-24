"""Transformer modules for DeepEarth model.

Implements efficient transformer architecture with RoPE position embeddings
and optimizations inspired by DeepSeek.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .configs import TransformerConfig


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE (Rotary Position Embeddings).
    
    Implements efficient attention mechanism with rotary position embeddings
    for better extrapolation to longer sequences.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_dim // config.n_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections (no bias for efficiency)
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Initialize RoPE frequencies
        self._init_rope()
    
    def _init_rope(self):
        """Initialize RoPE frequency tensor."""
        dim = self.head_dim
        max_seq_len = 8192  # Maximum sequence length
        
        # Compute frequencies
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs).float()
        
        # Create sin/cos embeddings
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis)
    
    def apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary position embeddings."""
        # Reshape to complex
        x = x.float().reshape(*x.shape[:-1], -1, 2)
        x = torch.view_as_complex(x)
        
        # Apply rotation
        freqs = self.freqs_cis[:seq_len]
        x = x * freqs.unsqueeze(0).unsqueeze(0)
        
        # Back to real
        x = torch.view_as_real(x)
        x = x.reshape(*x.shape[:-2], -1)
        
        return x.type_as(x)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of multi-head attention.
        
        Args:
            x: (B, N, D) input tensor
            mask: (B, N) boolean mask
            
        Returns:
            output: (B, N, D) output tensor
        """
        B, N, D = x.shape
        
        # Linear projections and reshape for multi-head
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to queries and keys
        q = self.apply_rope(q, N)
        k = self.apply_rope(k, N)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        output = torch.matmul(attn, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(B, N, D)
        output = self.out_proj(output)
        output = self.resid_dropout(output)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and MLP.
    
    Uses pre-normalization for training stability and includes
    a feed-forward MLP with GELU activation.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention
        self.attention = MultiHeadAttention(config)
        
        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # MLP
        mlp_dim = int(config.hidden_dim * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of transformer block.
        
        Args:
            x: (B, N, D) input tensor
            mask: (B, N) boolean mask
            
        Returns:
            output: (B, N, D) output tensor
        """
        # Self-attention with residual
        x = x + self.attention(self.norm1(x), mask=mask)
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class Transformer(nn.Module):
    """Full transformer model for cross-modal fusion.
    
    Implements a transformer with RoPE position embeddings and
    efficient attention patterns inspired by DeepSeek architecture.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer.
        
        Args:
            x: (B, N, D) input tensor
            mask: (B, N) boolean mask
            
        Returns:
            output: (B, N, D) output tensor
        """
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)
        
        # Final normalization
        x = self.norm(x)
        
        return x
