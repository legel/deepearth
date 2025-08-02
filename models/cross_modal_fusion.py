"""
Cross-Modal Fusion for Universal Tokens in DeepEarth
Implements hierarchical fusion with spatial-temporal awareness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass


@dataclass
class FusionConfig:
    """Configuration for cross-modal fusion"""
    universal_dim: int = 2048
    num_fusion_layers: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    use_rotary_embeddings: bool = True
    use_gated_mlp: bool = True
    cross_attention_freq: int = 3  # Cross-attention every N layers
    spatial_aware: bool = True
    temporal_aware: bool = True
    max_seq_length: int = 8192


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings for spatial-temporal awareness"""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute embeddings for common sequence lengths
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Precompute sin/cos embeddings"""
        t = torch.arange(self.max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to input tensor"""
        seq_len = x.shape[seq_dim]
        
        if seq_len > self.max_seq_len:
            # Extend embeddings if needed
            self._extend_embeddings(seq_len)
        
        return (
            self.cos_cached[:seq_len, :],
            self.sin_cached[:seq_len, :]
        )
    
    def _extend_embeddings(self, seq_len: int):
        """Extend embeddings for longer sequences"""
        t = torch.arange(seq_len).float().to(self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
        self.max_seq_len = seq_len


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper for rotary embeddings"""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys"""
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class SpatialTemporalEmbedding(nn.Module):
    """Learnable spatial-temporal embeddings for universal tokens"""
    
    def __init__(self, universal_dim: int, max_spatial_resolution: int = 64):
        super().__init__()
        self.universal_dim = universal_dim
        
        # Spatial position embeddings (2D)
        self.spatial_embed_x = nn.Parameter(
            torch.randn(1, max_spatial_resolution, universal_dim // 4) * 0.02
        )
        self.spatial_embed_y = nn.Parameter(
            torch.randn(1, max_spatial_resolution, universal_dim // 4) * 0.02
        )
        
        # Temporal embeddings (continuous)
        self.temporal_embed = nn.Sequential(
            nn.Linear(1, universal_dim // 2),
            nn.GELU(),
            nn.Linear(universal_dim // 2, universal_dim // 2)
        )
        
        # Modality type embeddings
        self.modality_embeds = nn.ParameterDict()
    
    def add_modality(self, name: str):
        """Add embedding for new modality"""
        self.modality_embeds[name] = nn.Parameter(
            torch.randn(1, 1, self.universal_dim) * 0.02
        )
    
    def forward(
        self,
        tokens: torch.Tensor,
        spatial_positions: Optional[torch.Tensor] = None,
        temporal_positions: Optional[torch.Tensor] = None,
        modality_name: Optional[str] = None
    ) -> torch.Tensor:
        """Add spatial-temporal-modal embeddings to tokens"""
        B, N, D = tokens.shape
        
        embeddings = torch.zeros_like(tokens)
        
        # Add spatial embeddings if provided
        if spatial_positions is not None:
            # spatial_positions: (B, N, 2) with x, y coordinates
            x_idx = (spatial_positions[..., 0] * self.spatial_embed_x.shape[1]).long()
            y_idx = (spatial_positions[..., 1] * self.spatial_embed_y.shape[1]).long()
            
            x_embed = self.spatial_embed_x[:, x_idx].squeeze(0)
            y_embed = self.spatial_embed_y[:, y_idx].squeeze(0)
            
            spatial_embed = torch.cat([x_embed, y_embed], dim=-1)
            embeddings[..., :D//2] += spatial_embed
        
        # Add temporal embeddings if provided
        if temporal_positions is not None:
            # temporal_positions: (B, N, 1) normalized time
            temp_embed = self.temporal_embed(temporal_positions)
            embeddings[..., D//2:] += temp_embed
        
        # Add modality embeddings
        if modality_name is not None and modality_name in self.modality_embeds:
            embeddings += self.modality_embeds[modality_name].expand(B, N, -1)
        
        return tokens + embeddings


class GatedMLP(nn.Module):
    """Gated MLP as used in modern transformers"""
    
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class MultiHeadCrossAttention(nn.Module):
    """Multi-head attention with support for cross-modal attention"""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.universal_dim // config.num_heads
        
        # Projections
        self.q_proj = nn.Linear(config.universal_dim, config.universal_dim, bias=False)
        self.k_proj = nn.Linear(config.universal_dim, config.universal_dim, bias=False)
        self.v_proj = nn.Linear(config.universal_dim, config.universal_dim, bias=False)
        self.out_proj = nn.Linear(config.universal_dim, config.universal_dim, bias=False)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Rotary embeddings
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_seq_length)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of attention
        
        Args:
            query: (B, N_q, D)
            key: (B, N_k, D) - if None, self-attention
            value: (B, N_v, D) - if None, self-attention
            attention_mask: (B, N_q, N_k) attention mask
            is_causal: Whether to apply causal masking
        """
        B, N_q, D = query.shape
        
        # Self-attention if key/value not provided
        if key is None:
            key = query
        if value is None:
            value = query
        
        N_k = key.shape[1]
        
        # Linear projections
        Q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings if enabled
        if self.config.use_rotary_embeddings:
            cos, sin = self.rotary_emb(Q, seq_dim=2)
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Apply masks
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))
        
        if is_causal:
            causal_mask = torch.triu(torch.ones(N_q, N_k, device=scores.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, N_q, D)
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output


class FusionLayer(nn.Module):
    """Single fusion layer with self/cross attention and MLP"""
    
    def __init__(self, config: FusionConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Determine if this layer does cross-attention
        self.use_cross_attention = (layer_idx % config.cross_attention_freq == 0)
        
        # Self-attention
        self.self_attn = MultiHeadCrossAttention(config)
        self.self_attn_norm = nn.LayerNorm(config.universal_dim, eps=config.layer_norm_eps)
        
        # Cross-attention (conditional)
        if self.use_cross_attention:
            self.cross_attn = MultiHeadCrossAttention(config)
            self.cross_attn_norm = nn.LayerNorm(config.universal_dim, eps=config.layer_norm_eps)
        
        # MLP
        if config.use_gated_mlp:
            self.mlp = GatedMLP(config.universal_dim, config.mlp_ratio, config.dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.universal_dim, int(config.universal_dim * config.mlp_ratio)),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(int(config.universal_dim * config.mlp_ratio), config.universal_dim),
                nn.Dropout(config.dropout)
            )
        
        self.mlp_norm = nn.LayerNorm(config.universal_dim, eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of fusion layer
        
        Args:
            hidden_states: (B, N, D) current hidden states
            encoder_hidden_states: (B, M, D) states from other modalities for cross-attention
            attention_mask: Self-attention mask
            encoder_attention_mask: Cross-attention mask
        """
        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attn_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        
        # Cross-attention (if enabled)
        if self.use_cross_attention and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attn_norm(hidden_states)
            hidden_states = self.cross_attn(
                query=hidden_states,
                key=encoder_hidden_states,
                value=encoder_hidden_states,
                attention_mask=encoder_attention_mask
            )
            hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class CrossModalFusion(nn.Module):
    """Main cross-modal fusion module for DeepEarth"""
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # Spatial-temporal-modal embeddings
        self.st_embedding = SpatialTemporalEmbedding(config.universal_dim)
        
        # Fusion layers
        self.layers = nn.ModuleList([
            FusionLayer(config, i) for i in range(config.num_fusion_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(config.universal_dim, eps=config.layer_norm_eps)
        
        # Pooling strategies
        self.register_buffer('cls_token', torch.randn(1, 1, config.universal_dim) * 0.02)
    
    def forward(
        self,
        modality_tokens: Dict[str, torch.Tensor],
        spatial_positions: Optional[Dict[str, torch.Tensor]] = None,
        temporal_positions: Optional[Dict[str, torch.Tensor]] = None,
        return_all_layers: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse tokens from multiple modalities
        
        Args:
            modality_tokens: Dict mapping modality names to token tensors (B, N_i, D)
            spatial_positions: Optional dict of spatial positions for each modality
            temporal_positions: Optional dict of temporal positions for each modality
            return_all_layers: Whether to return intermediate representations
            
        Returns:
            Dict containing:
                - fused_representation: (B, D) pooled representation
                - all_tokens: (B, N_total, D) all tokens after fusion
                - modality_tokens: Dict of updated modality-specific tokens
                - layer_outputs: List of outputs from each layer (if requested)
        """
        B = next(iter(modality_tokens.values())).shape[0]
        
        # Add spatial-temporal-modal embeddings
        embedded_tokens = {}
        for name, tokens in modality_tokens.items():
            # Ensure modality embedding exists
            if name not in self.st_embedding.modality_embeds:
                self.st_embedding.add_modality(name)
            
            # Apply embeddings
            embedded = self.st_embedding(
                tokens,
                spatial_positions.get(name) if spatial_positions else None,
                temporal_positions.get(name) if temporal_positions else None,
                name
            )
            embedded_tokens[name] = embedded
        
        # Concatenate all tokens with CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        all_tokens = [cls_tokens]
        
        # Track token boundaries for later separation
        token_boundaries = {'cls': (0, 1)}
        current_idx = 1
        
        for name, tokens in embedded_tokens.items():
            all_tokens.append(tokens)
            num_tokens = tokens.shape[1]
            token_boundaries[name] = (current_idx, current_idx + num_tokens)
            current_idx += num_tokens
        
        # Concatenate all tokens
        hidden_states = torch.cat(all_tokens, dim=1)  # (B, N_total, D)
        
        # Apply fusion layers
        layer_outputs = []
        
        for i, layer in enumerate(self.layers):
            # For cross-attention layers, use all other modalities as context
            if layer.use_cross_attention:
                # Simple strategy: use all tokens as context
                # More sophisticated: separate by modality
                hidden_states = layer(
                    hidden_states,
                    encoder_hidden_states=hidden_states,
                    attention_mask=None,  # Could add custom masks
                    encoder_attention_mask=None
                )
            else:
                hidden_states = layer(hidden_states)
            
            if return_all_layers:
                layer_outputs.append(hidden_states)
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Extract representations
        outputs = {
            'fused_representation': hidden_states[:, 0],  # CLS token
            'all_tokens': hidden_states,
            'modality_tokens': {}
        }
        
        # Separate modality-specific tokens
        for name, (start, end) in token_boundaries.items():
            if name != 'cls':
                outputs['modality_tokens'][name] = hidden_states[:, start:end]
        
        if return_all_layers:
            outputs['layer_outputs'] = layer_outputs
        
        return outputs


class HierarchicalFusion(nn.Module):
    """Hierarchical fusion for handling multiple scales and modalities"""
    
    def __init__(
        self,
        config: FusionConfig,
        num_levels: int = 3,
        downscale_factor: int = 2
    ):
        super().__init__()
        self.config = config
        self.num_levels = num_levels
        
        # Create fusion modules for each level
        self.level_fusions = nn.ModuleList([
            CrossModalFusion(config) for _ in range(num_levels)
        ])
        
        # Downsampling between levels
        self.downsamplers = nn.ModuleList([
            nn.Conv1d(
                config.universal_dim,
                config.universal_dim,
                kernel_size=downscale_factor,
                stride=downscale_factor
            ) for _ in range(num_levels - 1)
        ])
        
        # Upsampling for combining levels
        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose1d(
                config.universal_dim,
                config.universal_dim,
                kernel_size=downscale_factor,
                stride=downscale_factor
            ) for _ in range(num_levels - 1)
        ])
        
        # Final fusion
        self.final_fusion = nn.Linear(
            config.universal_dim * num_levels,
            config.universal_dim
        )
    
    def forward(
        self,
        modality_tokens: Dict[str, torch.Tensor],
        spatial_positions: Optional[Dict[str, torch.Tensor]] = None,
        temporal_positions: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Hierarchical fusion across multiple scales
        """
        level_outputs = []
        current_tokens = modality_tokens
        
        # Process each level
        for level in range(self.num_levels):
            # Fuse at current level
            fusion_output = self.level_fusions[level](
                current_tokens,
                spatial_positions,
                temporal_positions
            )
            level_outputs.append(fusion_output['fused_representation'])
            
            # Downsample for next level (except last)
            if level < self.num_levels - 1:
                downsampled_tokens = {}
                for name, tokens in fusion_output['modality_tokens'].items():
                    # Reshape for Conv1d: (B, N, D) -> (B, D, N)
                    tokens_reshaped = tokens.transpose(1, 2)
                    downsampled = self.downsamplers[level](tokens_reshaped)
                    # Reshape back: (B, D, N) -> (B, N, D)
                    downsampled_tokens[name] = downsampled.transpose(1, 2)
                
                current_tokens = downsampled_tokens
                
                # Adjust positions if provided
                if spatial_positions is not None:
                    spatial_positions = {
                        name: pos[:, ::2] for name, pos in spatial_positions.items()
                    }
        
        # Combine representations from all levels
        # Upsample lower levels to match highest resolution
        combined_representations = [level_outputs[0]]
        
        for level in range(1, self.num_levels):
            # Progressively upsample
            upsampled = level_outputs[level]
            for upsample_idx in range(level):
                upsampled = self.upsamplers[upsample_idx](
                    upsampled.unsqueeze(-1)
                ).squeeze(-1)
            combined_representations.append(upsampled)
        
        # Concatenate and fuse
        multi_scale_representation = torch.cat(combined_representations, dim=-1)
        final_representation = self.final_fusion(multi_scale_representation)
        
        return {
            'fused_representation': final_representation,
            'level_representations': level_outputs,
            'multi_scale_representation': multi_scale_representation
        }


# Example usage
if __name__ == "__main__":
    # Configuration
    config = FusionConfig(
        universal_dim=2048,
        num_fusion_layers=24,
        num_heads=16,
        cross_attention_freq=3
    )
    
    # Create fusion module
    fusion = CrossModalFusion(config)
    
    # Example tokens from different modalities
    batch_size = 2
    modality_tokens = {
        'vision': torch.randn(batch_size, 16, 2048),  # 16 visual tokens
        'language': torch.randn(batch_size, 8, 2048),  # 8 language tokens
        'spatial': torch.randn(batch_size, 4, 2048),   # 4 spatial tokens
    }
    
    # Optional position information
    spatial_positions = {
        'vision': torch.rand(batch_size, 16, 2),  # x, y positions for each token
        'spatial': torch.rand(batch_size, 4, 2),
    }
    
    temporal_positions = {
        'vision': torch.rand(batch_size, 16, 1),
        'language': torch.rand(batch_size, 8, 1),
        'spatial': torch.rand(batch_size, 4, 1),
    }
    
    # Forward pass
    outputs = fusion(
        modality_tokens,
        spatial_positions,
        temporal_positions
    )
    
    print("Fusion outputs:")
    print(f"Fused representation: {outputs['fused_representation'].shape}")
    print(f"All tokens: {outputs['all_tokens'].shape}")
    for name, tokens in outputs['modality_tokens'].items():
        print(f"{name} tokens: {tokens.shape}")
