# Updated vision_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepseek_components import (
    DeepSeekConfig, 
    DeepSeekTransformer, 
    DeepSeekMLP, 
    DeepSeekMoE,
    DeepseekV3RMSNorm
)

class DeepSeekVisionDecoder(nn.Module):
    def __init__(self, config: ModalityDecoderConfig):
        super().__init__()
        self.config = config
        
        # Input projection - can be MLP or MoE based on config
        if config.use_moe and hasattr(config, 'input_use_moe') and config.input_use_moe:
            # Use MoE for input projection when dealing with diverse inputs
            input_moe_config = DeepSeekConfig(
                hidden_size=config.input_dim,
                intermediate_size=config.intermediate_size,
                n_routed_experts=config.num_experts // 2,  # Fewer experts for input
                num_experts_per_tok=2,
                moe_intermediate_size=config.intermediate_size // 4,
                n_shared_experts=1,  # One shared expert for common patterns
            )
            self.input_projection = DeepSeekMoE(input_moe_config)
        else:
            # Standard MLP for input projection
            mlp_config = DeepSeekConfig(
                hidden_size=config.input_dim,
                intermediate_size=config.intermediate_size
            )
            self.input_projection = DeepSeekMLP(mlp_config)
        
        # Projection layer to match dimensions if needed
        if config.input_dim != config.output_dim:
            self.dim_projection = nn.Linear(config.input_dim, config.output_dim)
        else:
            self.dim_projection = None
        
        # Create DeepSeek transformer config with potential MoE layers
        deepseek_config = DeepSeekConfig(
            hidden_size=config.output_dim,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            num_key_value_heads=config.num_key_value_heads,
            # MoE configuration
            n_routed_experts=config.num_experts if config.use_moe else None,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size or (config.intermediate_size // 4),
            n_shared_experts=config.n_shared_experts if hasattr(config, 'n_shared_experts') else None,
            # MoE layer frequency (e.g., every other layer)
            first_k_dense_replace=0,  # Start MoE from first layer
            moe_layer_freq=config.moe_layer_freq if hasattr(config, 'moe_layer_freq') else 1,
            # Other configs
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout_prob,
        )
        
        # DeepSeek Transformer (will use MoE based on config)
        self.transformer = DeepSeekTransformer(deepseek_config)
        
        # Positional embeddings for patches
        max_patches = 1024  # Support up to 32x32 patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, max_patches, config.output_dim) * 0.02
        )
        
        # Token generation strategy
        if config.num_tokens > 1:
            # Learnable query tokens for multi-token output
            self.query_tokens = nn.Parameter(
                torch.randn(1, config.num_tokens, config.output_dim) * 0.02
            )
            
            # Use DeepSeek components for token generation too
            if config.use_moe:
                # MoE for diverse token generation
                token_moe_config = DeepSeekConfig(
                    hidden_size=config.output_dim,
                    n_routed_experts=min(config.num_experts, config.num_tokens * 2),
                    num_experts_per_tok=2,
                    moe_intermediate_size=config.output_dim * 2,
                )
                self.token_generator = DeepSeekMoE(token_moe_config)
            else:
                # MLP for token generation
                token_mlp_config = DeepSeekConfig(
                    hidden_size=config.output_dim,
                    intermediate_size=config.output_dim * 3,
                )
                self.token_generator = DeepSeekMLP(token_mlp_config)
            
            # Cross-attention for token generation
            self.cross_attention = nn.MultiheadAttention(
                config.output_dim,
                config.num_heads,
                dropout=config.attention_dropout_prob,
                batch_first=True
            )
            self.cross_norm = DeepseekV3RMSNorm(config.output_dim)
        
        # Output normalization using DeepSeek's RMSNorm
        self.output_norm = DeepseekV3RMSNorm(config.output_dim)
        
        # Optional spatial pooling for single token output
        if config.num_tokens == 1:
            # Even pooling can use MoE for diverse aggregation strategies
            if config.use_moe:
                pool_moe_config = DeepSeekConfig(
                    hidden_size=config.output_dim,
                    n_routed_experts=4,  # Different pooling strategies
                    num_experts_per_tok=1,
                    moe_intermediate_size=config.output_dim,
                )
                self.pool_mlp = DeepSeekMoE(pool_moe_config)
            else:
                pool_config = DeepSeekConfig(
                    hidden_size=config.output_dim,
                    intermediate_size=config.output_dim * 2,
                )
                self.pool_mlp = DeepSeekMLP(pool_config)
    
    def forward(
        self,
        patch_embeddings: torch.Tensor,
        spatial_positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode patch embeddings to universal tokens
        
        Args:
            patch_embeddings: (B, N_patches, D_native) patch features from vision backbone
            spatial_positions: Optional (B, N_patches, 2) normalized x,y positions
            attention_mask: Optional (B, N_patches) mask for valid patches
            
        Returns:
            universal_tokens: (B, K, D_universal) where K = num_tokens
        """
        B, N, _ = patch_embeddings.shape
        
        # Project patches through MLP or MoE
        hidden_states = self.input_projection(patch_embeddings)
        
        # Project dimensions if needed
        if self.dim_projection is not None:
            hidden_states = self.dim_projection(hidden_states)
        
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
        
        # Optional: Add spatial position encoding if provided
        if spatial_positions is not None:
            spatial_encoding = self._encode_spatial_positions(spatial_positions)
            hidden_states = hidden_states + spatial_encoding
        
        # Apply transformer (with MoE layers if configured)
        if attention_mask is not None:
            # Convert boolean mask to attention mask format
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        transformed = self.transformer(hidden_states, attention_mask=attention_mask)
        
        # Generate output tokens based on configuration
        if self.config.num_tokens > 1:
            # Multi-token output using cross-attention and MoE/MLP
            output_tokens = self._generate_multi_tokens(transformed, attention_mask)
        else:
            # Single token output through MoE/MLP pooling
            output_tokens = self._generate_single_token(transformed, attention_mask)
        
        # Final normalization
        output_tokens = self.output_norm(output_tokens)
        
        return output_tokens
    
    def _generate_multi_tokens(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate multiple tokens using cross-attention and MoE/MLP"""
        B = hidden_states.shape[0]
        
        # Expand query tokens for batch
        queries = self.query_tokens.expand(B, -1, -1)  # (B, K, D)
        
        # Cross-attention from queries to transformed patches
        if attention_mask is not None:
            key_padding_mask = attention_mask.squeeze(1).squeeze(1) == float('-inf')
        else:
            key_padding_mask = None
        
        attended_tokens, _ = self.cross_attention(
            query=queries,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask
        )
        
        # Residual connection and normalization
        attended_tokens = self.cross_norm(queries + attended_tokens)
        
        # Further process through MoE/MLP for token specialization
        output_tokens = self.token_generator(attended_tokens)
        output_tokens = attended_tokens + output_tokens  # Residual
        
        return output_tokens
    
    def _generate_single_token(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate single token through MoE/MLP pooling"""
        B, N, D = hidden_states.shape
        
        # First aggregate information
        if attention_mask is not None and attention_mask.numel() > 0:
            # Masked average pooling
            mask = (attention_mask.squeeze(1).squeeze(1) != float('-inf')).float()
            mask = mask.unsqueeze(-1)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            # Attention-weighted pooling
            attn_weights = torch.softmax(hidden_states.norm(dim=-1), dim=-1)
            pooled = torch.einsum('bn,bnd->bd', attn_weights, hidden_states)
        
        # Process through MoE/MLP for final token
        output_token = self.pool_mlp(pooled)
        output_token = pooled + output_token  # Residual
        
        return output_token.unsqueeze(1)  # (B, 1, D)
    
    def _encode_spatial_positions(
        self, 
        spatial_positions: torch.Tensor
    ) -> torch.Tensor:
        """Encode spatial positions (x, y) coordinates into embeddings"""
        B, N, _ = spatial_positions.shape
        D = self.config.output_dim
        
        # Sinusoidal encoding in 2D space
        div_term = torch.exp(
            torch.arange(0, D // 4, dtype=torch.float32) * 
            -(torch.log(torch.tensor(10000.0)) / (D // 4))
        ).to(spatial_positions.device)
        
        x_pos = spatial_positions[:, :, 0:1]
        y_pos = spatial_positions[:, :, 1:2]
        
        x_sin = torch.sin(x_pos * div_term)
        x_cos = torch.cos(x_pos * div_term)
        y_sin = torch.sin(y_pos * div_term)
        y_cos = torch.cos(y_pos * div_term)
        
        spatial_encoding = torch.cat([x_sin, x_cos, y_sin, y_cos], dim=-1)
        
        if spatial_encoding.shape[-1] < D:
            padding = D - spatial_encoding.shape[-1]
            spatial_encoding = F.pad(spatial_encoding, (0, padding))
        
        return spatial_encoding * 0.1