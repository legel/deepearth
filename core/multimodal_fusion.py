# deepearth/core/multimodal_fusion.py
"""
Multimodal Fusion Networks for DeepEarth
════════════════════════════════════════

These networks handle the critical task of translating between modality-specific
representations and the universal embedding space where cross-modal learning occurs.

Architecture Philosophy:
    Each Earth observation modality has unique characteristics and dimensionalities.
    Rather than forcing all modalities into a fixed representation, we learn
    flexible projections that preserve modality-specific information while
    enabling cross-modal interaction.

Bidirectional Design:
    Forward:  Modality Space → Universal Space (for encoding observations)
    Backward: Universal Space → Modality Space (for generating predictions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from transformers import PerceiverConfig, PerceiverModel


class MultimodalFusionNetwork(nn.Module):
    """
    Bidirectional fusion network for multimodal Earth observations.
    
    This network maintains separate projection pathways for each encoder/modality
    pair, learning to translate between their specific representations and the
    universal space where patterns can be discovered across modalities.
    
    Key capabilities:
    - Variable input/output dimensions per modality
    - Support for both 1D (vectors) and 2D (sequences) inputs
    - Learnable projections that preserve semantic structure
    - Efficient attention-based fusion using Perceiver architecture
    """
    
    def __init__(
        self,
        encoder_configs: Dict[int, Dict],
        universal_dim: int = 507,
        device: torch.device = torch.device('cuda'),
        config=None
    ):
        """
        Initialize fusion network.

        Args:
            encoder_configs: Configuration for each encoder
                {encoder_id: {'name': str, 'input_dim': int, ...}}
            universal_dim: Target dimension for universal space
            device: Device for computation
            config: Optional DeepEarthConfig for projector settings
        """
        super().__init__()

        self.universal_dim = universal_dim
        self.encoder_configs = encoder_configs
        self.device = device
        self.config = config
        
        print(f"\n{'='*70}")
        print(f"Initializing Multimodal Fusion Network")
        print(f"{'='*70}")
        print(f"Universal dimension: {universal_dim}")
        print(f"Number of encoders: {len(encoder_configs)}")
        
        # ═══════════════════════════════════════════════════════════
        # Create projection networks
        # ═══════════════════════════════════════════════════════════
        
        self.to_universal = nn.ModuleDict()    # Modality → Universal
        self.from_universal = nn.ModuleDict()  # Universal → Modality
        
        for encoder_id, config in encoder_configs.items():
            encoder_name = config['name']
            input_dim = config['input_dim']

            print(f"\n  Encoder {encoder_id} ({encoder_name}):")
            print(f"    Input dimension: {input_dim}")

            # Skip Earth4D (encoder_id=1) as it's handled directly by spacetime encoder
            if encoder_id == 1:
                print(f"    → Earth4D handled by spacetime encoder (trainable)")
                continue

            # Forward projection (modality → universal)
            self.to_universal[str(encoder_id)] = self._create_projection(
                input_dim, universal_dim, config, f"{encoder_name}_to_universal"
            )

            # Backward projection (universal → modality)
            self.from_universal[str(encoder_id)] = self._create_projection(
                universal_dim, input_dim, config, f"{encoder_name}_from_universal"
            )

            print(f"    ✓ Created bidirectional projections")
        
        print(f"\n{'='*70}\n")
    
    def _create_projection(
        self,
        input_dim: int,
        output_dim: int,
        encoder_config: Dict,
        name: str
    ) -> nn.Module:
        """
        Create a Perceiver-based projection network.

        The Perceiver architecture is ideal for dimension projection because:
        1. It can handle variable input sizes efficiently
        2. The cross-attention mechanism preserves important information
        3. The iterative processing refines the representation

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            encoder_config: Encoder-specific configuration
            name: Name for logging

        Returns:
            PerceiverProjection module
        """
        # Use lightweight projector config if available, otherwise use defaults
        if hasattr(self.config, 'projector_config') and self.config.projector_config:
            proj_cfg = self.config.projector_config
            num_latents = proj_cfg.get('num_latents', 64)
            latent_dim = proj_cfg.get('latent_dim', 64)
            num_blocks = proj_cfg.get('num_blocks', 2)
            num_cross_heads = proj_cfg.get('num_cross_attention_heads', 2)
            num_self_heads = proj_cfg.get('num_self_attention_heads', 2)
            dropout = proj_cfg.get('dropout', 0.1)
        else:
            # Fallback to adaptive sizing based on input dimension
            if input_dim < 128:
                num_latents = 16
                latent_dim = 64
                num_blocks = 1
            elif input_dim < 512:
                num_latents = 32
                latent_dim = 64
                num_blocks = 2
            else:
                num_latents = 64
                latent_dim = 128
                num_blocks = 2
            num_cross_heads = 2
            num_self_heads = 2
            dropout = 0.1
        
        # Configure Perceiver with lightweight settings
        # Ensure dimensions are divisible by number of heads
        d_model = max(input_dim, 64)

        # Make sure d_model is divisible by num_cross_heads
        if d_model % num_cross_heads != 0:
            d_model = ((d_model // num_cross_heads) + 1) * num_cross_heads

        # Similarly for latent_dim and self-attention heads
        if latent_dim % num_self_heads != 0:
            latent_dim = ((latent_dim // num_self_heads) + 1) * num_self_heads

        perceiver_config = PerceiverConfig(
            num_latents=num_latents,
            d_latents=latent_dim,
            d_model=d_model,
            num_blocks=num_blocks,
            num_self_attends_per_block=1,
            num_self_attention_heads=num_self_heads,  # Use lightweight setting
            num_cross_attention_heads=num_cross_heads,  # Use lightweight setting
            qk_channels=None,
            v_channels=None,
            cross_attention_shape_for_attention="kv",
            self_attention_widening_factor=1,
            cross_attention_widening_factor=1,
            attention_probs_dropout_prob=dropout,
            use_query_residual=True
        )
        
        return PerceiverProjection(
            perceiver_config, 
            input_dim, 
            output_dim, 
            self.device,
            name
        )
    
    def project_to_universal(
        self,
        x: torch.Tensor,
        encoder_id: int,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Project from modality space to universal space.
        
        Args:
            x: Input tensor [batch, ...dims...]
            encoder_id: Which encoder's projection to use
            return_attention: Whether to return attention weights
            
        Returns:
            Universal representation [batch, universal_dim] or
            [batch, n_tokens, universal_dim] for sequence inputs
        """
        if str(encoder_id) not in self.to_universal:
            raise ValueError(f"Unknown encoder ID: {encoder_id}")
        
        return self.to_universal[str(encoder_id)](x, return_attention)
    
    def project_from_universal(
        self,
        x: torch.Tensor,
        encoder_id: int,
        target_shape: Optional[Tuple] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Project from universal space back to modality space.
        
        Args:
            x: Universal representation [batch, universal_dim]
            encoder_id: Which encoder's projection to use
            target_shape: Optional shape to reshape output
            return_attention: Whether to return attention weights
            
        Returns:
            Modality representation with original dimensions
        """
        if str(encoder_id) not in self.from_universal:
            raise ValueError(f"Unknown encoder ID: {encoder_id}")
        
        output = self.from_universal[str(encoder_id)](x, return_attention)
        
        # Reshape if target shape provided
        if target_shape is not None and output.shape != target_shape:
            try:
                output = output.view(target_shape)
            except RuntimeError:
                print(f"Warning: Could not reshape {output.shape} to {target_shape}")
        
        return output
    
    def get_projection_info(self, encoder_id: int) -> Dict:
        """Get information about projections for an encoder."""
        if str(encoder_id) not in self.to_universal:
            return {}
        
        to_universal = self.to_universal[str(encoder_id)]
        from_universal = self.from_universal[str(encoder_id)]
        
        return {
            'encoder_id': encoder_id,
            'encoder_name': self.encoder_configs[encoder_id]['name'],
            'input_dim': self.encoder_configs[encoder_id]['input_dim'],
            'universal_dim': self.universal_dim,
            'to_universal_params': sum(p.numel() for p in to_universal.parameters()),
            'from_universal_params': sum(p.numel() for p in from_universal.parameters())
        }


class PerceiverProjection(nn.Module):
    """
    Perceiver-based projection between arbitrary dimensions.
    
    This module uses the Perceiver architecture's cross-attention mechanism
    to project between dimensions while preserving important information.
    The latent bottleneck forces the model to learn compressed representations
    that capture the most salient features.
    
    Architecture:
        Input → Linear Projection → Cross-Attention with Latents
              → Self-Attention (Latents) → Output Projection
    """
    
    def __init__(
        self,
        config: PerceiverConfig,
        input_dim: int,
        output_dim: int,
        device: torch.device,
        name: str = "projection"
    ):
        """
        Initialize Perceiver projection.
        
        Args:
            config: Perceiver configuration
            input_dim: Input dimension
            output_dim: Output dimension
            device: Computation device
            name: Name for logging
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.name = name
        
        # ═══════════════════════════════════════════════════════════
        # Input processing
        # ═══════════════════════════════════════════════════════════

        # Project input to Perceiver's expected dimension
        self.input_projection = nn.Linear(input_dim, config.d_model)
        
        # Layer normalization for stable training
        self.input_norm = nn.LayerNorm(config.d_model)
        
        # ═══════════════════════════════════════════════════════════
        # Perceiver model (handles cross and self attention)
        # ═══════════════════════════════════════════════════════════
        
        self.perceiver = PerceiverModel(config)
        
        # ═══════════════════════════════════════════════════════════
        # Output processing
        # ═══════════════════════════════════════════════════════════
        
        # Project from latents to output dimension
        # We flatten the latent array for projection
        latent_total_dim = config.num_latents * config.d_latents
        
        self.output_projection = nn.Sequential(
            nn.Linear(latent_total_dim, output_dim * 2),
            nn.GELU(),
            nn.LayerNorm(output_dim * 2),
            nn.Dropout(config.attention_probs_dropout_prob),
            nn.Linear(output_dim * 2, output_dim)
        )
        
        # Move to device
        self.to(device)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Project input through Perceiver.
        
        The forward pass:
        1. Projects input to Perceiver dimension
        2. Cross-attention: latents query the input
        3. Self-attention: latents refine their representation
        4. Projects latents to output dimension
        
        Args:
            x: Input tensor [batch, dim] or [batch, seq, dim]
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor [batch, output_dim] or 
            [batch, seq, output_dim] for sequence inputs
        """
        original_shape = x.shape
        
        # ───────────────────────────────────────────────────────
        # Handle different input dimensions
        # ───────────────────────────────────────────────────────
        
        if x.dim() == 2:
            # [batch, dim] → [batch, 1, dim]
            x = x.unsqueeze(1)
            squeeze_output = True
        elif x.dim() == 3:
            # [batch, seq, dim] - keep as is
            squeeze_output = False
        else:
            # Handle higher dimensions by flattening
            batch_size = x.shape[0]
            x = x.view(batch_size, -1, self.input_dim)
            squeeze_output = False
        
        batch_size, seq_len, _ = x.shape
        
        # ───────────────────────────────────────────────────────
        # Project input to Perceiver dimension
        # ───────────────────────────────────────────────────────
        
        x_projected = self.input_projection(x)
        x_projected = self.input_norm(x_projected)
        
        # ───────────────────────────────────────────────────────
        # Process through Perceiver
        # ───────────────────────────────────────────────────────
        
        outputs = self.perceiver(
            inputs=x_projected,
            attention_mask=None,
            head_mask=None,
            output_attentions=return_attention,
            output_hidden_states=False,
            return_dict=True
        )
        
        # Get latent representation
        latents = outputs.last_hidden_state  # [batch, num_latents, d_latents]
        
        # ───────────────────────────────────────────────────────
        # Project to output dimension
        # ───────────────────────────────────────────────────────
        
        # Flatten latents for projection
        latents_flat = latents.reshape(batch_size, -1)
        
        # Project to output
        output = self.output_projection(latents_flat)  # [batch, output_dim]
        
        # ───────────────────────────────────────────────────────
        # Handle sequence outputs
        # ───────────────────────────────────────────────────────
        
        if seq_len > 1 and not squeeze_output:
            # For sequence inputs, we can either:
            # 1. Repeat the output for each position (default)
            # 2. Learn position-specific outputs (future enhancement)
            output = output.unsqueeze(1).expand(-1, seq_len, -1)
        elif squeeze_output:
            # Remove sequence dimension if we added it
            pass  # Already [batch, output_dim]
        
        if return_attention and outputs.attentions is not None:
            return output, outputs.attentions
        else:
            return output
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_device(self) -> torch.device:
        """Get device of module."""
        return next(self.parameters()).device
