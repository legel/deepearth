"""
Defines the complete, end-to-end DeepEarth model, integrating all
components from encoding to simulation.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from encoders.universal_encoder import create_universal_encoder
from models.cross_modal_fusion import CrossModalFusion, FusionConfig
from core.inductive_simulator import create_inductive_simulator

class DeepEarthModel(nn.Module):
    """
    The full DeepEarth model, orchestrating the universal encoder,
    cross-modal fusion, and the inductive simulator.
    """
    def __init__(self, universal_encoder, cross_modal_fusion, inductive_simulator):
        super().__init__()
        self.universal_encoder = universal_encoder
        self.cross_modal_fusion = cross_modal_fusion
        self.inductive_simulator = inductive_simulator
        # Simple decoders to project back from universal dim for reconstruction
        self.decoders = nn.ModuleDict()

    def forward(
        self,
        mask_config: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for training or inference.
        Handles encoding, masking, fusion, and simulation.
        """
        # 1. Encode all inputs to universal token space
        # The encoder expects a dictionary of inputs
        universal_tokens = self.universal_encoder(kwargs)

        # 2. Apply masking to the universal tokens
        masked_tokens = {}
        original_tokens = {}
        if mask_config:
            for name, tokens in universal_tokens.items():
                original_tokens[name] = tokens
                if name in mask_config and mask_config[name] > 0.0:
                    # Simple masking: replace with a learnable mask token
                    # A real implementation would use more advanced masking from train.py
                    mask_token = self.get_mask_token_for(tokens)
                    masked_tokens[name] = mask_token.expand_as(tokens)
                else:
                    masked_tokens[name] = tokens
        else:
            masked_tokens = universal_tokens

        # 3. Fuse tokens from all modalities
        fusion_outputs = self.cross_modal_fusion(masked_tokens)
        fused_representation = fusion_outputs['all_tokens']

        # 4. Run the inductive simulation
        simulated_outputs = self.inductive_simulator(fused_representation)
        simulated_features = simulated_outputs['simulated_features']

        # 5. Decode back to modality-specific embeddings for loss calculation
        outputs = {'reconstructions': {}}
        for name, original in original_tokens.items():
            if name not in self.decoders:
                # Dynamically create a decoder for this modality
                self.decoders[name] = nn.Linear(
                    self.inductive_simulator.config.hidden_dim,
                    original.shape[-1]
                ).to(fused_representation.device)

            # Find the corresponding part in the simulated sequence
            # This is a simplification; a real model uses token boundaries
            start_idx = 1 # Skip CLS token
            modality_simulated = simulated_features[:, start_idx : start_idx + original.shape[1]]
            
            reconstruction = self.decoders[name](modality_simulated)
            outputs['reconstructions'][name] = reconstruction
            
        return outputs
        
    def get_mask_token_for(self, x):
        # Helper to create a learnable mask token if it doesn't exist
        if not hasattr(self, '_mask_token'):
            self._mask_token = nn.Parameter(torch.randn(1, 1, x.shape[-1])).to(x.device)
        return self._mask_token


def create_full_deepearth_model(**kwargs):
    """Factory function to build the complete DeepEarth model."""
    universal_encoder = create_universal_encoder(**kwargs)
    
    # Config for fusion must match the encoder's output dimension
    fusion_config = FusionConfig(
        universal_dim=universal_encoder.config.universal_dim
    )
    cross_modal_fusion = CrossModalFusion(fusion_config)
    
    # Simulator's input must match the fusion output
    inductive_simulator = create_inductive_simulator(
        hidden_dim=fusion_config.universal_dim,
        preset=kwargs.get("preset", "standard")
    )
    
    model = DeepEarthModel(
        universal_encoder,
        cross_modal_fusion,
        inductive_simulator
    )
    
    # Move the entire composed model to the correct device
    if 'device' in kwargs:
        model = model.to(kwargs['device'])
        
    return model
