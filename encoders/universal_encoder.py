"""
Universal Encoder Infrastructure for DeepEarth
Handles extraction and projection of native embeddings to universal token space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class EncoderConfig:
    """Configuration for modality encoders"""
    name: str
    native_dim: int  # Native embedding dimension from pretrained model
    universal_dim: int = 2048  # Universal token dimension
    num_tokens_per_sample: int = 1  # How many universal tokens per input
    projection_type: str = "mlp"  # "mlp", "linear", "attention"
    projection_depth: int = 2
    use_layernorm: bool = True
    dropout: float = 0.1
    freeze_backbone: bool = True
    pooling_strategy: str = "mean"  # For multi-token inputs: "mean", "max", "cls", "attention"


class ModalityEncoder(ABC):
    """Abstract base class for modality-specific encoders"""
    
    @abstractmethod
    def extract_native_embeddings(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract native embeddings from pretrained model"""
        pass
    
    @abstractmethod
    def get_native_dim(self) -> int:
        """Return native embedding dimension"""
        pass


class VJEPAEncoder(ModalityEncoder):
    """V-JEPA 2 Vision Encoder with native embedding extraction"""
    
    def __init__(self, model_name: str = "vjepa2-base", extract_layers: List[int] = [-1]):
        # Import the actual V-JEPA model
        from encoders.vision.vjepa2_extractor import VJEPA2Extractor
        
        self.model = VJEPA2Extractor()  # Remove pretrained argument
        self.extract_layers = extract_layers
        
        # Try to get native dimension from model
        if hasattr(self.model, 'hidden_size'):
            self.native_dim = self.model.hidden_size
        elif hasattr(self.model, 'embed_dim'):
            self.native_dim = self.model.embed_dim
        else:
            # Default to 768 for base model
            self.native_dim = 768
        
        # Register hooks for intermediate representations
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to extract intermediate representations"""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Hook into specific transformer layers
        for idx in self.extract_layers:
            if idx == -1:
                # Final layer
                self.model.encoder.register_forward_hook(get_activation('final'))
            else:
                # Intermediate layers
                self.model.encoder.layers[idx].register_forward_hook(
                    get_activation(f'layer_{idx}')
                )
    
    def extract_native_embeddings(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract native V-JEPA embeddings at multiple scales
        
        Args:
            images: (B, C, H, W) input images
            
        Returns:
            Dict containing:
                - patch_embeddings: (B, N_patches, D) patch-level features
                - global_embedding: (B, D) global image representation
                - multiscale_features: List of features at different layers
        """
        # Clear previous activations
        self.activations.clear()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(images)
        
        # Extract different representations
        embeddings = {
            'patch_embeddings': outputs,  # (B, N_patches, D)
            'global_embedding': outputs.mean(dim=1),  # (B, D)
            'multiscale_features': []
        }
        
        # Collect multiscale features from hooks
        for name, activation in self.activations.items():
            embeddings['multiscale_features'].append(activation)
        
        return embeddings
    
    def get_native_dim(self) -> int:
        return self.native_dim


class LanguageEncoder(ModalityEncoder):
    """Language model encoder with native embedding extraction"""
    
    def __init__(self, model_name: str = "deepseek-v3", extract_layers: List[int] = [-1]):
        from encoders.language.deepseek_v3_encoder import DeepSeekV3Encoder
        
        self.model = DeepSeekV3Encoder()  # Remove arguments, let it use defaults
        self.extract_layers = extract_layers
        
        # Try to get native dimension
        if hasattr(self.model, 'hidden_size'):
            self.native_dim = self.model.hidden_size
        elif hasattr(self.model, 'out_dim'):
            self.native_dim = self.model.out_dim
        else:
            # Default to 4096 for large models
            self.native_dim = 4096
        
        # For extracting intermediate representations
        self.activations = {}
    
    def extract_native_embeddings(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract native language model embeddings
        
        Returns:
            Dict containing:
                - token_embeddings: (B, L, D) all token representations
                - pooled_embedding: (B, D) pooled sequence representation
                - layer_embeddings: List of embeddings from different layers
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Extract representations
        all_hidden_states = outputs.hidden_states  # Tuple of (B, L, D)
        last_hidden_state = outputs.last_hidden_state
        
        embeddings = {
            'token_embeddings': last_hidden_state,
            'pooled_embedding': self._pool_tokens(last_hidden_state, attention_mask),
            'layer_embeddings': [all_hidden_states[i] for i in self.extract_layers]
        }
        
        return embeddings
    
    def _pool_tokens(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool token embeddings into sequence representation"""
        if attention_mask is None:
            return token_embeddings.mean(dim=1)
        
        # Masked mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_native_dim(self) -> int:
        return self.native_dim


class UniversalProjector(nn.Module):
    """Projects native embeddings to universal token space"""
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        
        # Build projection network
        if config.projection_type == "linear":
            self.projector = nn.Linear(config.native_dim, config.universal_dim)
        
        elif config.projection_type == "mlp":
            layers = []
            in_dim = config.native_dim
            hidden_dim = (config.native_dim + config.universal_dim) // 2
            
            for i in range(config.projection_depth):
                out_dim = hidden_dim if i < config.projection_depth - 1 else config.universal_dim
                
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim) if config.use_layernorm else nn.Identity(),
                    nn.GELU() if i < config.projection_depth - 1 else nn.Identity(),
                    nn.Dropout(config.dropout) if i < config.projection_depth - 1 else nn.Identity()
                ])
                in_dim = out_dim
            
            self.projector = nn.Sequential(*layers)
        
        elif config.projection_type == "attention":
            # Cross-attention based projection
            self.query_proj = nn.Linear(config.universal_dim, config.universal_dim)
            self.key_proj = nn.Linear(config.native_dim, config.universal_dim)
            self.value_proj = nn.Linear(config.native_dim, config.universal_dim)
            self.output_proj = nn.Linear(config.universal_dim, config.universal_dim)
            
            # Learnable query tokens
            self.query_tokens = nn.Parameter(
                torch.randn(1, config.num_tokens_per_sample, config.universal_dim) * 0.02
            )
        
        # Token generation strategy
        if config.num_tokens_per_sample > 1:
            self.token_generator = self._create_token_generator()
    
    def _create_token_generator(self):
        """Create module for generating multiple tokens from single embedding"""
        if self.config.projection_type == "attention":
            # Already handled by query tokens
            return None
        else:
            # Linear projection to multiple tokens
            return nn.Linear(
                self.config.universal_dim,
                self.config.universal_dim * self.config.num_tokens_per_sample
            )
    
    def forward(
        self, 
        native_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Project native embeddings to universal tokens
        
        Args:
            native_embeddings: (B, N, D_native) or (B, D_native)
            mask: Optional attention mask
            
        Returns:
            universal_tokens: (B, K, D_universal) where K = num_tokens_per_sample
        """
        # Handle both 2D and 3D inputs
        if native_embeddings.dim() == 2:
            native_embeddings = native_embeddings.unsqueeze(1)
            single_token = True
        else:
            single_token = False
        
        B, N, D = native_embeddings.shape
        
        if self.config.projection_type == "attention":
            # Cross-attention projection
            Q = self.query_tokens.expand(B, -1, -1)  # (B, K, D_universal)
            Q = self.query_proj(Q)
            
            K = self.key_proj(native_embeddings)  # (B, N, D_universal)
            V = self.value_proj(native_embeddings)  # (B, N, D_universal)
            
            # Compute attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.config.universal_dim)
            
            if mask is not None:
                scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            universal_tokens = torch.matmul(attn_weights, V)  # (B, K, D_universal)
            universal_tokens = self.output_proj(universal_tokens)
            
        else:
            # MLP or linear projection
            if not single_token and self.config.pooling_strategy != "none":
                # Pool multiple input tokens to single representation
                native_embeddings = self._pool_embeddings(native_embeddings, mask)
            
            # Project to universal dimension
            universal_tokens = self.projector(native_embeddings)  # (B, 1, D_universal)
            
            # Generate multiple tokens if needed
            if self.config.num_tokens_per_sample > 1 and self.token_generator is not None:
                universal_tokens = self.token_generator(universal_tokens)
                B, _, _ = universal_tokens.shape
                universal_tokens = universal_tokens.reshape(
                    B, self.config.num_tokens_per_sample, self.config.universal_dim
                )
        
        return universal_tokens
    
    def _pool_embeddings(
        self, 
        embeddings: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pool multiple embeddings based on strategy"""
        if self.config.pooling_strategy == "mean":
            if mask is None:
                return embeddings.mean(dim=1, keepdim=True)
            else:
                mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1, keepdim=True)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1, keepdim=True), min=1e-9)
                return sum_embeddings / sum_mask
        
        elif self.config.pooling_strategy == "max":
            if mask is not None:
                embeddings = embeddings.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            return embeddings.max(dim=1, keepdim=True)[0]
        
        elif self.config.pooling_strategy == "cls":
            return embeddings[:, 0:1, :]  # First token
        
        else:
            return embeddings


class UniversalEncoderModule(nn.Module):
    """Complete universal encoder module combining extraction and projection"""
    
    def __init__(
        self,
        modality_configs: Dict[str, EncoderConfig],
        share_projectors: bool = False
    ):
        super().__init__()
        self.modality_configs = modality_configs
        
        # Initialize modality-specific encoders
        self.encoders = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        
        # Shared projector for all modalities (if requested)
        if share_projectors:
            # Find max native dim
            max_native_dim = max(config.native_dim for config in modality_configs.values())
            shared_config = EncoderConfig(
                name="shared",
                native_dim=max_native_dim,
                universal_dim=list(modality_configs.values())[0].universal_dim
            )
            self.shared_projector = UniversalProjector(shared_config)
            
            # Add linear adapters to match dimensions
            self.adapters = nn.ModuleDict({
                name: nn.Linear(config.native_dim, max_native_dim)
                for name, config in modality_configs.items()
            })
        
        for name, config in modality_configs.items():
            # Initialize encoder based on modality
            if name == "vision":
                self.encoders[name] = VJEPAEncoder()
            elif name == "language":
                self.encoders[name] = LanguageEncoder()
            # Add more modalities as needed
            
            # Initialize projector
            if not share_projectors:
                self.projectors[name] = UniversalProjector(config)
            
            # Freeze backbone if requested
            if config.freeze_backbone and name in self.encoders:
                for param in self.encoders[name].model.parameters():
                    param.requires_grad = False
    
    def forward(
        self,
        inputs: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multiple modalities to universal token space
        
        Args:
            inputs: Dict mapping modality names to their inputs
            
        Returns:
            Dict mapping modality names to universal tokens (B, K, D_universal)
        """
        universal_tokens = {}
        
        for name, modality_input in inputs.items():
            if name not in self.encoders:
                continue
            
            # Extract native embeddings
            if name == "vision":
                native_embeds = self.encoders[name].extract_native_embeddings(modality_input)
                # Use patch embeddings by default
                native_embeds = native_embeds['patch_embeddings']
            
            elif name == "language":
                if isinstance(modality_input, dict):
                    native_embeds = self.encoders[name].extract_native_embeddings(
                        modality_input['input_ids'],
                        modality_input.get('attention_mask')
                    )
                    native_embeds = native_embeds['token_embeddings']
                else:
                    # Assume it's just input_ids
                    native_embeds = self.encoders[name].extract_native_embeddings(modality_input)
                    native_embeds = native_embeds['token_embeddings']
            
            # Project to universal space
            if hasattr(self, 'shared_projector'):
                # Adapt dimensions first
                native_embeds_flat = native_embeds.reshape(-1, native_embeds.shape[-1])
                adapted = self.adapters[name](native_embeds_flat)
                adapted = adapted.reshape(*native_embeds.shape[:-1], -1)
                universal = self.shared_projector(adapted)
            else:
                universal = self.projectors[name](native_embeds)
            
            universal_tokens[name] = universal
        
        return universal_tokens


class UniversalDecoder(nn.Module):
    """Decoder from universal token space back to modality-specific space"""
    
    def __init__(
        self,
        target_configs: Dict[str, Dict[str, int]],
        universal_dim: int = 2048
    ):
        super().__init__()
        self.universal_dim = universal_dim
        
        # Create decoders for each target modality
        self.decoders = nn.ModuleDict()
        
        for name, config in target_configs.items():
            target_dim = config['dim']
            decoder_type = config.get('type', 'mlp')
            
            if decoder_type == 'mlp':
                self.decoders[name] = nn.Sequential(
                    nn.Linear(universal_dim, universal_dim // 2),
                    nn.LayerNorm(universal_dim // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(universal_dim // 2, target_dim)
                )
            elif decoder_type == 'linear':
                self.decoders[name] = nn.Linear(universal_dim, target_dim)
    
    def forward(
        self,
        universal_tokens: torch.Tensor,
        target_modality: str
    ) -> torch.Tensor:
        """Decode universal tokens to target modality"""
        if target_modality not in self.decoders:
            raise ValueError(f"No decoder for modality: {target_modality}")
        
        return self.decoders[target_modality](universal_tokens)


# Example usage
if __name__ == "__main__":
    # Configure encoders
    configs = {
        "vision": EncoderConfig(
            name="vision",
            native_dim=768,  # V-JEPA base
            universal_dim=2048,
            num_tokens_per_sample=4,  # Multiple tokens for image
            projection_type="attention",
            freeze_backbone=True
        ),
        "language": EncoderConfig(
            name="language", 
            native_dim=4096,  # DeepSeek large
            universal_dim=2048,
            num_tokens_per_sample=1,
            projection_type="mlp",
            freeze_backbone=True
        )
    }
    
    # Create universal encoder
    universal_encoder = UniversalEncoderModule(configs)
    
    # Example inputs
    batch_size = 2
    inputs = {
        "vision": torch.randn(batch_size, 3, 224, 224),
        "language": {
            "input_ids": torch.randint(0, 1000, (batch_size, 32)),
            "attention_mask": torch.ones(batch_size, 32)
        }
    }
    
    # Encode to universal space
    universal_tokens = universal_encoder(inputs)
    
    print("Universal token shapes:")
    for name, tokens in universal_tokens.items():
        print(f"{name}: {tokens.shape}")
