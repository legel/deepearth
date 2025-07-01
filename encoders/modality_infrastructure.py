"""
DeepEarth Modality Infrastructure
Comprehensive system for extracting native embeddings and decoding to universal tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, List, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import logging

# DeepSeek components (from your deepseek_components module)
from deepseek_components import (
    DeepSeekConfig,
    DeepSeekTransformer,
    DeepSeekMLP,
    DeepSeekMoE,
    DeepseekV3RMSNorm
)

# External model imports
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
import timm


@dataclass
class UniversalTokenConfig:
    """Configuration for universal token generation"""
    universal_dim: int = 2048  # Target dimension for all modalities
    max_tokens_per_modality: int = 16  # Maximum universal tokens per modality
    
    # Token generation strategies
    token_generation_strategy: str = "adaptive"  # "fixed", "adaptive", "hierarchical"
    min_tokens: int = 1
    
    # Pooling strategies for single token
    pooling_strategy: str = "weighted"  # "mean", "max", "weighted", "learned"


@dataclass
class ModalityDecoderConfig:
    """Enhanced configuration for modality decoders"""
    name: str
    input_dim: int  # Native dimension from encoder
    output_dim: int  # Universal dimension
    num_tokens: int = 1  # Number of universal tokens
    
    # DeepSeek Transformer settings
    num_layers: int = 4
    num_heads: int = 16
    num_key_value_heads: Optional[int] = None
    intermediate_size: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    
    # MoE configuration
    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: Optional[int] = None
    n_shared_experts: Optional[int] = 1
    moe_layer_freq: int = 1
    
    # Advanced features
    use_cross_attention: bool = True  # For multi-token generation
    use_position_encoding: bool = True
    dropout_prob: float = 0.1
    
    def to_deepseek_config(self) -> DeepSeekConfig:
        """Convert to DeepSeek configuration"""
        return DeepSeekConfig(
            hidden_size=self.output_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads or self.num_heads,
            intermediate_size=self.intermediate_size,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            # MoE settings
            n_routed_experts=self.num_experts if self.use_moe else None,
            num_experts_per_tok=self.num_experts_per_tok,
            moe_intermediate_size=self.moe_intermediate_size or (self.intermediate_size // 4),
            n_shared_experts=self.n_shared_experts,
            moe_layer_freq=self.moe_layer_freq,
            first_k_dense_replace=0,
        )


class BaseModalityExtractor(ABC):
    """Abstract base class for modality feature extraction"""
    
    @abstractmethod
    def extract_native_embeddings(self, inputs: Any) -> Dict[str, torch.Tensor]:
        """Extract native embeddings from pretrained model"""
        pass
    
    @abstractmethod
    def get_native_dim(self) -> int:
        """Get the native embedding dimension"""
        pass


class VJEPA2Extractor(BaseModalityExtractor):
    """V-JEPA2 vision feature extractor"""
    
    def __init__(self, model_name: str = "vjepa2_base", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self.logger = logging.getLogger('DeepEarth.VJEPA2')
        
        # Initialize V-JEPA2 model
        self._load_model()
        
    def _load_model(self):
        """Load V-JEPA2 model"""
        # This is a placeholder - replace with actual V-JEPA2 loading
        # For now, using a Vision Transformer as proxy
        if self.model_name == "vjepa2_base":
            self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.native_dim = 768
        elif self.model_name == "vjepa2_large":
            self.model = timm.create_model('vit_large_patch16_224', pretrained=True)
            self.native_dim = 1024
        else:
            raise ValueError(f"Unknown V-JEPA2 model: {self.model_name}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.logger.info(f"Loaded V-JEPA2 model: {self.model_name}")
        
    def extract_native_embeddings(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from V-JEPA2
        
        Args:
            images: (B, C, H, W) image tensor
            
        Returns:
            Dictionary with:
                - patch_embeddings: (B, N_patches, D) patch-level features
                - global_embedding: (B, D) global image representation
                - cls_embedding: (B, D) CLS token if available
        """
        with torch.no_grad():
            # Get patch embeddings
            x = self.model.patch_embed(images)
            B, N, D = x.shape
            
            # Add CLS token if model has one
            if hasattr(self.model, 'cls_token'):
                cls_tokens = self.model.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
            
            # Add position embeddings
            if hasattr(self.model, 'pos_embed'):
                x = x + self.model.pos_embed[:, :x.size(1)]
            
            # Pass through transformer blocks
            for blk in self.model.blocks:
                x = blk(x)
            
            # Apply norm
            x = self.model.norm(x)
            
            # Extract different representations
            if hasattr(self.model, 'cls_token'):
                cls_embedding = x[:, 0]
                patch_embeddings = x[:, 1:]
            else:
                cls_embedding = x.mean(dim=1)  # Global average pooling
                patch_embeddings = x
            
            global_embedding = patch_embeddings.mean(dim=1)
            
            return {
                'patch_embeddings': patch_embeddings,
                'global_embedding': global_embedding,
                'cls_embedding': cls_embedding
            }
    
    def get_native_dim(self) -> int:
        return self.native_dim


class LanguageModelExtractor(BaseModalityExtractor):
    """Language model feature extractor supporting multiple model families"""
    
    def __init__(self, model_name: str = "deepseek-ai/deepseek-llm-7b-base", 
                 device: str = "cuda", precision: str = "fp16"):
        self.device = device
        self.model_name = model_name
        self.precision = precision
        self.logger = logging.getLogger('DeepEarth.LanguageModel')
        
        self._load_model()
        
    def _load_model(self):
        """Load language model and tokenizer"""
        # Determine dtype
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16
        }
        torch_dtype = dtype_map.get(self.precision, torch.float16)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        
        self.model.eval()
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Get native dimension
        if hasattr(self.model.config, 'hidden_size'):
            self.native_dim = self.model.config.hidden_size
        else:
            # Fallback: try to infer from model
            dummy_input = torch.randint(0, 1000, (1, 10)).to(self.device)
            with torch.no_grad():
                output = self.model(dummy_input, output_hidden_states=True)
                self.native_dim = output.last_hidden_state.shape[-1]
                
        self.logger.info(f"Loaded language model: {self.model_name} (dim={self.native_dim})")
        
    def extract_native_embeddings(self, 
                                 input_ids: torch.Tensor,
                                 attention_mask: Optional[torch.Tensor] = None,
                                 extract_layers: List[int] = [-1]) -> Dict[str, torch.Tensor]:
        """
        Extract features from language model
        
        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L) attention mask
            extract_layers: Which layers to extract (-1 for last)
            
        Returns:
            Dictionary with token and sequence embeddings
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get all hidden states
            all_hidden_states = outputs.hidden_states
            
            # Extract specified layers
            layer_embeddings = []
            for layer_idx in extract_layers:
                layer_embeddings.append(all_hidden_states[layer_idx])
            
            # Token-level embeddings from last layer
            token_embeddings = all_hidden_states[-1]
            
            # Sequence-level: weighted mean pooling
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                sequence_embedding = sum_embeddings / sum_mask
            else:
                sequence_embedding = token_embeddings.mean(dim=1)
            
            return {
                'token_embeddings': token_embeddings,
                'sequence_embedding': sequence_embedding,
                'layer_embeddings': layer_embeddings,
                'attention_mask': attention_mask
            }
    
    def get_native_dim(self) -> int:
        return self.native_dim
    
    def tokenize(self, texts: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """Tokenize text inputs"""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **kwargs
        ).to(self.device)


class UniversalModalityDecoder(nn.Module):
    """
    Universal decoder using DeepSeek components
    Decodes native embeddings to universal token space
    """
    
    def __init__(self, config: ModalityDecoderConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(f'DeepEarth.Decoder.{config.name}')
        
        # Build decoder architecture
        self._build_input_projection()
        self._build_transformer()
        self._build_output_projection()
        self._build_token_generation()
        
        # Initialize
        self._init_weights()
        
    def _build_input_projection(self):
        """Build input projection with MoE"""
        if self.config.use_moe:
            # MoE for diverse inputs
            input_config = DeepSeekConfig(
                hidden_size=self.config.input_dim,
                n_routed_experts=self.config.num_experts // 2,
                num_experts_per_tok=2,
                moe_intermediate_size=self.config.output_dim // 2,
                n_shared_experts=1,
            )
            self.input_projection = DeepSeekMoE(input_config)
            self.input_norm = DeepseekV3RMSNorm(self.config.input_dim)
            
            # Additional projection to match dimensions
            self.dim_projection = nn.Linear(self.config.input_dim, self.config.output_dim)
        else:
            # Simple linear projection
            self.input_projection = nn.Linear(self.config.input_dim, self.config.output_dim)
            self.input_norm = None
            self.dim_projection = None
            
    def _build_transformer(self):
        """Build DeepSeek transformer"""
        transformer_config = self.config.to_deepseek_config()
        self.transformer = DeepSeekTransformer(transformer_config)
        
        # Position encodings for sequences
        if self.config.use_position_encoding:
            max_len = 1024  # Max sequence length
            self.position_embeddings = nn.Parameter(
                torch.randn(1, max_len, self.config.output_dim) * 0.02
            )
        else:
            self.position_embeddings = None
            
    def _build_output_projection(self):
        """Build output projection"""
        self.output_norm = DeepseekV3RMSNorm(self.config.output_dim)
        
    def _build_token_generation(self):
        """Build components for multi-token generation"""
        if self.config.num_tokens > 1:
            # Learnable query tokens
            self.query_tokens = nn.Parameter(
                torch.randn(1, self.config.num_tokens, self.config.output_dim) * 0.02
            )
            
            if self.config.use_cross_attention:
                # Cross-attention for token generation
                self.cross_attention = nn.MultiheadAttention(
                    self.config.output_dim,
                    self.config.num_heads,
                    dropout=self.config.dropout_prob,
                    batch_first=True
                )
                self.cross_norm = DeepseekV3RMSNorm(self.config.output_dim)
                
            # Token specialization with MoE
            if self.config.use_moe:
                token_config = DeepSeekConfig(
                    hidden_size=self.config.output_dim,
                    n_routed_experts=min(self.config.num_experts, self.config.num_tokens * 2),
                    num_experts_per_tok=2,
                    moe_intermediate_size=self.config.output_dim,
                )
                self.token_specializer = DeepSeekMoE(token_config)
            else:
                mlp_config = DeepSeekConfig(
                    hidden_size=self.config.output_dim,
                    intermediate_size=self.config.output_dim * 2,
                )
                self.token_specializer = DeepSeekMLP(mlp_config)
                
        else:
            # Single token generation
            if self.config.use_moe:
                pool_config = DeepSeekConfig(
                    hidden_size=self.config.output_dim,
                    n_routed_experts=4,
                    num_experts_per_tok=1,
                    moe_intermediate_size=self.config.output_dim // 2,
                )
                self.pooling_mlp = DeepSeekMoE(pool_config)
            else:
                pool_config = DeepSeekConfig(
                    hidden_size=self.config.output_dim,
                    intermediate_size=self.config.output_dim,
                )
                self.pooling_mlp = DeepSeekMLP(pool_config)
                
    def _init_weights(self):
        """Initialize weights"""
        # DeepSeek components handle their own initialization
        # Initialize only our custom parameters
        if hasattr(self, 'dim_projection') and self.dim_projection is not None:
            nn.init.xavier_uniform_(self.dim_projection.weight)
            if self.dim_projection.bias is not None:
                nn.init.zeros_(self.dim_projection.bias)
                
    def forward(self, 
                native_embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_all_tokens: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode native embeddings to universal tokens
        
        Args:
            native_embeddings: (B, N, D_native) or (B, D_native)
            attention_mask: (B, N) for sequences
            return_all_tokens: Return intermediate tokens
            
        Returns:
            universal_tokens: (B, K, D_universal) or dict with additional info
        """
        # Handle both sequence and single embedding inputs
        if native_embeddings.dim() == 2:
            native_embeddings = native_embeddings.unsqueeze(1)
            single_input = True
        else:
            single_input = False
            
        B, N, D = native_embeddings.shape
        
        # Input projection
        if self.config.use_moe:
            if self.input_norm is not None:
                native_embeddings = self.input_norm(native_embeddings)
            hidden_states = self.input_projection(native_embeddings)
            hidden_states = self.dim_projection(hidden_states)
        else:
            hidden_states = self.input_projection(native_embeddings)
            
        # Add position embeddings
        if self.position_embeddings is not None and N > 1:
            if N <= self.position_embeddings.shape[1]:
                hidden_states = hidden_states + self.position_embeddings[:, :N]
            else:
                # Interpolate for longer sequences
                pos_embed = F.interpolate(
                    self.position_embeddings.transpose(1, 2),
                    size=N,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
                hidden_states = hidden_states + pos_embed
                
        # Apply transformer
        if attention_mask is not None:
            # Convert to transformer format
            attn_mask = attention_mask.float()
            attn_mask = attn_mask.masked_fill(attention_mask == 0, float('-inf'))
            attn_mask = attn_mask.masked_fill(attention_mask == 1, float(0.0))
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        else:
            attn_mask = None
            
        transformed = self.transformer(hidden_states, attention_mask=attn_mask)
        
        # Generate output tokens
        if self.config.num_tokens > 1:
            output_tokens = self._generate_multi_tokens(transformed, attention_mask)
        else:
            output_tokens = self._generate_single_token(transformed, attention_mask, single_input)
            
        # Final normalization
        output_tokens = self.output_norm(output_tokens)
        
        if return_all_tokens:
            return {
                'universal_tokens': output_tokens,
                'transformed_features': transformed,
                'attention_weights': None  # Could add attention extraction
            }
        else:
            return output_tokens
            
    def _generate_multi_tokens(self, 
                              hidden_states: torch.Tensor,
                              attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate multiple universal tokens"""
        B = hidden_states.shape[0]
        
        # Initialize with learnable queries
        tokens = self.query_tokens.expand(B, -1, -1)
        
        if self.config.use_cross_attention:
            # Cross-attention to sequence
            if attention_mask is not None:
                key_padding_mask = (attention_mask == 0)
                if key_padding_mask.dim() == 3:
                    key_padding_mask = key_padding_mask.squeeze(1).squeeze(1)
            else:
                key_padding_mask = None
                
            attended, _ = self.cross_attention(
                query=tokens,
                key=hidden_states,
                value=hidden_states,
                key_padding_mask=key_padding_mask
            )
            
            # Residual and norm
            tokens = self.cross_norm(tokens + attended)
            
        else:
            # Simple pooling + expansion
            if attention_mask is not None and attention_mask.dim() > 1:
                mask = attention_mask.float()
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(-1)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            else:
                pooled = hidden_states.mean(dim=1)
                
            # Broadcast to all tokens
            pooled_expanded = pooled.unsqueeze(1).expand(-1, self.config.num_tokens, -1)
            tokens = tokens + pooled_expanded
            
        # Specialize tokens
        specialized = self.token_specializer(tokens)
        output_tokens = tokens + specialized  # Residual
        
        return output_tokens
        
    def _generate_single_token(self, 
                              hidden_states: torch.Tensor,
                              attention_mask: Optional[torch.Tensor] = None,
                              single_input: bool = False) -> torch.Tensor:
        """Generate single universal token"""
        B, N, D = hidden_states.shape
        
        if single_input or N == 1:
            # Already single token
            pooled = hidden_states.squeeze(1)
        else:
            # Pool sequence
            if attention_mask is not None and attention_mask.numel() > 0:
                # Masked pooling
                if attention_mask.dim() == 4:
                    mask = (attention_mask.squeeze(1).squeeze(1) != float('-inf')).float()
                else:
                    mask = attention_mask.float()
                mask = mask.unsqueeze(-1)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            else:
                # Attention-weighted pooling
                weights = torch.softmax(hidden_states.norm(dim=-1), dim=-1)
                pooled = torch.einsum('bn,bnd->bd', weights, hidden_states)
                
        # Refine through MoE/MLP
        refined = self.pooling_mlp(pooled)
        output = pooled + refined  # Residual
        
        return output.unsqueeze(1)  # (B, 1, D)


class DeepEarthModalityProcessor:
    """
    High-level interface for modality processing
    Combines extractors and decoders
    """
    
    def __init__(self, 
                 modality_name: str,
                 extractor: BaseModalityExtractor,
                 decoder_config: Optional[ModalityDecoderConfig] = None,
                 universal_config: Optional[UniversalTokenConfig] = None):
        
        self.modality_name = modality_name
        self.extractor = extractor
        self.universal_config = universal_config or UniversalTokenConfig()
        self.logger = logging.getLogger(f'DeepEarth.Processor.{modality_name}')
        
        # Create decoder config if not provided
        if decoder_config is None:
            decoder_config = ModalityDecoderConfig(
                name=modality_name,
                input_dim=extractor.get_native_dim(),
                output_dim=self.universal_config.universal_dim,
                num_tokens=self._determine_num_tokens(modality_name)
            )
            
        self.decoder = UniversalModalityDecoder(decoder_config)
        self.decoder_config = decoder_config
        
        self.logger.info(f"Initialized {modality_name} processor: "
                        f"{extractor.get_native_dim()}D -> "
                        f"{decoder_config.num_tokens}x{decoder_config.output_dim}D")
        
    def _determine_num_tokens(self, modality_name: str) -> int:
        """Determine number of tokens based on modality type"""
        if self.universal_config.token_generation_strategy == "fixed":
            return self.universal_config.max_tokens_per_modality
        elif self.universal_config.token_generation_strategy == "adaptive":
            # Heuristics based on modality
            if "vision" in modality_name or "image" in modality_name:
                return min(16, self.universal_config.max_tokens_per_modality)
            elif "video" in modality_name:
                return min(32, self.universal_config.max_tokens_per_modality)
            elif "language" in modality_name or "text" in modality_name:
                return min(4, self.universal_config.max_tokens_per_modality)
            else:
                return self.universal_config.min_tokens
        else:
            return self.universal_config.min_tokens
            
    def process(self, 
                inputs: Any,
                return_native: bool = False,
                return_details: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process inputs through full pipeline
        
        Args:
            inputs: Raw inputs for the modality
            return_native: Include native embeddings in output
            return_details: Return detailed information
            
        Returns:
            Universal tokens or dictionary with details
        """
        # Extract native embeddings
        native_embeddings = self.extractor.extract_native_embeddings(inputs)
        
        # Select which embeddings to decode
        if hasattr(self.decoder_config, 'embedding_selection'):
            selected = native_embeddings[self.decoder_config.embedding_selection]
        else:
            # Default selection based on modality
            if 'patch_embeddings' in native_embeddings and self.decoder_config.num_tokens > 1:
                selected = native_embeddings['patch_embeddings']
            elif 'token_embeddings' in native_embeddings:
                selected = native_embeddings['token_embeddings']
            elif 'sequence_embedding' in native_embeddings:
                selected = native_embeddings['sequence_embedding']
            else:
                selected = native_embeddings['global_embedding']
                
        # Get attention mask if available
        attention_mask = native_embeddings.get('attention_mask', None)
        
        # Decode to universal tokens
        if return_details:
            decoder_output = self.decoder(selected, attention_mask, return_all_tokens=True)
            universal_tokens = decoder_output['universal_tokens']
        else:
            universal_tokens = self.decoder(selected, attention_mask)
            
        # Prepare output
        if return_details:
            output = {
                'universal_tokens': universal_tokens,
                'modality_name': self.modality_name,
                'num_tokens': universal_tokens.shape[1],
                'token_dim': universal_tokens.shape[2],
            }
            if return_native:
                output['native_embeddings'] = native_embeddings
            if isinstance(decoder_output, dict):
                output.update(decoder_output)
            return output
        elif return_native:
            return universal_tokens, native_embeddings
        else:
            return universal_tokens


# Factory functions for common configurations

def create_vision_processor(
    model_name: str = "vjepa2_base",
    num_tokens: int = 16,
    universal_dim: int = 2048,
    use_moe: bool = True
) -> DeepEarthModalityProcessor:
    """Create a vision modality processor"""
    
    # Initialize V-JEPA2 extractor
    extractor = VJEPA2Extractor(model_name=model_name)
    
    # Create decoder config
    decoder_config = ModalityDecoderConfig(
        name="vision",
        input_dim=extractor.get_native_dim(),
        output_dim=universal_dim,
        num_tokens=num_tokens,
        num_layers=6,  # Deeper for spatial understanding
        num_heads=16,
        intermediate_size=universal_dim * 4,
        use_moe=use_moe,
        num_experts=8,
        use_cross_attention=True,  # Important for patch processing
        use_position_encoding=True  # Spatial positions matter
    )
    
    return DeepEarthModalityProcessor("vision", extractor, decoder_config)


def create_language_processor(
    model_name: str = "deepseek-ai/deepseek-llm-7b-base",
    num_tokens: int = 4,
    universal_dim: int = 2048,
    use_moe: bool = True,
    precision: str = "fp16"
) -> DeepEarthModalityProcessor:
    """Create a language modality processor"""
    
    # Initialize language model extractor
    extractor = LanguageModelExtractor(model_name=model_name, precision=precision)
    
    # Create decoder config
    decoder_config = ModalityDecoderConfig(
        name="language",
        input_dim=extractor.get_native_dim(),
        output_dim=universal_dim,
        num_tokens=num_tokens,
        num_layers=4,
        num_heads=16,
        intermediate_size=universal_dim * 4,
        use_moe=use_moe,
        num_experts=8,
        use_cross_attention=True,
        embedding_selection='token_embeddings'  # Use token-level features
    )
    
    return DeepEarthModalityProcessor("language", extractor, decoder_config)


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Vision Processing
    print("="*80)
    print("Example 1: Vision Processing with V-JEPA2")
    print("="*80)
    
    # Create vision processor
    vision_processor = create_vision_processor(
        model_name="vjepa2_base",
        num_tokens=16,  # 4x4 grid of tokens
        universal_dim=2048,
        use_moe=True
    )
    
    # Process an image
    dummy_image = torch.randn(2, 3, 224, 224).cuda()  # Batch of 2 images
    
    # Get universal tokens
    vision_tokens = vision_processor.process(dummy_image)
    print(f"Vision tokens shape: {vision_tokens.shape}")  # (2, 16, 2048)
    
    # Get detailed output
    vision_details = vision_processor.process(
        dummy_image, 
        return_native=True, 
        return_details=True
    )
    print(f"Native patch embeddings: {vision_details['native_embeddings']['patch_embeddings'].shape}")
    print(f"Universal tokens: {vision_details['universal_tokens'].shape}")
    
    # Example 2: Language Processing
    print("\n" + "="*80)
    print("Example 2: Language Processing with DeepSeek")
    print("="*80)
    
    # Create language processor
    language_processor = create_language_processor(
        model_name="deepseek-ai/deepseek-llm-7b-base",
        num_tokens=4,  # 4 semantic tokens
        universal_dim=2048,
        use_moe=True,
        precision="fp16"
    )
    
    # Process text
    texts = [
        "The soil moisture levels indicate optimal conditions for planting.",
        "Satellite imagery shows early signs of crop stress in the northern fields."
    ]
    
    # Tokenize
    tokenized = language_processor.extractor.tokenize(texts)
    
    # Get universal tokens
    language_tokens = language_processor.process(tokenized)
    print(f"Language tokens shape: {language_tokens.shape}")  # (2, 4, 2048)
    
    # Example 3: Multi-Modal Integration
    print("\n" + "="*80)
    print("Example 3: Multi-Modal Token Integration")
    print("="*80)
    
    # Combine tokens from different modalities
    all_tokens = torch.cat([
        vision_tokens,    # (2, 16, 2048)
        language_tokens   # (2, 4, 2048)
    ], dim=1)  # (2, 20, 2048)
    
    print(f"Combined tokens shape: {all_tokens.shape}")
    print(f"Total tokens per sample: {all_tokens.shape[1]}")
    print(f"  - Vision tokens: {vision_tokens.shape[1]}")
    print(f"  - Language tokens: {language_tokens.shape[1]}")


# Advanced configurations for specific use cases

def create_satellite_vision_processor(**kwargs) -> DeepEarthModalityProcessor:
    """Create processor optimized for satellite imagery"""
    
    extractor = VJEPA2Extractor(model_name="vjepa2_large")  # Larger model for fine details
    
    decoder_config = ModalityDecoderConfig(
        name="satellite_vision",
        input_dim=extractor.get_native_dim(),
        output_dim=kwargs.get('universal_dim', 2048),
        num_tokens=kwargs.get('num_tokens', 64),  # 8x8 grid for high resolution
        num_layers=8,  # Deeper for complex spatial patterns
        num_heads=16,
        intermediate_size=kwargs.get('universal_dim', 2048) * 4,
        use_moe=True,
        num_experts=16,  # More experts for diverse terrain types
        num_experts_per_tok=4,
        use_cross_attention=True,
        use_position_encoding=True
    )
    
    return DeepEarthModalityProcessor("satellite_vision", extractor, decoder_config)


def create_agricultural_language_processor(**kwargs) -> DeepEarthModalityProcessor:
    """Create processor for agricultural text analysis"""
    
    # Use specialized model if available
    model_name = kwargs.get('model_name', 'deepseek-ai/deepseek-llm-33b-base')
    
    extractor = LanguageModelExtractor(
        model_name=model_name,
        precision=kwargs.get('precision', 'fp32')  # Higher precision for technical text
    )
    
    decoder_config = ModalityDecoderConfig(
        name="agricultural_language",
        input_dim=extractor.get_native_dim(),
        output_dim=kwargs.get('universal_dim', 2048),
        num_tokens=kwargs.get('num_tokens', 8),  # More tokens for technical detail
        num_layers=6,
        num_heads=16,
        intermediate_size=kwargs.get('universal_dim', 2048) * 4,
        use_moe=True,
        num_experts=8,
        n_shared_experts=2,  # Shared experts for common agricultural terms
        embedding_selection='token_embeddings'
    )
    
    return DeepEarthModalityProcessor("agricultural_language", extractor, decoder_config)


class MultiModalProcessor:
    """
    Orchestrates multiple modality processors for DeepEarth
    """
    
    def __init__(self, universal_config: Optional[UniversalTokenConfig] = None):
        self.universal_config = universal_config or UniversalTokenConfig()
        self.processors = {}
        self.logger = logging.getLogger('DeepEarth.MultiModal')
        
    def add_processor(self, name: str, processor: DeepEarthModalityProcessor):
        """Add a modality processor"""
        self.processors[name] = processor
        self.logger.info(f"Added {name} processor")
        
    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process a batch containing multiple modalities
        
        Args:
            batch: Dict with modality names as keys and inputs as values
            
        Returns:
            Dict with modality names as keys and universal tokens as values
        """
        results = {}
        
        for modality_name, inputs in batch.items():
            if modality_name in self.processors:
                results[modality_name] = self.processors[modality_name].process(inputs)
            else:
                self.logger.warning(f"No processor for modality: {modality_name}")
                
        return results
    
    def get_total_tokens(self, batch_results: Dict[str, torch.Tensor]) -> int:
        """Calculate total number of tokens across all modalities"""
        total = 0
        for tokens in batch_results.values():
            total += tokens.shape[1]
        return total
    
    def create_token_sequence(self, 
                             batch_results: Dict[str, torch.Tensor],
                             add_modality_embeddings: bool = True) -> torch.Tensor:
        """
        Create a unified token sequence from all modalities
        
        Args:
            batch_results: Dict of modality tokens
            add_modality_embeddings: Add learnable modality type embeddings
            
        Returns:
            Unified token sequence (B, N_total, D)
        """
        # Collect all tokens
        all_tokens = []
        modality_indices = []
        
        for i, (modality_name, tokens) in enumerate(batch_results.items()):
            all_tokens.append(tokens)
            # Track which modality each token belongs to
            modality_indices.extend([i] * tokens.shape[1])
            
        # Concatenate
        unified_tokens = torch.cat(all_tokens, dim=1)
        
        if add_modality_embeddings and hasattr(self, 'modality_embeddings'):
            # Add modality-specific embeddings
            B, N, D = unified_tokens.shape
            modality_emb = self.modality_embeddings(torch.tensor(modality_indices).to(unified_tokens.device))
            modality_emb = modality_emb.unsqueeze(0).expand(B, -1, -1)
            unified_tokens = unified_tokens + modality_emb
            
        return unified_tokens


# Integration with DeepEarth main architecture

class DeepEarthModalityIntegration:
    """
    Integration layer between modality processors and DeepEarth model
    """
    
    def __init__(self, config: 'DeepEarthConfig'):
        self.config = config
        self.logger = logging.getLogger('DeepEarth.Integration')
        
        # Initialize processors based on config
        self.processors = self._initialize_processors()
        
        # Multi-modal orchestrator
        self.multi_modal = MultiModalProcessor()
        for name, processor in self.processors.items():
            self.multi_modal.add_processor(name, processor)
            
    def _initialize_processors(self) -> Dict[str, DeepEarthModalityProcessor]:
        """Initialize processors based on DeepEarth config"""
        processors = {}
        
        # Check for vision modalities
        if any('vision' in name or 'image' in name for name in self.config.modality_configs):
            processors['vision'] = create_vision_processor(
                universal_dim=self.config.cross_modal_fusion_config.dim
            )
            
        # Check for language modalities
        if any('text' in name or 'language' in name for name in self.config.modality_configs):
            processors['language'] = create_language_processor(
                universal_dim=self.config.cross_modal_fusion_config.dim
            )
            
        return processors
    
    def prepare_batch_for_deepearth(self, raw_batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Convert raw inputs to universal tokens for DeepEarth
        
        Args:
            raw_batch: Dict with raw inputs for each modality
            
        Returns:
            Dict compatible with DeepEarth model forward()
        """
        # Process each modality
        universal_tokens = self.multi_modal.process_batch(raw_batch)
        
        # Create unified sequence
        token_sequence = self.multi_modal.create_token_sequence(universal_tokens)
        
        # Prepare DeepEarth format
        deepearth_batch = {
            'modality_tokens': token_sequence,
            'modality_masks': self._create_modality_masks(universal_tokens),
            'token_counts': {name: tokens.shape[1] for name, tokens in universal_tokens.items()}
        }
        
        return deepearth_batch
    
    def _create_modality_masks(self, universal_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create masks for each modality"""
        masks = {}
        for name, tokens in universal_tokens.items():
            B = tokens.shape[0]
            masks[name] = torch.ones(B, dtype=torch.bool, device=tokens.device)
        return masks


# Standalone usage example
def example_deepearth_integration():
    """Complete example of modality processing for DeepEarth"""
    
    print("="*80)
    print("DeepEarth Modality Infrastructure Example")
    print("="*80)
    
    # 1. Create individual processors
    vision_proc = create_satellite_vision_processor(num_tokens=32)
    language_proc = create_agricultural_language_processor(num_tokens=8)
    
    # 2. Create multi-modal processor
    multi_modal = MultiModalProcessor()
    multi_modal.add_processor('satellite_imagery', vision_proc)
    multi_modal.add_processor('field_notes', language_proc)
    
    # 3. Prepare batch
    batch = {
        'satellite_imagery': torch.randn(4, 3, 512, 512).cuda(),  # 4 high-res images
        'field_notes': language_proc.extractor.tokenize([
            "Corn field showing signs of nitrogen deficiency in northwest corner",
            "Irrigation system functioning normally, soil moisture at 65%",
            "Pest activity detected near eastern boundary, recommend inspection",
            "Overall crop health good, estimated yield 180 bushels per acre"
        ])
    }
    
    # 4. Process
    results = multi_modal.process_batch(batch)
    
    print("\nProcessing Results:")
    for modality, tokens in results.items():
        print(f"  {modality}: {tokens.shape}")
        
    # 5. Create unified sequence
    unified = multi_modal.create_token_sequence(results)
    print(f"\nUnified token sequence: {unified.shape}")
    print(f"Total tokens per sample: {unified.shape[1]}")
    
    # 6. Token statistics
    print("\nToken allocation:")
    total = unified.shape[1]
    for modality, tokens in results.items():
        count = tokens.shape[1]
        percent = (count / total) * 100
        print(f"  {modality}: {count} tokens ({percent:.1f}%)")


if __name__ == "__main__":
    example_deepearth_integration()