"""
Flexible Language Encoder for DeepEarth
Supports multiple models (DeepSeek, Llama) with various sizes
Uses DeepSeek Transformers for decoding to universal space
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer, LlamaModel
import os
from deepseek_components import (
    DeepSeekConfig, 
    DeepSeekTransformer, 
    DeepSeekMLP, 
    DeepSeekMoE,
    DeepseekV3RMSNorm
)

@dataclass
class LanguageModelConfig:
    """Configuration for language models"""
    model_family: str  # "deepseek", "llama", "mistral"
    model_size: str    # "7b", "13b", "33b", "70b"
    precision: str     # "fp32", "fp16", "int8", "int4"
    device_map: str    # "auto", "cuda:0", "cpu"
    cache_dir: Optional[str] = None
    freeze_backbone: bool = True
    
    def get_model_name(self) -> str:
        """Get the full model name for loading"""
        model_names = {
            ("deepseek", "7b"): "deepseek-ai/deepseek-llm-7b-base",
            ("deepseek", "33b"): "deepseek-ai/deepseek-llm-33b-base", 
            ("deepseek", "67b"): "deepseek-ai/deepseek-llm-67b-base",
            ("llama", "7b"): "meta-llama/Llama-2-7b-hf",
            ("llama", "13b"): "meta-llama/Llama-2-13b-hf",
            ("llama", "70b"): "meta-llama/Llama-2-70b-hf",
            ("mistral", "7b"): "mistralai/Mistral-7B-v0.1",
        }
        return model_names.get((self.model_family, self.model_size), "deepseek-ai/deepseek-llm-7b-base")
    
    def get_hidden_size(self) -> int:
        """Get the hidden size for the model"""
        hidden_sizes = {
            "7b": 4096,
            "13b": 5120,
            "33b": 7168,
            "67b": 8192,
            "70b": 8192,
        }
        return hidden_sizes.get(self.model_size, 4096)


@dataclass
class ModalityDecoderConfig:
    """Configuration for modality-specific decoders using DeepSeek Transformers"""
    name: str
    input_dim: int  # Native dimension from pretrained encoder
    output_dim: int  # Universal dimension (e.g., 2048)
    num_tokens: int = 1  # Number of output tokens
    
    # DeepSeek Transformer config
    num_layers: int = 4
    num_heads: int = 8
    num_key_value_heads: Optional[int] = None  # For GQA
    intermediate_size: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    
    # MoE configuration
    use_moe: bool = False  # Use Mixture of Experts
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: Optional[int] = None
    n_shared_experts: Optional[int] = None
    moe_layer_freq: int = 1  # MoE every N layers
    
    # Additional MoE configs
    scoring_func: str = "softmax"
    norm_topk_prob: bool = True
    input_use_moe: bool = False  # Use MoE for input projection
    output_use_moe: bool = False  # Use MoE for output generation
    
    def to_deepseek_config(self) -> "DeepSeekConfig":
        """Convert to DeepSeek configuration"""
        return DeepSeekConfig(
            hidden_size=self.output_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads or self.num_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_dropout_prob=self.attention_dropout_prob,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            # MoE configs
            n_routed_experts=self.num_experts if self.use_moe else None,
            num_experts_per_tok=self.num_experts_per_tok,
            moe_intermediate_size=self.moe_intermediate_size or (self.intermediate_size // 4),
            n_shared_experts=self.n_shared_experts,
            moe_layer_freq=self.moe_layer_freq,
            first_k_dense_replace=0,
            scoring_func=self.scoring_func,
            norm_topk_prob=self.norm_topk_prob,
        )


class FlexibleLanguageEncoder(nn.Module):
    """
    Flexible language encoder supporting multiple models and sizes
    
    Example:
        # Use small model for development
        encoder = FlexibleLanguageEncoder(
            LanguageModelConfig(model_family="deepseek", model_size="7b", precision="int8")
        )
        
        # Use large model for production
        encoder = FlexibleLanguageEncoder(
            LanguageModelConfig(model_family="llama", model_size="70b", precision="fp16")
        )
    """
    
    def __init__(self, config: LanguageModelConfig):
        super().__init__()
        self.config = config
        
        # Load model and tokenizer
        model_name = config.get_model_name()
        print(f"Loading language model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=config.cache_dir,
            trust_remote_code=True
        )
        
        # Load model with appropriate precision
        self.model = self._load_model(model_name, config)
        
        # Freeze if requested
        if config.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get native dimension
        self.native_dim = config.get_hidden_size()
    
    def _load_model(self, model_name: str, config: LanguageModelConfig) -> nn.Module:
        """Load model with specified precision and device mapping"""
        
        # Precision settings
        torch_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(config.precision, torch.float32)
        
        # Quantization settings
        load_in_8bit = config.precision == "int8"
        load_in_4bit = config.precision == "int4"
        
        # Model-specific loading
        if config.model_family == "deepseek":
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=config.cache_dir,
                torch_dtype=torch_dtype,
                device_map=config.device_map,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                trust_remote_code=True
            )
        elif config.model_family == "llama":
            model = LlamaModel.from_pretrained(
                model_name,
                cache_dir=config.cache_dir,
                torch_dtype=torch_dtype,
                device_map=config.device_map,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )
        else:
            # Generic loading
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=config.cache_dir,
                torch_dtype=torch_dtype,
                device_map=config.device_map,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )
        
        return model
    
    def extract_native_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        extract_layers: List[int] = [-1]
    ) -> Dict[str, torch.Tensor]:
        """Extract native embeddings from language model"""
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract representations
        all_hidden_states = outputs.hidden_states  # Tuple of (B, L, D)
        last_hidden_state = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else all_hidden_states[-1]
        
        embeddings = {
            'token_embeddings': last_hidden_state,
            'pooled_embedding': self._pool_tokens(last_hidden_state, attention_mask),
            'layer_embeddings': [all_hidden_states[i] for i in extract_layers if i < len(all_hidden_states)]
        }
        
        return embeddings
    
    def _pool_tokens(self, token_embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Pool token embeddings into sequence representation"""
        if attention_mask is None:
            return token_embeddings.mean(dim=1)
        
        # Masked mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def tokenize(self, texts: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """Tokenize texts for the model"""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **kwargs
        )


class DeepSeekLanguageDecoder(nn.Module):
    """
    Language modality decoder using DeepSeek Transformer architecture
    Decodes native embeddings into universal token space with attention
    """
    
    def __init__(self, config: ModalityDecoderConfig):
        super().__init__()
        self.config = config
        
        # Input projection - can use MoE for diverse language inputs
        if config.input_use_moe and config.use_moe:
            input_moe_config = DeepSeekConfig(
                hidden_size=config.input_dim,
                n_routed_experts=min(config.num_experts // 2, 4),
                num_experts_per_tok=2,
                moe_intermediate_size=config.intermediate_size // 2,
                n_shared_experts=1,
                scoring_func=config.scoring_func,
                norm_topk_prob=config.norm_topk_prob,
            )
            self.input_projection = DeepSeekMoE(input_moe_config)
            # Additional linear to match dimensions
            self.dim_match = nn.Linear(config.input_dim, config.output_dim)
        else:
            # Standard linear projection
            self.input_projection = nn.Linear(config.input_dim, config.output_dim)
            self.dim_match = None
        
        # Position embeddings for sequence modeling
        self.position_embeddings = nn.Parameter(
            torch.randn(1, 512, config.output_dim) * 0.02  # Max 512 tokens
        )
        
        # DeepSeek Transformer for decoding
        deepseek_config = config.to_deepseek_config()
        self.transformer = DeepSeekTransformer(deepseek_config)
        
        # Token generation mechanism
        if config.num_tokens > 1:
            # Learnable query tokens
            self.query_tokens = nn.Parameter(
                torch.randn(1, config.num_tokens, config.output_dim) * 0.02
            )
            
            # MoE for token specialization if configured
            if config.output_use_moe and config.use_moe:
                token_moe_config = DeepSeekConfig(
                    hidden_size=config.output_dim,
                    n_routed_experts=min(config.num_experts, config.num_tokens * 2),
                    num_experts_per_tok=2,
                    moe_intermediate_size=config.output_dim,
                    scoring_func=config.scoring_func,
                )
                self.token_specializer = DeepSeekMoE(token_moe_config)
            else:
                # MLP for token specialization
                token_mlp_config = DeepSeekConfig(
                    hidden_size=config.output_dim,
                    intermediate_size=config.output_dim * 2,
                )
                self.token_specializer = DeepSeekMLP(token_mlp_config)
            
            # Cross attention for query-to-sequence
            self.cross_attention = nn.MultiheadAttention(
                config.output_dim,
                config.num_heads,
                dropout=config.attention_dropout_prob,
                batch_first=True
            )
            self.cross_norm = DeepseekV3RMSNorm(config.output_dim)
        
        # Output normalization
        self.output_norm = DeepseekV3RMSNorm(config.output_dim)
        
        # Pooling strategy for single token
        if config.num_tokens == 1:
            if config.use_moe:
                # MoE for different pooling strategies
                pool_moe_config = DeepSeekConfig(
                    hidden_size=config.output_dim,
                    n_routed_experts=4,  # Different aggregation strategies
                    num_experts_per_tok=1,
                    moe_intermediate_size=config.output_dim // 2,
                )
                self.pool_module = DeepSeekMoE(pool_moe_config)
            else:
                pool_mlp_config = DeepSeekConfig(
                    hidden_size=config.output_dim,
                    intermediate_size=config.output_dim,
                )
                self.pool_module = DeepSeekMLP(pool_mlp_config)
    
    def forward(
        self,
        native_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode native embeddings to universal tokens
        
        Args:
            native_embeddings: (B, N, D_native) native encoder outputs
            attention_mask: Optional attention mask
            
        Returns:
            universal_tokens: (B, K, D_universal) where K = num_tokens
        """
        B, N, _ = native_embeddings.shape
        
        # Project to universal dimension
        if self.config.input_use_moe and self.config.use_moe:
            hidden_states = self.input_projection(native_embeddings)
            hidden_states = self.dim_match(hidden_states)
        else:
            hidden_states = self.input_projection(native_embeddings)
        
        # Add position embeddings
        if N <= self.position_embeddings.shape[1]:
            hidden_states = hidden_states + self.position_embeddings[:, :N]
        else:
            # Interpolate for longer sequences
            pos_embed = nn.functional.interpolate(
                self.position_embeddings.transpose(1, 2),
                size=N,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            hidden_states = hidden_states + pos_embed
        
        # Apply transformer with potential MoE layers
        if attention_mask is not None:
            # Convert to transformer format
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        transformed = self.transformer(hidden_states, attention_mask=attention_mask)
        
        # Generate output tokens
        if self.config.num_tokens > 1:
            output_tokens = self._generate_multi_tokens(transformed, attention_mask)
        else:
            output_tokens = self._generate_single_token(transformed, attention_mask)
        
        # Final normalization
        output_tokens = self.output_norm(output_tokens)
        
        return output_tokens
    
    def _generate_multi_tokens(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate multiple specialized tokens"""
        B = hidden_states.shape[0]
        
        # Expand query tokens
        queries = self.query_tokens.expand(B, -1, -1)
        
        # Cross-attention to sequence
        if attention_mask is not None:
            key_padding_mask = attention_mask.squeeze(1).squeeze(1) == float('-inf')
        else:
            key_padding_mask = None
        
        attended, _ = self.cross_attention(
            query=queries,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask
        )
        
        # Residual and normalize
        attended = self.cross_norm(queries + attended)
        
        # Specialize tokens through MoE/MLP
        specialized = self.token_specializer(attended)
        output_tokens = attended + specialized  # Residual
        
        return output_tokens
    
    def _generate_single_token(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate single aggregated token"""
        B, N, D = hidden_states.shape
        
        # Intelligent pooling
        if attention_mask is not None and attention_mask.numel() > 0:
            # Masked pooling
            mask = (attention_mask.squeeze(1).squeeze(1) != float('-inf')).float()
            mask = mask.unsqueeze(-1)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            # Attention-weighted pooling based on norm
            weights = torch.softmax(hidden_states.norm(dim=-1), dim=-1)
            pooled = torch.einsum('bn,bnd->bd', weights, hidden_states)
        
        # Process through MoE/MLP
        refined = self.pool_module(pooled)
        output = pooled + refined  # Residual
        
        return output.unsqueeze(1)  # (B, 1, D)


class LanguageModalityProcessor(nn.Module):
    """
    Complete language modality processor
    Combines flexible language encoder with DeepSeek decoder
    """
    
    def __init__(
        self,
        language_config: LanguageModelConfig,
        decoder_config: ModalityDecoderConfig
    ):
        super().__init__()
        
        # Language encoder (frozen pretrained model)
        self.encoder = FlexibleLanguageEncoder(language_config)
        
        # DeepSeek decoder to universal space
        decoder_config.input_dim = self.encoder.native_dim
        self.decoder = DeepSeekLanguageDecoder(decoder_config)
        
        # Store configs
        self.language_config = language_config
        self.decoder_config = decoder_config
    
    def forward(
        self,
        texts: Union[str, List[str], Dict[str, torch.Tensor]],
        return_native: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Process language input to universal tokens
        
        Args:
            texts: Raw text, list of texts, or pre-tokenized dict
            return_native: Whether to also return native embeddings
            
        Returns:
            universal_tokens: (B, K, D_universal) universal tokens
            native_embeddings: Optional dict of native embeddings
        """
        # Handle different input types
        if isinstance(texts, (str, list)):
            # Tokenize
            encoded = self.encoder.tokenize(texts)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
        else:
            # Assume pre-tokenized
            input_ids = texts['input_ids']
            attention_mask = texts.get('attention_mask')
        
        # Extract native embeddings
        with torch.no_grad() if self.encoder.config.freeze_backbone else torch.enable_grad():
            native_embeddings = self.encoder.extract_native_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Decode to universal space
        universal_tokens = self.decoder(
            native_embeddings['token_embeddings'],
            attention_mask=attention_mask
        )
        
        if return_native:
            return universal_tokens, native_embeddings
        else:
            return universal_tokens


# Factory functions for common configurations
def create_language_processor(
    model_size: str = "7b",
    model_family: str = "deepseek",
    precision: str = "int8",
    num_universal_tokens: int = 1,
    universal_dim: int = 2048,
    device: str = "auto",
    use_moe: bool = False,
    num_experts: int = 8
) -> LanguageModalityProcessor:
    """
    Create a language processor with sensible defaults
    
    Args:
        model_size: Size of language model ("7b", "13b", "70b", etc.)
        model_family: Model family ("deepseek", "llama", "mistral")
        precision: Model precision ("fp32", "fp16", "int8", "int4")
        num_universal_tokens: Number of universal tokens to generate
        universal_dim: Dimension of universal token space
        device: Device placement strategy
        use_moe: Whether to use MoE in the decoder
        num_experts: Number of experts if using MoE
        
    Returns:
        LanguageModalityProcessor ready for use
    """
    # Language model config
    language_config = LanguageModelConfig(
        model_family=model_family,
        model_size=model_size,
        precision=precision,
        device_map=device if device != "auto" else "auto",
        freeze_backbone=True
    )
    
    # Decoder config with MoE options
    decoder_config = ModalityDecoderConfig(
        name="language",
        input_dim=language_config.get_hidden_size(),
        output_dim=universal_dim,
        num_tokens=num_universal_tokens,
        num_layers=4,
        num_heads=16 if universal_dim >= 1024 else 8,
        intermediate_size=universal_dim * 4,
        use_moe=use_moe,
        num_experts=num_experts,
        num_experts_per_tok=2,
        moe_intermediate_size=universal_dim // 2,
        input_use_moe=use_moe and num_universal_tokens > 1,
        output_use_moe=use_moe and num_universal_tokens > 1,
        moe_layer_freq=2 if use_moe else 1,  # MoE every other layer
    )
    
    return LanguageModalityProcessor(language_config, decoder_config)


# Convenience functions
def create_local_language_processor(**kwargs) -> LanguageModalityProcessor:
    """Create processor optimized for local/edge deployment (7B int8)"""
    return create_language_processor(
        model_size="7b",
        precision="int8",
        use_moe=False,  # Keep simple for edge
        **kwargs
    )


def create_cloud_language_processor(**kwargs) -> LanguageModalityProcessor:
    """Create processor for cloud deployment (70B fp16)"""
    return create_language_processor(
        model_size="70b",
        precision="fp16",
        use_moe=True,  # Can afford MoE in cloud
        num_experts=8,
        **kwargs
    )


def create_agricultural_language_processor(**kwargs) -> LanguageModalityProcessor:
    """Create high-precision processor for agricultural applications"""
    return create_language_processor(
        model_size="33b",
        model_family="deepseek",
        precision="fp32",
        num_universal_tokens=4,  # More tokens for detailed analysis
        use_moe=True,  # MoE for diverse agricultural contexts
        num_experts=8,  # Different crop types, conditions, etc.
        **kwargs
    )


def create_multilingual_processor(**kwargs) -> LanguageModalityProcessor:
    """Create processor optimized for multilingual content"""
    return create_language_processor(
        model_size="13b",
        precision="fp16",
        num_universal_tokens=2,
        use_moe=True,  # MoE helps with language diversity
        num_experts=16,  # More experts for language variety
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Development setup - small and fast
    dev_processor = create_local_language_processor()
    
    # Production setup - large and accurate with MoE
    prod_processor = create_cloud_language_processor()
    
    # Test processing
    texts = [
        "The soil moisture levels indicate optimal conditions for planting.",
        "Satellite imagery shows early signs of crop stress in the northern fields."
    ]
    
    # Process to universal tokens
    universal_tokens = dev_processor(texts)
    print(f"Universal tokens shape: {universal_tokens.shape}")
    
    # Get both universal and native embeddings
    universal, native = dev_processor(texts, return_native=True)
    print(f"Native token embeddings: {native['token_embeddings'].shape}")
    print(f"Native pooled embedding: {native['pooled_embedding'].shape}")