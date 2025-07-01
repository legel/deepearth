"""
Flexible Language Encoder for DeepEarth
Supports multiple models (DeepSeek, Llama) with various sizes
Uses DeepSeek Transformers for decoding to universal space
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer, LlamaModel, LlamaTokenizer
import os

from deepseek.modeling import DeepSeekTransformer, DeepSeekConfig
from deepseek.modules import DeepSeekMLP, DeepSeekMoE



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
    intermediate_size: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    use_moe: bool = False  # Use Mixture of Experts
    num_experts: int = 8
    num_experts_per_tok: int = 2
    
    def to_deepseek_config(self) -> "DeepSeekConfig":
        """Convert to DeepSeek configuration"""
        return DeepSeekConfig(
            hidden_size=self.output_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_dropout_prob=self.attention_dropout_prob,
            use_moe=self.use_moe,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
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


class DeepSeekModalityDecoder(nn.Module):
    """
    Modality decoder using DeepSeek Transformer architecture
    Decodes native embeddings into universal token space with attention
    """
    
    def __init__(self, config: ModalityDecoderConfig):
        super().__init__()
        self.config = config
        
        # Input projection to match DeepSeek hidden size
        self.input_projection = nn.Linear(config.input_dim, config.output_dim)
        
        # DeepSeek Transformer for decoding
        deepseek_config = config.to_deepseek_config()
        self.transformer = DeepSeekTransformer(deepseek_config)
        
        # Learnable query tokens if generating multiple outputs
        if config.num_tokens > 1:
            self.query_tokens = nn.Parameter(
                torch.randn(1, config.num_tokens, config.output_dim) * 0.02
            )
        
        # Output layer normalization
        self.output_norm = nn.LayerNorm(config.output_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
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
        
        # Project to DeepSeek dimension
        hidden_states = self.input_projection(native_embeddings)  # (B, N, D_universal)
        
        if self.config.num_tokens > 1:
            # Prepend learnable query tokens
            queries = self.query_tokens.expand(B, -1, -1)  # (B, K, D_universal)
            hidden_states = torch.cat([queries, hidden_states], dim=1)  # (B, K+N, D_universal)
            
            # Adjust attention mask
            if attention_mask is not None:
                query_mask = torch.ones(B, self.config.num_tokens, device=attention_mask.device)
                attention_mask = torch.cat([query_mask, attention_mask], dim=1)
        
        # Apply DeepSeek Transformer
        transformer_outputs = self.transformer(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )
        
        # Extract output tokens
        if self.config.num_tokens > 1:
            # Take the query tokens as output
            output_tokens = transformer_outputs[:, :self.config.num_tokens]
        else:
            # Pool all tokens for single output
            if attention_mask is not None:
                # Masked mean pooling
                mask = attention_mask.unsqueeze(-1).float()
                output_tokens = (transformer_outputs * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
            else:
                output_tokens = transformer_outputs.mean(dim=1, keepdim=True)
        
        # Final normalization
        output_tokens = self.output_norm(output_tokens)
        
        return output_tokens


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
        self.decoder = DeepSeekModalityDecoder(decoder_config)
    
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
    device: str = "auto"
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
    
    # Decoder config
    decoder_config = ModalityDecoderConfig(
        name="language",
        input_dim=language_config.get_hidden_size(),
        output_dim=universal_dim,
        num_tokens=num_universal_tokens,
        num_layers=4,
        num_heads=16 if universal_dim >= 1024 else 8,
        intermediate_size=universal_dim * 4,
        use_moe=False  # Can enable for larger decoders
    )
    
    return LanguageModalityProcessor(language_config, decoder_config)


# Convenience functions
def create_local_language_processor(**kwargs) -> LanguageModalityProcessor:
    """Create processor optimized for local/edge deployment (7B int8)"""
    return create_language_processor(
        model_size="7b",
        precision="int8",
        **kwargs
    )


def create_cloud_language_processor(**kwargs) -> LanguageModalityProcessor:
    """Create processor for cloud deployment (70B fp16)"""
    return create_language_processor(
        model_size="70b",
        precision="fp16",
        **kwargs
    )


def create_agricultural_language_processor(**kwargs) -> LanguageModalityProcessor:
    """Create high-precision processor for agricultural applications"""
    return create_language_processor(
        model_size="33b",
        model_family="deepseek",
        precision="fp32",
        num_universal_tokens=4,  # More tokens for detailed analysis
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Development setup - small and fast
    dev_processor = create_local_language_processor()
    
    # Production setup - large and accurate
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