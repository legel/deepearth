"""
Universal Encoder V3 for DeepEarth
Uses modality infrastructure with DeepSeek Transformers
"""

import torch
import torch.nn as nn
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass
import logging

# Import the modality infrastructure
from encoders.modality_infrastructure import (
    create_vision_processor,
    create_language_processor,
    create_satellite_vision_processor,
    create_agricultural_language_processor,
    DeepEarthModalityProcessor,
    UniversalModalityDecoder,
    MultiModalProcessor,
    UniversalTokenConfig,
    BaseModalityExtractor,
    VJEPA2Extractor as InfraVJEPA2Extractor,
    LanguageModelExtractor
)

# Import modality config
from encoders.modality_config import ModalityDecoderConfig, get_preset_config

# Import DeepSeek components
from encoders.deepseek_components import (
    DeepSeekConfig,
    DeepSeekTransformer,
    DeepSeekMLP,
    DeepSeekMoE,
    DeepseekV3RMSNorm
)

# Use the existing VJEPA2 extractor
from encoders.vision.vjepa2_extractor import VJEPA2Extractor


@dataclass
class UniversalEncoderConfig:
    """Configuration for universal encoder"""
    universal_dim: int = 2048
    
    # Vision settings
    vision_backbone: str = "vjepa2"
    vision_model_name: str = "facebook/vjepa2-vitg-fpc64-384"
    vision_size: str = "base"
    vision_tokens: int = 16  # 4x4 grid
    
    # Language settings
    language_model: str = "deepseek-ai/deepseek-llm-7b-base"
    language_size: str = "7b"
    language_precision: str = "int8"
    language_tokens: int = 4
    
    # Additional modality defaults
    default_decoder_layers: int = 4
    default_decoder_heads: int = 8
    use_moe_for_complex_modalities: bool = True
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True


class VJEPAExtractorAdapter(BaseModalityExtractor):
    """Adapter to use the existing VJEPA2Extractor with our infrastructure"""
    
    def __init__(self, model_name: str = "facebook/vjepa2-vitg-fpc64-384", device: str = "cuda"):
        self.extractor = VJEPA2Extractor(
            model_name=model_name,
            device=device,
            use_fp16=True
        )
        self.native_dim = self.extractor.patch_dim  # 1408
        
    def extract_native_embeddings(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features compatible with infrastructure format"""
        features_list = []
        
        # Process each image in batch
        for i in range(images.shape[0]):
            # Convert tensor to PIL if needed
            if isinstance(images, torch.Tensor):
                from PIL import Image
                import numpy as np
                img_array = images[i].cpu().numpy()
                if img_array.shape[0] == 3:  # CHW -> HWC
                    img_array = img_array.transpose(1, 2, 0)
                img_array = (img_array * 255).astype(np.uint8)
                pil_image = Image.fromarray(img_array)
            else:
                pil_image = images[i]
                
            # Extract features
            features = self.extractor.extract_features(pil_image)
            if features is not None:
                features_list.append(features)
                
        # Stack features
        if features_list:
            # features shape: [batch, 4608, 1408]
            batch_features = torch.stack(features_list)
            
            # Reshape to separate spatial and temporal
            B = batch_features.shape[0]
            # 4608 = 576 spatial × 8 temporal
            spatial_patches = 576
            temporal_frames = 8
            
            # Get first temporal frame as spatial features
            features_reshaped = batch_features.view(B, temporal_frames, spatial_patches, self.native_dim)
            spatial_features = features_reshaped[:, 0, :, :]  # [B, 576, 1408]
            
            # Reshape spatial to grid
            grid_size = int(spatial_patches ** 0.5)  # 24
            patch_embeddings = spatial_features.view(B, grid_size, grid_size, self.native_dim)
            patch_embeddings = patch_embeddings.view(B, -1, self.native_dim)  # [B, 576, 1408]
            
            # Global features
            global_embedding = spatial_features.mean(dim=1)  # [B, 1408]
            cls_embedding = spatial_features[:, 0, :]  # [B, 1408]
            
            return {
                'patch_embeddings': patch_embeddings,
                'global_embedding': global_embedding,
                'cls_embedding': cls_embedding
            }
        else:
            raise ValueError("No features extracted")
            
    def get_native_dim(self) -> int:
        return self.native_dim


class AdditionalModalityProcessor(nn.Module):
    """
    Generic processor for additional modalities using DeepSeek decoders
    Used for weather, soil, species, etc.
    """
    
    def __init__(self, config: ModalityDecoderConfig):
        super().__init__()
        self.config = config
        self.decoder = UniversalModalityDecoder(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process modality data to universal tokens"""
        return self.decoder(x)


class UniversalEncoderV3(nn.Module):
    """
    Universal Encoder using the modular infrastructure
    
    Key improvements:
    - Uses modality_infrastructure for all processing
    - Integrates with existing VJEPA2 extractor
    - All decoders use DeepSeek Transformers
    - Easy registration of new modalities
    """
    
    def __init__(self, config: UniversalEncoderConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger('DeepEarth.UniversalEncoder')
        
        # Initialize multi-modal processor
        self.multi_modal = MultiModalProcessor()
        
        # Create vision processor with VJEPA2
        self._init_vision_processor()
        
        # Create language processor
        self._init_language_processor()
        
        # Registry for additional modalities
        self.additional_processors = nn.ModuleDict()
        
        # Initialize common Earth observation modalities
        self._init_earth_modalities()
        
    def _init_vision_processor(self):
        """Initialize vision processor with VJEPA2"""
        # Create adapter for existing VJEPA2 extractor
        extractor = VJEPAExtractorAdapter(
            model_name=self.config.vision_model_name,
            device=self.config.device
        )
        
        # Get config for vision
        decoder_config = get_preset_config("vision_satellite") if self.config.vision_tokens > 16 else get_preset_config("vision_standard")
        decoder_config.input_dim = extractor.get_native_dim()
        decoder_config.output_dim = self.config.universal_dim
        decoder_config.num_tokens = self.config.vision_tokens
        
        # Create processor
        vision_processor = DeepEarthModalityProcessor(
            modality_name="vision",
            extractor=extractor,
            decoder_config=decoder_config
        )
        
        self.multi_modal.add_processor("vision", vision_processor)
        self.logger.info(f"Initialized vision processor: {extractor.get_native_dim()}D -> {self.config.vision_tokens}x{self.config.universal_dim}D")
        
    def _init_language_processor(self):
        """Initialize language processor"""
        # Create language extractor
        extractor = LanguageModelExtractor(
            model_name=self.config.language_model,
            device=self.config.device,
            precision=self.config.language_precision
        )
        
        # Get config for language
        decoder_config = get_preset_config("language_standard")
        decoder_config.input_dim = extractor.get_native_dim()
        decoder_config.output_dim = self.config.universal_dim
        decoder_config.num_tokens = self.config.language_tokens
        
        # Create processor
        language_processor = DeepEarthModalityProcessor(
            modality_name="language",
            extractor=extractor,
            decoder_config=decoder_config
        )
        
        self.multi_modal.add_processor("language", language_processor)
        self.logger.info(f"Initialized language processor: {extractor.get_native_dim()}D -> {self.config.language_tokens}x{self.config.universal_dim}D")
        
    def _init_earth_modalities(self):
        """Initialize common Earth observation modalities"""
        earth_modalities = {
            'weather': get_preset_config('weather'),
            'weather_forecast': get_preset_config('weather_forecast'),
            'soil': get_preset_config('soil'),
            'species': get_preset_config('species'),
            'ndvi_timeseries': get_preset_config('ndvi_timeseries'),
        }
        
        for name, config in earth_modalities.items():
            config.output_dim = self.config.universal_dim
            processor = AdditionalModalityProcessor(config)
            self.additional_processors[name] = processor
            self.logger.info(f"Registered modality '{name}': {config.input_dim}D -> {config.num_tokens}x{config.output_dim}D")
            
    def register_modality(
        self,
        name: str,
        input_dim: int,
        num_tokens: int = 1,
        use_moe: bool = None,
        preset: Optional[str] = None,
        **kwargs
    ):
        """
        Register a new modality with automatic DeepSeek decoder creation
        
        Args:
            name: Modality name
            input_dim: Input dimension
            num_tokens: Number of universal tokens to generate
            use_moe: Whether to use MoE (auto-determined if None)
            preset: Use a preset configuration
            **kwargs: Additional config options
        """
        if preset:
            config = get_preset_config(preset)
            config.name = name
            config.input_dim = input_dim
            config.output_dim = self.config.universal_dim
            if num_tokens is not None:
                config.num_tokens = num_tokens
        else:
            # Create custom config
            config = ModalityDecoderConfig(
                name=name,
                input_dim=input_dim,
                output_dim=self.config.universal_dim,
                num_tokens=num_tokens,
                num_layers=kwargs.get('num_layers', self.config.default_decoder_layers),
                num_heads=kwargs.get('num_heads', self.config.default_decoder_heads),
                intermediate_size=kwargs.get('intermediate_size', self.config.universal_dim * 4),
                use_moe=use_moe if use_moe is not None else (
                    self.config.use_moe_for_complex_modalities and 
                    (input_dim > 100 or num_tokens > 4)
                ),
                **kwargs
            )
        
        processor = AdditionalModalityProcessor(config)
        self.additional_processors[name] = processor
        
        self.logger.info(f"Registered modality '{name}': {config.input_dim}D -> {config.num_tokens}x{config.output_dim}D (MoE={config.use_moe})")
        
    def forward(
        self,
        inputs: Dict[str, Any],
        return_native: bool = False,
        return_details: bool = False
    ) -> Union[Dict[str, torch.Tensor], tuple]:
        """
        Process multiple modalities to universal token space
        
        Args:
            inputs: Dict mapping modality names to inputs
            return_native: Whether to return native embeddings
            return_details: Whether to return detailed information
            
        Returns:
            Dict mapping modality names to universal tokens
        """
        outputs = {}
        native_embeddings = {} if return_native else None
        details = {} if return_details else None
        
        # Process core modalities through multi_modal processor
        core_inputs = {}
        for name in ['vision', 'language']:
            if name in inputs:
                core_inputs[name] = inputs[name]
                
        if core_inputs:
            core_results = self.multi_modal.process_batch(core_inputs)
            outputs.update(core_results)
            
            # Get native embeddings if requested
            if return_native:
                for name, processor in self.multi_modal.processors.items():
                    if name in core_inputs:
                        _, native = processor.process(core_inputs[name], return_native=True)
                        native_embeddings[name] = native
                        
        # Process additional modalities
        for name, data in inputs.items():
            if name not in ['vision', 'language'] and name in self.additional_processors:
                outputs[name] = self.additional_processors[name](data)
                
        if return_details:
            # Add token counts and dimensions
            for name, tokens in outputs.items():
                details[name] = {
                    'shape': tokens.shape,
                    'num_tokens': tokens.shape[1],
                    'token_dim': tokens.shape[2]
                }
                
        if return_native and return_details:
            return outputs, native_embeddings, details
        elif return_native:
            return outputs, native_embeddings
        elif return_details:
            return outputs, details
        else:
            return outputs
            
    def list_modalities(self) -> List[str]:
        """List all registered modalities"""
        core = list(self.multi_modal.processors.keys())
        additional = list(self.additional_processors.keys())
        return core + additional
        
    def get_modality_info(self) -> Dict[str, Dict]:
        """Get information about all modalities"""
        info = {}
        
        # Core modalities
        for name, processor in self.multi_modal.processors.items():
            info[name] = {
                'type': 'core',
                'input_dim': processor.decoder_config.input_dim,
                'output_dim': processor.decoder_config.output_dim,
                'num_tokens': processor.decoder_config.num_tokens,
                'use_moe': processor.decoder_config.use_moe
            }
            
        # Additional modalities
        for name, processor in self.additional_processors.items():
            info[name] = {
                'type': 'additional',
                'input_dim': processor.config.input_dim,
                'output_dim': processor.config.output_dim,
                'num_tokens': processor.config.num_tokens,
                'use_moe': processor.config.use_moe
            }
            
        return info


def create_universal_encoder(
    vision_model: str = "facebook/vjepa2-vitg-fpc64-384",
    vision_tokens: int = 16,
    language_model: str = "deepseek-ai/deepseek-llm-7b-base",
    language_tokens: int = 4,
    language_precision: str = "int8",
    universal_dim: int = 2048,
    device: str = "cuda",
    **kwargs
) -> UniversalEncoderV3:
    """
    Create a universal encoder with specified settings
    
    Args:
        vision_model: V-JEPA2 model to use
        vision_tokens: Number of vision tokens (e.g., 16 for 4x4 grid)
        language_model: Language model to use
        language_tokens: Number of language tokens
        language_precision: Precision for language model
        universal_dim: Universal token dimension
        device: Device to run on
        **kwargs: Additional config options
        
    Returns:
        Configured UniversalEncoderV3
    """
    config = UniversalEncoderConfig(
        universal_dim=universal_dim,
        vision_model_name=vision_model,
        vision_tokens=vision_tokens,
        language_model=language_model,
        language_tokens=language_tokens,
        language_precision=language_precision,
        device=device,
        **kwargs
    )
    
    return UniversalEncoderV3(config)


# Convenience functions for specific use cases
def create_agricultural_encoder(**kwargs) -> UniversalEncoderV3:
    """Create encoder optimized for agricultural monitoring"""
    encoder = create_universal_encoder(
        vision_tokens=32,  # Higher resolution for field analysis
        language_tokens=8,  # More tokens for technical descriptions
        language_precision="fp32",  # Higher precision for agricultural terms
        **kwargs
    )
    
    # Register agricultural-specific modalities
    encoder.register_modality("crop_health", input_dim=10, num_tokens=2)
    encoder.register_modality("irrigation", input_dim=5, num_tokens=1)
    encoder.register_modality("pest_detection", input_dim=20, num_tokens=4, use_moe=True)
    
    return encoder


def create_satellite_encoder(**kwargs) -> UniversalEncoderV3:
    """Create encoder optimized for satellite imagery"""
    encoder = create_universal_encoder(
        vision_tokens=64,  # 8x8 grid for high-res satellite
        language_tokens=2,  # Fewer tokens for coordinates/metadata
        **kwargs
    )
    
    # Register satellite-specific modalities
    encoder.register_modality("hyperspectral", input_dim=224, preset="hyperspectral")
    encoder.register_modality("radar", input_dim=4, num_tokens=4)  # SAR bands
    encoder.register_modality("elevation", input_dim=1, num_tokens=1)
    
    return encoder


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create standard encoder
    encoder = create_universal_encoder(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Register custom modalities
    encoder.register_modality("temperature", input_dim=1, num_tokens=1)
    encoder.register_modality("precipitation", input_dim=1, num_tokens=1)
    encoder.register_modality("ndvi", input_dim=1, num_tokens=1)
    
    # Process multimodal inputs
    batch_size = 2
    inputs = {
        'vision': torch.randn(batch_size, 3, 224, 224),
        'language': ["Drought conditions observed in northern fields", 
                    "Healthy crop growth with optimal moisture"],
        'weather': torch.randn(batch_size, 5),
        'soil': torch.randn(batch_size, 10),
        'temperature': torch.randn(batch_size, 1),
        'ndvi': torch.randn(batch_size, 1)
    }
    
    # Get universal tokens
    print("Processing inputs...")
    universal_tokens, details = encoder(inputs, return_details=True)
    
    print("\nUniversal token shapes:")
    for name, info in details.items():
        print(f"  {name}: {info['shape']} ({info['num_tokens']} tokens)")
    
    # Get modality information
    print(f"\nRegistered modalities ({len(encoder.list_modalities())} total):")
    mod_info = encoder.get_modality_info()
    for name, info in mod_info.items():
        print(f"  {name}: {info['input_dim']}D -> {info['num_tokens']}x{info['output_dim']}D (MoE={info['use_moe']})")
    
    # Test agricultural encoder
    print("\n" + "="*80)
    print("Agricultural Encoder Example")
    print("="*80)
    
    ag_encoder = create_agricultural_encoder()
    
    ag_inputs = {
        'vision': torch.randn(1, 3, 512, 512),  # High-res field image
        'language': ["Corn field showing nitrogen deficiency, recommend fertilizer application"],
        'crop_health': torch.randn(1, 10),
        'irrigation': torch.randn(1, 5),
        'pest_detection': torch.randn(1, 20)
    }
    
    ag_tokens = ag_encoder(ag_inputs)
    
    print("\nAgricultural tokens:")
    for name, tokens in ag_tokens.items():
        print(f"  {name}: {tokens.shape}")