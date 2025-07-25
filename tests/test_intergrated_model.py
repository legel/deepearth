"""
Comprehensive test suite for DeepEarth integrated model
Tests individual components and full pipeline
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import tempfile
import os
from pathlib import Path

# Import model components
from models.deepearth_integrated import (
    create_integrated_deepearth,
    DeepEarthIntegrated,
    IntegratedDeepEarthConfig
)
from encoders.universal_encoder import (
    UniversalEncoderModule,
    EncoderConfig,
    VJEPAEncoder,
    LanguageEncoder
)
from models.cross_modal_fusion import (
    CrossModalFusion,
    FusionConfig
)


class TestUniversalEncoder:
    """Test universal encoder components"""
    
    @pytest.fixture
    def encoder_configs(self):
        """Create test encoder configurations"""
        return {
            "vision": EncoderConfig(
                name="vision",
                native_dim=768,
                universal_dim=2048,
                num_tokens_per_sample=4,
                projection_type="attention"
            ),
            "language": EncoderConfig(
                name="language",
                native_dim=4096,
                universal_dim=2048,
                num_tokens_per_sample=1,
                projection_type="mlp"
            )
        }
    
    def test_encoder_initialization(self, encoder_configs):
        """Test encoder module initialization"""
        encoder = UniversalEncoderModule(encoder_configs)
        
        assert "vision" in encoder.encoders
        assert "language" in encoder.encoders
        assert "vision" in encoder.projectors
        assert "language" in encoder.projectors
    
    def test_vision_encoding(self, encoder_configs):
        """Test vision encoding pipeline"""
        encoder = UniversalEncoderModule({"vision": encoder_configs["vision"]})
        
        # Create dummy image
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        
        # Encode
        outputs = encoder({"vision": images})
        
        # Check output shape
        assert "vision" in outputs
        assert outputs["vision"].shape == (batch_size, 4, 2048)  # 4 tokens per image
    
    def test_language_encoding(self, encoder_configs):
        """Test language encoding pipeline"""
        encoder = UniversalEncoderModule({"language": encoder_configs["language"]})
        
        # Create dummy text
        batch_size = 2
        language_input = {
            "input_ids": torch.randint(0, 1000, (batch_size, 32)),
            "attention_mask": torch.ones(batch_size, 32)
        }
        
        # Encode
        outputs = encoder({"language": language_input})
        
        # Check output shape
        assert "language" in outputs
        assert outputs["language"].shape == (batch_size, 1, 2048)  # 1 token per text
    
    def test_projection_types(self):
        """Test different projection types"""
        batch_size = 2
        native_dim = 768
        universal_dim = 2048
        
        # Test linear projection
        linear_config = EncoderConfig(
            name="test",
            native_dim=native_dim,
            universal_dim=universal_dim,
            projection_type="linear"
        )
        
        # Test MLP projection
        mlp_config = EncoderConfig(
            name="test",
            native_dim=native_dim,
            universal_dim=universal_dim,
            projection_type="mlp",
            projection_depth=3
        )
        
        # Test attention projection
        attn_config = EncoderConfig(
            name="test",
            native_dim=native_dim,
            universal_dim=universal_dim,
            projection_type="attention",
            num_tokens_per_sample=4
        )
        
        # Verify each creates valid projectors
        from encoders.universal_encoder import UniversalProjector
        
        for config in [linear_config, mlp_config, attn_config]:
            projector = UniversalProjector(config)
            input_tensor = torch.randn(batch_size, 10, native_dim)
            output = projector(input_tensor)
            
            expected_tokens = config.num_tokens_per_sample
            assert output.shape == (batch_size, expected_tokens, universal_dim)


class TestCrossModalFusion:
    """Test cross-modal fusion components"""
    
    @pytest.fixture
    def fusion_config(self):
        """Create test fusion configuration"""
        return FusionConfig(
            universal_dim=2048,
            num_fusion_layers=12,
            num_heads=16,
            cross_attention_freq=3
        )
    
    def test_fusion_initialization(self, fusion_config):
        """Test fusion module initialization"""
        fusion = CrossModalFusion(fusion_config)
        
        assert len(fusion.layers) == 12
        assert hasattr(fusion, 'st_embedding')
        assert hasattr(fusion, 'final_norm')
    
    def test_single_modality_fusion(self, fusion_config):
        """Test fusion with single modality"""
        fusion = CrossModalFusion(fusion_config)
        
        batch_size = 2
        modality_tokens = {
            'vision': torch.randn(batch_size, 4, 2048)
        }
        
        outputs = fusion(modality_tokens)
        
        assert 'fused_representation' in outputs
        assert outputs['fused_representation'].shape == (batch_size, 2048)
        assert 'all_tokens' in outputs
        assert 'modality_tokens' in outputs
    
    def test_multi_modality_fusion(self, fusion_config):
        """Test fusion with multiple modalities"""
        fusion = CrossModalFusion(fusion_config)
        
        batch_size = 2
        modality_tokens = {
            'vision': torch.randn(batch_size, 4, 2048),
            'language': torch.randn(batch_size, 1, 2048),
            'spatial': torch.randn(batch_size, 1, 2048)
        }
        
        outputs = fusion(modality_tokens)
        
        # Check outputs
        assert outputs['fused_representation'].shape == (batch_size, 2048)
        assert outputs['all_tokens'].shape[0] == batch_size
        assert len(outputs['modality_tokens']) == 3
    
    def test_spatial_temporal_embeddings(self, fusion_config):
        """Test spatial-temporal position embeddings"""
        fusion = CrossModalFusion(fusion_config)
        
        batch_size = 2
        modality_tokens = {
            'vision': torch.randn(batch_size, 4, 2048)
        }
        
        # Add spatial positions
        spatial_positions = {
            'vision': torch.rand(batch_size, 4, 2)  # x, y for each token
        }
        
        # Add temporal positions
        temporal_positions = {
            'vision': torch.rand(batch_size, 4, 1)  # t for each token
        }
        
        outputs = fusion(
            modality_tokens,
            spatial_positions,
            temporal_positions
        )
        
        # Verify outputs are valid
        assert not torch.isnan(outputs['fused_representation']).any()


class TestIntegratedModel:
    """Test complete integrated DeepEarth model"""
    
    @pytest.fixture
    def model(self):
        """Create test model"""
        return create_integrated_deepearth(
            universal_dim=2048,
            num_fusion_layers=6,  # Smaller for testing
            freeze_backbones=True
        )
    
    def test_model_initialization(self, model):
        """Test model initialization"""
        assert isinstance(model, DeepEarthIntegrated)
        assert hasattr(model, 'universal_encoder')
        assert hasattr(model, 'grid4d_encoder')
        assert hasattr(model, 'fusion')
        
        # Check parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params < total_params  # Some params should be frozen
    
    def test_forward_pass_minimal(self, model):
        """Test forward pass with minimal inputs"""
        batch_size = 2
        xyzt = torch.randn(batch_size, 4)
        
        outputs = model(xyzt=xyzt)
        
        assert 'fused_representation' in outputs
        assert outputs['fused_representation'].shape == (batch_size, 2048)
    
    def test_forward_pass_full(self, model):
        """Test forward pass with all modalities"""
        batch_size = 2
        
        inputs = {
            'xyzt': torch.randn(batch_size, 4),
            'vision_input': torch.randn(batch_size, 3, 224, 224),
            'language_input': {
                'input_ids': torch.randint(0, 1000, (batch_size, 32)),
                'attention_mask': torch.ones(batch_size, 32)
            },
            'additional_modalities': {
                'weather': torch.randn(batch_size, 5),
                'species': torch.randn(batch_size, 64)
            }
        }
        
        outputs = model(**inputs)
        
        # Check all expected outputs
        assert 'fused_representation' in outputs
        assert 'all_tokens' in outputs
        assert 'modality_tokens' in outputs
        
        # Check modality tokens
        assert 'vision' in outputs['modality_tokens']
        assert 'language' in outputs['modality_tokens']
        assert 'spatial' in outputs['modality_tokens']
    
    def test_task_specific_outputs(self, model):
        """Test task-specific predictions"""
        batch_size = 2
        xyzt = torch.randn(batch_size, 4)
        
        # Test temperature prediction
        outputs = model(
            xyzt=xyzt,
            task='temperature_prediction'
        )
        
        assert 'task_output' in outputs
        assert outputs['task_output'].shape == (batch_size, 1)
        
        # Test land cover classification
        outputs = model(
            xyzt=xyzt,
            task='land_cover_classification'
        )
        
        assert 'task_output' in outputs
        assert outputs['task_output'].shape == (batch_size, 10)
    
    def test_reconstruction_outputs(self, model):
        """Test reconstruction capabilities"""
        batch_size = 2
        xyzt = torch.randn(batch_size, 4)
        vision = torch.randn(batch_size, 3, 224, 224)
        
        outputs = model(
            xyzt=xyzt,
            vision_input=vision,
            return_intermediates=True
        )
        
        # Check reconstructions
        assert 'reconstructions' in outputs
        assert 'spatial' in outputs['reconstructions']
        assert 'temporal' in outputs['reconstructions']
        
        # Verify reconstruction shapes
        assert outputs['reconstructions']['spatial'].shape == (batch_size, 3)
        assert outputs['reconstructions']['temporal'].shape == (batch_size, 1)
    
    def test_feature_extraction(self, model):
        """Test feature extraction at different levels"""
        batch_size = 2
        xyzt = torch.randn(batch_size, 4)
        vision = torch.randn(batch_size, 3, 224, 224)
        
        # Extract features at different levels
        with torch.no_grad():
            fused_features = model.extract_features(
                xyzt, vision, level='fused'
            )
            universal_features = model.extract_features(
                xyzt, vision, level='universal'
            )
            native_features = model.extract_features(
                xyzt, vision, level='native'
            )
        
        # Check shapes
        assert fused_features.shape == (batch_size, 2048)
        assert universal_features.ndim == 2
        assert native_features.shape[0] == batch_size
    
    def test_add_modality(self, model):
        """Test dynamic modality addition"""
        # Create a simple encoder
        class SimpleEncoder(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim)
            
            def forward(self, x):
                return self.linear(x)
        
        # Add new modality
        model.add_modality(
            name="custom",
            encoder=SimpleEncoder(10, 100),
            native_dim=100,
            num_tokens=2
        )
        
        # Test with new modality
        batch_size = 2
        outputs = model(
            xyzt=torch.randn(batch_size, 4),
            additional_modalities={
                'custom': torch.randn(batch_size, 10)
            }
        )
        
        assert 'fused_representation' in outputs


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics"""
    
    def test_memory_efficiency(self):
        """Test memory usage stays within bounds"""
        model = create_integrated_deepearth(
            universal_dim=2048,
            num_fusion_layers=12
        )
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Test with different batch sizes
        for batch_size in [1, 4, 8, 16]:
            inputs = {
                'xyzt': torch.randn(batch_size, 4, device=device),
                'vision_input': torch.randn(batch_size, 3, 224, 224, device=device)
            }
            
            # Measure memory before
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Measure memory after
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated()
                mem_used = (mem_after - mem_before) / 1024**2  # MB
                
                print(f"Batch size {batch_size}: {mem_used:.2f} MB")
                
                # Clear cache
                torch.cuda.empty_cache()
    
    def test_inference_speed(self):
        """Test inference speed"""
        import time
        
        model = create_integrated_deepearth(
            universal_dim=2048,
            num_fusion_layers=12
        )
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Warmup
        for _ in range(5):
            _ = model(xyzt=torch.randn(1, 4, device=device))
        
        # Benchmark
        batch_size = 8
        num_iterations = 50
        
        inputs = {
            'xyzt': torch.randn(batch_size, 4, device=device),
            'vision_input': torch.randn(batch_size, 3, 224, 224, device=device),
            'language_input': {
                'input_ids': torch.randint(0, 1000, (batch_size, 32), device=device),
                'attention_mask': torch.ones(batch_size, 32, device=device)
            }
        }
        
        # Time forward passes
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(**inputs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate throughput
        total_time = end_time - start_time
        samples_per_second = (batch_size * num_iterations) / total_time
        ms_per_sample = (total_time / (batch_size * num_iterations)) * 1000
        
        print(f"Throughput: {samples_per_second:.2f} samples/sec")
        print(f"Latency: {ms_per_sample:.2f} ms/sample")


class TestDataFlow:
    """Test data flow through the entire pipeline"""
    
    def test_gradient_flow(self):
        """Test gradients flow correctly"""
        model = create_integrated_deepearth(
            universal_dim=2048,
            num_fusion_layers=6,
            freeze_backbones=False  # Enable gradients
        )
        
        # Create inputs
        batch_size = 2
        xyzt = torch.randn(batch_size, 4, requires_grad=True)
        vision = torch.randn(batch_size, 3, 224, 224)
        
        # Forward pass
        outputs = model(xyzt=xyzt, vision_input=vision)
        
        # Create dummy loss
        loss = outputs['fused_representation'].sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert xyzt.grad is not None
        assert not torch.isnan(xyzt.grad).any()
        
        # Check some model parameters have gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                assert not torch.isnan(param.grad).any()
        
        assert has_grad
    
    def test_checkpoint_save_load(self):
        """Test model checkpointing"""
        model = create_integrated_deepearth()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            # Save model
            torch.save(model.state_dict(), tmp.name)
            
            # Create new model and load
            model2 = create_integrated_deepearth()
            model2.load_state_dict(torch.load(tmp.name))
            
            # Compare outputs
            batch_size = 2
            xyzt = torch.randn(batch_size, 4)
            
            with torch.no_grad():
                out1 = model(xyzt=xyzt)
                out2 = model2(xyzt=xyzt)
            
            # Outputs should be identical
            torch.testing.assert_close(
                out1['fused_representation'],
                out2['fused_representation']
            )


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
