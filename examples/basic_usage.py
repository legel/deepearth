"""Basic usage example for DeepEarth model components."""

import torch
from deepearth.models import (
    DeepEarthConfig,
    ModalityConfig,
    Grid4DEncoder,
    ModalityEncoder,
    ModalityDecoder,
    SpatiotemporalDecoder,
    Transformer
)


def test_components():
    """Test all DeepEarth components."""
    
    # Create configuration
    config = DeepEarthConfig(
        hidden_dim=768,
        n_heads=12,
        n_layers=6,
        n_spatial_levels=8,
        n_temporal_levels=4
    )
    
    # Add modality configurations
    config.modality_configs = {
        'temperature': ModalityConfig(
            name='temperature',
            encoding_type='continuous_values',
            input_type='numerical',
            column_names=['temp_celsius', 'temp_variance']
        ),
        'species': ModalityConfig(
            name='species',
            encoding_type='learned_embedding',
            input_type='categorical',
            column_name='species_name',
            embed_dim=64
        )
    }
    
    print("Testing DeepEarth Components\n" + "="*50)
    
    # Test Grid4D Encoder
    print("\n1. Testing Grid4D Encoder...")
    grid4d = Grid4DEncoder(config)
    batch_size = 32
    xyzt = torch.rand(batch_size, 4)  # Random spatiotemporal coordinates
    coord_embeddings = grid4d(xyzt)
    print(f"   Input shape: {xyzt.shape}")
    print(f"   Output shape: {coord_embeddings.shape}")
    print(f"   ✓ Grid4D encoding successful")
    
    # Test Modality Encoder
    print("\n2. Testing Modality Encoder...")
    temp_encoder = ModalityEncoder(
        modality_name="temperature",
        input_dim=2,  # temp_celsius, temp_variance
        config=config,
        encoder_config=config.modality_encoder_config
    )
    temp_data = torch.randn(batch_size, 2)
    temp_embeddings = temp_encoder(temp_data)
    print(f"   Input shape: {temp_data.shape}")
    print(f"   Output shape: {temp_embeddings.shape}")
    print(f"   ✓ Modality encoding successful")
    
    # Test Cross-modal Transformer
    print("\n3. Testing Cross-modal Transformer...")
    transformer = Transformer(config.cross_modal_fusion_config)
    # Simulate tokens from different modalities
    n_tokens = 3  # coord, temperature, species
    tokens = torch.randn(batch_size, n_tokens, config.hidden_dim)
    fused_embeddings = transformer(tokens)
    print(f"   Input shape: {tokens.shape}")
    print(f"   Output shape: {fused_embeddings.shape}")
    print(f"   ✓ Cross-modal fusion successful")
    
    # Test Modality Decoder
    print("\n4. Testing Modality Decoder...")
    temp_decoder = ModalityDecoder(
        modality_name="temperature",
        output_dim=2,
        config=config
    )
    reconstructed_temp = temp_decoder(fused_embeddings[:, 1])  # Use temperature token
    print(f"   Input shape: {fused_embeddings[:, 1].shape}")
    print(f"   Output shape: {reconstructed_temp.shape}")
    print(f"   ✓ Modality decoding successful")
    
    # Test Spatiotemporal Decoder
    print("\n5. Testing Spatiotemporal Decoder...")
    spatial_decoder = SpatiotemporalDecoder('spatial', output_dim=3, config=config)
    temporal_decoder = SpatiotemporalDecoder('temporal', output_dim=1, config=config)
    
    reconstructed_xyz = spatial_decoder(fused_embeddings[:, 0])  # Use coord token
    reconstructed_t = temporal_decoder(fused_embeddings[:, 0])
    print(f"   Spatial output shape: {reconstructed_xyz.shape}")
    print(f"   Temporal output shape: {reconstructed_t.shape}")
    print(f"   ✓ Spatiotemporal decoding successful")
    
    # Memory usage
    print("\n6. Model Statistics:")
    total_params = sum(p.numel() for p in [
        *grid4d.parameters(),
        *temp_encoder.parameters(),
        *transformer.parameters(),
        *temp_decoder.parameters(),
        *spatial_decoder.parameters(),
        *temporal_decoder.parameters()
    ])
    print(f"   Total parameters: {total_params:,}")
    print(f"   Memory usage: ~{total_params * 4 / 1024**2:.1f} MB (float32)")
    
    print("\n✓ All components working correctly!")


def example_forward_pass():
    """Example of a complete forward pass."""
    print("\n\nExample Forward Pass\n" + "="*50)
    
    # Setup
    config = DeepEarthConfig()
    batch_size = 16
    
    # Create model components
    grid4d = Grid4DEncoder(config)
    modality_encoder = ModalityEncoder(
        "climate", input_dim=5, config=config, 
        encoder_config=config.modality_encoder_config
    )
    transformer = Transformer(config.cross_modal_fusion_config)
    modality_decoder = ModalityDecoder("climate", output_dim=5, config=config)
    
    # Create sample data
    coords = torch.rand(batch_size, 4)  # (x, y, z, t)
    climate_data = torch.randn(batch_size, 5)  # 5 climate variables
    
    print("1. Encoding coordinates...")
    coord_embeddings = grid4d(coords)
    print(f"   Coord embeddings: {coord_embeddings.shape}")
    
    print("\n2. Encoding modality data...")
    climate_embeddings = modality_encoder(climate_data)
    print(f"   Climate embeddings: {climate_embeddings.shape}")
    
    print("\n3. Cross-modal fusion...")
    # Stack embeddings as token sequence
    tokens = torch.stack([
        coord_embeddings,
        climate_embeddings
    ], dim=1)
    fused = transformer(tokens)
    print(f"   Fused embeddings: {fused.shape}")
    
    print("\n4. Decoding...")
    reconstructed_climate = modality_decoder(fused[:, 1])
    print(f"   Reconstructed climate: {reconstructed_climate.shape}")
    
    print("\n✓ Forward pass complete!")


if __name__ == "__main__":
    test_components()
    example_forward_pass()
