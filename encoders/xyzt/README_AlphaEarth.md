# Earth4D to AlphaEarth Training Pipeline

Train a DeepEarth model that uses the Earth4D spatiotemporal encoder to predict Google DeepMind's AlphaEarth 64D geospatial embeddings from (latitude, longitude, elevation, time) coordinates.

## Features

- **Automatic Data Download**: Downloads the AlphaEarth dataset (3.2M samples) from Google Cloud Storage if not present
- **GPU-Optimized Training**: Fully GPU-resident dataset for maximum training speed
- **Multi-Resolution Hash Encoding**: Uses Earth4D's planetary-scale encoder with 162D features
- **Real-time Visualization**: Generates training progress visualizations using autoencoder projections
- **Production Ready**: Achieves 3.61% MAPE on AlphaEarth embeddings

## Quick Start

### Basic Training

```bash
# Auto-downloads data and trains with default settings
python earth4d_to_alphaearth.py
```

The script will:
1. Download the AlphaEarth dataset (~800MB) to `./data/alphaearth/`
2. Train for 100 epochs with optimal hyperparameters
3. Save checkpoints and visualizations to `./earth4d_alphaearth_outputs/`

### Custom Training

```bash
# Train with custom settings
python earth4d_to_alphaearth.py \
    --epochs 200 \
    --batch-size 20000 \
    --learning-rate 1e-3 \
    --output-dir ./my_experiment
```

### Using Your Own Data

```bash
# Use custom AlphaEarth files
python earth4d_to_alphaearth.py \
    --metadata /path/to/metadata.csv \
    --embeddings /path/to/embeddings.pt
```

## Data Format

The pipeline expects:
- **Metadata CSV**: Columns must include `latitude`, `longitude`, `elevation_m`, `event_date`
- **Embeddings Tensor**: PyTorch tensor of shape `[N, 64]` with AlphaEarth embeddings

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--metadata` | Auto-download | Path to metadata CSV file |
| `--embeddings` | Auto-download | Path to embeddings .pt file |
| `--data-dir` | `./data/alphaearth` | Directory for auto-downloaded data |
| `--force-download` | False | Force re-download even if data exists |
| `--output-dir` | `./earth4d_alphaearth_outputs` | Output directory for results |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 10000 | Training batch size |
| `--learning-rate` | 1e-3 | Learning rate |
| `--train-split` | 0.95 | Fraction of data for training |
| `--max-samples` | None | Limit number of samples (for testing) |
| `--device` | cuda/cpu | Device for training |
| `--seed` | 42 | Random seed |
| `--skip-nan-test` | False | Skip NaN testing (faster but risky) |

## Model Architecture

The model consists of:
1. **Earth4D Encoder**: 162D spatiotemporal features (48D spatial + 114D temporal)
   - 24 spatial levels: ~100km to 9.5cm resolution
   - 19 temporal levels: ~73 days to 3.6 hours
2. **MLP Decoder**: 3-layer network with LayerNorm and Dropout
3. **Output**: 64D AlphaEarth embeddings with Tanh activation

## Performance

- **Training Time**: ~2 hours for 200 epochs on L4 GPU (24GB)
- **Memory Usage**: ~3.8GB GPU memory during training
- **Final MAPE**: 3.61% on test set (160K samples)
- **Model Size**: ~20MB (includes 17MB Earth4D encoder)

## Output Files

The training pipeline generates:

```
earth4d_alphaearth_outputs/
├── checkpoints/
│   └── checkpoint_epoch_*.pt     # Model checkpoints
├── predictions/
│   ├── ground_truth.csv          # Test set ground truth
│   ├── test_coordinates.csv      # Test set coordinates
│   └── epoch_*.csv               # Predictions per epoch
├── visualizations/
│   ├── bay_area_epoch_*_rgb.png  # RGB visualizations
│   └── bay_area_epoch_*_error.png # Error heatmaps
├── final_model.pt                # Final trained model
├── training_curves.png           # Loss and MAPE curves
└── final_bay_area_*.png         # Final visualizations
```

## Visualization

The pipeline generates two types of visualizations:
1. **RGB Projections**: AlphaEarth embeddings projected to RGB using autoencoder
2. **Error Heatmaps**: Spatial distribution of prediction errors

To create a training progression video:
```bash
python create_training_video.py \
    --input-dir ./earth4d_alphaearth_outputs/visualizations \
    --output training_progression.mp4
```

## Requirements

```bash
# Core dependencies
torch >= 2.0.0
pandas
numpy
matplotlib
tqdm

# Earth4D encoder (included)
# HashEncoder CUDA extension (must be compiled)
```

## Citation

If you use this code, please cite:

```bibtex
@software{earth4d2024,
  title = {Earth4D: Spatiotemporal Hash Encoding for Planetary-Scale Learning},
  author = {DeepEarth Team},
  year = {2024},
  url = {https://github.com/deepearth/earth4d}
}
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Google DeepMind for the AlphaEarth embeddings
- NVIDIA for the InstantNGP hash encoding technique
- GBIF for the biodiversity observations