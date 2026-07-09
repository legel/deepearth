"""
Earth4D LFMC Benchmark
======================
Species-aware Live Fuel Moisture Content prediction using Earth4D positional encoding.

Usage:
    python -m benchmarks.lfmc.train --epochs 100 --output-dir ./outputs

Quick start:
    from benchmarks.lfmc import FullyGPUDataset, SpeciesAwareLFMCModel, get_ai2_splits

    dataset = FullyGPUDataset(device='cuda')
    splits = get_ai2_splits(dataset)
    model = SpeciesAwareLFMCModel(dataset.n_species).to('cuda')
"""

from .constants import MAX_LFMC_VALUE, AI2_LFMC_CSV_URL, DEFAULT_AI2_CSV_PATH
from .data import FullyGPUDataset, get_ai2_splits, compute_lfmc_metrics
from .model import SpeciesAwareLFMCModel
from .utils import MetricsEMA, export_test_predictions_csv
from .visualization import (
    create_error_histogram,
    create_geospatial_visualization,
    create_temporal_visualization,
    create_combined_scientific_figure
)

__all__ = [
    # Constants
    'MAX_LFMC_VALUE',
    'AI2_LFMC_CSV_URL',
    'DEFAULT_AI2_CSV_PATH',
    # Data
    'FullyGPUDataset',
    'get_ai2_splits',
    'compute_lfmc_metrics',
    # Model
    'SpeciesAwareLFMCModel',
    # Utils
    'MetricsEMA',
    'export_test_predictions_csv',
    # Visualization
    'create_error_histogram',
    'create_geospatial_visualization',
    'create_temporal_visualization',
    'create_combined_scientific_figure',
]
