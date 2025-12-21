"""
LFMC Utility Functions.

EMA tracking, table printing, and CSV export utilities.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

from .constants import MAX_LFMC_VALUE


class ExponentialMovingAverage:
    """Track exponential moving average of metrics."""

    def __init__(self, alpha: float = 0.1):
        """
        Initialize EMA with smoothing factor alpha.

        Args:
            alpha: Smoothing factor (0 < alpha <= 1). Higher = more responsive.
        """
        self.alpha = alpha
        self.ema = None

    def update(self, value: float) -> float:
        """Update EMA with new value."""
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
        return self.ema

    def get(self) -> float:
        """Get current EMA value."""
        return self.ema if self.ema is not None else 0.0


class MetricsEMA:
    """Track EMAs for all metrics."""

    def __init__(self, alpha: float = 0.1):
        """
        Initialize metrics EMA tracker.

        Args:
            alpha: Smoothing factor for all EMAs
        """
        self.emas = defaultdict(lambda: ExponentialMovingAverage(alpha))
        self.sample_predictions = defaultdict(list)

    def update(self, metrics_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        Update all EMAs with new metrics.

        Args:
            metrics_dict: Dictionary of metric name -> value

        Returns:
            Dictionary of EMA values with '_ema' suffix
        """
        ema_dict = {}
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                ema_dict[f"{key}_ema"] = self.emas[key].update(value)
        return ema_dict

    def get_all(self) -> Dict[str, float]:
        """Get all current EMA values."""
        return {key: ema.get() for key, ema in self.emas.items()}


def print_predictions_table(
    test_gt: np.ndarray,
    test_pred: np.ndarray,
    test_types: List[str]
) -> None:
    """
    Print predictions in a clean table format with degeneracy indicators.

    Args:
        test_gt: Ground truth values (normalized [0, 1])
        test_pred: Predicted values (normalized [0, 1])
        test_types: List of 'U' (unique) or 'M' (multi-species)
    """
    print("  ┌───────────────────────────────────────────────────────────────┐", flush=True)
    print("  │  Sample Test Predictions (UNIQUE vs MULTI-species)          │", flush=True)
    print("  ├───────────────────────────────────────────────────────────────┤", flush=True)

    for i in range(min(5, len(test_gt))):
        # Denormalize for display
        gt_val = test_gt[i] * MAX_LFMC_VALUE
        pred_val = test_pred[i] * MAX_LFMC_VALUE
        err = abs(gt_val - pred_val)
        tag = "UNIQUE" if test_types[i] == 'U' else "MULTI"

        line = f"  │  {i+1}. {tag:6s}: {gt_val:6.1f}% → {pred_val:6.1f}% (Δ{err:5.1f}pp)  │"
        print(line, flush=True)

    print("  └───────────────────────────────────────────────────────────────┘", flush=True)
    print("    (UNIQUE=Single species, MULTI=Multiple species at same location)", flush=True)


def export_test_predictions_csv(
    dataset,
    test_predictions: np.ndarray,
    test_ground_truth: np.ndarray,
    test_indices,
    output_dir: Path
) -> Path:
    """
    Export test predictions and ground truth to CSV for analysis.

    Args:
        dataset: Dataset object with coords, species_idx, is_degenerate, idx_to_species
        test_predictions: Predictions for test samples (normalized [0, 1])
        test_ground_truth: Ground truth for test samples (normalized [0, 1])
        test_indices: Indices of test samples
        output_dir: Output directory

    Returns:
        Path to exported CSV
    """
    # Denormalize to LFMC percentage points
    preds_denorm = test_predictions * MAX_LFMC_VALUE
    gts_denorm = test_ground_truth * MAX_LFMC_VALUE

    # Extract coordinates and metadata
    coords = dataset.coords[test_indices].cpu().numpy()
    lats = coords[:, 0]
    lons = coords[:, 1]
    elevs = coords[:, 2]
    times_norm = coords[:, 3]

    # Get species information
    species_indices = dataset.species_idx[test_indices].cpu().numpy()
    species_names = [dataset.idx_to_species[idx] for idx in species_indices]

    # Get degeneracy flags
    is_degenerate = dataset.is_degenerate[test_indices].cpu().numpy()

    # Calculate errors
    errors = preds_denorm - gts_denorm
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    # Create DataFrame
    export_df = pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'elevation_m': elevs,
        'time_normalized': times_norm,
        'species': species_names,
        'is_multi_species_location': is_degenerate,
        'ground_truth_lfmc_pct': gts_denorm,
        'predicted_lfmc_pct': preds_denorm,
        'error_pp': errors,
        'absolute_error_pp': abs_errors,
        'squared_error': squared_errors
    })

    # Sort by absolute error (descending)
    export_df = export_df.sort_values('absolute_error_pp', ascending=False)

    # Save to CSV
    csv_path = output_dir / 'test_predictions.csv'
    export_df.to_csv(csv_path, index=False, float_format='%.4f')

    print(f"  Exported {len(export_df):,} test predictions to: {csv_path}", flush=True)

    # Print summary statistics
    print(f"\n  CSV Summary Statistics:", flush=True)
    print(f"    LFMC range: [{gts_denorm.min():.1f}%, {gts_denorm.max():.1f}%]", flush=True)
    print(f"    Absolute Error percentiles:", flush=True)
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(abs_errors, p)
        print(f"      {p:2d}th: {val:5.1f}pp", flush=True)

    # Show worst predictions
    worst_10 = sorted(abs_errors, reverse=True)[:10]
    print(f"    Worst 10 absolute errors (pp):", flush=True)
    print(f"      {', '.join([f'{e:.1f}' for e in worst_10])}", flush=True)

    # Compute overall metrics
    mean_squared_error = np.mean(squared_errors)
    mean_rmse = np.sqrt(mean_squared_error)
    mean_mae = np.mean(abs_errors)

    print(f"\n  Overall Metrics on Test Set:", flush=True)
    print(f"    RMSE: {mean_rmse:.1f}pp", flush=True)
    print(f"    MAE:  {mean_mae:.1f}pp", flush=True)

    return csv_path
