"""
LFMC Visualization Suite.

Provides geospatial, temporal, and error distribution visualizations
for Live Fuel Moisture Content prediction analysis.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.dates as mdates

from .constants import MAX_LFMC_VALUE

# Package directory for relative paths
PACKAGE_DIR = Path(__file__).parent.parent.parent


def create_error_histogram(csv_path: Path, output_dir: Path) -> None:
    """
    Create histogram showing distribution of absolute errors across test dataset.

    Args:
        csv_path: Path to test_predictions.csv
        output_dir: Output directory for saving histogram
    """
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Get absolute errors and predictions/ground truth
    abs_errors_original = df['absolute_error_pp'].values
    predictions = df['predicted_lfmc_pct'].values
    ground_truth = df['ground_truth_lfmc_pct'].values

    # Compute R²
    ss_res = np.sum((ground_truth - predictions) ** 2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Cap errors at 100pp for histogram display
    abs_errors = np.clip(abs_errors_original, 0, 100)
    n_over_100 = np.sum(abs_errors_original > 100)

    # Compute statistics (on original uncapped data)
    mean_error = np.mean(abs_errors_original)
    median_error = np.median(abs_errors_original)
    rmse = np.sqrt(np.mean(df['squared_error'].values))
    mae = np.mean(abs_errors_original)
    max_error = np.max(abs_errors_original)
    min_error = np.min(abs_errors_original)

    # Create histogram with 100 bins from 0-100pp
    n_bins = 100
    bins = np.linspace(0, 100, n_bins + 1)

    # Create figure (height = 1/5 of width for compressed vertical display)
    fig, ax = plt.subplots(1, 1, figsize=(16, 3.2))

    # Plot histogram with Turbo colormap (matching geospatial plot)
    counts, bins_out, patches = ax.hist(abs_errors, bins=bins, edgecolor='black', linewidth=0.5)

    # Color bars using Turbo colormap (same as geospatial plot: 0-50pp scale)
    norm = plt.Normalize(vmin=0, vmax=50)
    cmap = cm.get_cmap('turbo')

    for i, patch in enumerate(patches):
        # Get the center value of this bin
        bin_center = (bins[i] + bins[i+1]) / 2
        # Clip to 50pp max (same as geospatial)
        bin_value = min(bin_center, 50.0)
        # Set color based on normalized value
        patch.set_facecolor(cmap(norm(bin_value)))

    # Calculate MAE percentiles for subtitle
    mae_25th = np.percentile(abs_errors_original, 25)
    mae_median = np.median(abs_errors_original)
    mae_75th = np.percentile(abs_errors_original, 75)

    # Labels and title (matching font sizes from other plots)
    ax.set_xlabel('Absolute Error (percentage points)', fontsize=12)
    ax.set_ylabel('Number of Test Samples', fontsize=12)

    # Title with LFMC definition for readers unfamiliar with LFMC
    title_text = 'Distribution of Live Fuel Moisture Content (LFMC) Prediction Errors on Test Set\n'
    ax.text(0.5, 1.08, title_text.strip(), transform=ax.transAxes,
            fontsize=13, ha='center', va='bottom')

    # Second line with metrics (smaller font, italicized)
    subtitle_text = f'LFMC% = (wet mass - dry mass) / dry mass × 100  |  '
    subtitle_text += f'Total Samples: {len(abs_errors_original):,} | RMSE: {rmse:.1f}pp | '
    subtitle_text += f'MAE: {mae:.1f}pp (25th: {mae_25th:.1f}pp, Median: {mae_median:.1f}pp, 75th: {mae_75th:.1f}pp) | '
    subtitle_text += f'R²: {r2:.3f}'
    if n_over_100 > 0:
        subtitle_text += f' | {n_over_100} samples >100pp'
    ax.text(0.5, 1.02, subtitle_text, transform=ax.transAxes,
            fontsize=9, ha='center', va='bottom', style='italic')

    # Set x-axis limits and tick marks every 5 percentage points
    ax.set_xlim(0, 100)
    ax.set_xticks([i for i in range(0, 101, 5)])

    # No gridlines - clean white background
    ax.grid(False)

    # Format y-axis with comma separators
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Save metrics to JSON file
    metrics_dict = {
        'summary_metrics': {
            'rmse_pp': float(rmse),
            'mae_pp': float(mae),
            'r2': float(r2),
            'total_samples': int(len(abs_errors_original)),
            'samples_over_100pp': int(n_over_100)
        },
        'error_distribution': {
            'min_error_pp': float(min_error),
            'percentile_25_pp': float(np.percentile(abs_errors_original, 25)),
            'median_pp': float(median_error),
            'mean_pp': float(mean_error),
            'percentile_75_pp': float(np.percentile(abs_errors_original, 75)),
            'percentile_95_pp': float(np.percentile(abs_errors_original, 95)),
            'max_error_pp': float(max_error)
        },
        'mae_distribution': {
            'mae_25th_percentile_pp': float(mae_25th),
            'mae_median_pp': float(mae_median),
            'mae_75th_percentile_pp': float(mae_75th)
        },
        'lfmc_definition': {
            'formula': 'LFMC% = (wet mass - dry mass) / dry mass × 100',
            'error_calculation': 'Percentage Points (pp) = |Predicted LFMC% - Ground Truth LFMC%|',
            'histogram_note': 'Histogram capped at 100pp for display'
        }
    }

    metrics_json_path = output_dir / 'error_histogram_metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"  Metrics saved to: {metrics_json_path}", flush=True)

    # Tight layout
    plt.tight_layout()

    # Save figure
    histogram_path = output_dir / 'error_distribution_histogram.png'
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Error histogram saved to: {histogram_path}", flush=True)
    print(f"  Error range: [{min_error:.1f}pp, {max_error:.1f}pp]", flush=True)
    print(f"  Errors >100pp: {n_over_100} samples (binned at 100+)", flush=True)
    print(f"  Number of bins: {n_bins} (0-100pp)", flush=True)


def create_geospatial_visualization(
    dataset,
    test_predictions: np.ndarray,
    test_ground_truth: np.ndarray,
    test_indices,
    output_dir: Path,
    epoch: str = "final",
    train_samples: Optional[int] = None,
    shapefile_dir: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create geospatial visualization of LFMC prediction errors across CONUS.

    Args:
        dataset: Dataset object with coords attribute
        test_predictions: Predictions for test samples (normalized [0,1])
        test_ground_truth: Ground truth for test samples (normalized [0,1])
        test_indices: Indices of test samples
        output_dir: Output directory
        epoch: Epoch number or "final"
        train_samples: Number of training samples (for title)
        shapefile_dir: Directory containing US shapefiles (defaults to package shapefiles/)

    Returns:
        Tuple of (grid_avg_errors, grid_counts)
    """
    # Import geopandas for US boundaries
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas is required for US state boundaries. Install with: pip install geopandas")

    # Default shapefile location
    if shapefile_dir is None:
        shapefile_dir = PACKAGE_DIR / 'shapefiles'

    # Extract coordinates for test samples
    coords = dataset.coords[test_indices].cpu().numpy()
    lats = coords[:, 0]
    lons = coords[:, 1]

    # Denormalize predictions and targets to LFMC percentage points
    test_predictions_denorm = test_predictions * MAX_LFMC_VALUE
    test_ground_truth_denorm = test_ground_truth * MAX_LFMC_VALUE

    # Calculate errors in LFMC percentage points
    errors = np.abs(test_predictions_denorm - test_ground_truth_denorm)

    # Define CONUS boundaries
    lon_min, lon_max = -125, -66
    lat_min, lat_max = 24, 50

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Load and plot US shapefile
    shapefile_path = shapefile_dir / 'cb_2018_us_state_20m.shp'
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found at {shapefile_path}. Please ensure shapefiles are downloaded.")

    # Read shapefile
    states = gpd.read_file(shapefile_path)
    # Filter to continental US (exclude Alaska, Hawaii, territories)
    states_conus = states[
        ~states['NAME'].isin(['Alaska', 'Hawaii', 'Puerto Rico',
                             'United States Virgin Islands', 'Guam',
                             'American Samoa', 'Northern Mariana Islands'])
    ]
    # Plot state boundaries with light grey, thin lines
    states_conus.boundary.plot(ax=ax, color='#CCCCCC', linewidth=0.4, alpha=0.8)
    # Plot overall US boundary slightly darker
    states_conus.dissolve().boundary.plot(ax=ax, color='#AAAAAA', linewidth=0.5, alpha=0.9)

    # Create 50x50 km grid (approximately 0.45 degrees)
    grid_size = 0.45
    lon_bins = np.arange(lon_min, lon_max, grid_size)
    lat_bins = np.arange(lat_min, lat_max, grid_size)

    # Bin the data
    grid_errors = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
    grid_counts = np.zeros((len(lat_bins)-1, len(lon_bins)-1))

    for i in range(len(lats)):
        if lon_min <= lons[i] <= lon_max and lat_min <= lats[i] <= lat_max:
            lon_idx = np.digitize(lons[i], lon_bins) - 1
            lat_idx = np.digitize(lats[i], lat_bins) - 1
            if 0 <= lon_idx < len(lon_bins)-1 and 0 <= lat_idx < len(lat_bins)-1:
                grid_errors[lat_idx, lon_idx] += errors[i]
                grid_counts[lat_idx, lon_idx] += 1

    # Calculate average errors per bin
    with np.errstate(divide='ignore', invalid='ignore'):
        grid_avg_errors = grid_errors / grid_counts
        grid_avg_errors[grid_counts == 0] = np.nan

    # Plot binned data
    valid_bins = ~np.isnan(grid_avg_errors)
    nonzero_counts = grid_counts[grid_counts > 0]

    if np.any(valid_bins) and len(nonzero_counts) > 0:
        # Log scale for sizes
        log_counts = np.log1p(grid_counts)
        log_counts[grid_counts == 0] = 0

        # Normalize sizes
        min_log = np.min(log_counts[log_counts > 0]) if np.any(log_counts > 0) else 1
        max_log = np.max(log_counts)

        # Size range
        size_min, size_max = 10, 68

        # Plot each grid cell
        for i in range(len(lat_bins)-1):
            for j in range(len(lon_bins)-1):
                if not np.isnan(grid_avg_errors[i, j]) and grid_counts[i, j] > 0:
                    lon_center = (lon_bins[j] + lon_bins[j+1]) / 2
                    lat_center = (lat_bins[i] + lat_bins[i+1]) / 2

                    if max_log > min_log:
                        size_norm = (log_counts[i, j] - min_log) / (max_log - min_log)
                        size = size_min + (size_max - size_min) * size_norm
                    else:
                        size = size_max

                    error_value = min(grid_avg_errors[i, j], 50.0)

                    ax.scatter(lon_center, lat_center, s=size,
                               c=[error_value], cmap='turbo',
                               vmin=0, vmax=50,
                               alpha=1.0, edgecolors='black', linewidth=0.5)

        # Add colorbar
        norm = plt.Normalize(vmin=0, vmax=50)
        sm = plt.cm.ScalarMappable(cmap='turbo', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Average LFMC Error (pp)')
        cbar.set_ticks([0, 10, 20, 30, 40, 50])
        cbar.set_ticklabels(['0', '10', '20', '30', '40', '50+'])

        # Create size legend
        count_min = int(np.min(nonzero_counts))
        count_max = int(np.max(nonzero_counts))
        count_25 = int(np.percentile(nonzero_counts, 25))
        count_75 = int(np.percentile(nonzero_counts, 75))

        legend_sizes = [count_min, count_25, count_75, count_max]
        legend_labels = [f'{c:,} samples' for c in legend_sizes]
        legend_handles = []

        for count in legend_sizes:
            log_count = np.log1p(count)
            if max_log > min_log:
                size_norm = (log_count - min_log) / (max_log - min_log)
                size = size_min + (size_max - size_min) * size_norm
            else:
                size = size_max
            legend_handles.append(plt.scatter([], [], s=size, c='grey', alpha=1.0,
                                             edgecolors='black', linewidth=0.5))

        size_legend = ax.legend(legend_handles, legend_labels,
                               title='Sample Count', loc='upper right',
                               frameon=True, fancybox=True, shadow=True)
        ax.add_artist(size_legend)

    # Set labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    test_count = len(test_indices) if hasattr(test_indices, '__len__') else 0
    if train_samples and str(epoch).isdigit():
        title_text = f'Test Performance on {test_count:,} Samples after training for {epoch} epochs on {train_samples:,} samples'
    elif str(epoch).isdigit():
        title_text = f'Test Performance on {test_count:,} Samples after {epoch} epochs'
    else:
        title_text = f'Test Performance on {test_count:,} Samples'

    ax.set_title(f'Earth4D LFMC Prediction Error - Geospatial Distribution\n'
                 f'100km × 100km Grid Bins\n'
                 f'{title_text}', fontsize=14)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.grid(False)

    # Add statistics
    if len(test_predictions) > 0:
        preds_denorm = test_predictions * MAX_LFMC_VALUE if test_predictions.max() <= 1.0 else test_predictions
        gts_denorm = test_ground_truth * MAX_LFMC_VALUE if test_ground_truth.max() <= 1.0 else test_ground_truth

        errors_calc = preds_denorm - gts_denorm
        abs_errors = np.abs(errors_calc)
        rmse = np.sqrt(np.mean(errors_calc ** 2))
        mae = np.mean(abs_errors)

        ss_res = np.sum((preds_denorm - gts_denorm) ** 2)
        ss_tot = np.sum((gts_denorm - np.mean(gts_denorm)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        stats_text = (f'Test Dataset:\n'
                     f'RMSE: {rmse:.1f}pp\n'
                     f'MAE: {mae:.1f}pp\n'
                     f'R²: {r2:.3f}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               ha='left', va='top', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9,
                        edgecolor='black', linewidth=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / f'geospatial_error_map_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()

    return grid_avg_errors, grid_counts


def create_temporal_visualization(
    dataset,
    all_predictions: Dict[str, np.ndarray],
    all_ground_truth: Dict[str, np.ndarray],
    all_indices: Dict[str, Any],
    output_dir: Path,
    epoch: str = "final",
    total_epochs: Optional[int] = None,
    train_samples: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Create temporal visualization of LFMC predictions vs ground truth with monthly binning.

    Args:
        dataset: Dataset object with coords attribute
        all_predictions: Dictionary with test split predictions
        all_ground_truth: Dictionary with test split ground truth
        all_indices: Dictionary with test split indices
        output_dir: Output directory
        epoch: Epoch number or "final"
        total_epochs: Total epochs trained (for title)
        train_samples: Number of training samples (for title)

    Returns:
        Array of absolute errors, or None if no data
    """
    # Combine all test sets
    all_preds = []
    all_gts = []
    all_times = []
    all_sources = []

    for split_name in ['test']:
        if split_name in all_predictions and len(all_predictions[split_name]) > 0:
            preds = all_predictions[split_name]
            gts = all_ground_truth[split_name]
            indices = all_indices[split_name]

            coords = dataset.coords[indices].cpu().numpy()
            times = coords[:, 3]

            all_preds.extend(preds)
            all_gts.extend(gts)
            all_times.extend(times)
            all_sources.extend([split_name] * len(preds))

    if len(all_preds) == 0:
        print("No test data available for temporal visualization", flush=True)
        return None

    all_preds = np.array(all_preds) * MAX_LFMC_VALUE
    all_gts = np.array(all_gts) * MAX_LFMC_VALUE
    all_times = np.array(all_times)

    # Convert normalized times to actual dates
    base_date = datetime(2015, 1, 1)
    end_date = datetime(2025, 1, 1)
    total_days = (end_date - base_date).days
    dates = [base_date + timedelta(days=int(t * total_days)) for t in all_times]

    # Monthly binning
    monthly_data = defaultdict(lambda: {'preds': [], 'gts': [], 'sources': []})

    for i, date in enumerate(dates):
        month_start = datetime(date.year, date.month, 1)
        monthly_data[month_start]['preds'].append(all_preds[i])
        monthly_data[month_start]['gts'].append(all_gts[i])
        monthly_data[month_start]['sources'].append(all_sources[i])

    months = sorted(monthly_data.keys())

    # Filter months with at least 5 samples
    months_filtered = []
    month_predictions = []
    month_ground_truths = []
    month_pred_medians = []
    month_gt_medians = []
    month_positions = []

    for month in months:
        preds = np.array(monthly_data[month]['preds'])
        gts = np.array(monthly_data[month]['gts'])

        if len(preds) >= 5:
            months_filtered.append(month)
            month_predictions.append(preds)
            month_ground_truths.append(gts)
            month_pred_medians.append(np.median(preds))
            month_gt_medians.append(np.median(gts))
            month_positions.append(mdates.date2num(month))

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))

    if len(months_filtered) > 0:
        offset = 5

        gt_positions_offset = []
        pred_positions_offset = []

        for i, pos in enumerate(month_positions):
            # Ground truth violin (ember red)
            gt_parts = ax.violinplot([month_ground_truths[i]],
                                     positions=[pos - offset],
                                     widths=8.4,
                                     showmeans=False, showmedians=False, showextrema=False)

            for pc in gt_parts['bodies']:
                pc.set_facecolor('#B22222')
                pc.set_edgecolor('#8B0000')
                pc.set_alpha(0.6)
                pc.set_linewidth(0.5)

            # Prediction violin (navy blue)
            pred_parts = ax.violinplot([month_predictions[i]],
                                       positions=[pos + offset],
                                       widths=8.4,
                                       showmeans=False, showmedians=False, showextrema=False)

            for pc in pred_parts['bodies']:
                pc.set_facecolor('#000080')
                pc.set_edgecolor('#000050')
                pc.set_alpha(0.6)
                pc.set_linewidth(0.5)

            gt_positions_offset.append(pos - offset)
            pred_positions_offset.append(pos + offset)

        # Plot median dots
        for pos, gt_median in zip(gt_positions_offset, month_gt_medians):
            ax.scatter(pos, gt_median, c='#B22222', s=40, zorder=6,
                      edgecolors='#8B0000', linewidth=0.5)

        for pos, pred_median in zip(pred_positions_offset, month_pred_medians):
            ax.scatter(pos, pred_median, c='#000080', s=40, zorder=6,
                      edgecolors='#000050', linewidth=0.5)

        # Connection lines
        ax.plot(gt_positions_offset, month_gt_medians, color='#B22222', linewidth=0.5,
                alpha=0.4, zorder=2, linestyle='-')
        ax.plot(pred_positions_offset, month_pred_medians, color='#000080', linewidth=0.5,
                alpha=0.4, zorder=2, linestyle='-')

    # Format axis
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Legend
    gt_violin = mpatches.Patch(color='#B22222', alpha=0.6, label='Ground Truth Distribution')
    gt_median = plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='#B22222', markeredgecolor='#8B0000',
                          markersize=6, label='Ground Truth Median')
    pred_violin = mpatches.Patch(color='#000080', alpha=0.6, label='Prediction Distribution')
    pred_median = plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor='#000080', markeredgecolor='#000050',
                            markersize=6, label='Prediction Median')

    ax.legend(handles=[gt_violin, gt_median, pred_violin, pred_median],
             loc='upper right', frameon=True, fancybox=True, shadow=True)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('LFMC (%)', fontsize=12)

    # Title
    total_test_samples = sum(len(indices) if hasattr(indices, '__len__') else 0
                            for indices in all_indices.values())

    if train_samples and str(epoch).isdigit():
        title_text = f'Test Performance on {total_test_samples:,} Samples after training for {epoch} epochs on {train_samples:,} samples'
    elif str(epoch).isdigit():
        title_text = f'Test Performance on {total_test_samples:,} Samples after {epoch} epochs'
    else:
        title_text = f'Test Performance on {total_test_samples:,} Samples'

    ax.set_title(f'Earth4D LFMC Predictions - Monthly Temporal Evolution (Test Set)\n'
                 f'Ground Truth and Prediction Distributions\n'
                 f'{title_text}', fontsize=14)

    ax.grid(False)

    # Y-axis limits
    if len(months_filtered) > 0:
        all_values = np.concatenate(month_predictions + month_ground_truths)
        y_min = max(0, np.min(all_values) - 20)
        y_max = min(600, np.max(all_values) + 20)
        ax.set_ylim(y_min, y_max)

    # Statistics
    if len(months_filtered) > 0:
        all_predictions_flat = np.concatenate(month_predictions)
        all_ground_truths_flat = np.concatenate(month_ground_truths)

        errors_calc = all_predictions_flat - all_ground_truths_flat
        abs_errors = np.abs(errors_calc)
        rmse = np.sqrt(np.mean(errors_calc ** 2))
        mae = np.mean(abs_errors)

        ss_res = np.sum((all_predictions_flat - all_ground_truths_flat) ** 2)
        ss_tot = np.sum((all_ground_truths_flat - np.mean(all_ground_truths_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        stats_text = (f'Test Dataset:\n'
                     f'RMSE: {rmse:.1f}pp\n'
                     f'MAE: {mae:.1f}pp\n'
                     f'R²: {r2:.3f}')
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                ha='left', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9,
                         edgecolor='black', linewidth=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / f'temporal_predictions_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()

    if len(months_filtered) > 0:
        all_predictions_full = np.concatenate(month_predictions)
        all_ground_truths_full = np.concatenate(month_ground_truths)
        abs_errors_full = np.abs(all_predictions_full - all_ground_truths_full)
        return abs_errors_full
    return None


def create_combined_scientific_figure(
    dataset,
    test_predictions: np.ndarray,
    test_ground_truth: np.ndarray,
    test_indices,
    all_predictions: Dict[str, np.ndarray],
    all_ground_truth: Dict[str, np.ndarray],
    all_indices: Dict[str, Any],
    output_dir: Path,
    epoch: str = "final",
    train_samples: Optional[int] = None,
    shapefile_dir: Optional[Path] = None
) -> None:
    """
    Create combined geospatial + temporal scientific figure with (A) and (B) labels.

    Args:
        dataset: Dataset object with coords attribute
        test_predictions: Test predictions for geospatial plot
        test_ground_truth: Test ground truth for geospatial plot
        test_indices: Test indices for geospatial plot
        all_predictions: Dict of all predictions for temporal plot
        all_ground_truth: Dict of all ground truth for temporal plot
        all_indices: Dict of all indices for temporal plot
        output_dir: Output directory
        epoch: Epoch number or "final"
        train_samples: Number of training samples
        shapefile_dir: Directory containing US shapefiles
    """
    from matplotlib.gridspec import GridSpec

    # Import geopandas
    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError("geopandas is required for US state boundaries. Install with: pip install geopandas")

    if shapefile_dir is None:
        shapefile_dir = PACKAGE_DIR / 'shapefiles'

    # Dimension calculations
    lon_min, lon_max = -125, -66
    lat_min, lat_max = 24, 50

    lat_center = (lat_min + lat_max) / 2
    cos_lat = np.cos(np.radians(lat_center))

    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    geo_aspect_physical = (lon_span * cos_lat) / lat_span

    geo_width_inches = 10.0
    geo_height_inches = geo_width_inches / geo_aspect_physical

    temp_aspect = 18.0 / 8.0
    temp_height_inches = geo_height_inches
    temp_width_inches = temp_height_inches * temp_aspect

    total_width = geo_width_inches + temp_width_inches + 2.0
    total_height = geo_height_inches

    # Create figure
    fig = plt.figure(figsize=(total_width, total_height))
    gs = GridSpec(1, 2, figure=fig,
                 width_ratios=[geo_width_inches, temp_width_inches],
                 wspace=0.08, hspace=0)

    ax_geo = fig.add_subplot(gs[0])

    # Geospatial data
    coords = dataset.coords[test_indices].cpu().numpy()
    lats, lons = coords[:, 0], coords[:, 1]
    test_preds_denorm = test_predictions * MAX_LFMC_VALUE
    test_gts_denorm = test_ground_truth * MAX_LFMC_VALUE
    errors = np.abs(test_preds_denorm - test_gts_denorm)

    # Grid binning
    grid_size = 1.35
    lon_bins = np.arange(lon_min, lon_max, grid_size)
    lat_bins = np.arange(lat_min, lat_max, grid_size)
    grid_errors = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
    grid_counts = np.zeros((len(lat_bins)-1, len(lon_bins)-1))

    for i in range(len(lats)):
        if lon_min <= lons[i] <= lon_max and lat_min <= lats[i] <= lat_max:
            lon_idx = np.digitize(lons[i], lon_bins) - 1
            lat_idx = np.digitize(lats[i], lat_bins) - 1
            if 0 <= lon_idx < len(lon_bins)-1 and 0 <= lat_idx < len(lat_bins)-1:
                grid_errors[lat_idx, lon_idx] += errors[i]
                grid_counts[lat_idx, lon_idx] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        grid_avg_errors = grid_errors / grid_counts
        grid_avg_errors[grid_counts == 0] = np.nan

    # Plot grid cells
    if np.any(~np.isnan(grid_avg_errors)):
        log_counts = np.log1p(grid_counts)
        log_counts[grid_counts == 0] = 0
        min_log = np.min(log_counts[log_counts > 0]) if np.any(log_counts > 0) else 1
        max_log = np.max(log_counts)
        size_min, size_max = 30, 112

        for i in range(len(lat_bins)-1):
            for j in range(len(lon_bins)-1):
                if not np.isnan(grid_avg_errors[i, j]) and grid_counts[i, j] > 0:
                    lon_center = (lon_bins[j] + lon_bins[j+1]) / 2
                    lat_center = (lat_bins[i] + lat_bins[i+1]) / 2
                    if max_log > min_log:
                        size_norm = (log_counts[i, j] - min_log) / (max_log - min_log)
                        size = size_min + (size_max - size_min) * size_norm
                    else:
                        size = size_max
                    error_value = min(grid_avg_errors[i, j], 50.0)
                    ax_geo.scatter(lon_center, lat_center, s=size, c=[error_value],
                                 cmap='turbo', vmin=0, vmax=50, alpha=1.0,
                                 edgecolors='black', linewidth=0.5)

    # Plot US boundaries
    shapefile_path = shapefile_dir / 'cb_2018_us_state_20m.shp'
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found at {shapefile_path}")

    states = gpd.read_file(shapefile_path)
    states_conus = states[
        ~states['NAME'].isin(['Alaska', 'Hawaii', 'Puerto Rico',
                             'United States Virgin Islands', 'Guam',
                             'American Samoa', 'Northern Mariana Islands'])
    ]
    states_conus.boundary.plot(ax=ax_geo, color='#CCCCCC', linewidth=0.4, alpha=0.8)
    states_conus.dissolve().boundary.plot(ax=ax_geo, color='#AAAAAA', linewidth=0.5, alpha=0.9)

    # Colorbar
    norm = plt.Normalize(vmin=0, vmax=50)
    sm = plt.cm.ScalarMappable(cmap='turbo', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_geo, label='Avg Error (pp)', fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 10, 20, 30, 40, 50])
    cbar.set_ticklabels(['0', '10', '20', '30', '40', '50+'])
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('Avg Error (pp)', fontsize=14)

    # Axis settings
    lat_range_current = lat_max - lat_min
    lat_center_point = (lat_min + lat_max) / 2
    lat_range_expanded = lat_range_current * 1.35
    lat_min_expanded = lat_center_point - lat_range_expanded / 2
    lat_max_expanded = lat_center_point + lat_range_expanded / 2

    ax_geo.set_xlim(lon_min, lon_max)
    ax_geo.set_ylim(lat_min_expanded, lat_max_expanded)
    ax_geo.set_aspect(1.0 / cos_lat)
    ax_geo.grid(False)
    ax_geo.set_xlabel('Longitude', fontsize=14)
    ax_geo.set_ylabel('Latitude', fontsize=14)
    ax_geo.tick_params(axis='both', labelsize=13)

    ax_geo.text(0.02, 0.98, '(A)',
               transform=ax_geo.transAxes, fontsize=14, weight='bold',
               ha='left', va='top',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

    # Panel B: Temporal
    ax_temp = fig.add_subplot(gs[1])

    all_preds, all_gts, all_times = [], [], []
    for split_name in ['test']:
        if split_name in all_predictions and len(all_predictions[split_name]) > 0:
            preds = all_predictions[split_name]
            gts = all_ground_truth[split_name]
            indices = all_indices[split_name]
            coords = dataset.coords[indices].cpu().numpy()
            times = coords[:, 3]
            all_preds.extend(preds)
            all_gts.extend(gts)
            all_times.extend(times)

    if len(all_preds) > 0:
        all_preds = np.array(all_preds) * MAX_LFMC_VALUE
        all_gts = np.array(all_gts) * MAX_LFMC_VALUE

        base_date = datetime(2015, 1, 1)
        end_date = datetime(2025, 1, 1)
        total_days = (end_date - base_date).days
        dates = [base_date + timedelta(days=int(t * total_days)) for t in all_times]

        monthly_data = defaultdict(lambda: {'preds': [], 'gts': []})
        for i, date in enumerate(dates):
            month_start = datetime(date.year, date.month, 1)
            monthly_data[month_start]['preds'].append(all_preds[i])
            monthly_data[month_start]['gts'].append(all_gts[i])

        months = sorted(monthly_data.keys())
        month_predictions, month_ground_truths, month_positions = [], [], []

        for month in months:
            preds = np.array(monthly_data[month]['preds'])
            gts = np.array(monthly_data[month]['gts'])
            if len(preds) >= 5:
                month_predictions.append(preds)
                month_ground_truths.append(gts)
                month_positions.append(mdates.date2num(month))

        if len(month_predictions) > 0:
            offset = 5
            gt_medians_list, pred_medians_list = [], []
            gt_positions_list, pred_positions_list = [], []

            for i, pos in enumerate(month_positions):
                gt_parts = ax_temp.violinplot([month_ground_truths[i]],
                                            positions=[pos - offset],
                                            widths=8.4, showmeans=False,
                                            showmedians=False, showextrema=False)
                for pc in gt_parts['bodies']:
                    pc.set_facecolor('#B22222')
                    pc.set_edgecolor('#8B0000')
                    pc.set_alpha(0.6)
                    pc.set_linewidth(0.5)

                pred_parts = ax_temp.violinplot([month_predictions[i]],
                                              positions=[pos + offset],
                                              widths=8.4, showmeans=False,
                                              showmedians=False, showextrema=False)
                for pc in pred_parts['bodies']:
                    pc.set_facecolor('#000080')
                    pc.set_edgecolor('#000050')
                    pc.set_alpha(0.6)
                    pc.set_linewidth(0.5)

                gt_medians_list.append(np.median(month_ground_truths[i]))
                pred_medians_list.append(np.median(month_predictions[i]))
                gt_positions_list.append(pos - offset)
                pred_positions_list.append(pos + offset)

            # Median points
            for pos, gt_median in zip(gt_positions_list, gt_medians_list):
                ax_temp.scatter(pos, gt_median, c='#B22222', s=40, zorder=6,
                              edgecolors='#8B0000', linewidth=0.5)
            for pos, pred_median in zip(pred_positions_list, pred_medians_list):
                ax_temp.scatter(pos, pred_median, c='#000080', s=40, zorder=6,
                              edgecolors='#000050', linewidth=0.5)

            ax_temp.xaxis_date()
            ax_temp.xaxis.set_major_locator(mdates.YearLocator())
            ax_temp.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax_temp.set_xlabel('Date', fontsize=14)
            ax_temp.set_ylabel('LFMC (%)', fontsize=14)
            ax_temp.yaxis.tick_right()
            ax_temp.yaxis.set_label_position('right')
            ax_temp.tick_params(axis='both', labelsize=13)
            ax_temp.grid(False)

            # Legend
            gt_violin = mpatches.Patch(color='#B22222', alpha=0.6, label='Ground Truth Distribution')
            gt_median_marker = plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor='#B22222', markeredgecolor='#8B0000',
                                        markersize=6, label='Ground Truth Median')
            pred_violin = mpatches.Patch(color='#000080', alpha=0.6, label='Prediction Distribution')
            pred_median_marker = plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor='#000080', markeredgecolor='#000050',
                                          markersize=6, label='Prediction Median')

            ax_temp.legend(handles=[gt_violin, gt_median_marker, pred_violin, pred_median_marker],
                         loc='upper right', frameon=True, fancybox=True, shadow=False, fontsize=12)

    ax_temp.text(0.02, 0.98, '(B)',
                transform=ax_temp.transAxes, fontsize=14, weight='bold',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

    plt.savefig(output_dir / f'combined_scientific_figure_epoch_{epoch}.png',
                dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  Combined scientific figure saved to: combined_scientific_figure_epoch_{epoch}.png", flush=True)
