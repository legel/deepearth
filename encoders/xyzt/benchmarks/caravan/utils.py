"""
Utility functions for Caravan benchmark.
"""

import numpy as np
import torch
from typing import List


class MetricsEMA:
    """Exponential moving average for smoothing metrics during training."""

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Smoothing factor (0 = no update, 1 = replace completely)
        """
        self.alpha = alpha
        self.ema_values = {}

    def update(self, metrics: dict) -> dict:
        """
        Update EMA values and return smoothed metrics.

        Args:
            metrics: Dictionary of current metric values

        Returns:
            Dictionary with EMA-smoothed metrics (keys prefixed with 'ema_')
        """
        ema_metrics = {}

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                ema_key = f'ema_{key}'

                if ema_key not in self.ema_values:
                    self.ema_values[ema_key] = value
                else:
                    self.ema_values[ema_key] = (
                        self.alpha * value +
                        (1 - self.alpha) * self.ema_values[ema_key]
                    )

                ema_metrics[ema_key] = self.ema_values[ema_key]

        return ema_metrics


def print_sample_predictions(true_values: List[float], pred_values: List[float], prefix: str = ""):
    """
    Print a table of sample predictions for monitoring.

    Args:
        true_values: Ground truth values
        pred_values: Predicted values
        prefix: Optional prefix for display
    """
    if len(true_values) == 0:
        return

    print(f"\n  {prefix}Sample Predictions (mm/day):", flush=True)
    print(f"    {'True':>8s} | {'Pred':>8s} | {'Error':>8s}", flush=True)
    print(f"    {'-'*8}-+-{'-'*8}-+-{'-'*8}", flush=True)

    for true_val, pred_val in zip(true_values[:5], pred_values[:5]):
        error = pred_val - true_val
        print(f"    {true_val:8.2f} | {pred_val:8.2f} | {error:+8.2f}", flush=True)
