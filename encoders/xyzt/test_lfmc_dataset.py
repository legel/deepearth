#!/usr/bin/env python3
"""
Test script for LFMC Dataset Loader
====================================

Comprehensive tests for lfmc_dataset.py functionality.

Run this script to validate:
1. Loading different feature combinations
2. Merging correctness
3. .pt caching functionality
4. PyTorch dataset conversion
5. Train/test splitting

Usage:
    python test_lfmc_dataset.py

Output:
    test_results.log - Detailed log file with all test output

Author: Claude Code
Date: 2025-10-28
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Import the dataset loader
from lfmc_dataset import LFMCDataset, TORCH_AVAILABLE

# Global log file
LOG_FILE = "test_results.log"
log_handle = None

def log(message, to_console=True):
    """Write message to both log file and console."""
    global log_handle

    if log_handle is None:
        log_handle = open(LOG_FILE, 'w', encoding='utf-8')
        # Write header
        log_handle.write(f"LFMC Dataset Loader Test Results\n")
        log_handle.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_handle.write(f"="*80 + "\n\n")

    # Write to log file
    log_handle.write(message + "\n")
    log_handle.flush()

    # Write to console
    if to_console:
        print(message, flush=True)

def test_basic_loading():
    """Test 1: Basic loading of base dataset."""
    log("\n" + "="*60)
    log("TEST 1: Basic Loading (Base Only)")
    log("="*60)

    try:
        dataset = LFMCDataset(verbose=True, use_cache=False)

        # Validate
        assert len(dataset) > 0, "Dataset is empty!"
        assert dataset.data is not None, "Data not loaded!"
        assert 'sample_id' in dataset.data.columns, "Missing sample_id column!"
        assert 'lfmc_percent' in dataset.data.columns, "Missing target column!"

        log("\n[PASS] Test 1: Basic loading works")
        return True

    except Exception as e:
        log(f"\n[FAIL] Test 1 failed: {e}")
        import traceback
        import io
        error_output = io.StringIO()
        traceback.print_exc(file=error_output)
        log(error_output.getvalue())
        return False


def test_aef_loading():
    """Test 2: Loading with AlphaEarth features."""
    log("\n" + "="*60)
    log("TEST 2: Loading with AEF")
    log("="*60)

    try:
        dataset = LFMCDataset(use_aef=True, verbose=True, use_cache=False)

        # Validate
        assert len(dataset) > 0, "Dataset is empty!"

        # Check for AEF columns
        aef_cols = [c for c in dataset.data.columns if c.startswith('aef_')]
        assert len(aef_cols) == 64, f"Expected 64 AEF columns, got {len(aef_cols)}"

        log(f"\n  Found {len(aef_cols)} AEF embedding dimensions")
        log("[PASS] Test 2: AEF loading works")
        return True

    except Exception as e:
        log(f"\n[FAIL] Test 2 failed: {e}")
        import traceback
        import io
        error_output = io.StringIO()
        traceback.print_exc(file=error_output)
        log(error_output.getvalue())
        return False


def test_daymet_loading():
    """Test 3: Loading with Daymet weather features."""
    log("\n" + "="*60)
    log("TEST 3: Loading with Daymet")
    log("="*60)

    try:
        dataset = LFMCDataset(use_daymet=True, verbose=True, use_cache=False)

        # Validate
        assert len(dataset) > 0, "Dataset is empty!"

        # Check for Daymet columns
        daymet_cols = [c for c in dataset.data.columns
                      if any(x in c for x in ['prcp', 'tmin', 'tmax', 'srad', 'vp', 'dayl', 'swe'])]
        assert len(daymet_cols) > 0, "No Daymet columns found"

        log(f"\n  Found {len(daymet_cols)} Daymet weather features")
        log("[PASS] Test 3: Daymet loading works")
        return True

    except Exception as e:
        log(f"\n[FAIL] Test 3 failed: {e}")
        import traceback
        import io
        error_output = io.StringIO()
        traceback.print_exc(file=error_output)
        log(error_output.getvalue())
        return False


def test_all_features():
    """Test 4: Loading all features together."""
    log("\n" + "="*60)
    log("TEST 4: Loading All Features (Base + AEF + Daymet)")
    log("="*60)

    try:
        dataset = LFMCDataset(use_aef=True, use_daymet=True, verbose=True, use_cache=False)

        # Validate
        assert len(dataset) > 0, "Dataset is empty!"

        aef_cols = [c for c in dataset.data.columns if c.startswith('aef_')]
        daymet_cols = [c for c in dataset.data.columns
                      if any(x in c for x in ['prcp', 'tmin', 'tmax', 'srad', 'vp', 'dayl', 'swe'])]

        log(f"\n  Total samples: {len(dataset):,}")
        log(f"  Total columns: {len(dataset.data.columns)}")
        log(f"  AEF features: {len(aef_cols)}")
        log(f"  Daymet features: {len(daymet_cols)}")
        log(f"  Total features: {len(dataset.feature_columns)}")

        assert len(aef_cols) == 64, "Missing AEF features"
        assert len(daymet_cols) > 0, "Missing Daymet features"

        log("[PASS] Test 4: All features loading works")
        return True

    except Exception as e:
        log(f"\n[FAIL] Test 4 failed: {e}")
        import traceback
        import io
        error_output = io.StringIO()
        traceback.print_exc(file=error_output)
        log(error_output.getvalue())
        return False


def test_caching():
    """Test 5: .pt caching functionality."""
    log("\n" + "="*60)
    log("TEST 5: Caching Functionality")
    log("="*60)

    if not TORCH_AVAILABLE:
        log("[SKIP] PyTorch not available, skipping cache test")
        return True

    try:
        # First load: create cache
        log("\n  First load (creating cache)...")
        t0 = time.time()
        dataset1 = LFMCDataset(use_aef=True, use_cache=True, verbose=False)
        t1 = time.time()
        load_time_1 = t1 - t0

        # Second load: use cache
        log("  Second load (using cache)...")
        t0 = time.time()
        dataset2 = LFMCDataset(use_aef=True, use_cache=True, verbose=False)
        t2 = time.time()
        load_time_2 = t2 - t0

        log(f"\n  First load:  {load_time_1:.2f}s")
        log(f"  Second load: {load_time_2:.2f}s")
        log(f"  Speedup:     {load_time_1/load_time_2:.1f}x")

        # Validate data is identical
        assert len(dataset1) == len(dataset2), "Different lengths!"
        assert list(dataset1.data.columns) == list(dataset2.data.columns), "Different columns!"

        # Check cache files exist
        cache_dir = Path("./data/cache")
        assert (cache_dir / "lfmc_base.pt").exists(), "Base cache not created"
        assert (cache_dir / "lfmc_aef.pt").exists(), "AEF cache not created"

        log("[PASS] Test 5: Caching works")
        return True

    except Exception as e:
        log(f"\n[FAIL] Test 5 failed: {e}")
        import traceback
        import io
        error_output = io.StringIO()
        traceback.print_exc(file=error_output)
        log(error_output.getvalue())
        return False


def test_splitting():
    """Test 6: Train/test splitting."""
    log("\n" + "="*60)
    log("TEST 6: Train/Test Splitting")
    log("="*60)

    try:
        dataset = LFMCDataset(use_aef=True, verbose=False)

        # Create splits
        splits = dataset.create_splits(
            temporal_frac=0.05,
            spatial_frac=0.05,
            random_frac=0.05,
            random_seed=42
        )

        # Validate
        assert 'train' in splits, "Missing train split"
        assert 'temporal' in splits, "Missing temporal split"
        assert 'spatial' in splits, "Missing spatial split"
        assert 'random' in splits, "Missing random split"

        # Check no overlap
        train_set = set(splits['train'])
        temporal_set = set(splits['temporal'])
        spatial_set = set(splits['spatial'])
        random_set = set(splits['random'])

        assert len(train_set & temporal_set) == 0, "Train/temporal overlap!"
        assert len(train_set & spatial_set) == 0, "Train/spatial overlap!"
        assert len(train_set & random_set) == 0, "Train/random overlap!"

        # Check all indices covered
        all_indices = train_set | temporal_set | spatial_set | random_set
        assert len(all_indices) == len(dataset), "Missing indices!"

        log(f"\n  Train:    {len(splits['train']):,} samples")
        log(f"  Temporal: {len(splits['temporal']):,} samples")
        log(f"  Spatial:  {len(splits['spatial']):,} samples")
        log(f"  Random:   {len(splits['random']):,} samples")
        log(f"  Total:    {len(all_indices):,} samples")

        log("[PASS] Test 6: Splitting works")
        return True

    except Exception as e:
        log(f"\n[FAIL] Test 6 failed: {e}")
        import traceback
        import io
        error_output = io.StringIO()
        traceback.print_exc(file=error_output)
        log(error_output.getvalue())
        return False


def test_pytorch_conversion():
    """Test 7: PyTorch dataset conversion."""
    log("\n" + "="*60)
    log("TEST 7: PyTorch Dataset Conversion")
    log("="*60)

    if not TORCH_AVAILABLE:
        log("[SKIP] PyTorch not available")
        return True

    try:
        import torch
        from torch.utils.data import DataLoader

        dataset = LFMCDataset(use_aef=True, use_daymet=True, verbose=False)

        # Create splits
        splits = dataset.create_splits(random_seed=42)

        # Convert to PyTorch
        train_dataset = dataset.to_torch_dataset(indices=splits['train'])

        log(f"\n  PyTorch dataset size: {len(train_dataset)}")

        # Test getting a sample
        sample = train_dataset[0]
        coords, features, species_idx, target = sample

        log(f"  Sample shapes:")
        log(f"    Coords:      {coords.shape}")
        log(f"    Features:    {features.shape}")
        log(f"    Species idx: {species_idx}")
        log(f"    Target:      {target}")

        # Test DataLoader
        loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        batch = next(iter(loader))

        log(f"\n  Batch shapes (batch_size=32):")
        log(f"    Coords:   {batch[0].shape}")
        log(f"    Features: {batch[1].shape}")
        log(f"    Species:  {batch[2].shape}")
        log(f"    Targets:  {batch[3].shape}")

        assert batch[0].shape[0] == 32, "Wrong batch size for coords"
        assert batch[1].shape[0] == 32, "Wrong batch size for features"
        assert batch[2].shape[0] == 32, "Wrong batch size for species"
        assert batch[3].shape[0] == 32, "Wrong batch size for targets"

        log("[PASS] Test 7: PyTorch conversion works")
        return True

    except Exception as e:
        log(f"\n[FAIL] Test 7 failed: {e}")
        import traceback
        import io
        error_output = io.StringIO()
        traceback.print_exc(file=error_output)
        log(error_output.getvalue())
        return False


def test_statistics():
    """Test 8: Statistics and metadata."""
    log("\n" + "="*60)
    log("TEST 8: Statistics and Metadata")
    log("="*60)

    try:
        dataset = LFMCDataset(use_aef=True, use_daymet=True, verbose=False)

        stats = dataset.get_statistics()

        log("\n  Dataset Statistics:")
        log(f"    Samples:     {stats['n_samples']:,}")
        log(f"    Features:    {stats['n_features']}")
        log(f"    Species:     {stats['n_species']}")
        log(f"    LFMC range:  {stats['lfmc_min']:.1f}% - {stats['lfmc_max']:.1f}%")
        log(f"    LFMC mean:   {stats['lfmc_mean']:.1f}%")
        log(f"    LFMC std:    {stats['lfmc_std']:.1f}%")
        log(f"    Date range:  {stats['date_min']} to {stats['date_max']}")
        log(f"    Lat range:   {stats['lat_range'][0]:.2f} to {stats['lat_range'][1]:.2f}")
        log(f"    Lon range:   {stats['lon_range'][0]:.2f} to {stats['lon_range'][1]:.2f}")
        log(f"    Has AEF:     {stats['has_aef']}")
        log(f"    Has Daymet:  {stats['has_daymet']}")

        # Validate
        assert stats['n_samples'] > 0, "Zero samples"
        assert stats['n_species'] > 0, "Zero species"
        assert 0 <= stats['lfmc_min'] <= stats['lfmc_max'] <= 600, "Invalid LFMC range"

        log("[PASS] Test 8: Statistics work")
        return True

    except Exception as e:
        log(f"\n[FAIL] Test 8 failed: {e}")
        import traceback
        import io
        error_output = io.StringIO()
        traceback.print_exc(file=error_output)
        log(error_output.getvalue())
        return False


def run_all_tests():
    """Run all tests and report results."""
    log("\n" + "="*80)
    log(" "*20 + "LFMC DATASET LOADER - TEST SUITE")
    log("="*80)

    tests = [
        ("Basic Loading", test_basic_loading),
        ("AEF Loading", test_aef_loading),
        ("Daymet Loading", test_daymet_loading),
        ("All Features", test_all_features),
        ("Caching", test_caching),
        ("Splitting", test_splitting),
        ("PyTorch Conversion", test_pytorch_conversion),
        ("Statistics", test_statistics)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            log(f"\n[ERROR] Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    log("\n" + "="*80)
    log(" "*30 + "TEST SUMMARY")
    log("="*80)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        log(f"  {status} {test_name}")

    log("\n" + "="*80)
    log(f"  Results: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    log("="*80)

    if passed == total:
        log("\n  🎉 All tests passed!")
        return 0
    else:
        log(f"\n  ⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    result = run_all_tests()

    # Close log file
    if log_handle:
        log_handle.write(f"\n\nTest log saved to: {LOG_FILE}\n")
        log_handle.close()
        print(f"\n📝 Full test log saved to: {LOG_FILE}")

    sys.exit(result)
