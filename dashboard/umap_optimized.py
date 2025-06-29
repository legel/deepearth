#!/usr/bin/env python3
"""
Simplified optimized UMAP implementation for DeepEarth.
Focus on JIT pre-warming and practical optimizations.
"""

import numpy as np
import umap
import time
import os
import pickle
import logging
from functools import lru_cache
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache for warmed UMAP state
_UMAP_WARMED = False
_WARMUP_TIME = None

def warm_up_umap():
    """Pre-compile UMAP's Numba functions to avoid first-run overhead."""
    global _UMAP_WARMED, _WARMUP_TIME
    
    if _UMAP_WARMED:
        return _WARMUP_TIME
    
    logger.info("🔥 Warming up UMAP JIT compilation...")
    start_time = time.time()
    
    # Create synthetic data similar to our use case
    # 576 points with 1408 features
    dummy_data = np.random.randn(100, 100).astype(np.float32)
    dummy_data = dummy_data / np.linalg.norm(dummy_data, axis=1, keepdims=True)
    
    # Run UMAP with minimal epochs to trigger compilation
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=3,
        min_dist=0.1,
        n_epochs=2,  # Minimal
        metric='cosine',
        init='random',
        random_state=42,
        verbose=False
    )
    _ = reducer.fit_transform(dummy_data)
    
    # Also warm up with euclidean metric
    reducer_euclidean = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        n_epochs=2,
        metric='euclidean',
        init='random',
        random_state=42,
        verbose=False
    )
    _ = reducer_euclidean.fit_transform(dummy_data)
    
    _WARMUP_TIME = time.time() - start_time
    _UMAP_WARMED = True
    logger.info(f"✅ UMAP warmup completed in {_WARMUP_TIME:.2f}s")
    
    return _WARMUP_TIME

class OptimizedUMAP:
    """
    Optimized UMAP wrapper with:
    - Automatic JIT pre-warming
    - Result caching
    - Consistent API with original UMAP
    """
    
    def __init__(self, cache_dir="/tmp/umap_cache", **umap_kwargs):
        self.cache_dir = cache_dir
        self.umap_kwargs = umap_kwargs
        os.makedirs(cache_dir, exist_ok=True)
        
        # Ensure UMAP is warmed
        warm_up_umap()
    
    def _get_cache_key(self, data):
        """Generate a cache key from data and parameters."""
        # Create a hash from data shape, sum, and a sample
        data_summary = f"{data.shape}_{data.sum():.6f}_{data[0].sum():.6f}"
        param_str = "_".join(f"{k}{v}" for k, v in sorted(self.umap_kwargs.items()))
        
        # Use SHA256 for consistent hashing
        combined = f"{data_summary}_{param_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def fit_transform(self, data):
        """Fit and transform with caching."""
        # Check memory cache first
        cache_key = self._get_cache_key(data)
        
        # Check disk cache
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                logger.info(f"📂 Loaded UMAP from cache: {cache_key}")
                return result
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        
        # Compute UMAP
        logger.info(f"🧮 Computing UMAP (n={data.shape[0]}, d={data.shape[1]})")
        start_time = time.time()
        
        reducer = umap.UMAP(**self.umap_kwargs)
        result = reducer.fit_transform(data)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ UMAP computed in {elapsed:.2f}s")
        
        # Cache result
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            logger.info(f"💾 Cached UMAP result: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
        
        return result

def benchmark_optimized_umap():
    """Compare optimized vs standard UMAP performance."""
    print("\nUMAP Optimization Benchmark")
    print("="*60)
    
    # Test data matching DeepEarth
    n_samples = 576
    n_features = 1408
    test_data = np.random.randn(n_samples, n_features).astype(np.float32)
    test_data = test_data / np.linalg.norm(test_data, axis=1, keepdims=True)
    
    # Standard UMAP parameters
    umap_params = {
        'n_components': 3,
        'n_neighbors': 15,
        'min_dist': 0.1,
        'n_epochs': 30,
        'metric': 'cosine',
        'init': 'random',
        'random_state': 42
    }
    
    # Test 1: Vanilla UMAP (cold start)
    print("\n1. Vanilla UMAP (first run):")
    start = time.time()
    vanilla_reducer = umap.UMAP(**umap_params)
    result_vanilla = vanilla_reducer.fit_transform(test_data)
    vanilla_time = time.time() - start
    print(f"   Time: {vanilla_time:.3f}s")
    
    # Test 2: Optimized UMAP (pre-warmed)
    print("\n2. Optimized UMAP (pre-warmed):")
    opt_umap = OptimizedUMAP(**umap_params)
    
    start = time.time()
    result_opt = opt_umap.fit_transform(test_data)
    opt_time = time.time() - start
    print(f"   Time: {opt_time:.3f}s")
    print(f"   Speedup: {vanilla_time/opt_time:.2f}x")
    
    # Test 3: Cached result
    print("\n3. Optimized UMAP (cached):")
    start = time.time()
    result_cached = opt_umap.fit_transform(test_data)
    cached_time = time.time() - start
    print(f"   Time: {cached_time:.3f}s")
    print(f"   Speedup: {vanilla_time/cached_time:.0f}x")
    
    # Test 4: Second vanilla run (JIT now compiled)
    print("\n4. Vanilla UMAP (second run):")
    start = time.time()
    vanilla_reducer2 = umap.UMAP(**umap_params)
    result_vanilla2 = vanilla_reducer2.fit_transform(test_data)
    vanilla_time2 = time.time() - start
    print(f"   Time: {vanilla_time2:.3f}s")
    
    # Verify results are similar
    diff = np.mean(np.abs(result_vanilla - result_opt))
    print(f"\n5. Result difference: {diff:.6f} (should be small)")
    
    print("\nSummary:")
    print(f"  - JIT compilation overhead: {vanilla_time - vanilla_time2:.2f}s")
    print(f"  - Pre-warming saves: {vanilla_time - opt_time:.2f}s on first run")
    print(f"  - Caching provides: {vanilla_time/cached_time:.0f}x speedup")
    print("="*60)

# Integration function for DeepEarth
def integrate_with_deepearth():
    """
    Integration code for deepearth_dashboard.py
    
    Add this to the imports:
        from umap_optimized import OptimizedUMAP, warm_up_umap
    
    In app initialization:
        # Pre-warm UMAP on startup
        warm_up_umap()
    
    Replace UMAP creation in get_umap_rgb():
        # OLD:
        reducer = umap.UMAP(...)
        
        # NEW:
        reducer = OptimizedUMAP(
            cache_dir="/tmp/deepearth_umap_cache",
            n_components=3,
            n_neighbors=15,
            min_dist=0.1,
            n_epochs=30,
            metric='cosine',
            init='random',
            random_state=42
        )
        coords_3d = reducer.fit_transform(features_flat)
    """
    print("\nIntegration Instructions:")
    print("="*60)
    print(integrate_with_deepearth.__doc__)

if __name__ == "__main__":
    # Warm up on module load
    warm_up_umap()
    
    # Run benchmark
    benchmark_optimized_umap()
    
    # Show integration instructions
    integrate_with_deepearth()