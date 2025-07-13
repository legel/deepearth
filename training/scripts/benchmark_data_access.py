#!/usr/bin/env python3
"""
Performance Benchmarking: Direct Import vs Flask API Access

Measures latency and validates data consistency between direct Python import 
and Flask API for ML training data access patterns.

    🔬 Direct Import ──► ⚡ <50ms target
    📡 Flask API ─────► ⚡ <100ms target
    
Usage:
    python benchmark_data_access.py --batch-size 64 --runs 10
"""

import time
import statistics
import argparse
import sys
import os
import json
import requests
import numpy as np
import torch
from pathlib import Path

# Add dashboard to path for direct imports
dashboard_path = Path(__file__).parent.parent.parent / "dashboard"
sys.path.insert(0, str(dashboard_path))

from data_cache import UnifiedDataCache
from services.training_data import get_training_batch, get_available_observation_ids


class PerformanceBenchmark:
    """
    🎯 ML Training Data Access Benchmark
    
    Comprehensive performance analysis comparing direct Python imports
    vs Flask API calls for training data retrieval.
    """
    
    def __init__(self, config_path: str, flask_url: str = "http://localhost:5000"):
        """
        Initialize benchmark with data cache and API endpoint.
        
        Args:
            config_path: Path to dashboard configuration file
            flask_url: Base URL for Flask API server
        """
        self.config_path = config_path
        self.flask_url = flask_url
        self.cache = None
        self.available_ids = []
        
    def setup(self):
        """Initialize data cache and load available observation IDs."""
        print("🔧 Setting up benchmark environment...")
        
        # Change to dashboard directory for proper relative paths
        original_cwd = os.getcwd()
        dashboard_dir = Path(__file__).parent.parent.parent / "dashboard"
        os.chdir(dashboard_dir)
        
        # Initialize cache (same as dashboard)
        try:
            self.cache = UnifiedDataCache(self.config_path)
            print(f"✅ Loaded data cache from {self.config_path}")
        except Exception as e:
            print(f"❌ Failed to load cache: {e}")
            os.chdir(original_cwd)  # Restore directory
            return False
        
        # Get available observation IDs
        try:
            self.available_ids = get_available_observation_ids(
                self.cache, 
                has_vision=True, 
                has_language=True,
                limit=1000  # Limit for testing
            )
            print(f"✅ Found {len(self.available_ids)} available observation IDs")
        except Exception as e:
            print(f"❌ Failed to load observation IDs: {e}")
            return False
        
        return True
    
    def benchmark_direct_import(self, batch_size: int, num_runs: int = 5) -> dict:
        """
        Benchmark direct Python import performance.
        
        Args:
            batch_size: Number of observations per batch
            num_runs: Number of benchmark runs for averaging
            
        Returns:
            dict: Performance metrics and timing data
        """
        print(f"\n🚀 Benchmarking Direct Import (batch_size={batch_size}, runs={num_runs})")
        
        if batch_size > len(self.available_ids):
            print(f"⚠️  Reducing batch size to {len(self.available_ids)} (available data)")
            batch_size = len(self.available_ids)
        
        timings = []
        memory_usage = []
        
        for run in range(num_runs):
            # Select random batch
            batch_ids = np.random.choice(self.available_ids, batch_size, replace=False).tolist()
            
            # Time the direct import call
            start_time = time.perf_counter()
            
            try:
                batch_data = get_training_batch(
                    self.cache,
                    batch_ids,
                    include_vision=True,
                    include_language=True,
                    device='cpu'
                )
                
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000
                timings.append(elapsed_ms)
                
                # Estimate memory usage (rough)
                memory_mb = self._estimate_memory_usage(batch_data)
                memory_usage.append(memory_mb)
                
                print(f"  Run {run+1}: {elapsed_ms:.1f}ms, ~{memory_mb:.1f}MB")
                
            except Exception as e:
                print(f"  Run {run+1}: FAILED - {e}")
                continue
        
        return {
            'method': 'direct_import',
            'batch_size': batch_size,
            'num_runs': len(timings),
            'timings_ms': timings,
            'memory_mb': memory_usage,
            'mean_time_ms': statistics.mean(timings) if timings else 0,
            'std_time_ms': statistics.stdev(timings) if len(timings) > 1 else 0,
            'mean_memory_mb': statistics.mean(memory_usage) if memory_usage else 0,
            'last_batch_data': batch_data if timings else None
        }
    
    def benchmark_flask_api(self, batch_size: int, num_runs: int = 5) -> dict:
        """
        Benchmark Flask API performance.
        
        Args:
            batch_size: Number of observations per batch
            num_runs: Number of benchmark runs for averaging
            
        Returns:
            dict: Performance metrics and timing data
        """
        print(f"\n📡 Benchmarking Flask API (batch_size={batch_size}, runs={num_runs})")
        
        if batch_size > len(self.available_ids):
            print(f"⚠️  Reducing batch size to {len(self.available_ids)} (available data)")
            batch_size = len(self.available_ids)
        
        timings = []
        memory_usage = []
        api_url = f"{self.flask_url}/api/training/batch"
        
        # Test API connectivity
        try:
            health_response = requests.get(f"{self.flask_url}/api/health", timeout=5)
            if health_response.status_code != 200:
                print(f"❌ API health check failed: {health_response.status_code}")
                return self._empty_results('flask_api', batch_size)
        except Exception as e:
            print(f"❌ Cannot connect to Flask API at {self.flask_url}: {e}")
            return self._empty_results('flask_api', batch_size)
        
        for run in range(num_runs):
            # Select random batch
            batch_ids = np.random.choice(self.available_ids, batch_size, replace=False).tolist()
            
            payload = {
                "observation_ids": batch_ids,
                "include_vision": True,
                "include_language": True
            }
            
            # Time the API call
            start_time = time.perf_counter()
            
            try:
                response = requests.post(
                    api_url,
                    json=payload,
                    timeout=120,  # Allow time for large batches
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    batch_data = response.json()
                    end_time = time.perf_counter()
                    elapsed_ms = (end_time - start_time) * 1000
                    timings.append(elapsed_ms)
                    
                    # Estimate memory usage from JSON response
                    memory_mb = len(response.content) / (1024 * 1024)
                    memory_usage.append(memory_mb)
                    
                    print(f"  Run {run+1}: {elapsed_ms:.1f}ms, ~{memory_mb:.1f}MB")
                    
                else:
                    print(f"  Run {run+1}: API ERROR - {response.status_code}: {response.text}")
                    continue
                    
            except Exception as e:
                print(f"  Run {run+1}: FAILED - {e}")
                continue
        
        return {
            'method': 'flask_api',
            'batch_size': batch_size,
            'num_runs': len(timings),
            'timings_ms': timings,
            'memory_mb': memory_usage,
            'mean_time_ms': statistics.mean(timings) if timings else 0,
            'std_time_ms': statistics.stdev(timings) if len(timings) > 1 else 0,
            'mean_memory_mb': statistics.mean(memory_usage) if memory_usage else 0,
            'last_batch_data': batch_data if timings else None
        }
    
    def validate_data_consistency(self, direct_data: dict, api_data: dict) -> dict:
        """
        Validate that direct import and API return identical data.
        
        Args:
            direct_data: Data from direct import
            api_data: Data from Flask API
            
        Returns:
            dict: Validation results
        """
        print("\n🔍 Validating Data Consistency...")
        
        validations = {
            'species_match': False,
            'image_urls_match': False, 
            'locations_match': False,
            'timestamps_match': False,
            'language_embeddings_match': False,
            'vision_embeddings_match': False,
            'errors': []
        }
        
        try:
            # Species names
            if direct_data['species'] == api_data['species']:
                validations['species_match'] = True
                print("  ✅ Species names match")
            else:
                validations['errors'].append("Species names differ")
                print("  ❌ Species names differ")
            
            # Image URLs
            if direct_data['image_urls'] == api_data['image_urls']:
                validations['image_urls_match'] = True
                print("  ✅ Image URLs match")
            else:
                validations['errors'].append("Image URLs differ")
                print("  ❌ Image URLs differ")
            
            # Locations (convert tensors to lists for comparison)
            direct_locations = direct_data['locations'].tolist()
            if np.allclose(direct_locations, api_data['locations'], rtol=1e-5):
                validations['locations_match'] = True
                print("  ✅ Locations match")
            else:
                validations['errors'].append("Locations differ")
                print("  ❌ Locations differ")
            
            # Timestamps
            direct_timestamps = direct_data['timestamps'].tolist()
            if direct_timestamps == api_data['timestamps']:
                validations['timestamps_match'] = True
                print("  ✅ Timestamps match")
            else:
                validations['errors'].append("Timestamps differ")
                print("  ❌ Timestamps differ")
            
            # Language embeddings
            if 'language_embeddings' in direct_data and 'language_embeddings' in api_data:
                direct_lang = direct_data['language_embeddings'].numpy()
                api_lang = np.array(api_data['language_embeddings'])
                if np.allclose(direct_lang, api_lang, rtol=1e-5):
                    validations['language_embeddings_match'] = True
                    print("  ✅ Language embeddings match")
                else:
                    validations['errors'].append("Language embeddings differ")
                    print("  ❌ Language embeddings differ")
            
            # Vision embeddings
            if 'vision_embeddings' in direct_data and 'vision_embeddings' in api_data:
                direct_vision = direct_data['vision_embeddings'].numpy()
                api_vision = np.array(api_data['vision_embeddings'])
                if np.allclose(direct_vision, api_vision, rtol=1e-5):
                    validations['vision_embeddings_match'] = True
                    print("  ✅ Vision embeddings match")
                else:
                    validations['errors'].append("Vision embeddings differ")
                    print("  ❌ Vision embeddings differ")
                    
        except Exception as e:
            validations['errors'].append(f"Validation error: {e}")
            print(f"  ❌ Validation error: {e}")
        
        return validations
    
    def print_summary(self, direct_results: dict, api_results: dict, validation: dict):
        """Print comprehensive benchmark summary."""
        print("\n" + "="*60)
        print("🎯 BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"\n📊 Performance Comparison (Batch Size: {direct_results['batch_size']})")
        print(f"Direct Import:  {direct_results['mean_time_ms']:.1f}ms ± {direct_results['std_time_ms']:.1f}ms")
        print(f"Flask API:      {api_results['mean_time_ms']:.1f}ms ± {api_results['std_time_ms']:.1f}ms")
        
        if direct_results['mean_time_ms'] > 0 and api_results['mean_time_ms'] > 0:
            overhead = api_results['mean_time_ms'] - direct_results['mean_time_ms']
            overhead_pct = (overhead / direct_results['mean_time_ms']) * 100
            print(f"API Overhead:   +{overhead:.1f}ms ({overhead_pct:.1f}%)")
        
        print(f"\n💾 Memory Usage")
        print(f"Direct Import:  {direct_results['mean_memory_mb']:.1f}MB")
        print(f"Flask API:      {api_results['mean_memory_mb']:.1f}MB")
        
        print(f"\n🔍 Data Validation")
        if not validation['errors']:
            print("✅ All data matches between methods")
        else:
            print("❌ Data inconsistencies found:")
            for error in validation['errors']:
                print(f"  • {error}")
        
        print(f"\n🎯 Recommendations")
        if direct_results['mean_time_ms'] < 50:
            print("✅ Direct import meets <50ms target")
        else:
            print("⚠️  Direct import exceeds 50ms target")
            
        if api_results['mean_time_ms'] < 100:
            print("✅ Flask API meets <100ms target")
        else:
            print("⚠️  Flask API exceeds 100ms target")
    
    def _estimate_memory_usage(self, batch_data: dict) -> float:
        """Estimate memory usage of batch data in MB."""
        total_bytes = 0
        
        if 'language_embeddings' in batch_data:
            total_bytes += batch_data['language_embeddings'].nbytes
        if 'vision_embeddings' in batch_data:
            total_bytes += batch_data['vision_embeddings'].nbytes
        if 'locations' in batch_data:
            total_bytes += batch_data['locations'].nbytes
        if 'timestamps' in batch_data:
            total_bytes += batch_data['timestamps'].nbytes
            
        return total_bytes / (1024 * 1024)
    
    def _empty_results(self, method: str, batch_size: int) -> dict:
        """Return empty results structure for failed benchmarks."""
        return {
            'method': method,
            'batch_size': batch_size,
            'num_runs': 0,
            'timings_ms': [],
            'memory_mb': [],
            'mean_time_ms': 0,
            'std_time_ms': 0,
            'mean_memory_mb': 0,
            'last_batch_data': None
        }


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Benchmark ML training data access performance")
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('--runs', type=int, default=5, help='Number of benchmark runs')
    parser.add_argument('--config', type=str, help='Path to dashboard config file')
    parser.add_argument('--flask-url', type=str, default='http://localhost:5000', help='Flask API URL')
    
    args = parser.parse_args()
    
    # Auto-detect config file if not provided
    if not args.config:
        dashboard_path = Path(__file__).parent.parent.parent / "dashboard"
        potential_configs = [
            dashboard_path / "config.json",
            dashboard_path / "dashboard_config.json",
            dashboard_path / "central_florida_config.json"
        ]
        
        for config_path in potential_configs:
            if config_path.exists():
                args.config = str(config_path)
                break
        
        if not args.config:
            print("❌ No config file found. Please specify with --config")
            return 1
    
    print(f"🎯 DeepEarth ML Training Data Benchmark")
    print(f"Config: {args.config}")
    print(f"Flask URL: {args.flask_url}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Runs: {args.runs}")
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark(args.config, args.flask_url)
    
    if not benchmark.setup():
        print("❌ Benchmark setup failed")
        return 1
    
    # Run benchmarks
    direct_results = benchmark.benchmark_direct_import(args.batch_size, args.runs)
    api_results = benchmark.benchmark_flask_api(args.batch_size, args.runs)
    
    # Validate data consistency
    validation = {'errors': ['No data to validate']}
    if (direct_results.get('last_batch_data') and 
        api_results.get('last_batch_data')):
        validation = benchmark.validate_data_consistency(
            direct_results['last_batch_data'],
            api_results['last_batch_data']
        )
    
    # Print summary
    benchmark.print_summary(direct_results, api_results, validation)
    
    return 0


if __name__ == "__main__":
    exit(main())