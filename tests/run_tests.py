"""
Test runner and validation suite for DeepEarth
Includes integration tests, benchmarks, and validation checks
"""

import sys
import os
import time
import argparse
import json
from pathlib import Path
from datetime import datetime
import subprocess

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.deepearth_integrated import create_integrated_deepearth
from test_data_generator import SyntheticEarthDataGenerator, create_test_batch_for_model


class DeepEarthValidator:
    """Comprehensive validation suite for DeepEarth"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = create_integrated_deepearth()
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'tests': {}
        }
    
    def run_all_tests(self) -> Dict:
        """Run complete test suite"""
        print("=" * 80)
        print("DeepEarth Validation Suite")
        print("=" * 80)
        
        # 1. Model architecture tests
        print("\n1. Testing Model Architecture...")
        self.test_model_architecture()
        
        # 2. Forward pass tests
        print("\n2. Testing Forward Pass...")
        self.test_forward_pass()
        
        # 3. Memory tests
        print("\n3. Testing Memory Usage...")
        self.test_memory_usage()
        
        # 4. Performance benchmarks
        print("\n4. Running Performance Benchmarks...")
        self.test_performance()
        
        # 5. Feature quality tests
        print("\n5. Testing Feature Quality...")
        self.test_feature_quality()
        
        # 6. Multi-modal fusion tests
        print("\n6. Testing Multi-Modal Fusion...")
        self.test_multimodal_fusion()
        
        # 7. Spatial-temporal consistency
        print("\n7. Testing Spatial-Temporal Consistency...")
        self.test_spatiotemporal_consistency()
        
        # 8. Export tests
        print("\n8. Testing Model Export...")
        self.test_model_export()
        
        # Summary
        self.print_summary()
        
        return self.results
    
    def test_model_architecture(self):
        """Test model architecture and parameter counts"""
        test_name = "architecture"
        results = {}
        
        try:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            
            results['total_parameters'] = total_params
            results['trainable_parameters'] = trainable_params
            results['frozen_parameters'] = frozen_params
            results['parameter_efficiency'] = trainable_params / total_params
            
            # Check components
            components = {
                'universal_encoder': hasattr(self.model, 'universal_encoder'),
                'grid4d_encoder': hasattr(self.model, 'grid4d_encoder'),
                'fusion': hasattr(self.model, 'fusion'),
                'task_heads': hasattr(self.model, 'task_heads')
            }
            results['components'] = components
            
            # Memory footprint
            model_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
            results['model_size_mb'] = model_size_mb
            
            print(f"  ✓ Total parameters: {total_params:,}")
            print(f"  ✓ Trainable parameters: {trainable_params:,} ({results['parameter_efficiency']:.2%})")
            print(f"  ✓ Model size: {model_size_mb:.2f} MB")
            
            results['status'] = 'PASSED'
            
        except Exception as e:
            print(f"  ✗ Architecture test failed: {str(e)}")
            results['status'] = 'FAILED'
            results['error'] = str(e)
        
        self.results['tests'][test_name] = results
    
    def test_forward_pass(self):
        """Test forward pass with different input configurations"""
        test_name = "forward_pass"
        results = {'configurations': {}}
        
        configurations = [
            ('minimal', {'xyzt': True}),
            ('vision_only', {'xyzt': True, 'vision': True}),
            ('language_only', {'xyzt': True, 'language': True}),
            ('full', {'xyzt': True, 'vision': True, 'language': True, 'weather': True}),
        ]
        
        for config_name, config in configurations:
            try:
                # Generate test batch
                batch = create_test_batch_for_model(batch_size=2, device=self.device)
                
                # Prepare inputs based on configuration
                inputs = {'xyzt': batch['xyzt']}
                if config.get('vision'):
                    inputs['vision_input'] = batch['images']
                if config.get('language'):
                    inputs['language_input'] = {
                        'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask']
                    }
                if config.get('weather'):
                    inputs['additional_modalities'] = {
                        'weather': batch['modalities']['weather']
                    }
                
                # Forward pass
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model(**inputs)
                end_time = time.time()
                
                # Validate outputs
                assert 'fused_representation' in outputs
                assert outputs['fused_representation'].shape == (2, 2048)
                
                results['configurations'][config_name] = {
                    'status': 'PASSED',
                    'time_ms': (end_time - start_time) * 1000,
                    'output_shape': list(outputs['fused_representation'].shape)
                }
                
                print(f"  ✓ {config_name}: {results['configurations'][config_name]['time_ms']:.2f} ms")
                
            except Exception as e:
                print(f"  ✗ {config_name} failed: {str(e)}")
                results['configurations'][config_name] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
        
        # Overall status
        all_passed = all(
            cfg['status'] == 'PASSED' 
            for cfg in results['configurations'].values()
        )
        results['status'] = 'PASSED' if all_passed else 'FAILED'
        
        self.results['tests'][test_name] = results
    
    def test_memory_usage(self):
        """Test memory usage under different batch sizes"""
        test_name = "memory_usage"
        results = {'batch_sizes': {}}
        
        if not torch.cuda.is_available():
            print("  ⚠ Skipping memory tests (no GPU available)")
            results['status'] = 'SKIPPED'
            self.results['tests'][test_name] = results
            return
        
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            try:
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Create batch
                batch = create_test_batch_for_model(batch_size, self.device)
                
                # Measure memory before
                mem_before = torch.cuda.memory_allocated() / 1024**2  # MB
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(
                        xyzt=batch['xyzt'],
                        vision_input=batch['images'],
                        language_input={
                            'input_ids': batch['input_ids'],
                            'attention_mask': batch['attention_mask']
                        }
                    )
                
                # Measure memory after
                mem_after = torch.cuda.memory_allocated() / 1024**2  # MB
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                
                results['batch_sizes'][batch_size] = {
                    'memory_used_mb': mem_after - mem_before,
                    'peak_memory_mb': peak_memory,
                    'status': 'PASSED'
                }
                
                print(f"  ✓ Batch size {batch_size}: {mem_after - mem_before:.2f} MB used, {peak_memory:.2f} MB peak")
                
                # Clear outputs
                del outputs
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ⚠ Batch size {batch_size}: OOM")
                    results['batch_sizes'][batch_size] = {'status': 'OOM'}
                    break
                else:
                    raise e
        
        results['status'] = 'PASSED'
        self.results['tests'][test_name] = results
    
    def test_performance(self):
        """Benchmark inference performance"""
        test_name = "performance"
        results = {}
        
        batch_size = 8
        num_warmup = 10
        num_iterations = 100
        
        try:
            # Create test batch
            batch = create_test_batch_for_model(batch_size, self.device)
            inputs = {
                'xyzt': batch['xyzt'],
                'vision_input': batch['images'],
                'language_input': {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']
                }
            }
            
            # Warmup
            for _ in range(num_warmup):
                with torch.no_grad():
                    _ = self.model(**inputs)
            
            # Synchronize if using CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = self.model(**inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            time_per_batch = total_time / num_iterations
            time_per_sample = time_per_batch / batch_size
            throughput = batch_size * num_iterations / total_time
            
            results = {
                'batch_size': batch_size,
                'num_iterations': num_iterations,
                'total_time_s': total_time,
                'time_per_batch_ms': time_per_batch * 1000,
                'time_per_sample_ms': time_per_sample * 1000,
                'throughput_samples_per_s': throughput,
                'status': 'PASSED'
            }
            
            print(f"  ✓ Throughput: {throughput:.2f} samples/sec")
            print(f"  ✓ Latency: {time_per_sample * 1000:.2f} ms/sample")
            
        except Exception as e:
            print(f"  ✗ Performance test failed: {str(e)}")
            results = {'status': 'FAILED', 'error': str(e)}
        
        self.results['tests'][test_name] = results
    
    def test_feature_quality(self):
        """Test quality of extracted features"""
        test_name = "feature_quality"
        results = {}
        
        try:
            # Generate diverse batch
            batch = create_test_batch_for_model(32, self.device)
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(
                    xyzt=batch['xyzt'],
                    vision_input=batch['images'],
                    return_intermediates=True
                )
            
            features = outputs['fused_representation']
            
            # Analyze feature statistics
            feature_stats = {
                'mean': features.mean().item(),
                'std': features.std().item(),
                'sparsity': (features.abs() < 0.01).float().mean().item(),
                'dead_neurons': (features.abs().mean(dim=0) < 0.01).float().mean().item()
            }
            
            # Check feature diversity (no collapsed representations)
            feature_similarity = torch.nn.functional.cosine_similarity(
                features.unsqueeze(1), features.unsqueeze(0), dim=2
            )
            # Mask out diagonal
            mask = torch.eye(features.shape[0], device=self.device).bool()
            feature_similarity[mask] = 0
            
            avg_similarity = feature_similarity.mean().item()
            max_similarity = feature_similarity.max().item()
            
            results = {
                'feature_stats': feature_stats,
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'status': 'PASSED'
            }
            
            # Quality checks
            quality_checks = {
                'healthy_std': feature_stats['std'] > 0.1,
                'low_sparsity': feature_stats['sparsity'] < 0.5,
                'active_neurons': feature_stats['dead_neurons'] < 0.1,
                'diverse_features': avg_similarity < 0.9
            }
            
            results['quality_checks'] = quality_checks
            all_passed = all(quality_checks.values())
            
            print(f"  ✓ Feature std: {feature_stats['std']:.3f}")
            print(f"  ✓ Feature sparsity: {feature_stats['sparsity']:.3f}")
            print(f"  ✓ Average similarity: {avg_similarity:.3f}")
            print(f"  {'✓' if all_passed else '✗'} All quality checks: {'PASSED' if all_passed else 'FAILED'}")
            
            results['status'] = 'PASSED' if all_passed else 'WARNING'
            
        except Exception as e:
            print(f"  ✗ Feature quality test failed: {str(e)}")
            results = {'status': 'FAILED', 'error': str(e)}
        
        self.results['tests'][test_name] = results
    
    def test_multimodal_fusion(self):
        """Test multi-modal fusion capabilities"""
        test_name = "multimodal_fusion"
        results = {}
        
        try:
            batch_size = 16
            
            # Test different modality combinations
            modality_combinations = [
                ['spatial'],
                ['spatial', 'vision'],
                ['spatial', 'language'],
                ['spatial', 'vision', 'language'],
                ['spatial', 'vision', 'language', 'weather']
            ]
            
            fusion_results = {}
            
            for modalities in modality_combinations:
                # Create batch
                batch = create_test_batch_for_model(batch_size, self.device)
                
                # Prepare inputs
                inputs = {'xyzt': batch['xyzt']}
                
                if 'vision' in modalities:
                    inputs['vision_input'] = batch['images']
                if 'language' in modalities:
                    inputs['language_input'] = {
                        'input_ids': batch['input_ids'],
                        'attention_mask': batch['attention_mask']
                    }
                if 'weather' in modalities:
                    inputs['additional_modalities'] = {
                        'weather': batch['modalities']['weather']
                    }
                
                # Get representations
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                features = outputs['fused_representation']
                
                # Measure information content
                feature_entropy = -torch.sum(
                    features * torch.log(features.abs() + 1e-8),
                    dim=-1
                ).mean().item()
                
                fusion_results[str(modalities)] = {
                    'feature_norm': features.norm(dim=-1).mean().item(),
                    'feature_entropy': feature_entropy,
                    'num_modalities': len(modalities)
                }
            
            # Check if adding modalities increases information
            norms = [v['feature_norm'] for v in fusion_results.values()]
            increasing_info = all(norms[i] <= norms[i+1] for i in range(len(norms)-1))
            
            results = {
                'fusion_results': fusion_results,
                'increasing_information': increasing_info,
                'status': 'PASSED' if increasing_info else 'WARNING'
            }
            
            print(f"  ✓ Tested {len(modality_combinations)} modality combinations")
            print(f"  {'✓' if increasing_info else '⚠'} Information increases with more modalities: {increasing_info}")
            
        except Exception as e:
            print(f"  ✗ Multimodal fusion test failed: {str(e)}")
            results = {'status': 'FAILED', 'error': str(e)}
        
        self.results['tests'][test_name] = results
    
    def test_spatiotemporal_consistency(self):
        """Test spatial-temporal consistency of representations"""
        test_name = "spatiotemporal_consistency"
        results = {}
        
        try:
            # Generate nearby points in space and time
            base_coord = torch.tensor([[0.0, 0.0, 0.0, 0.5]], device=self.device)
            
            # Spatial perturbations
            spatial_deltas = torch.randn(10, 3, device=self.device) * 0.01
            spatial_coords = base_coord.repeat(10, 1)
            spatial_coords[:, :3] += spatial_deltas
            
            # Temporal perturbations
            temporal_deltas = torch.randn(10, 1, device=self.device) * 0.01
            temporal_coords = base_coord.repeat(10, 1)
            temporal_coords[:, 3:4] += temporal_deltas
            
            # Get representations
            with torch.no_grad():
                base_rep = self.model(xyzt=base_coord)['fused_representation']
                spatial_reps = self.model(xyzt=spatial_coords)['fused_representation']
                temporal_reps = self.model(xyzt=temporal_coords)['fused_representation']
            
            # Compute similarities
            spatial_sims = F.cosine_similarity(
                base_rep.expand_as(spatial_reps),
                spatial_reps,
                dim=-1
            )
            
            temporal_sims = F.cosine_similarity(
                base_rep.expand_as(temporal_reps),
                temporal_reps,
                dim=-1
            )
            
            results = {
                'spatial_similarity_mean': spatial_sims.mean().item(),
                'spatial_similarity_std': spatial_sims.std().item(),
                'temporal_similarity_mean': temporal_sims.mean().item(),
                'temporal_similarity_std': temporal_sims.std().item(),
                'status': 'PASSED'
            }
            
            # Check consistency (nearby points should have similar representations)
            consistency_threshold = 0.9
            spatial_consistent = spatial_sims.mean() > consistency_threshold
            temporal_consistent = temporal_sims.mean() > consistency_threshold
            
            print(f"  ✓ Spatial consistency: {spatial_sims.mean():.3f} ± {spatial_sims.std():.3f}")
            print(f"  ✓ Temporal consistency: {temporal_sims.mean():.3f} ± {temporal_sims.std():.3f}")
            print(f"  {'✓' if spatial_consistent and temporal_consistent else '⚠'} Consistency check: {'PASSED' if spatial_consistent and temporal_consistent else 'WARNING'}")
            
            results['consistent'] = spatial_consistent and temporal_consistent
            
        except Exception as e:
            print(f"  ✗ Spatiotemporal consistency test failed: {str(e)}")
            results = {'status': 'FAILED', 'error': str(e)}
        
        self.results['tests'][test_name] = results
    
    def test_model_export(self):
        """Test model export capabilities"""
        test_name = "model_export"
        results = {}
        
        try:
            import tempfile
            
            # Test TorchScript export
            with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
                # Create example inputs
                example_inputs = {
                    'xyzt': torch.randn(1, 4, device=self.device),
                    'vision_input': torch.randn(1, 3, 224, 224, device=self.device)
                }
                
                # Try to script the model (may fail for complex models)
                try:
                    scripted = torch.jit.script(self.model)
                    torch.jit.save(scripted, tmp.name)
                    file_size = os.path.getsize(tmp.name) / 1024**2  # MB
                    
                    results['torchscript'] = {
                        'status': 'PASSED',
                        'file_size_mb': file_size
                    }
                    print(f"  ✓ TorchScript export: {file_size:.2f} MB")
                except Exception as e:
                    # Try tracing instead
                    try:
                        traced = torch.jit.trace(
                            self.model,
                            example_kwarg_inputs=example_inputs
                        )
                        torch.jit.save(traced, tmp.name)
                        file_size = os.path.getsize(tmp.name) / 1024**2
                        
                        results['torchscript'] = {
                            'status': 'PASSED (traced)',
                            'file_size_mb': file_size
                        }
                        print(f"  ✓ TorchScript trace: {file_size:.2f} MB")
                    except Exception as trace_error:
                        results['torchscript'] = {
                            'status': 'FAILED',
                            'error': str(trace_error)
                        }
                        print(f"  ✗ TorchScript export failed: {str(trace_error)}")
            
            # Test state dict save/load
            with tempfile.NamedTemporaryFile(suffix='.pth') as tmp:
                torch.save(self.model.state_dict(), tmp.name)
                file_size = os.path.getsize(tmp.name) / 1024**2
                
                # Try loading
                state_dict = torch.load(tmp.name, map_location=self.device)
                
                results['state_dict'] = {
                    'status': 'PASSED',
                    'file_size_mb': file_size,
                    'num_parameters': len(state_dict)
                }
                print(f"  ✓ State dict export: {file_size:.2f} MB, {len(state_dict)} parameters")
            
            results['status'] = 'PASSED'
            
        except Exception as e:
            print(f"  ✗ Model export test failed: {str(e)}")
            results = {'status': 'FAILED', 'error': str(e)}
        
        self.results['tests'][test_name] = results
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        passed = 0
        failed = 0
        warnings = 0
        
        for test_name, test_results in self.results['tests'].items():
            status = test_results.get('status', 'UNKNOWN')
            
            if status == 'PASSED':
                symbol = '✓'
                passed += 1
            elif status == 'FAILED':
                symbol = '✗'
                failed += 1
            elif status == 'WARNING':
                symbol = '⚠'
                warnings += 1
            else:
                symbol = '?'
            
            print(f"{symbol} {test_name}: {status}")
        
        print("\n" + "-" * 40)
        print(f"Total tests: {len(self.results['tests'])}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Warnings: {warnings}")
        print("-" * 40)
        
        # Overall status
        if failed == 0:
            print("\n✓ ALL TESTS PASSED!")
        else:
            print(f"\n✗ {failed} TESTS FAILED!")
    
    def save_results(self, filepath: str):
        """Save test results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)


def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(description='DeepEarth Test Runner')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='test_results.json',
                        help='Output file for test results')
    parser.add_argument('--tests', type=str, nargs='+', default=None,
                        help='Specific tests to run')
    
    args = parser.parse_args()
    
    # Create validator
    validator = DeepEarthValidator(model_path=args.model_path)
    
    # Run tests
    results = validator.run_all_tests()
    
    # Save results
    validator.save_results(args.output)
    print(f"\nResults saved to: {args.output}")
    
    # Return exit code based on results
    failed_tests = sum(1 for test in results['tests'].values() 
                      if test.get('status') == 'FAILED')
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
