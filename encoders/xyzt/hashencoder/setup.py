#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from setuptools import setup, Extension
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess

def get_cuda_version():
    """Get CUDA version from nvcc"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    version = line.split('release')[1].split(',')[0].strip()
                    return version
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None

def get_compute_capabilities():
    """Auto-detect compute capabilities from available GPUs"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            caps = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    # Convert format like "8.6" to "86"
                    cap = line.strip().replace('.', '')
                    if cap not in caps:
                        caps.append(cap)
            return caps
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Fallback to common architectures
    return ['75']  # RTX 20xx, Tesla V100

def check_prerequisites():
    """Check if all prerequisites are met"""
    errors = []
    
    # Set environment variable to allow CUDA version mismatch
    # This is sometimes needed on HPC systems where exact versions don't match
    os.environ['TORCH_CUDA_VERSION_CHECK'] = '0'
    
    # Check CUDA installation
    cuda_home = os.environ.get('CUDA_HOME')
    if not cuda_home:
        errors.append("CUDA_HOME environment variable is not set")
    elif not os.path.exists(cuda_home):
        errors.append(f"CUDA_HOME path does not exist: {cuda_home}")
    
    # Check nvcc
    cuda_version = get_cuda_version()
    if not cuda_version:
        errors.append("nvcc not found in PATH")
    else:
        print(f"Found CUDA version: {cuda_version}")
        # Warn if version is old
        try:
            major, minor = map(int, cuda_version.split('.')[:2])
            if major < 11 or (major == 11 and minor < 7):
                print(f"Warning: CUDA {cuda_version} detected. Recommended: >= 11.7")
        except ValueError:
            pass
    
    # Check PyTorch CUDA support
    if not torch.cuda.is_available():
        errors.append("PyTorch CUDA support not available")
    else:
        torch_cuda_version = torch.version.cuda
        print(f"PyTorch built with CUDA: {torch_cuda_version}")
    
    if errors:
        print("Prerequisites check failed:")
        for error in errors:
            print(f"  - {error}")
        print("\nTo fix on ASU Sol:")
        print("  module load cuda/11.7")
        print("  export CUDA_HOME=$CUDA_ROOT")
        sys.exit(1)
    
    print("Prerequisites check passed")

if __name__ == '__main__':
    print("Building HashEncoder CUDA extension...")
    
    # Check prerequisites
    check_prerequisites()
    
    # Get source directory
    src_path = Path(__file__).parent / 'src'
    if not src_path.exists():
        print(f"Source directory not found: {src_path}")
        sys.exit(1)
    
    # Source files
    sources = [
        str(src_path / 'hashencoder.cu'),
        str(src_path / 'bindings.cpp')
    ]
    
    # Verify source files exist
    for src in sources:
        if not Path(src).exists():
            print(f"Source file not found: {src}")
            sys.exit(1)
    
    # Get compute capabilities
    compute_caps = get_compute_capabilities()
    arch_list = ';'.join([f'{cap[0]}.{cap[1]}' for cap in compute_caps])
    
    # Set TORCH_CUDA_ARCH_LIST if not already set
    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list
        print(f"Setting TORCH_CUDA_ARCH_LIST={arch_list}")
    
    # Compiler flags
    cxx_flags = ['-O3', '-std=c++17']
    nvcc_flags = [
        '-O3',
        '--use_fast_math',
        '-Xcompiler', '-fPIC',
        '-std=c++17',
        '-allow-unsupported-compiler',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__', 
        '-U__CUDA_NO_HALF2_OPERATORS__',
    ]
    
    # Create extension
    ext_module = CUDAExtension(
        name='encoders.xyzt.hashencoder.hashencoder_backend',
        sources=sources,
        extra_compile_args={
            'cxx': cxx_flags,
            'nvcc': nvcc_flags
        },
        include_dirs=[str(src_path)],
    )
    
    # Setup
    setup(
        name='hashencoder',
        ext_modules=[ext_module],
        cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)},
        zip_safe=False,
    )
    
    print("Build completed successfully!")