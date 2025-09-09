import os
import sys
import importlib.util

# Try to import the pre-built extension first
try:
    # Import the compiled extension directly
    from . import hashencoder_backend as _backend
    print("Using pre-built hashencoder extension")
except ImportError as e:
    print(f"Pre-built extension not found: {e}")
    print("Falling back to JIT compilation...")
    
    # Fallback to JIT compilation
    from torch.utils.cpp_extension import load
    from pathlib import Path
    
    Path('./tmp_build/').mkdir(parents=True, exist_ok=True)
    
    _src_path = os.path.dirname(os.path.abspath(__file__))
    
    _backend = load(name='_hash_encoder',
                    extra_cflags=['-O3', '-std=c++17'],
                    extra_cuda_cflags=[
                        '-O3', '-std=c++17', '-allow-unsupported-compiler',
                        '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
                        '--use_fast_math', '-Xcompiler', '-fPIC'
                    ],
                    sources=[os.path.join(_src_path, 'src', f) for f in [
                        'hashencoder.cu',
                        'bindings.cpp',
                    ]],
                    build_directory='./tmp_build/',
                    verbose=True,
                    )

__all__ = ['_backend']