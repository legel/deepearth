import os
import sys

# Try to import the installed CUDA extension
try:
    import hashencoder_cuda as _backend
except ImportError:
    # Try to import from local directory
    import os
    import sys
    module_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(module_dir, 'hashencoder_cuda.so')

    if os.path.exists(module_path):
        # Load the compiled module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("hashencoder_cuda", module_path)
        _backend = importlib.util.module_from_spec(spec)
        sys.modules["hashencoder_cuda"] = _backend
        spec.loader.exec_module(_backend)
    else:
        # Last resort: JIT compilation (for development only)
        print("Warning: hashencoder_cuda.so not found. Attempting JIT compilation...")
        print("Build with: cd hashencoder && python3 setup.py build_ext --inplace")

        from torch.utils.cpp_extension import load
        from pathlib import Path

        _src_path = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(_src_path, 'build')
        Path(build_dir).mkdir(parents=True, exist_ok=True)

        _backend = load(
            name='hashencoder_cuda',
            sources=[
                os.path.join(_src_path, 'src', 'hashencoder.cu'),
                os.path.join(_src_path, 'src', 'bindings.cpp'),
            ],
            extra_cflags=['-O3', '-std=c++17'],
            extra_cuda_cflags=[
                '-O3', '-std=c++17',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_HALF2_OPERATORS__',
                '--use_fast_math',
            ],
            build_directory=build_dir,
            verbose=False,
        )

__all__ = ['_backend']
