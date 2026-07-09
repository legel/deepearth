"""Build the HashEncoder CUDA extension as a Python package."""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import multiprocessing

this_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(this_dir, 'src')

num_threads = str(multiprocessing.cpu_count() * 2)   # parallel nvcc: template instantiation dominates compile time

nvcc_flags = [
    '-O3', '-std=c++17',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
    '--use_fast_math',
    '--expt-relaxed-constexpr',
    '--threads', num_threads,
]

ext_modules = [
    CUDAExtension(
        name='hashencoder_cuda',
        sources=[
            os.path.join(src_dir, 'hashencoder.cu'),
            os.path.join(src_dir, 'bindings.cpp'),
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': nvcc_flags,
        },
    )
]

setup(
    name='hashencoder',
    version='1.0.0',
    description='Multi-resolution hash encoding for Earth4D',
    long_description='',
    py_modules=['hashgrid', 'backend', '__init__'],
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    },
    python_requires='>=3.7',
)
