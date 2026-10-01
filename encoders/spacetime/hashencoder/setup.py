"""Build the HashEncoder CUDA extension."""

from multiprocessing import cpu_count
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

nvcc_flags = [
    '-O3', '-std=c++17',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
    '--use_fast_math',
    '--expt-relaxed-constexpr',
    '--threads', str(cpu_count() * 2),
]

ext_modules = [
    CUDAExtension(
        name='hashencoder_cuda',
        sources=[
            'src/hashencoder.cu',
            'src/bindings.cpp',
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
