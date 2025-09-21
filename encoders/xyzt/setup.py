"""
Earth4D Installation Script

Installs Earth4D with its CUDA hash encoder extension.
"""

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
import subprocess
import sys
import os
import shutil

# Read the README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

class BuildHashEncoder(build_ext):
    """Custom build command to compile the CUDA extension."""
    def run(self):
        # Build the hash encoder
        print("\n" + "="*60)
        print("Building HashEncoder CUDA extension...")
        print("="*60)

        hashencoder_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hashencoder')

        # Check if already built
        compiled_path = os.path.join(hashencoder_dir, 'hashencoder_cuda.so')
        if os.path.exists(compiled_path):
            print(f"✓ CUDA extension already compiled: {compiled_path}")
            print(f"  Size: {os.path.getsize(compiled_path) / 1024 / 1024:.1f} MB")
            super().run()
            return

        # Build the extension
        result = subprocess.run(
            [sys.executable, 'setup.py', 'build_ext', '--inplace'],
            cwd=hashencoder_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("⚠️  Warning: Could not build CUDA extension")
            print("Error details:")
            print(result.stderr)
            print("\nEarth4D will compile on first use (may take a few minutes)")
        else:
            print("✅ HashEncoder CUDA extension built successfully!")
            if os.path.exists(compiled_path):
                print(f"  Location: {compiled_path}")
                print(f"  Size: {os.path.getsize(compiled_path) / 1024 / 1024:.1f} MB")

        super().run()

class CustomInstall(install):
    """Custom install command that ensures CUDA extension is built."""
    def run(self):
        self.run_command('build_ext')
        super().run()

setup(
    name='earth4d',
    version='1.0.0',
    author='Earth4D Team',
    author_email='noreply@anthropic.com',
    description='Multi-resolution hash encoding for planetary-scale deep learning',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/earth4d',
    packages=find_packages(),
    py_modules=['earth4d'],
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
        'ninja',  # Required for CUDA compilation
    ],
    extras_require={
        'dev': [
            'pytest',
            'black',
            'flake8',
        ],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: GIS',
    ],
    keywords='deep-learning, spatiotemporal, earth-observation, hash-encoding, cuda, gpu',
    cmdclass={
        'build_ext': BuildHashEncoder,
        'install': CustomInstall,
    },
    package_data={
        'hashencoder': ['*.cu', '*.cuh', '*.cpp', '*.h', 'hashencoder_cuda.so'],
        'analysis': ['*.md', '*.py'],
    },
    include_package_data=True,
    zip_safe=False,  # Cannot zip because of the compiled CUDA extension
)