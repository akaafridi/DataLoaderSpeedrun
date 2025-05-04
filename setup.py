import os
from setuptools import setup, find_packages, Extension

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Check if PyTorch is installed for C++ extension
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension
    
    # Define C++ extension
    cpp_extension = [
        CppExtension(
            name="grok3_io_opt.fastfetch",
            sources=["grok3_io_opt/cpp_extension/fastfetch.cpp"],
            extra_compile_args=["-O3"],
        )
    ]
    cmdclass = {"build_ext": BuildExtension}
except ImportError:
    cpp_extension = []
    cmdclass = {}
    print("PyTorch not installed. C++ extensions will not be built.")

# Package setup
setup(
    name="grok3-io-opt",
    version="0.1.0",
    author="Replit Grok IO",
    author_email="example@example.com",
    description="PyTorch DataLoader optimization for improved I/O performance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/grok3-io-opt",
    packages=find_packages(include=["grok3_io_opt", "grok3_io_opt.*"]),
    ext_modules=cpp_extension,
    cmdclass=cmdclass,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-benchmark", "black", "flake8", "isort"],
    },
    entry_points={
        "console_scripts": [
            "grok3-io-opt=grok3_io_opt.cli:main",
        ],
    },
)