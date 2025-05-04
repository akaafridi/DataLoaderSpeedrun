"""
Grok3 I/O Optimization for PyTorch DataLoader
---------------------------------------------

This module provides optimized dataset classes and utility functions
for improving I/O performance in PyTorch DataLoaders.
"""

from .cached_dataset import CachedDataset
from .async_dataset import AsyncPrefetchDataset

try:
    from .fastfetch import fetch_tensor_zero_copy
except ImportError:
    def fetch_tensor_zero_copy(*args, **kwargs):
        raise ImportError(
            "C++ extension not available. Run 'pip install --editable .' first."
        )

__version__ = "0.1.0"
__all__ = ["CachedDataset", "AsyncPrefetchDataset", "fetch_tensor_zero_copy"]