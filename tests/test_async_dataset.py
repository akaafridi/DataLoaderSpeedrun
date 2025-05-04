"""
Tests for AsyncPrefetchDataset implementation.
"""
import time

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from grok3_io_opt import AsyncPrefetchDataset


def test_async_dataset_creation():
    """Test creating an AsyncPrefetchDataset."""
    # Create sample data
    data = torch.randn(100, 10)
    base_dataset = TensorDataset(data)
    
    # Create AsyncPrefetchDataset
    dataset = AsyncPrefetchDataset(base_dataset, queue_size=3, num_workers=1)
    
    # Check length
    assert len(dataset) == 100
    
    # Clean up
    dataset.__del__()


def test_async_dataset_getitem():
    """Test __getitem__ method of AsyncPrefetchDataset."""
    # Create sample data
    data = torch.randn(100, 10)
    base_dataset = TensorDataset(data)
    
    # Create AsyncPrefetchDataset
    dataset = AsyncPrefetchDataset(base_dataset, queue_size=3, num_workers=1)
    
    # Wait for prefetching to start
    time.sleep(0.1)
    
    # Get item
    item = dataset[0]
    
    # Check item
    assert torch.all(torch.eq(item[0], data[0]))
    
    # Clean up
    dataset.__del__()


def test_async_dataset_dataloader():
    """Test AsyncPrefetchDataset with DataLoader."""
    # Create sample data
    data = torch.randn(100, 10)
    base_dataset = TensorDataset(data)
    
    # Create AsyncPrefetchDataset
    dataset = AsyncPrefetchDataset(base_dataset, queue_size=10, num_workers=2)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=10)
    
    # Check if batches can be retrieved
    batch_count = 0
    for batch in dataloader:
        # Check batch shape
        assert batch[0].shape == torch.Size([10, 10])
        batch_count += 1
    
    # Check batch count
    assert batch_count == 10
    
    # Clean up
    dataset.__del__()