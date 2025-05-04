"""
Tests for CachedDataset implementation.
"""
import pytest
import torch
from torch.utils.data import DataLoader

from grok3_io_opt import CachedDataset


def test_cached_dataset_creation():
    """Test creating a CachedDataset."""
    # Create sample data
    data = torch.randn(100, 10)
    
    # Create CachedDataset
    dataset = CachedDataset(data)
    
    # Check length
    assert len(dataset) == 100


def test_cached_dataset_getitem():
    """Test __getitem__ method of CachedDataset."""
    # Create sample data
    data = torch.randn(100, 10)
    
    # Create CachedDataset
    dataset = CachedDataset(data)
    
    # Get item
    item = dataset[0]
    
    # Check item
    assert torch.all(torch.eq(item, data[0]))


def test_cached_dataset_dataloader():
    """Test CachedDataset with DataLoader."""
    # Create sample data
    data = torch.randn(100, 10)
    
    # Create CachedDataset
    dataset = CachedDataset(data)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=10)
    
    # Check if batches can be retrieved
    batch_count = 0
    for batch in dataloader:
        # Check batch shape
        assert batch.shape == torch.Size([10, 10])
        batch_count += 1
    
    # Check batch count
    assert batch_count == 10


def test_cached_dataset_to_device():
    """Test to_device method of CachedDataset."""
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping device test")
    
    # Create sample data
    data = torch.randn(100, 10)
    
    # Create CachedDataset
    dataset = CachedDataset(data)
    
    # Move to device
    device = torch.device("cuda:0")
    dataset.to_device(device)
    
    # Check device
    assert dataset[0].device.type == "cuda"