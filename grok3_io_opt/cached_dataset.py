"""
CachedDataset module for optimized memory access.
"""
import torch
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    """
    Dataset implementation that preloads and pins all data in memory.
    
    This optimization:
    1. Preloads all data into memory at initialization
    2. Pins data in memory (page-locked) to enable faster GPU transfers
    3. Avoids memory copies during batch fetch
    
    Args:
        data (torch.Tensor): The tensor data to cache
        pin_memory (bool): Whether to pin the data in memory (default: True)
    """
    def __init__(self, data, pin_memory=True):
        """Initialize the CachedDataset with tensor data.
        
        Args:
            data (torch.Tensor): The data to be cached
            pin_memory (bool): Whether to pin memory for faster GPU transfers
        """
        self.data = data
        if pin_memory and torch.cuda.is_available():
            # Pin memory for faster host-to-device transfers
            self.data = self.data.pin_memory()
        self.device = 'cpu'
        
    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get an item by index.
        
        Args:
            idx (int): The index of the item to retrieve
            
        Returns:
            torch.Tensor: The tensor at the given index
        """
        # Direct tensor access without copying
        if self.device != 'cpu':
            return self.data[idx].to(self.device, non_blocking=True)
        return self.data[idx]
    
    def to_device(self, device):
        """
        Move the cached data to a specific device.
        
        Args:
            device: The device to move the data to (e.g., 'cuda:0')
        """
        if device != 'cpu':
            self.data = self.data.to(device, non_blocking=True)
        self.device = device
        return self