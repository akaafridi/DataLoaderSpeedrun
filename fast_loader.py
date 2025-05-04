import torch
from torch.utils.data import Dataset
import numpy as np

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
        self.data = data
        
        # Create a copy of the data to avoid modifying the original
        if isinstance(data, torch.Tensor):
            # For PyTorch tensors, clone and optionally pin memory
            self.cached_data = data.clone()
            if pin_memory and torch.cuda.is_available():
                self.cached_data = self.cached_data.pin_memory()
        elif isinstance(data, np.ndarray):
            # For NumPy arrays, convert to PyTorch tensor
            self.cached_data = torch.from_numpy(data.copy())
            if pin_memory and torch.cuda.is_available():
                self.cached_data = self.cached_data.pin_memory()
        else:
            raise TypeError("Data must be either a PyTorch tensor or NumPy array")
            
        # Pre-compute indices for faster access
        self.indices = list(range(len(self.cached_data)))
    
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, idx):
        # Return a view of the cached data (avoids copy when possible)
        # Using .as_strided() for advanced users to create a view without copy
        # Standard approach would be to use tensor[idx]
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer")
            
        return self.cached_data[idx]
    
    def to_device(self, device):
        """
        Move the cached data to a specific device.
        
        Args:
            device: The device to move the data to (e.g., 'cuda:0')
        """
        self.cached_data = self.cached_data.to(device)
        return self
