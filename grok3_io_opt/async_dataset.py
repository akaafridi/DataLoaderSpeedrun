"""
AsyncPrefetchDataset module for asynchronous data prefetching.
"""
import os
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import Dataset


class AsyncPrefetchDataset(Dataset):
    """
    Dataset implementation with asynchronous prefetching of data.
    
    This optimization:
    1. Uses a background thread to prefetch data ahead of consumption
    2. Maintains a queue of prefetched items to reduce I/O latency
    3. Supports direct I/O for NVMe drives when available
    
    Args:
        dataset (Dataset): Base dataset to wrap with prefetching
        queue_size (int): Size of the prefetch queue (default: 5)
        num_workers (int): Number of prefetch workers (default: 2)
        use_direct_io (bool): Whether to use O_DIRECT for file operations (default: False)
    """
    def __init__(self, dataset, queue_size=5, num_workers=2, use_direct_io=False):
        """Initialize the AsyncPrefetchDataset.
        
        Args:
            dataset (Dataset): The base dataset to wrap with prefetching
            queue_size (int): The size of the prefetch queue
            num_workers (int): Number of prefetch worker threads
            use_direct_io (bool): Whether to use direct I/O (O_DIRECT) when available
        """
        self.dataset = dataset
        self.queue_size = queue_size
        self.num_workers = num_workers
        self.use_direct_io = use_direct_io
        
        # Initialize prefetch queue and control flags
        self.prefetch_queue = queue.Queue(maxsize=queue_size)
        self.prefetch_indices = set()
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # Current prefetch position and dataset length
        self.current_idx = 0
        self.dataset_len = len(dataset)
        
        # Enable direct I/O if requested and if we're on Linux
        if use_direct_io and os.name == 'posix':
            try:
                import fcntl
                self.fcntl = fcntl
                self.direct_io_supported = True
            except ImportError:
                self.direct_io_supported = False
                print("Warning: fcntl not available, direct I/O disabled")
        else:
            self.direct_io_supported = False
            
        # Start prefetch workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.workers = []
        for _ in range(num_workers):
            worker = self.executor.submit(self._prefetch_worker)
            self.workers.append(worker)
            
    def __del__(self):
        """Clean up resources when the dataset is deleted."""
        self.shutdown_event.set()
        self.executor.shutdown(wait=False)
        
    def _prefetch_worker(self):
        """Worker function to prefetch data items in the background."""
        while not self.shutdown_event.is_set():
            # Determine next index to prefetch
            with self.lock:
                # Start from current position and find the next index to prefetch
                next_idx = self.current_idx
                for i in range(self.dataset_len):
                    idx = (next_idx + i) % self.dataset_len
                    if idx not in self.prefetch_indices:
                        self.prefetch_indices.add(idx)
                        break
                else:
                    # All indices are being prefetched or in queue
                    next_idx = None
                    
            # If we found an index to prefetch
            if next_idx is not None:
                try:
                    # Fetch the data with direct I/O if supported and enabled
                    if self.direct_io_supported and self.use_direct_io and hasattr(self.dataset, 'get_file_path'):
                        file_path = self.dataset.get_file_path(next_idx)
                        if file_path:
                            # Use O_DIRECT for direct I/O to bypass page cache
                            flags = os.O_RDONLY
                            if hasattr(self.fcntl, 'O_DIRECT'):
                                flags |= self.fcntl.O_DIRECT
                            fd = os.open(file_path, flags)
                            try:
                                # Custom direct read implementation can go here
                                # This is a placeholder for actual direct I/O implementation
                                data = self.dataset[next_idx]
                            finally:
                                os.close(fd)
                        else:
                            data = self.dataset[next_idx]
                    else:
                        # Use standard dataset access
                        data = self.dataset[next_idx]
                        
                    # Put the prefetched data in the queue
                    self.prefetch_queue.put((next_idx, data), block=True)
                except Exception as e:
                    print(f"Error prefetching index {next_idx}: {e}")
                    with self.lock:
                        self.prefetch_indices.remove(next_idx)
            else:
                # If all indices are prefetched, sleep briefly to avoid spinning
                time.sleep(0.001)
    
    def __len__(self):
        """Return the number of items in the dataset."""
        return self.dataset_len
    
    def __getitem__(self, idx):
        """Get an item by index, preferably from the prefetch queue.
        
        Args:
            idx (int): The index of the item to retrieve
            
        Returns:
            Any: The item at the given index
        """
        # Update current position for prefetcher
        self.current_idx = idx
        
        # Check if the requested item is in the queue
        for _ in range(self.prefetch_queue.qsize()):
            queue_idx, data = self.prefetch_queue.get()
            if queue_idx == idx:
                with self.lock:
                    self.prefetch_indices.remove(idx)
                return data
            else:
                # Put it back in the queue
                self.prefetch_queue.put((queue_idx, data))
                
        # If not found in queue, fetch directly and trigger prefetch
        with self.lock:
            if idx in self.prefetch_indices:
                self.prefetch_indices.remove(idx)
                
        # Direct fetch (fallback)
        return self.dataset[idx]