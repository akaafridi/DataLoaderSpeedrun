#!/bin/bash
set -e

echo "=== Building grok3-io-opt package ==="
pip install -e .

echo "=== Testing CachedDataset implementation ==="
python -c "
import torch
from grok3_io_opt import CachedDataset
from torch.utils.data import DataLoader

# Create sample data
data = torch.randn(100, 10)

# Create CachedDataset
dataset = CachedDataset(data)
print(f'Dataset length: {len(dataset)}')

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=10)

# Iterate through batches
for i, batch in enumerate(dataloader):
    print(f'Batch {i} shape: {batch.shape}')
    if i >= 2:
        break
"

echo "=== Testing AsyncPrefetchDataset implementation ==="
python -c "
import torch
from grok3_io_opt import AsyncPrefetchDataset
from torch.utils.data import TensorDataset, DataLoader

# Create sample data
data = torch.randn(100, 10)
base_dataset = TensorDataset(data)

# Create AsyncPrefetchDataset
dataset = AsyncPrefetchDataset(base_dataset, queue_size=5, num_workers=2)
print(f'Dataset length: {len(dataset)}')

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=10)

# Iterate through batches
for i, batch in enumerate(dataloader):
    print(f'Batch {i} shape: {batch[0].shape}')
    if i >= 2:
        break

# Clean up
dataset.__del__()
"

echo "=== Running benchmark ==="
python -m grok3_io_opt.benchmark --multi-seed

echo "=== All tests completed successfully ==="