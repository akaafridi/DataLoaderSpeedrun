"""
Benchmark script for grok3-io-opt package.

This script measures the performance and memory usage of different data loading approaches:
1. Baseline PyTorch DataLoader
2. CachedDataset optimization
3. AsyncPrefetchDataset optimization
4. C++ extension optimization (if available)

Usage:
    python -m grok3_io_opt.benchmark [--multi-seed] [--batch-size BATCH_SIZE] [--dataset-size DATASET_SIZE]
"""

import argparse
import logging
import time
import random
import statistics
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import psutil
import torch
from torch.utils.data import Dataset, DataLoader

from grok3_io_opt import CachedDataset, AsyncPrefetchDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try importing the C++ extension
try:
    from grok3_io_opt.fastfetch import fetch_tensor_zero_copy
    HAS_CPP_EXTENSION = True
except ImportError:
    HAS_CPP_EXTENSION = False
    logger.warning("C++ extension not available. Run 'pip install -e .' first.")

# Set default parameters
DEFAULT_BATCH_SIZE = 64
DEFAULT_DATASET_SIZE = (2000, 50)  # Rows, Features
DEFAULT_NUM_WORKERS = 0
DEFAULT_EPOCHS = 2
DEFAULT_SEEDS = [42, 123, 456]


class BaselineDataset(Dataset):
    """Basic dataset that loads tensors from memory."""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class CppExtensionDataset(Dataset):
    """Dataset that uses C++ extension for zero-copy data fetching."""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if HAS_CPP_EXTENSION:
            return fetch_tensor_zero_copy(self.data, idx)
        else:
            return self.data[idx]


def get_memory_usage():
    """Get current memory usage of the process in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def create_sample_data(seed=None, dataset_size=None):
    """Create sample data for benchmarking."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    if dataset_size is None:
        dataset_size = DEFAULT_DATASET_SIZE
        
    logger.info(f"Creating sample data: {dataset_size[0]}x{dataset_size[1]} tensor")
    return torch.randn(dataset_size)


def benchmark_dataset(dataset_class, data=None, seed=None, batch_size=None, epochs=None, name="Unnamed"):
    """Generic benchmark for any dataset implementation."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
        
    if epochs is None:
        epochs = DEFAULT_EPOCHS
        
    if data is None:
        data = create_sample_data(seed)
        
    # Create dataset
    initial_memory = get_memory_usage()
    dataset = dataset_class(data)
    dataset_memory = get_memory_usage()
    memory_increase = dataset_memory - initial_memory
    logger.info(f"{name} Dataset Memory: {dataset_memory:.2f} MB (Increase: {memory_increase:.2f} MB)")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=DEFAULT_NUM_WORKERS
    )
    
    # Execute benchmark
    logger.info(f"Running {name} benchmark...")
    times = []
    peak_memory = dataset_memory
    
    total_start = time.time()
    for epoch in range(epochs):
        start = time.time()
        for batch_idx, batch in enumerate(dataloader):
            # Simulate some processing
            _ = batch.mean()
            
        end = time.time()
        elapsed = (end - start) * 1000  # Convert to ms
        times.append(elapsed)
        logger.info(f"Epoch {epoch+1}/{epochs} - Time: {elapsed:.2f}ms")
        
        # Track memory
        mem = get_memory_usage()
        if mem > peak_memory:
            peak_memory = mem
            
    total_end = time.time()
    total_time = (total_end - total_start) * 1000  # Convert to ms
    
    # Calculate metrics
    items_per_second = (len(dataset) * epochs) / ((total_end - total_start))
    ms_per_batch = total_time / (len(dataloader) * epochs)
    
    # Memory increase from initial
    memory_increase = peak_memory - initial_memory
    
    # Log results
    logger.info(f"{name} Results:")
    logger.info(f"  Total time: {total_time:.2f}ms")
    logger.info(f"  Items/second: {items_per_second:.2f}")
    logger.info(f"  Ms/batch: {ms_per_batch:.2f}ms")
    logger.info(f"  Peak memory: {peak_memory:.2f} MB (Increase: {memory_increase:.2f} MB)")
    
    # Return metrics
    return total_time, items_per_second, ms_per_batch, memory_increase


def benchmark_baseline(data=None, seed=None, batch_size=None, epochs=None):
    """Benchmark baseline DataLoader."""
    return benchmark_dataset(
        BaselineDataset, 
        data=data, 
        seed=seed, 
        batch_size=batch_size, 
        epochs=epochs,
        name="Baseline"
    )


def benchmark_cached_dataset(data=None, seed=None, batch_size=None, epochs=None):
    """Benchmark CachedDataset optimization."""
    return benchmark_dataset(
        CachedDataset, 
        data=data, 
        seed=seed, 
        batch_size=batch_size, 
        epochs=epochs,
        name="CachedDataset"
    )


def benchmark_async_dataset(data=None, seed=None, batch_size=None, epochs=None):
    """Benchmark AsyncPrefetchDataset optimization."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
        
    if epochs is None:
        epochs = DEFAULT_EPOCHS
        
    if data is None:
        data = create_sample_data(seed)
        
    # Create base dataset first
    base_dataset = BaselineDataset(data)
    
    # Create AsyncPrefetchDataset wrapping the base dataset
    initial_memory = get_memory_usage()
    dataset = AsyncPrefetchDataset(base_dataset, queue_size=5, num_workers=2)
    dataset_memory = get_memory_usage()
    memory_increase = dataset_memory - initial_memory
    logger.info(f"AsyncPrefetchDataset Memory: {dataset_memory:.2f} MB (Increase: {memory_increase:.2f} MB)")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=DEFAULT_NUM_WORKERS
    )
    
    # Execute benchmark
    logger.info(f"Running AsyncPrefetchDataset benchmark...")
    times = []
    peak_memory = dataset_memory
    
    # Warm up the prefetch queue
    time.sleep(0.1)
    
    total_start = time.time()
    for epoch in range(epochs):
        start = time.time()
        for batch_idx, batch in enumerate(dataloader):
            # Simulate some processing
            _ = batch.mean()
            
        end = time.time()
        elapsed = (end - start) * 1000  # Convert to ms
        times.append(elapsed)
        logger.info(f"Epoch {epoch+1}/{epochs} - Time: {elapsed:.2f}ms")
        
        # Track memory
        mem = get_memory_usage()
        if mem > peak_memory:
            peak_memory = mem
            
    total_end = time.time()
    total_time = (total_end - total_start) * 1000  # Convert to ms
    
    # Calculate metrics
    items_per_second = (len(dataset) * epochs) / ((total_end - total_start))
    ms_per_batch = total_time / (len(dataloader) * epochs)
    
    # Memory increase from initial
    memory_increase = peak_memory - initial_memory
    
    # Log results
    logger.info(f"AsyncPrefetchDataset Results:")
    logger.info(f"  Total time: {total_time:.2f}ms")
    logger.info(f"  Items/second: {items_per_second:.2f}")
    logger.info(f"  Ms/batch: {ms_per_batch:.2f}ms")
    logger.info(f"  Peak memory: {peak_memory:.2f} MB (Increase: {memory_increase:.2f} MB)")
    
    # Clean up (stop background threads)
    dataset.__del__()
    
    # Return metrics
    return total_time, items_per_second, ms_per_batch, memory_increase


def benchmark_cpp_extension(data=None, seed=None, batch_size=None, epochs=None):
    """Benchmark C++ extension optimization."""
    if not HAS_CPP_EXTENSION:
        logger.error("C++ extension not available. Skipping benchmark.")
        return None
        
    return benchmark_dataset(
        CppExtensionDataset, 
        data=data, 
        seed=seed, 
        batch_size=batch_size, 
        epochs=epochs,
        name="C++ Extension"
    )


def print_comparison(baseline, cached, async_prefetch, cpp_ext=None):
    """Print comparison of all benchmarks."""
    print("\n" + "=" * 60)
    print("=" * 19 + " BENCHMARK COMPARISON " + "=" * 19)
    print("=" * 60)
    print(f"{'Metric':<15} {'Baseline':<15} {'CachedDataset':<15} {'AsyncPrefetch':<15} {'C++ Extension':<15}")
    print("-" * 60)
    
    # Time
    baseline_time = baseline[0]
    cached_time = cached[0]
    async_time = async_prefetch[0] if async_prefetch else "N/A"
    cpp_time = cpp_ext[0] if cpp_ext else "N/A"
    print(f"{'Time (ms)':<15} {baseline_time:<15.2f} {cached_time:<15.2f} {async_time if isinstance(async_time, str) else async_time:.2f:<15} {cpp_time if isinstance(cpp_time, str) else cpp_time:.2f:<15}")
    
    # Items/second
    baseline_items = baseline[1]
    cached_items = cached[1]
    async_items = async_prefetch[1] if async_prefetch else "N/A"
    cpp_items = cpp_ext[1] if cpp_ext else "N/A"
    print(f"{'Items/sec':<15} {baseline_items:<15,.2f} {cached_items:<15,.2f} {async_items if isinstance(async_items, str) else async_items:.2f:<15} {cpp_items if isinstance(cpp_items, str) else cpp_items:.2f:<15}")
    
    # Ms/batch
    baseline_batch = baseline[2]
    cached_batch = cached[2]
    async_batch = async_prefetch[2] if async_prefetch else "N/A"
    cpp_batch = cpp_ext[2] if cpp_ext else "N/A"
    print(f"{'Ms/batch':<15} {baseline_batch:<15.2f} {cached_batch:<15.2f} {async_batch if isinstance(async_batch, str) else async_batch:.2f:<15} {cpp_batch if isinstance(cpp_batch, str) else cpp_batch:.2f:<15}")
    
    # Memory
    baseline_mem = baseline[3]
    cached_mem = cached[3]
    async_mem = async_prefetch[3] if async_prefetch else "N/A"
    cpp_mem = cpp_ext[3] if cpp_ext else "N/A"
    print(f"{'Memory (MB)':<15} {baseline_mem:<15.2f} {cached_mem:<15.2f} {async_mem if isinstance(async_mem, str) else async_mem:.2f:<15} {cpp_mem if isinstance(cpp_mem, str) else cpp_mem:.2f:<15}")
    
    # Speedups
    cached_speedup = baseline_batch / cached_batch
    print(f"CachedDataset Speedup: {cached_speedup:.2f}x faster per batch")
    print(f"CachedDataset Time Saved: {baseline_batch - cached_batch:.2f}ms per batch")
    
    if async_prefetch:
        async_speedup = baseline_batch / async_batch
        print(f"AsyncPrefetch Speedup: {async_speedup:.2f}x faster per batch")
        print(f"AsyncPrefetch Time Saved: {baseline_batch - async_batch:.2f}ms per batch")
    
    if cpp_ext:
        cpp_speedup = baseline_batch / cpp_batch
        print(f"C++ Extension Speedup: {cpp_speedup:.2f}x faster per batch")
        print(f"C++ Extension Time Saved: {baseline_batch - cpp_batch:.2f}ms per batch")
    
    # Memory comparison
    if cached_mem != 0 and baseline_mem != 0:
        mem_change_pct = ((cached_mem / baseline_mem) - 1) * 100
        if mem_change_pct < 0:
            print(f"CachedDataset Memory Change: {mem_change_pct:.2f}% compared to baseline")
        else:
            print(f"CachedDataset Memory Change: +{mem_change_pct:.2f}% compared to baseline")
    else:
        print(f"CachedDataset Memory Change: absolute change of {cached_mem:.2f} MB")
        
    if async_prefetch and async_mem != 0 and baseline_mem != 0:
        mem_change_pct = ((async_mem / baseline_mem) - 1) * 100
        if mem_change_pct < 0:
            print(f"AsyncPrefetch Memory Change: {mem_change_pct:.2f}% compared to baseline")
        else:
            print(f"AsyncPrefetch Memory Change: +{mem_change_pct:.2f}% compared to baseline")
    elif async_prefetch:
        print(f"AsyncPrefetch Memory Change: absolute change of {async_mem:.2f} MB")
        
    print("=" * 60)


def run_single_benchmark(seed=None, batch_size=None, dataset_size=None):
    """Run a single benchmark with all implementations using the same seed."""
    logger.info(f"Running benchmark with seed: {seed}")
    
    # Create shared data
    data = create_sample_data(seed, dataset_size)
    
    # Run benchmarks
    baseline = benchmark_baseline(data, seed, batch_size)
    cached = benchmark_cached_dataset(data, seed, batch_size)
    async_prefetch = benchmark_async_dataset(data, seed, batch_size)
    cpp_ext = benchmark_cpp_extension(data, seed, batch_size)
    
    # Print comparison
    print_comparison(baseline, cached, async_prefetch, cpp_ext)
    
    return baseline, cached, async_prefetch, cpp_ext


def run_multi_seed_benchmark(seeds=None, batch_size=None, dataset_size=None):
    """Run benchmarks with multiple seeds for reproducibility and calculate statistics."""
    if seeds is None:
        seeds = DEFAULT_SEEDS
        
    logger.info(f"Starting benchmarks with {len(seeds)} different seeds")
    logger.info("Running multi-seed benchmark for reproducibility...")
    
    baseline_results = []
    cached_results = []
    async_results = []
    cpp_results = []
    
    for i, seed in enumerate(seeds):
        logger.info(f"Run {i+1}/{len(seeds)} (Seed: {seed})")
        baseline, cached, async_prefetch, cpp_ext = run_single_benchmark(seed, batch_size, dataset_size)
        
        baseline_results.append(baseline)
        cached_results.append(cached)
        if async_prefetch:
            async_results.append(async_prefetch)
        if cpp_ext:
            cpp_results.append(cpp_ext)
    
    # Calculate statistics across runs
    print("\n" + "=" * 60)
    print("=" * 16 + " REPRODUCIBILITY STATISTICS " + "=" * 16)
    print("=" * 60)
    
    # Analyze time
    baseline_times = [r[2] for r in baseline_results]  # ms/batch metric
    cached_times = [r[2] for r in cached_results]
    
    baseline_avg = statistics.mean(baseline_times)
    baseline_stdev = statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0
    baseline_cv = (baseline_stdev / baseline_avg * 100) if baseline_avg > 0 else 0
    
    cached_avg = statistics.mean(cached_times)
    cached_stdev = statistics.stdev(cached_times) if len(cached_times) > 1 else 0
    cached_cv = (cached_stdev / cached_avg * 100) if cached_avg > 0 else 0
    
    print(f"Baseline Time (ms/batch): {baseline_avg:.2f} ± {baseline_stdev:.2f} (CV: {baseline_cv:.2f}%)")
    print(f"CachedDataset Time (ms/batch): {cached_avg:.2f} ± {cached_stdev:.2f} (CV: {cached_cv:.2f}%)")
    
    # Consistent speedup
    speedup = baseline_avg / cached_avg if cached_avg > 0 else 0
    print(f"Consistent Speedup: {speedup:.2f}x")
    
    if async_results:
        async_times = [r[2] for r in async_results]
        async_avg = statistics.mean(async_times)
        async_stdev = statistics.stdev(async_times) if len(async_times) > 1 else 0
        async_cv = (async_stdev / async_avg * 100) if async_avg > 0 else 0
        
        print(f"AsyncPrefetch Time (ms/batch): {async_avg:.2f} ± {async_stdev:.2f} (CV: {async_cv:.2f}%)")
        async_speedup = baseline_avg / async_avg if async_avg > 0 else 0
        print(f"AsyncPrefetch Speedup: {async_speedup:.2f}x")
    
    if cpp_results:
        cpp_times = [r[2] for r in cpp_results]
        cpp_avg = statistics.mean(cpp_times)
        cpp_stdev = statistics.stdev(cpp_times) if len(cpp_times) > 1 else 0
        cpp_cv = (cpp_stdev / cpp_avg * 100) if cpp_avg > 0 else 0
        
        print(f"C++ Extension Time (ms/batch): {cpp_avg:.2f} ± {cpp_stdev:.2f} (CV: {cpp_cv:.2f}%)")
        cpp_speedup = baseline_avg / cpp_avg if cpp_avg > 0 else 0
        print(f"C++ Extension Speedup: {cpp_speedup:.2f}x")
    
    # Memory stability
    print("\nMemory Usage Stability:")
    baseline_mems = [r[3] for r in baseline_results]
    cached_mems = [r[3] for r in cached_results]
    
    baseline_mem_avg = statistics.mean(baseline_mems)
    baseline_mem_stdev = statistics.stdev(baseline_mems) if len(baseline_mems) > 1 else 0
    baseline_mem_cv = (baseline_mem_stdev / baseline_mem_avg * 100) if baseline_mem_avg > 0 else 0
    
    cached_mem_avg = statistics.mean(cached_mems)
    cached_mem_stdev = statistics.stdev(cached_mems) if len(cached_mems) > 1 else 0
    cached_mem_cv = (cached_mem_stdev / cached_mem_avg * 100) if cached_mem_avg > 0 else 0
    
    print(f"Baseline Memory (MB): {baseline_mem_avg:.2f} ± {baseline_mem_stdev:.2f} (CV: {baseline_mem_cv:.2f}%)")
    print(f"CachedDataset Memory (MB): {cached_mem_avg:.2f} ± {cached_mem_stdev:.2f} (CV: {cached_mem_cv:.2f}%)")
    
    if baseline_mem_avg > 0:
        mem_overhead = ((cached_mem_avg / baseline_mem_avg) - 1) * 100
        print(f"Memory Overhead: {mem_overhead:.2f}%")
    else:
        print(f"Memory Overhead: {cached_mem_avg:.2f} MB absolute")
        
    if async_results:
        async_mems = [r[3] for r in async_results]
        async_mem_avg = statistics.mean(async_mems)
        async_mem_stdev = statistics.stdev(async_mems) if len(async_mems) > 1 else 0
        async_mem_cv = (async_mem_stdev / async_mem_avg * 100) if async_mem_avg > 0 else 0
        
        print(f"AsyncPrefetch Memory (MB): {async_mem_avg:.2f} ± {async_mem_stdev:.2f} (CV: {async_mem_cv:.2f}%)")
        if baseline_mem_avg > 0:
            async_mem_overhead = ((async_mem_avg / baseline_mem_avg) - 1) * 100
            print(f"AsyncPrefetch Memory Overhead: {async_mem_overhead:.2f}%")
        else:
            print(f"AsyncPrefetch Memory Overhead: {async_mem_avg:.2f} MB absolute")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch DataLoader optimizations")
    parser.add_argument("--multi-seed", action="store_true", help="Run benchmarks with multiple seeds")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for DataLoader")
    parser.add_argument("--dataset-size", type=str, default=f"{DEFAULT_DATASET_SIZE[0]},{DEFAULT_DATASET_SIZE[1]}", 
                      help="Dataset size as rows,features (e.g., '2000,50')")
    
    args = parser.parse_args()
    
    # Parse dataset size
    dataset_size = tuple(map(int, args.dataset_size.split(',')))
    
    if args.multi_seed:
        run_multi_seed_benchmark(batch_size=args.batch_size, dataset_size=dataset_size)
    else:
        run_single_benchmark(batch_size=args.batch_size, dataset_size=dataset_size)
        

if __name__ == "__main__":
    main()