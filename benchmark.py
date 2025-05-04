#!/usr/bin/env python3
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging
import gc
import os
import sys
import psutil
import random

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import optimized implementations
from fast_loader import CachedDataset

# Try to import C++ extension (may fail if not built)
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "cpp_extension"))
    from fastfetch import fetch_tensor_zero_copy
    cpp_extension_available = True
except ImportError:
    logger.warning("C++ extension not available. Run setup.py in cpp_extension/ first.")
    cpp_extension_available = False

# Constants
NUM_SAMPLES = 2000  # Reduced from 5000
FEATURE_DIM = 50    # Reduced from 100
BATCH_SIZE = 64
NUM_WORKERS = 1     # Reduced from 2
NUM_EPOCHS = 2      # Reduced from 3
NUM_RUNS = 3        # Reduced from 5
RANDOM_SEEDS = [42, 123, 456]  # Using only 3 seeds

class BaselineDataset(Dataset):
    """Basic dataset that loads tensors from memory."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Copy data when fetching items
        return self.data[idx].clone()

class CppExtensionDataset(Dataset):
    """Dataset that uses C++ extension for zero-copy data fetching."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Use C++ extension for zero-copy fetching
        return fetch_tensor_zero_copy(self.data, idx)

def get_memory_usage():
    """Get current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert bytes to MB

def create_sample_data(seed=None):
    """Create sample data for benchmarking."""
    if seed is not None:
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    logger.info(f"Creating sample data: {NUM_SAMPLES}x{FEATURE_DIM} tensor")
    return torch.randn(NUM_SAMPLES, FEATURE_DIM)

def benchmark_baseline(data=None, seed=None):
    """Benchmark baseline DataLoader."""
    if data is None:
        data = create_sample_data(seed)
    
    # Clear memory and measure baseline memory before creating dataset
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    base_memory = get_memory_usage()
    
    dataset = BaselineDataset(data)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True
    )
    
    # Measure memory after dataset creation
    init_memory = get_memory_usage()
    mem_increase = init_memory - base_memory
    logger.info(f"Baseline Dataset Memory: {init_memory:.2f} MB (Increase: {mem_increase:.2f} MB)")
    
    logger.info("Running baseline benchmark...")
    start_time = time.time()
    
    # Track peak memory during execution
    peak_memory = init_memory
    total_items = 0
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        for i, batch in enumerate(dataloader):
            # Simulate some processing on the batch
            _ = batch.mean()
            total_items += len(batch)
            
            # Update peak memory
            current_mem = get_memory_usage()
            peak_memory = max(peak_memory, current_mem)
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Time: {epoch_time*1000:.2f}ms")
    
    total_time = time.time() - start_time
    items_per_second = total_items / total_time
    ms_per_batch = (total_time * 1000) / (total_items / BATCH_SIZE)
    
    # Final memory measurements
    final_memory = get_memory_usage()
    mem_increase = peak_memory - base_memory
    
    logger.info(f"Baseline Results:")
    logger.info(f"  Total time: {total_time*1000:.2f}ms")
    logger.info(f"  Items/second: {items_per_second:.2f}")
    logger.info(f"  Ms/batch: {ms_per_batch:.2f}ms")
    logger.info(f"  Peak memory: {peak_memory:.2f} MB (Increase: {mem_increase:.2f} MB)")
    
    return total_time, items_per_second, ms_per_batch, peak_memory - base_memory

def benchmark_cached_dataset(data=None, seed=None):
    """Benchmark CachedDataset optimization."""
    if data is None:
        data = create_sample_data(seed)
    
    # Clear memory and measure baseline memory before creating dataset
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    base_memory = get_memory_usage()
    
    dataset = CachedDataset(data)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=False  # Already pinned in CachedDataset
    )
    
    # Measure memory after dataset creation
    init_memory = get_memory_usage()
    mem_increase = init_memory - base_memory
    logger.info(f"CachedDataset Memory: {init_memory:.2f} MB (Increase: {mem_increase:.2f} MB)")
    
    logger.info("Running CachedDataset benchmark...")
    start_time = time.time()
    
    # Track peak memory during execution
    peak_memory = init_memory
    total_items = 0
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        for i, batch in enumerate(dataloader):
            # Simulate some processing on the batch
            _ = batch.mean()
            total_items += len(batch)
            
            # Update peak memory
            current_mem = get_memory_usage()
            peak_memory = max(peak_memory, current_mem)
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Time: {epoch_time*1000:.2f}ms")
    
    total_time = time.time() - start_time
    items_per_second = total_items / total_time
    ms_per_batch = (total_time * 1000) / (total_items / BATCH_SIZE)
    
    # Final memory measurements
    final_memory = get_memory_usage()
    mem_increase = peak_memory - base_memory
    
    logger.info(f"CachedDataset Results:")
    logger.info(f"  Total time: {total_time*1000:.2f}ms")
    logger.info(f"  Items/second: {items_per_second:.2f}")
    logger.info(f"  Ms/batch: {ms_per_batch:.2f}ms")
    logger.info(f"  Peak memory: {peak_memory:.2f} MB (Increase: {mem_increase:.2f} MB)")
    
    return total_time, items_per_second, ms_per_batch, peak_memory - base_memory

def benchmark_cpp_extension(data=None, seed=None):
    """Benchmark C++ extension optimization."""
    if not cpp_extension_available:
        logger.error("C++ extension not available. Skipping benchmark.")
        return None, None, None, None
    
    if data is None:
        data = create_sample_data(seed)
    
    # Clear memory and measure baseline memory before creating dataset
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    base_memory = get_memory_usage()
    
    dataset = CppExtensionDataset(data)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True
    )
    
    # Measure memory after dataset creation
    init_memory = get_memory_usage()
    mem_increase = init_memory - base_memory
    logger.info(f"C++ Extension Dataset Memory: {init_memory:.2f} MB (Increase: {mem_increase:.2f} MB)")
    
    logger.info("Running C++ extension benchmark...")
    start_time = time.time()
    
    # Track peak memory during execution
    peak_memory = init_memory
    total_items = 0
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        for i, batch in enumerate(dataloader):
            # Simulate some processing on the batch
            _ = batch.mean()
            total_items += len(batch)
            
            # Update peak memory
            current_mem = get_memory_usage()
            peak_memory = max(peak_memory, current_mem)
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Time: {epoch_time*1000:.2f}ms")
    
    total_time = time.time() - start_time
    items_per_second = total_items / total_time
    ms_per_batch = (total_time * 1000) / (total_items / BATCH_SIZE)
    
    # Final memory measurements
    final_memory = get_memory_usage()
    mem_increase = peak_memory - base_memory
    
    logger.info(f"C++ Extension Results:")
    logger.info(f"  Total time: {total_time*1000:.2f}ms")
    logger.info(f"  Items/second: {items_per_second:.2f}")
    logger.info(f"  Ms/batch: {ms_per_batch:.2f}ms")
    logger.info(f"  Peak memory: {peak_memory:.2f} MB (Increase: {mem_increase:.2f} MB)")
    
    return total_time, items_per_second, ms_per_batch, peak_memory - base_memory

def print_comparison(baseline, cached, cpp_ext):
    """Print comparison of all benchmarks."""
    print("\n" + "="*60)
    print(" BENCHMARK COMPARISON ".center(60, "="))
    print("="*60)
    
    print(f"{'Metric':<15} {'Baseline':<15} {'CachedDataset':<15} {'C++ Extension':<15}")
    print("-"*60)
    
    # Only show results that are available
    if baseline[0] is not None:
        b_time, b_ips, b_ms, b_mem = baseline
    else:
        b_time, b_ips, b_ms, b_mem = 0, 0, 0, 0
        
    if cached[0] is not None:
        c_time, c_ips, c_ms, c_mem = cached
    else:
        c_time, c_ips, c_ms, c_mem = 0, 0, 0, 0
        
    if cpp_ext[0] is not None:
        cpp_time, cpp_ips, cpp_ms, cpp_mem = cpp_ext
    else:
        cpp_time, cpp_ips, cpp_ms, cpp_mem = 0, 0, 0, 0
    
    cpp_time_str = f"{(cpp_time*1000):,.2f}" if cpp_time else "N/A"
    print(f"{'Time (ms)':<15} {b_time*1000:,.2f}{'':<8} {c_time*1000:,.2f}{'':<8} {cpp_time_str}")
    cpp_ips_str = f"{cpp_ips:,.2f}" if cpp_ips else "N/A"
    print(f"{'Items/sec':<15} {b_ips:,.2f}{'':<8} {c_ips:,.2f}{'':<8} {cpp_ips_str}")
    
    cpp_ms_str = f"{cpp_ms:,.2f}" if cpp_ms else "N/A"
    print(f"{'Ms/batch':<15} {b_ms:,.2f}{'':<8} {c_ms:,.2f}{'':<8} {cpp_ms_str}")
    
    # Print memory metrics
    cpp_mem_str = f"{cpp_mem:.2f}" if cpp_ext[0] is not None else "N/A"
    print(f"{'Memory (MB)':<15} {b_mem:.2f}{'':<8} {c_mem:.2f}{'':<8} {cpp_mem_str}")
    
    # Calculate and print speedups if we have all results
    if baseline[0] and cached[0]:
        cached_speedup = b_ms / c_ms if c_ms > 0 else 0
        
        # Avoid division by zero for memory change calculation
        if b_mem > 0:
            cached_mem_change = ((c_mem - b_mem) / b_mem) * 100
            mem_change_str = f"{cached_mem_change:+.2f}% compared to baseline"
        else:
            # If baseline memory is 0, we can't calculate percentage change
            mem_change_str = f"absolute change of {c_mem - b_mem:.2f} MB"
            
        print(f"\nCachedDataset Speedup: {cached_speedup:.2f}x faster per batch")
        print(f"CachedDataset Time Saved: {b_ms - c_ms:.2f}ms per batch")
        print(f"CachedDataset Memory Change: {mem_change_str}")
    
    if baseline[0] and cpp_ext[0]:
        cpp_speedup = b_ms / cpp_ms if cpp_ms > 0 else 0
        
        # Avoid division by zero for memory change calculation
        if b_mem > 0:
            cpp_mem_change = ((cpp_mem - b_mem) / b_mem) * 100
            cpp_mem_change_str = f"{cpp_mem_change:+.2f}% compared to baseline"
        else:
            # If baseline memory is 0, we can't calculate percentage change
            cpp_mem_change_str = f"absolute change of {cpp_mem - b_mem:.2f} MB"
            
        print(f"C++ Extension Speedup: {cpp_speedup:.2f}x faster per batch")
        print(f"C++ Extension Time Saved: {b_ms - cpp_ms:.2f}ms per batch")
        print(f"C++ Extension Memory Change: {cpp_mem_change_str}")
    
    print("="*60)

def run_single_benchmark(seed=None):
    """Run a single benchmark with all implementations using the same seed."""
    logger.info(f"Running benchmark with seed: {seed}")
    
    # Create sample data with specific seed
    data = create_sample_data(seed)
    
    # Run all benchmarks with the same data
    baseline_results = benchmark_baseline(data, seed)
    cached_results = benchmark_cached_dataset(data, seed)
    cpp_results = benchmark_cpp_extension(data, seed)
    
    # Print comparison for this run
    print_comparison(baseline_results, cached_results, cpp_results)
    
    return baseline_results, cached_results, cpp_results

def main():
    """Run benchmarks with multiple seeds for reproducibility and calculate statistics."""
    logger.info(f"Starting benchmarks with {NUM_RUNS} different seeds")
    
    # Lists to store results from each run
    baseline_times = []
    baseline_mems = []
    cached_times = []
    cached_mems = []
    cpp_times = []
    cpp_mems = []
    
    # Run benchmarks with different seeds
    if len(sys.argv) > 1 and sys.argv[1] == "--multi-seed":
        logger.info("Running multi-seed benchmark for reproducibility...")
        
        for i, seed in enumerate(RANDOM_SEEDS):
            logger.info(f"Run {i+1}/{NUM_RUNS} (Seed: {seed})")
            baseline, cached, cpp = run_single_benchmark(seed)
            
            # Store results
            if baseline[0] is not None:
                baseline_times.append(baseline[2])  # ms_per_batch
                baseline_mems.append(baseline[3])   # memory usage
            
            if cached[0] is not None:
                cached_times.append(cached[2])      # ms_per_batch
                cached_mems.append(cached[3])       # memory usage
                
            if cpp[0] is not None:
                cpp_times.append(cpp[2])            # ms_per_batch
                cpp_mems.append(cpp[3])             # memory usage
        
        # Calculate and print statistics
        print("\n" + "="*60)
        print(" REPRODUCIBILITY STATISTICS ".center(60, "="))
        print("="*60)
        
        # Performance stability
        if baseline_times:
            baseline_avg = sum(baseline_times) / len(baseline_times)
            baseline_std = np.std(baseline_times)
            baseline_cv = (baseline_std / baseline_avg) * 100 if baseline_avg > 0 else 0
            print(f"Baseline Time (ms/batch): {baseline_avg:.2f} ± {baseline_std:.2f} (CV: {baseline_cv:.2f}%)")
        
        if cached_times:
            cached_avg = sum(cached_times) / len(cached_times)
            cached_std = np.std(cached_times)
            cached_cv = (cached_std / cached_avg) * 100 if cached_avg > 0 else 0
            print(f"CachedDataset Time (ms/batch): {cached_avg:.2f} ± {cached_std:.2f} (CV: {cached_cv:.2f}%)")
            
            # Calculate consistent speedup (if both values are available)
            if 'baseline_avg' in locals() and cached_avg > 0:
                speedup_avg = baseline_avg / cached_avg
                print(f"Consistent Speedup: {speedup_avg:.2f}x")
            else:
                print("Consistent Speedup: N/A (insufficient data)")
        
        if cpp_times:
            cpp_avg = sum(cpp_times) / len(cpp_times)
            cpp_std = np.std(cpp_times)
            cpp_cv = (cpp_std / cpp_avg) * 100 if cpp_avg > 0 else 0
            print(f"C++ Extension Time (ms/batch): {cpp_avg:.2f} ± {cpp_std:.2f} (CV: {cpp_cv:.2f}%)")
            
            # Calculate consistent speedup (if both values are available)
            if 'baseline_avg' in locals() and cpp_avg > 0:
                cpp_speedup_avg = baseline_avg / cpp_avg
                print(f"Consistent Speedup: {cpp_speedup_avg:.2f}x")
            else:
                print("Consistent Speedup: N/A (insufficient data)")
        
        # Memory usage stability
        print("\nMemory Usage Stability:")
        if baseline_mems:
            baseline_mem_avg = sum(baseline_mems) / len(baseline_mems)
            baseline_mem_std = np.std(baseline_mems)
            baseline_mem_cv = (baseline_mem_std / baseline_mem_avg) * 100 if baseline_mem_avg > 0 else 0
            print(f"Baseline Memory (MB): {baseline_mem_avg:.2f} ± {baseline_mem_std:.2f} (CV: {baseline_mem_cv:.2f}%)")
        
        if cached_mems:
            cached_mem_avg = sum(cached_mems) / len(cached_mems)
            cached_mem_std = np.std(cached_mems)
            cached_mem_cv = (cached_mem_std / cached_mem_avg) * 100 if cached_mem_avg > 0 else 0
            print(f"CachedDataset Memory (MB): {cached_mem_avg:.2f} ± {cached_mem_std:.2f} (CV: {cached_mem_cv:.2f}%)")
            
            # Calculate memory overhead (if both values are available)
            if 'baseline_mem_avg' in locals() and baseline_mem_avg > 0:
                mem_change_pct = ((cached_mem_avg - baseline_mem_avg) / baseline_mem_avg) * 100
                print(f"Memory Overhead: {mem_change_pct:+.2f}%")
            else:
                print(f"Memory Overhead: absolute change of {cached_mem_avg:.2f} MB")
        
        if cpp_mems:
            cpp_mem_avg = sum(cpp_mems) / len(cpp_mems)
            cpp_mem_std = np.std(cpp_mems)
            cpp_mem_cv = (cpp_mem_std / cpp_mem_avg) * 100 if cpp_mem_avg > 0 else 0
            print(f"C++ Extension Memory (MB): {cpp_mem_avg:.2f} ± {cpp_mem_std:.2f} (CV: {cpp_mem_cv:.2f}%)")
            
            # Calculate memory overhead (if both values are available)
            if 'baseline_mem_avg' in locals() and baseline_mem_avg > 0:
                cpp_mem_change_pct = ((cpp_mem_avg - baseline_mem_avg) / baseline_mem_avg) * 100
                print(f"Memory Overhead: {cpp_mem_change_pct:+.2f}%")
            else:
                print(f"Memory Overhead: absolute change of {cpp_mem_avg:.2f} MB")
        
        print("="*60)
    else:
        # Just run a single benchmark with default seed
        run_single_benchmark()

if __name__ == "__main__":
    main()
