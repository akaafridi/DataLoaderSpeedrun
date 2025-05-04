#!/usr/bin/env python
"""
Command-line interface for grok3-io-opt package.

Usage:
    grok3-io-opt benchmark [--multi-seed] [--batch-size BATCH_SIZE] [--dataset-size DATASET_SIZE]
    grok3-io-opt profile [--output OUTPUT] [--iterations ITERATIONS]
    grok3-io-opt check-extension
    grok3-io-opt --version

Options:
    -h --help                       Show this help message and exit
    --version                       Show version and exit
    --multi-seed                    Run benchmarks with multiple seeds for reproducibility
    --batch-size BATCH_SIZE         Batch size for DataLoader [default: 64]
    --dataset-size DATASET_SIZE     Dataset size as rows,features (e.g., '2000,50') [default: 2000,50]
    --output OUTPUT                 Output file for profiling results [default: profile_results.txt]
    --iterations ITERATIONS         Number of iterations for profiling [default: 10]
"""

import sys
import argparse
import logging
import time
from typing import List, Optional

from . import __version__
from .benchmark import run_single_benchmark, run_multi_seed_benchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_extension() -> bool:
    """Check if C++ extension is available."""
    try:
        from .fastfetch import fetch_tensor_zero_copy
        logger.info("C++ extension is available.")
        return True
    except ImportError:
        logger.error("C++ extension is not available.")
        logger.info("To build the extension, run: pip install -e .")
        return False


def profile(output: str = "profile_results.txt", iterations: int = 10) -> None:
    """
    Profile the performance of different dataset implementations.
    
    Args:
        output (str): Output file for profiling results
        iterations (int): Number of iterations for profiling
    """
    import torch
    import psutil
    from torch.utils.data import Dataset, DataLoader
    from .cached_dataset import CachedDataset
    from .async_dataset import AsyncPrefetchDataset
    
    logger.info(f"Profiling performance over {iterations} iterations...")
    
    # Create sample data
    data = torch.randn(5000, 100)
    
    # Define datasets
    class BaselineDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    
    baseline_dataset = BaselineDataset(data)
    cached_dataset = CachedDataset(data)
    async_dataset = AsyncPrefetchDataset(baseline_dataset, queue_size=10)
    
    # Profile each dataset
    results = {
        "baseline": {"times": [], "memory": []},
        "cached": {"times": [], "memory": []},
        "async": {"times": [], "memory": []},
    }
    
    # Baseline
    logger.info("Profiling baseline dataset...")
    for i in range(iterations):
        dataloader = DataLoader(baseline_dataset, batch_size=64, shuffle=True)
        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        start = time.time()
        for batch in dataloader:
            _ = batch.mean()
        end = time.time()
        end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        
        results["baseline"]["times"].append((end - start) * 1000)  # ms
        results["baseline"]["memory"].append(end_mem - start_mem)
    
    # CachedDataset
    logger.info("Profiling CachedDataset...")
    for i in range(iterations):
        dataloader = DataLoader(cached_dataset, batch_size=64, shuffle=True)
        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        start = time.time()
        for batch in dataloader:
            _ = batch.mean()
        end = time.time()
        end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        
        results["cached"]["times"].append((end - start) * 1000)  # ms
        results["cached"]["memory"].append(end_mem - start_mem)
    
    # AsyncPrefetchDataset
    logger.info("Profiling AsyncPrefetchDataset...")
    for i in range(iterations):
        dataloader = DataLoader(async_dataset, batch_size=64, shuffle=True)
        start_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        start = time.time()
        for batch in dataloader:
            _ = batch.mean()
        end = time.time()
        end_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        
        results["async"]["times"].append((end - start) * 1000)  # ms
        results["async"]["memory"].append(end_mem - start_mem)
    
    # Calculate averages
    def calc_avg(values):
        return sum(values) / len(values) if values else 0
    
    baseline_avg_time = calc_avg(results["baseline"]["times"])
    cached_avg_time = calc_avg(results["cached"]["times"])
    async_avg_time = calc_avg(results["async"]["times"])
    
    baseline_avg_mem = calc_avg(results["baseline"]["memory"])
    cached_avg_mem = calc_avg(results["cached"]["memory"])
    async_avg_mem = calc_avg(results["async"]["memory"])
    
    # Calculate speedups
    cached_speedup = baseline_avg_time / cached_avg_time if cached_avg_time > 0 else 0
    async_speedup = baseline_avg_time / async_avg_time if async_avg_time > 0 else 0
    
    # Write results to file
    with open(output, "w") as f:
        f.write("# Grok3 I/O Optimization Profiling Results\n\n")
        f.write(f"## Performance Metrics (Average over {iterations} iterations)\n\n")
        
        f.write("| Dataset | Time (ms) | Speedup | Memory (MB) |\n")
        f.write("|---------|-----------|---------|-------------|\n")
        f.write(f"| Baseline | {baseline_avg_time:.2f} | 1.00x | {baseline_avg_mem:.2f} |\n")
        f.write(f"| CachedDataset | {cached_avg_time:.2f} | {cached_speedup:.2f}x | {cached_avg_mem:.2f} |\n")
        f.write(f"| AsyncPrefetchDataset | {async_avg_time:.2f} | {async_speedup:.2f}x | {async_avg_mem:.2f} |\n\n")
        
        f.write("## Raw Results\n\n")
        f.write("### Baseline\n\n")
        f.write("| Iteration | Time (ms) | Memory (MB) |\n")
        f.write("|-----------|-----------|-------------|\n")
        for i, (time_ms, mem) in enumerate(zip(results["baseline"]["times"], results["baseline"]["memory"])):
            f.write(f"| {i+1} | {time_ms:.2f} | {mem:.2f} |\n")
        
        f.write("\n### CachedDataset\n\n")
        f.write("| Iteration | Time (ms) | Memory (MB) |\n")
        f.write("|-----------|-----------|-------------|\n")
        for i, (time_ms, mem) in enumerate(zip(results["cached"]["times"], results["cached"]["memory"])):
            f.write(f"| {i+1} | {time_ms:.2f} | {mem:.2f} |\n")
        
        f.write("\n### AsyncPrefetchDataset\n\n")
        f.write("| Iteration | Time (ms) | Memory (MB) |\n")
        f.write("|-----------|-----------|-------------|\n")
        for i, (time_ms, mem) in enumerate(zip(results["async"]["times"], results["async"]["memory"])):
            f.write(f"| {i+1} | {time_ms:.2f} | {mem:.2f} |\n")
    
    logger.info(f"Profiling results written to {output}")
    logger.info(f"Summary:")
    logger.info(f"- Baseline: {baseline_avg_time:.2f}ms per iteration, {baseline_avg_mem:.2f}MB memory")
    logger.info(f"- CachedDataset: {cached_avg_time:.2f}ms per iteration ({cached_speedup:.2f}x), {cached_avg_mem:.2f}MB memory")
    logger.info(f"- AsyncPrefetchDataset: {async_avg_time:.2f}ms per iteration ({async_speedup:.2f}x), {async_avg_mem:.2f}MB memory")


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Grok3 I/O Optimization tools for PyTorch DataLoader"
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--multi-seed", action="store_true", help="Run with multiple seeds")
    bench_parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DataLoader")
    bench_parser.add_argument("--dataset-size", type=str, default="2000,50", 
                             help="Dataset size as rows,features (e.g., '2000,50')")
    
    # Profile command
    profile_parser = subparsers.add_parser("profile", help="Profile performance")
    profile_parser.add_argument("--output", type=str, default="profile_results.txt", 
                              help="Output file for profiling results")
    profile_parser.add_argument("--iterations", type=int, default=10,
                              help="Number of iterations for profiling")
    
    # Check extension command
    subparsers.add_parser("check-extension", help="Check if C++ extension is available")
    
    # Parse arguments
    args = parser.parse_args(args)
    
    if args.version:
        print(f"grok3-io-opt version {__version__}")
        return 0
    
    if args.command == "benchmark":
        dataset_size = tuple(map(int, args.dataset_size.split(',')))
        if args.multi_seed:
            run_multi_seed_benchmark(batch_size=args.batch_size, dataset_size=dataset_size)
        else:
            run_single_benchmark(batch_size=args.batch_size, dataset_size=dataset_size)
        return 0
    
    elif args.command == "profile":
        profile(output=args.output, iterations=args.iterations)
        return 0
    
    elif args.command == "check-extension":
        check_extension()
        return 0
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())