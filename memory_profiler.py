"""Memory Profiling Utilities for GSC Network

Utilities to monitor memory usage and detect OOM issues early.
Can be used standalone or imported into other scripts.
"""

import psutil
import os
import time
import functools
import gc
from typing import Callable, Any


class MemoryMonitor:
    """Monitor memory usage throughout script execution"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.checkpoints = []
        self.start_memory = self.get_memory_mb()

    def get_memory_mb(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def get_memory_gb(self):
        """Get current memory usage in GB"""
        return self.get_memory_mb() / 1024

    def checkpoint(self, name: str, print_stats: bool = True):
        """Record a memory checkpoint"""
        mem_mb = self.get_memory_mb()
        delta = mem_mb - (self.checkpoints[-1][1] if self.checkpoints else self.start_memory)

        self.checkpoints.append((name, mem_mb, delta))

        if print_stats:
            print(f"[MEMORY] {name}")
            print(f"  Current: {mem_mb:.1f} MB ({mem_mb/1024:.2f} GB)")
            print(f"  Delta:   {delta:+.1f} MB")

        return mem_mb

    def summary(self):
        """Print summary of all checkpoints"""
        print("\n" + "="*70)
        print("MEMORY USAGE SUMMARY")
        print("="*70)
        print(f"{'Checkpoint':<40} {'Memory (MB)':<15} {'Delta (MB)':<15}")
        print("-"*70)
        print(f"{'START':<40} {self.start_memory:<15.1f} {'-':<15}")

        for name, mem_mb, delta in self.checkpoints:
            print(f"{name:<40} {mem_mb:<15.1f} {delta:+<15.1f}")

        if self.checkpoints:
            final_mem = self.checkpoints[-1][1]
            total_delta = final_mem - self.start_memory
            print("-"*70)
            print(f"{'TOTAL INCREASE':<40} {final_mem:<15.1f} {total_delta:+<15.1f}")
            print("="*70)

    def check_oom_risk(self, threshold_gb: float = 14.0):
        """Check if we're at risk of OOM"""
        current_gb = self.get_memory_gb()

        if current_gb > threshold_gb:
            print(f"\n⚠️  OOM RISK: Memory usage ({current_gb:.2f} GB) exceeds threshold ({threshold_gb:.1f} GB)")
            return True
        return False


def profile_memory(func: Callable) -> Callable:
    """Decorator to profile memory usage of a function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024

        print(f"\n[PROFILING] {func.__name__}()")
        print(f"  Memory before: {mem_before:.1f} MB")

        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        # Force garbage collection to get accurate reading
        gc.collect()

        mem_after = process.memory_info().rss / 1024 / 1024
        mem_delta = mem_after - mem_before

        print(f"  Memory after:  {mem_after:.1f} MB")
        print(f"  Delta:         {mem_delta:+.1f} MB")
        print(f"  Duration:      {duration:.2f}s")

        if mem_delta > 1000:  # >1GB increase
            print(f"  ⚠️  Large memory allocation detected!")

        return result

    return wrapper


def monitor_sparse_matrices(obj, name: str = "object"):
    """Monitor memory usage of sparse matrices in an object"""
    from scipy import sparse
    import numpy as np

    print(f"\n[SPARSE MATRIX ANALYSIS] {name}")

    total_sparse_mb = 0
    total_dense_mb = 0

    for attr_name in dir(obj):
        if attr_name.startswith('_'):
            continue

        try:
            attr = getattr(obj, attr_name)

            # Check for sparse matrices
            if sparse.issparse(attr):
                nnz = attr.nnz
                shape = attr.shape
                total_elements = shape[0] * shape[1] if len(shape) == 2 else np.prod(shape)
                sparsity = 100 * (1 - nnz / total_elements) if total_elements > 0 else 0

                # Estimate memory usage
                sparse_mb = (nnz * 16) / 1024 / 1024  # value + index
                dense_mb = (total_elements * 8) / 1024 / 1024  # would be if dense

                total_sparse_mb += sparse_mb
                total_dense_mb += dense_mb

                print(f"  {attr_name}:")
                print(f"    Shape: {shape}")
                print(f"    Non-zero: {nnz:,} / {total_elements:,} ({100-sparsity:.4f}%)")
                print(f"    Memory: {sparse_mb:.1f} MB (sparse) vs {dense_mb:.1f} MB (dense)")
                print(f"    Saved: {dense_mb - sparse_mb:.1f} MB")

            # Check for large numpy arrays
            elif isinstance(attr, np.ndarray):
                if attr.size > 10000:  # Only report large arrays
                    array_mb = attr.nbytes / 1024 / 1024
                    print(f"  {attr_name}:")
                    print(f"    Shape: {attr.shape}")
                    print(f"    Memory: {array_mb:.1f} MB (dense)")
                    total_dense_mb += array_mb

        except Exception:
            pass  # Skip attributes that can't be accessed

    print(f"\n  Total sparse matrices: {total_sparse_mb:.1f} MB")
    print(f"  Total dense arrays: {total_dense_mb:.1f} MB")
    print(f"  Total saved by sparsity: {total_dense_mb - total_sparse_mb:.1f} MB")


if __name__ == "__main__":
    print("Memory Profiler Utilities")
    print("="*70)
    print("\nUsage:")
    print("\n1. Basic monitoring:")
    print("   from memory_profiler import MemoryMonitor")
    print("   monitor = MemoryMonitor()")
    print("   monitor.checkpoint('After step 1')")
    print("   monitor.checkpoint('After step 2')")
    print("   monitor.summary()")
    print("\n2. Function profiling:")
    print("   from memory_profiler import profile_memory")
    print("   @profile_memory")
    print("   def my_function():")
    print("       ...")
    print("\n3. Sparse matrix analysis:")
    print("   from memory_profiler import monitor_sparse_matrices")
    print("   monitor_sparse_matrices(net, 'GscNet')")
