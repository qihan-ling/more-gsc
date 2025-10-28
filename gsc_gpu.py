"""
GPU wrapper for gsc.py

NOTE: Drop-in NumPy→CuPy replacement doesn't work due to import mechanics.
The monkey-patching approach fails because gsc.py has 'import numpy as np'
at module level, and changing gsc.np later doesn't affect that reference.

Current status: Uses CPU NumPy (same performance as regular gsc)

For GPU acceleration, we need to implement batch parallelization
(running multiple trials in parallel on GPU), which will provide
20-40x speedup instead of the 5-10x from array operations alone.

Usage:
    import gsc_gpu as gsc  # Currently uses CPU
"""

import sys
import warnings

# Detect GPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU detected (CuPy available)")
    print(f"  Device: {cp.cuda.Device()}")
    print(f"  Memory: {cp.cuda.Device().mem_info[1] / 1e9:.2f} GB total")
    print()
    print("  ⚠️  NOTE: Currently running on CPU")
    print("  Drop-in CuPy replacement has compatibility issues.")
    print("  GPU acceleration requires batch parallelization (future work).")
    print()
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    warnings.warn("CuPy not available. Install with: pip install cupy-cuda12x")

import numpy as np

# Import gsc normally (uses CPU NumPy)
import gsc

# Re-export everything from gsc
from gsc import *

# Utility functions (for future GPU work)
def to_gpu(arr):
    """Move numpy array to GPU (currently no-op)"""
    if GPU_AVAILABLE and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr

def to_cpu(arr):
    """Move cupy array to CPU"""
    if GPU_AVAILABLE and hasattr(arr, '__cuda_array_interface__'):
        return cp.asnumpy(arr)
    return arr

def print_gpu_memory():
    """Print GPU memory usage"""
    if GPU_AVAILABLE:
        device_info = cp.cuda.Device().mem_info
        device_free = device_info[0] / 1e9
        device_total = device_info[1] / 1e9
        device_used = device_total - device_free
        print(f"GPU Memory: {device_used:.2f} GB / {device_total:.2f} GB used")
    else:
        print("GPU not available")

# Export utilities
__all__ = [
    'to_gpu',
    'to_cpu',
    'print_gpu_memory',
    'GPU_AVAILABLE',
] + [name for name in dir(gsc) if not name.startswith('_')]

if GPU_AVAILABLE:
    print("gsc_gpu module loaded (GPU detected but using CPU for computation)")
else:
    print("gsc_gpu module loaded (No GPU, using CPU)")
