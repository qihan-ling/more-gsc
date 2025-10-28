"""
Simplified GPU wrapper - uses CPU for initialization, GPU for computation

This avoids compatibility issues by doing all parsing/setup on CPU,
then moving heavy arrays to GPU for computation.
"""

import sys
import warnings

# Try to import CuPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU (CuPy) detected and enabled")
    print(f"  Device: {cp.cuda.Device()}")
    print(f"  Memory: {cp.cuda.Device().mem_info[1] / 1e9:.2f} GB total")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    warnings.warn("CuPy not available, falling back to NumPy (CPU)")

import numpy as np

# Import gsc normally (uses CPU NumPy)
import gsc

# Re-export everything from gsc
from gsc import *

def to_gpu(arr):
    """Move numpy array to GPU"""
    if GPU_AVAILABLE and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr

def to_cpu(arr):
    """Move cupy array to CPU"""
    if GPU_AVAILABLE and hasattr(arr, '__cuda_array_interface__'):
        return cp.asnumpy(arr)
    return arr

# Patch GscNet to move arrays to GPU after initialization
if GPU_AVAILABLE:
    _original_GscNet_init = gsc.GscNet.__init__

    def gpu_init(self, *args, **kwargs):
        # Initialize on CPU
        _original_GscNet_init(self, *args, **kwargs)

        # Move large arrays to GPU
        print("  Moving arrays to GPU...", end='', flush=True)
        for attr in ['WC', 'bC', 'S']:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if isinstance(val, np.ndarray):
                    setattr(self, attr, cp.asarray(val))
        print(" done")

    gsc.GscNet.__init__ = gpu_init

    # Patch methods that create arrays to use GPU
    _original_reset = gsc.GscNet.reset

    def gpu_reset(self, mu=None, sd=0.):
        _original_reset(self, mu=mu, sd=sd)
        # Move state arrays to GPU
        for attr in ['actC', 'q', 'extC', 'scale_constants']:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if isinstance(val, np.ndarray):
                    setattr(self, attr, cp.asarray(val))

    gsc.GscNet.reset = gpu_reset

print(f"gsc_gpu_simple module loaded (GPU: {'ON' if GPU_AVAILABLE else 'OFF'})")
