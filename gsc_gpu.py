"""
GPU-accelerated wrapper for gsc.py using CuPy

This module provides a drop-in replacement for gsc that uses GPU acceleration
via CuPy. It monkey-patches numpy with cupy in the gsc module.

Usage:
    import gsc_gpu as gsc  # Use GPU
    # OR
    import gsc  # Use CPU (original)
"""

import sys
import importlib
import warnings

# Try to import CuPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU (CuPy) detected and enabled")
    print(f"  Device: {cp.cuda.Device()}")
    print(f"  Memory: {cp.cuda.Device().mem_info[1] / 1e9:.2f} GB total")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    warnings.warn("CuPy not available, falling back to NumPy (CPU). Install with: pip install cupy-cuda11x")

# Import numpy for operations that must stay on CPU
import numpy as np_cpu

# Create a hybrid module that uses NumPy for some operations, CuPy for others
class HybridNumPy:
    """
    Hybrid NumPy/CuPy module that uses GPU for heavy operations
    but keeps light operations on CPU to avoid transfer overhead
    """
    def __init__(self, gpu_module, cpu_module):
        self._gpu = gpu_module
        self._cpu = cpu_module
        self._GPU_AVAILABLE = GPU_AVAILABLE

    def __getattr__(self, name):
        # Keep random number generation on CPU for compatibility
        if name == 'random':
            return self._cpu.random
        # Keep these light operations on CPU
        elif name in ['array', 'zeros', 'ones', 'eye', 'arange', 'linspace']:
            # For array creation, use CPU then convert to GPU as needed
            return getattr(self._cpu, name)
        # Use GPU for heavy operations
        else:
            if self._GPU_AVAILABLE:
                return getattr(self._gpu, name)
            else:
                return getattr(self._cpu, name)

# Create hybrid numpy
if GPU_AVAILABLE:
    np_hybrid = HybridNumPy(cp, np_cpu)
else:
    np_hybrid = np_cpu

# Now import gsc and monkey-patch it to use hybrid NumPy
import gsc

# Store original numpy reference
gsc._np_original = gsc.np

# Replace numpy with hybrid module in gsc
gsc.np = np_hybrid

# Re-export all gsc classes and functions
from gsc import *

# Add GPU-specific utilities
class GPUContext:
    """Context manager for GPU operations with automatic memory management"""

    def __init__(self, device_id=0):
        self.device_id = device_id
        if GPU_AVAILABLE:
            self.device = cp.cuda.Device(device_id)

    def __enter__(self):
        if GPU_AVAILABLE:
            self.device.use()
            self.mempool = cp.get_default_memory_pool()
            self.pinned_mempool = cp.get_default_pinned_memory_pool()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if GPU_AVAILABLE:
            # Free unused GPU memory
            self.mempool.free_all_blocks()
            self.pinned_mempool.free_all_blocks()

def to_gpu(arr):
    """Move numpy array to GPU"""
    if GPU_AVAILABLE and isinstance(arr, np_cpu.ndarray):
        return cp.asarray(arr)
    return arr

def to_cpu(arr):
    """Move cupy array to CPU"""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if GPU_AVAILABLE:
        mempool = cp.get_default_memory_pool()
        used = mempool.used_bytes() / 1e9
        total = mempool.total_bytes() / 1e9
        device_info = cp.cuda.Device().mem_info
        device_free = device_info[0] / 1e9
        device_total = device_info[1] / 1e9
        return {
            'pool_used_gb': used,
            'pool_total_gb': total,
            'device_free_gb': device_free,
            'device_total_gb': device_total,
            'device_used_gb': device_total - device_free
        }
    return None

def print_gpu_memory():
    """Print GPU memory usage"""
    if GPU_AVAILABLE:
        info = get_gpu_memory_info()
        print(f"GPU Memory: {info['device_used_gb']:.2f} GB / {info['device_total_gb']:.2f} GB used")
        print(f"  Pool: {info['pool_used_gb']:.2f} GB used, {info['pool_total_gb']:.2f} GB allocated")
    else:
        print("GPU not available")

# Override pickle save/load to handle GPU arrays
_original_save_model = gsc.save_model
_original_load_model = gsc.load_model

def save_model(net, filename):
    """Save model, converting GPU arrays to CPU first"""
    if GPU_AVAILABLE:
        # Temporarily convert arrays to CPU for pickling
        with GPUContext():
            # Store original arrays
            arrays_to_restore = {}
            for attr in ['WC', 'bC', 'actC', 'extC', 'ep', 'q', 'scale_constants']:
                if hasattr(net, attr):
                    val = getattr(net, attr)
                    if isinstance(val, cp.ndarray):
                        arrays_to_restore[attr] = val
                        setattr(net, attr, to_cpu(val))

            # Save with CPU arrays
            _original_save_model(net, filename)

            # Restore GPU arrays
            for attr, val in arrays_to_restore.items():
                setattr(net, attr, val)
    else:
        _original_save_model(net, filename)

def load_model(filename, use_gpu=True):
    """Load model and optionally move to GPU"""
    net = _original_load_model(filename)

    if GPU_AVAILABLE and use_gpu:
        # Move arrays to GPU
        with GPUContext():
            for attr in ['WC', 'bC', 'actC', 'extC', 'ep', 'q', 'scale_constants']:
                if hasattr(net, attr):
                    val = getattr(net, attr)
                    if isinstance(val, np_cpu.ndarray):
                        setattr(net, attr, to_gpu(val))

    return net

# Monkey-patch the save/load functions
gsc.save_model = save_model
gsc.load_model = load_model

# Performance profiling utilities
class GPUProfiler:
    """Simple GPU profiler for benchmarking"""

    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        if GPU_AVAILABLE:
            cp.cuda.Stream.null.synchronize()
        self.start_time = self._get_time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if GPU_AVAILABLE:
            cp.cuda.Stream.null.synchronize()
        self.end_time = self._get_time()
        elapsed = self.elapsed()
        print(f"{self.name}: {elapsed:.3f} seconds")

    def _get_time(self):
        import time
        return time.perf_counter()

    def elapsed(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

# Export GPU utilities
__all__ = [
    'GPUContext',
    'to_gpu',
    'to_cpu',
    'get_gpu_memory_info',
    'print_gpu_memory',
    'GPUProfiler',
    'GPU_AVAILABLE',
    'save_model',
    'load_model',
] + [name for name in dir(gsc) if not name.startswith('_')]

print(f"gsc_gpu module loaded (GPU: {'ON' if GPU_AVAILABLE else 'OFF'})")
